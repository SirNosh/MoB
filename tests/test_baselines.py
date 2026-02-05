"""
Comprehensive baseline comparison for MoB validation.

This script runs all 4 baselines plus MoB on Split-MNIST and compares:
1. Naive Fine-tuning (lower bound)
2. Random Assignment (tests auction intelligence)
3. Monolithic EWC (tests architectural benefit)
4. Gated MoE (knockout test - gater forgetting)
5. MoB (our approach)

Success criteria: MoB > Monolithic EWC > Gated MoE >= Random Assignment >> Naive
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys
import json
from collections import defaultdict
from scipy import stats

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob import (
    PerBatchVCGAuction,
    ExpertPool,
    SimpleCNN,
    create_model,
    set_seed,
    BidLogger
)
from mob.baselines import (
    NaiveFineTuning,
    RandomAssignment,
    MonolithicEWC,
    GatedMoE
)


def create_split_mnist(num_tasks: int = 5, train: bool = True, batch_size: int = 32, replay_ratio: float = 0.0):
    """
    Create Split-MNIST dataset with replay mechanism to prevent catastrophic forgetting.

    Parameters:
    -----------
    num_tasks : int
        Number of tasks to split MNIST into
    train : bool
        Whether to use training or test set
    batch_size : int
        Batch size for DataLoader
    replay_ratio : float
        Ratio of previous task samples to include in current task (default 0.2 = 20%)

    Returns:
    --------
    tasks : list[DataLoader]
        List of DataLoaders, one for each task

    Note:
    -----
    This implementation includes replay from previous tasks to ensure that:
    1. All output neurons get trained throughout learning
    2. Models don't catastrophically forget by overwriting output weights
    3. Fair comparison between methods (all see same data distribution)
    """
    import random

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    tasks = []
    digits_per_task = 10 // num_tasks

    for task_id in range(num_tasks):
        start_digit = task_id * digits_per_task
        end_digit = start_digit + digits_per_task

        # Get current task samples
        current_indices = [
            i for i, (_, label) in enumerate(dataset)
            if start_digit <= label < end_digit
        ]

        # For tasks after the first, add replay samples from previous tasks
        if task_id > 0:
            # Get samples from all previous classes
            previous_indices = [
                i for i, (_, label) in enumerate(dataset)
                if label < start_digit
            ]

            # Randomly sample replay_ratio of current task size from previous tasks
            random.shuffle(previous_indices)
            replay_count = int(len(current_indices) * replay_ratio)
            replay_indices = previous_indices[:replay_count]

            # Combine current task samples with replay samples
            task_indices = current_indices + replay_indices
            random.shuffle(task_indices)  # Shuffle to mix current and replay samples
        else:
            task_indices = current_indices

        task_dataset = Subset(dataset, task_indices)
        task_loader = DataLoader(
            task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        tasks.append(task_loader)

    return tasks


def run_mob_experiment(train_tasks, test_tasks, config):
    """Run MoB experiment."""
    print("\n" + "="*60)
    print("Running MoB (Mixture of Bidders)")
    print("="*60)

    device = torch.device(config['device'])

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': config['alpha'],
        'beta': config['beta'],
        'lambda_ewc': config['lambda_ewc'],
        'forgetting_cost_scale': config['forgetting_cost_scale'],
        'dropout': 0.5
    }

    pool = ExpertPool(config['num_experts'], expert_config, device=device)
    auction = PerBatchVCGAuction(config['num_experts'])

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=config['learning_rate'])
        for expert in pool.experts
    ]

    # Create bid logger (stores alpha/beta once, not per-batch)
    bid_logger = BidLogger(
        num_experts=config['num_experts'],
        alpha=config['alpha'],
        beta=config['beta'],
        log_file=None  # Don't auto-save during training
    )

    # Track metrics
    task_accuracies = []
    final_accuracies = []

    # Training
    global_batch_idx = 0
    epochs_per_task = config.get('epochs_per_task', 1)
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training for {epochs_per_task} epoch(s)...")
        
        # Reset per-task statistics for proper logging
        for expert in pool.experts:
            expert.reset_task_statistics()
        
        winners_this_task = set()
        for epoch in range(epochs_per_task):
            for x, y in tqdm(task_loader, desc=f"Task {task_id + 1} Epoch {epoch + 1}/{epochs_per_task}", leave=False):
                # Collect bids with components
                bids, components = pool.collect_bids(x, y)

                # Run auction
                winner_id, payment, _ = auction.run_auction(bids)
                
                winners_this_task.add(winner_id)

                # Log bids for diagnostics
                bid_logger.log_batch(
                    batch_idx=global_batch_idx,
                    bids=bids,
                    components=components,
                    winner_id=winner_id,
                    task_id=task_id
                )

                # Train winner
                pool.train_winner(winner_id, x, y, optimizers)

                global_batch_idx += 1

        # Update EWC
        print(f"\nUpdating Fisher matrices for task specialists: {sorted(list(winners_this_task))}")
        for expert_id in winners_this_task:
            pool.experts[expert_id].update_after_task(task_loader, num_samples=200)

        # Evaluate on current task
        results = pool.evaluate_all(test_tasks[task_id])
        task_accuracies.append(results['ensemble_accuracy'])

    # Final evaluation on all tasks
    for task_id, test_loader in enumerate(test_tasks):
        results = pool.evaluate_all(test_loader)
        final_accuracies.append(results['ensemble_accuracy'])
        print(f"Task {task_id + 1} Final Accuracy: {results['ensemble_accuracy']:.4f}")

    avg_accuracy = np.mean(final_accuracies)
    forgetting = np.mean([
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]) if len(final_accuracies) > 1 else 0.0

    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average Forgetting: {forgetting:.4f}")

    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': forgetting,
        'bid_logger': bid_logger  # Include bid logger for diagnostics
    }


def run_naive_baseline(train_tasks, test_tasks, config):
    """Run Naive Fine-tuning baseline."""
    print("\n" + "="*60)
    print("Running Baseline 1: Naive Fine-tuning")
    print("="*60)

    device = torch.device(config['device'])
    # Use width_multiplier=2 to approximately match 4-expert systems' total parameter count
    # (width_multiplier scales both conv and fc layers, so grows faster than linear)
    model = create_model('simple_cnn', num_classes=10, input_channels=1, width_multiplier=2)
    baseline = NaiveFineTuning(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    task_accuracies = []
    final_accuracies = []

    # Training
    epochs_per_task = config.get('epochs_per_task', 1)
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training for {epochs_per_task} epoch(s)...")

        for epoch in range(epochs_per_task):
            for x, y in tqdm(task_loader, desc=f"Task {task_id + 1} Epoch {epoch + 1}/{epochs_per_task}", leave=False):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

        # Evaluate on current task
        results = baseline.evaluate(test_tasks[task_id])
        task_accuracies.append(results['accuracy'])

    # Final evaluation
    for task_id, test_loader in enumerate(test_tasks):
        results = baseline.evaluate(test_loader)
        final_accuracies.append(results['accuracy'])
        print(f"Task {task_id + 1} Final Accuracy: {results['accuracy']:.4f}")

    avg_accuracy = np.mean(final_accuracies)
    forgetting = np.mean([
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]) if len(final_accuracies) > 1 else 0.0

    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average Forgetting: {forgetting:.4f}")

    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': forgetting
    }


def run_random_baseline(train_tasks, test_tasks, config):
    """Run Random Assignment baseline."""
    print("\n" + "="*60)
    print("Running Baseline 2: Random Assignment")
    print("="*60)

    device = torch.device(config['device'])

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': config['alpha'],
        'beta': config['beta'],
        'lambda_ewc': config['lambda_ewc'],
        'forgetting_cost_scale': config.get('forgetting_cost_scale', 1.0),
        'dropout': 0.5
    }

    baseline = RandomAssignment(config['num_experts'], expert_config, device=device)

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=config['learning_rate'])
        for expert in baseline.experts
    ]

    task_accuracies = []
    final_accuracies = []

    # Training
    epochs_per_task = config.get('epochs_per_task', 1)
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training for {epochs_per_task} epoch(s)...")

        # Track which experts trained on this task (Issue 8 fix)
        trained_experts = set()
        for epoch in range(epochs_per_task):
            metrics = baseline.train_on_task(task_loader, optimizers)
            # Track experts that received batches
            for i, usage in enumerate(metrics.get('expert_usage', [])):
                if usage > 0:
                    trained_experts.add(i)

        # Only update Fisher for experts that actually trained (matches MoB behavior)
        print(f"Updating Fisher for trained experts: {sorted(list(trained_experts))}")
        for expert_id in trained_experts:
            baseline.experts[expert_id].update_after_task(task_loader, num_samples=200)

        # Evaluate on current task
        results = baseline.evaluate_all(test_tasks[task_id])
        task_accuracies.append(results['ensemble_accuracy'])

    # Final evaluation
    for task_id, test_loader in enumerate(test_tasks):
        results = baseline.evaluate_all(test_loader)
        final_accuracies.append(results['ensemble_accuracy'])
        print(f"Task {task_id + 1} Final Accuracy: {results['ensemble_accuracy']:.4f}")

    avg_accuracy = np.mean(final_accuracies)
    forgetting = np.mean([
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]) if len(final_accuracies) > 1 else 0.0

    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average Forgetting: {forgetting:.4f}")

    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': forgetting
    }


def run_monolithic_ewc_baseline(train_tasks, test_tasks, config):
    """Run Monolithic EWC baseline."""
    print("\n" + "="*60)
    print("Running Baseline 3: Monolithic EWC")
    print("="*60)

    device = torch.device(config['device'])
    # Use width_multiplier=2 to approximately match 4-expert systems' total parameter count
    # (width_multiplier scales both conv and fc layers, so grows faster than linear)
    model = create_model('simple_cnn', num_classes=10, input_channels=1, width_multiplier=2)
    baseline = MonolithicEWC(model, lambda_ewc=config['lambda_ewc'],forgetting_cost_scale=config['forgetting_cost_scale'],device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    task_accuracies = []
    final_accuracies = []

    # Training
    epochs_per_task = config.get('epochs_per_task', 1)
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training for {epochs_per_task} epoch(s)...")

        for epoch in range(epochs_per_task):
            baseline.train_on_task(task_loader, optimizer)

        baseline.update_after_task(task_loader, num_samples=200)

        # Evaluate on current task
        results = baseline.evaluate(test_tasks[task_id])
        task_accuracies.append(results['accuracy'])

    # Final evaluation
    for task_id, test_loader in enumerate(test_tasks):
        results = baseline.evaluate(test_loader)
        final_accuracies.append(results['accuracy'])
        print(f"Task {task_id + 1} Final Accuracy: {results['accuracy']:.4f}")

    avg_accuracy = np.mean(final_accuracies)
    forgetting = np.mean([
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]) if len(final_accuracies) > 1 else 0.0

    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average Forgetting: {forgetting:.4f}")

    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': forgetting
    }


def run_gated_moe_baseline(train_tasks, test_tasks, config):
    """Run Gated MoE baseline."""
    print("\n" + "="*60)
    print("Running Baseline 4: Gated MoE (Knockout Test)")
    print("="*60)

    device = torch.device(config['device'])

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 784,  # 28*28
        'dropout': 0.5
    }

    baseline = GatedMoE(config['num_experts'], expert_config, device=device)

    expert_optimizers = [
        torch.optim.Adam(expert.parameters(), lr=config['learning_rate'])
        for expert in baseline.expert_models
    ]
    gater_optimizer = torch.optim.Adam(baseline.gater.parameters(), lr=config['learning_rate'])

    task_accuracies = []
    final_accuracies = []

    # Training
    epochs_per_task = config.get('epochs_per_task', 1)
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training for {epochs_per_task} epoch(s)...")

        for epoch in range(epochs_per_task):
            baseline.train_on_task(task_loader, expert_optimizers, gater_optimizer)

        # Evaluate on current task
        results = baseline.evaluate_all(test_tasks[task_id])
        task_accuracies.append(results['gated_accuracy'])

    # Final evaluation
    for task_id, test_loader in enumerate(test_tasks):
        results = baseline.evaluate_all(test_loader)
        final_accuracies.append(results['gated_accuracy'])
        print(f"Task {task_id + 1} Final Accuracy: {results['gated_accuracy']:.4f}")

    avg_accuracy = np.mean(final_accuracies)
    forgetting = np.mean([
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]) if len(final_accuracies) > 1 else 0.0

    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average Forgetting: {forgetting:.4f}")

    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': forgetting
    }


def plot_comparison(results_dict, save_path='results/baseline_comparison.png'):
    """Create comprehensive comparison visualizations."""
    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results_dict.keys())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # Plot 1: Average Accuracy Comparison
    avg_accs = [results_dict[m]['avg_accuracy'] for m in methods]
    axes[0, 0].bar(range(len(methods)), avg_accs, color=colors[:len(methods)])
    axes[0, 0].set_xticks(range(len(methods)))
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Average Accuracy')
    axes[0, 0].set_title('Average Accuracy Across All Tasks')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.0])

    # Add value labels on bars
    for i, v in enumerate(avg_accs):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Forgetting Comparison
    forgetting = [results_dict[m]['forgetting'] for m in methods]
    axes[0, 1].bar(range(len(methods)), forgetting, color=colors[:len(methods)])
    axes[0, 1].set_xticks(range(len(methods)))
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Average Forgetting')
    axes[0, 1].set_title('Catastrophic Forgetting (Lower is Better)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(forgetting):
        axes[0, 1].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Per-Task Final Accuracy
    for i, method in enumerate(methods):
        final_accs = results_dict[method]['final_accuracies']
        axes[1, 0].plot(range(1, len(final_accs) + 1), final_accs,
                        marker='o', label=method, color=colors[i], linewidth=2)

    axes[1, 0].set_xlabel('Task ID')
    axes[1, 0].set_ylabel('Final Accuracy')
    axes[1, 0].set_title('Final Accuracy per Task')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.0])

    # Plot 4: Success Criteria Check
    expected_order = ['MoB', 'Monolithic EWC', 'Gated MoE', 'Random Assignment', 'Naive']
    actual_ranking = sorted(methods, key=lambda m: results_dict[m]['avg_accuracy'], reverse=True)

    axes[1, 1].text(0.1, 0.9, 'Success Criteria:', fontsize=14, fontweight='bold',
                    transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, 'Expected: MoB > Monolithic EWC > Gated MoE ≥ Random > Naive',
                    fontsize=10, transform=axes[1, 1].transAxes)

    axes[1, 1].text(0.1, 0.65, 'Actual Ranking:', fontsize=12, fontweight='bold',
                    transform=axes[1, 1].transAxes)

    for i, method in enumerate(actual_ranking):
        acc = results_dict[method]['avg_accuracy']
        y_pos = 0.55 - i * 0.08
        axes[1, 1].text(0.1, y_pos, f'{i+1}. {method}: {acc:.4f}',
                        fontsize=10, transform=axes[1, 1].transAxes)

    # Check success
    mob_best = actual_ranking[0] == 'MoB'
    axes[1, 1].text(0.1, 0.15, f'✓ MoB is best: {mob_best}' if mob_best else f'✗ MoB is NOT best',
                    fontsize=11, color='green' if mob_best else 'red',
                    fontweight='bold', transform=axes[1, 1].transAxes)

    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def main(seed=None):
    """Run comprehensive baseline comparison with a specific seed.

    Parameters:
    -----------
    seed : int, optional
        Random seed for reproducibility. If None, no seed is set.

    Returns:
    --------
    results : dict
        Dictionary of results for all methods.
    """
    if seed is not None:
        set_seed(seed)
        print("="*60)
        print(f"MoB Baseline Validation Experiment (Seed {seed})")
        print("="*60)
    else:
        print("="*60)
        print("MoB Baseline Validation Experiment")
        print("="*60)

    # Configuration
    config = {
        'num_experts': 4,
        'num_tasks': 5,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 10.0,  # EWC regularization strength
        'learning_rate': 0.001,
        'forgetting_cost_scale': 1.0,  # Not needed with bid normalization
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'epochs_per_task': 4  # Number of epochs to train on each task
    }

    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create datasets
    print("\nCreating Split-MNIST datasets...")
    train_tasks = create_split_mnist(config['num_tasks'], train=True, batch_size=config['batch_size'])
    test_tasks = create_split_mnist(config['num_tasks'], train=False, batch_size=config['batch_size'])

    # Run all experiments
    results = {}

    # 1. Naive (should be worst)
    results['Naive'] = run_naive_baseline(train_tasks, test_tasks, config)

    # 2. Random Assignment (should be better than naive)
    results['Random Assignment'] = run_random_baseline(train_tasks, test_tasks, config)

    # 3. Gated MoE (knockout test - should show gater forgetting)
    results['Gated MoE'] = run_gated_moe_baseline(train_tasks, test_tasks, config)

    # 4. Monolithic EWC (strong baseline)
    results['Monolithic EWC'] = run_monolithic_ewc_baseline(train_tasks, test_tasks, config)

    # 5. MoB (should be best!)
    results['MoB'] = run_mob_experiment(train_tasks, test_tasks, config)

    # Print bid diagnostics if available
    if 'bid_logger' in results['MoB']:
        print("\n" + "="*80)
        print("BID DIAGNOSTICS FOR MoB")
        print("="*80)
        results['MoB']['bid_logger'].print_diagnostics()

        # Save bid logs
        results['MoB']['bid_logger'].save_logs("mob_bid_diagnostics.json")

        # Create visualization if possible
        try:
            results['MoB']['bid_logger'].plot_bid_components(save_path="mob_bid_components.png")
        except Exception:
            pass  # Matplotlib not available

    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    ranking = sorted(results.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)

    print("\nRanking by Average Accuracy:")
    for rank, (method, metrics) in enumerate(ranking, 1):
        print(f"{rank}. {method:20s} - Avg Acc: {metrics['avg_accuracy']:.4f}, "
              f"Forgetting: {metrics['forgetting']:.4f}")

    # Check success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    print("Expected: MoB > Monolithic EWC > Gated MoE ≥ Random Assignment >> Naive")

    mob_acc = results['MoB']['avg_accuracy']
    mono_acc = results['Monolithic EWC']['avg_accuracy']
    gated_acc = results['Gated MoE']['avg_accuracy']
    random_acc = results['Random Assignment']['avg_accuracy']
    naive_acc = results['Naive']['avg_accuracy']

    checks = [
        ("MoB > Monolithic EWC", mob_acc > mono_acc),
        ("MoB > Gated MoE", mob_acc > gated_acc),
        ("MoB > Random Assignment", mob_acc > random_acc),
        ("MoB >> Naive", mob_acc > naive_acc + 0.05),
        ("Monolithic EWC > Naive", mono_acc > naive_acc),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Phase 1 validation successful!")
    else:
        print("\n⚠️  Some checks failed. Review results.")

    return results


def compute_statistics(all_results):
    """
    Compute statistics across multiple seeds.

    Parameters:
    -----------
    all_results : list[dict]
        List of results dictionaries from multiple seeds.

    Returns:
    --------
    stats : dict
        Statistics for each method.
    """
    methods = list(all_results[0].keys())
    stats = {}

    for method in methods:
        # Collect metrics across seeds
        avg_accs = [r[method]['avg_accuracy'] for r in all_results]
        forgetting = [r[method]['forgetting'] for r in all_results]

        # Final accuracies per task (matrix: seeds x tasks)
        final_accs_matrix = [r[method]['final_accuracies'] for r in all_results]

        stats[method] = {
            'avg_accuracy': {
                'mean': np.mean(avg_accs),
                'std': np.std(avg_accs, ddof=1),  # Sample std
                'min': np.min(avg_accs),
                'max': np.max(avg_accs),
                'values': avg_accs
            },
            'forgetting': {
                'mean': np.mean(forgetting),
                'std': np.std(forgetting, ddof=1),
                'min': np.min(forgetting),
                'max': np.max(forgetting),
                'values': forgetting
            },
            'final_accuracies': {
                'mean': np.mean(final_accs_matrix, axis=0).tolist(),
                'std': np.std(final_accs_matrix, axis=0, ddof=1).tolist()
            }
        }

    return stats


def perform_significance_tests(all_results, baseline_method='MoB'):
    """
    Perform statistical significance tests comparing baseline_method to others.

    Uses paired t-test to determine if differences are statistically significant.

    Parameters:
    -----------
    all_results : list[dict]
        List of results from multiple seeds.
    baseline_method : str
        Method to compare against (default: 'MoB').

    Returns:
    --------
    significance_results : dict
        Dictionary of p-values and significance indicators.
    """
    methods = [m for m in list(all_results[0].keys()) if m != baseline_method]
    baseline_accs = [r[baseline_method]['avg_accuracy'] for r in all_results]

    significance_results = {}

    for method in methods:
        method_accs = [r[method]['avg_accuracy'] for r in all_results]

        # Paired t-test (since same seeds used for both methods)
        t_stat, p_value = stats.ttest_rel(baseline_accs, method_accs)

        # Determine significance level
        if p_value < 0.001:
            significance = "***"  # Very significant
        elif p_value < 0.01:
            significance = "**"   # Significant
        elif p_value < 0.05:
            significance = "*"    # Marginally significant
        else:
            significance = "ns"   # Not significant

        significance_results[method] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance,
            'better_than_baseline': t_stat < 0  # negative t means baseline > method
        }

    return significance_results


def run_multi_seed_experiments(num_seeds=5):
    """
    Run baseline comparison experiments with multiple random seeds.

    This provides statistical validity by:
    - Estimating true performance (mean across seeds)
    - Measuring stability (std across seeds)
    - Testing significance (paired t-tests)

    Parameters:
    -----------
    num_seeds : int
        Number of random seeds to run (default: 5).
        Recommended: 5-10 for quick validation, 20-30 for publication.

    Returns:
    --------
    all_results : list[dict]
        Results from each seed.
    statistics : dict
        Aggregated statistics.
    significance : dict
        Statistical significance tests.
    """
    print("="*70)
    print(f"  MULTI-SEED BASELINE COMPARISON ({num_seeds} seeds)")
    print("="*70)
    print("\nWhy multiple seeds?")
    print("  - Neural network training is stochastic (random)")
    print("  - Single run can be lucky or unlucky")
    print("  - Multiple seeds estimate true performance")
    print("  - Standard deviation shows stability")
    print("  - Statistical tests show significance")
    print("="*70)

    # Generate seeds
    seeds = [42 + i for i in range(num_seeds)]

    all_results = []

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"SEED {seed_idx}/{num_seeds}: {seed}")
        print(f"{'='*70}")

        # Run experiment with this seed
        results = main(seed=seed)
        all_results.append(results)

        # Show quick summary
        print(f"\nSeed {seed} Results:")
        for method, metrics in results.items():
            print(f"  {method:20s}: Acc={metrics['avg_accuracy']:.4f}, "
                  f"Forget={metrics['forgetting']:.4f}")

    # Compute statistics
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    statistics = compute_statistics(all_results)

    print("\n📊 Results Across Seeds (Mean ± Std):")
    print("-" * 70)
    for method in sorted(statistics.keys(), key=lambda m: statistics[m]['avg_accuracy']['mean'], reverse=True):
        acc_mean = statistics[method]['avg_accuracy']['mean']
        acc_std = statistics[method]['avg_accuracy']['std']
        forg_mean = statistics[method]['forgetting']['mean']
        forg_std = statistics[method]['forgetting']['std']

        print(f"{method:20s}: Acc={acc_mean:.4f}±{acc_std:.4f}, "
              f"Forget={forg_mean:.4f}±{forg_std:.4f}")

    # Statistical significance tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE (vs MoB)")
    print("="*70)
    print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("-" * 70)

    significance = perform_significance_tests(all_results, baseline_method='MoB')

    for method, sig_results in significance.items():
        p_val = sig_results['p_value']
        sig_marker = sig_results['significance']
        better = "MoB > " if sig_results['better_than_baseline'] else "MoB ≤ "

        print(f"{method:20s}: p={p_val:.4f} {sig_marker:3s} ({better}{method})")

    # Save aggregated results
    os.makedirs('results', exist_ok=True)

    # Save raw results from all seeds
    with open('results/multi_seed_raw_results.json', 'w') as f:
        json.dump({
            'seeds': seeds,
            'num_seeds': num_seeds,
            'results_per_seed': all_results
        }, f, indent=2)

    # Save statistics
    with open('results/multi_seed_statistics.json', 'w') as f:
        json.dump({
            'statistics': statistics,
            'significance_tests': significance,
            'seeds': seeds,
            'num_seeds': num_seeds
        }, f, indent=2)

    print("\n📁 Results saved:")
    print("  - results/multi_seed_raw_results.json (all seeds)")
    print("  - results/multi_seed_statistics.json (statistics)")

    # Generate visualization with mean and std
    print("\n📊 Generating visualizations...")
    plot_multi_seed_comparison(statistics, save_path='results/multi_seed_comparison.png')

    # Success criteria check
    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECK (Mean Performance)")
    print("="*70)

    mob_acc = statistics['MoB']['avg_accuracy']['mean']
    mono_acc = statistics['Monolithic EWC']['avg_accuracy']['mean']
    gated_acc = statistics['Gated MoE']['avg_accuracy']['mean']
    random_acc = statistics['Random Assignment']['avg_accuracy']['mean']
    naive_acc = statistics['Naive']['avg_accuracy']['mean']

    checks = [
        ("MoB > Monolithic EWC", mob_acc > mono_acc,
         significance.get('Monolithic EWC', {}).get('significance', 'ns')),
        ("MoB > Gated MoE", mob_acc > gated_acc,
         significance.get('Gated MoE', {}).get('significance', 'ns')),
        ("MoB > Random Assignment", mob_acc > random_acc,
         significance.get('Random Assignment', {}).get('significance', 'ns')),
        ("MoB >> Naive", mob_acc > naive_acc + 0.05,
         significance.get('Naive', {}).get('significance', 'ns')),
    ]

    all_passed = True
    for check_name, passed, sig in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name} (p-value {sig})")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Phase 1 validation successful with statistical significance!")
    else:
        print("\n⚠️  Some checks failed. Review results.")

    return all_results, statistics, significance


def plot_multi_seed_comparison(statistics, save_path='results/multi_seed_comparison.png'):
    """
    Create visualization showing mean ± std across seeds.

    Parameters:
    -----------
    statistics : dict
        Statistics computed from compute_statistics().
    save_path : str
        Path to save figure.
    """
    methods = list(statistics.keys())
    method_names = list(statistics.keys())

    # Sort by mean accuracy
    method_names = sorted(method_names,
                         key=lambda m: statistics[m]['avg_accuracy']['mean'],
                         reverse=True)

    # Extract data
    avg_accs = [statistics[m]['avg_accuracy']['mean'] for m in method_names]
    avg_stds = [statistics[m]['avg_accuracy']['std'] for m in method_names]
    forg_means = [statistics[m]['forgetting']['mean'] for m in method_names]
    forg_stds = [statistics[m]['forgetting']['std'] for m in method_names]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Average Accuracy with error bars
    colors = ['#2ecc71' if m == 'MoB' else '#3498db' if m == 'Monolithic EWC'
              else '#e74c3c' if m == 'Naive' else '#95a5a6' for m in method_names]

    axes[0].barh(method_names, avg_accs, xerr=avg_stds, color=colors, alpha=0.7, capsize=5)
    axes[0].set_xlabel('Average Accuracy (Mean ± Std)', fontsize=12)
    axes[0].set_title('Method Performance Across Seeds', fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='Target: 0.9')
    axes[0].legend()

    # Right panel: Forgetting with error bars
    axes[1].barh(method_names, forg_means, xerr=forg_stds, color=colors, alpha=0.7, capsize=5)
    axes[1].set_xlabel('Average Forgetting (Mean ± Std)', fontsize=12)
    axes[1].set_title('Catastrophic Forgetting Across Seeds', fontsize=13, fontweight='bold')
    axes[1].axvline(x=0.05, color='gray', linestyle='--', alpha=0.5, label='Target: <0.05')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Multi-seed comparison plot saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MoB Baseline Validation')
    parser.add_argument('--seeds', type=int, default=1,
                       help='Number of random seeds to run (default: 1, recommended: 5-10)')
    parser.add_argument('--single-seed', type=int, default=None,
                       help='Run with a specific seed (overrides --seeds)')

    args = parser.parse_args()

    if args.single_seed is not None:
        # Single seed run
        set_seed(args.single_seed)
        results = main(seed=args.single_seed)

        # Save and visualize
        os.makedirs('results', exist_ok=True)
        
        # Remove non-serializable objects before saving
        serializable_results = {}
        for method, data in results.items():
            serializable_results[method] = {
                k: v for k, v in data.items() 
                if k != 'bid_logger'  # BidLogger is not JSON serializable
            }
        
        with open('results/baseline_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print("\nResults saved to results/baseline_results.json")
        plot_comparison(results)

    elif args.seeds > 1:
        # Multi-seed run
        all_results, statistics, significance = run_multi_seed_experiments(num_seeds=args.seeds)

    else:
        # Single run without seed
        results = main()

        # Save and visualize
        os.makedirs('results', exist_ok=True)
        with open('results/baseline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to results/baseline_results.json")
        plot_comparison(results)
