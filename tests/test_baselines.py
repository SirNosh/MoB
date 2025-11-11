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

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob import (
    PerBatchVCGAuction,
    ExpertPool,
    SimpleCNN,
    create_model
)
from mob.baselines import (
    NaiveFineTuning,
    RandomAssignment,
    MonolithicEWC,
    GatedMoE
)


def create_split_mnist(num_tasks: int = 5, train: bool = True, batch_size: int = 32):
    """Create Split-MNIST dataset."""
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

        indices = [
            i for i, (_, label) in enumerate(dataset)
            if start_digit <= label < end_digit
        ]

        task_dataset = Subset(dataset, indices)
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
        'dropout': 0.5
    }

    pool = ExpertPool(config['num_experts'], expert_config, device=device)
    auction = PerBatchVCGAuction(config['num_experts'])

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=config['learning_rate'])
        for expert in pool.experts
    ]

    # Track metrics
    task_accuracies = []
    final_accuracies = []

    # Training
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training...")

        for x, y in tqdm(task_loader, desc=f"Task {task_id + 1}", leave=False):
            bids, _ = pool.collect_bids(x, y)
            winner_id, payment, _ = auction.run_auction(bids)
            pool.train_winner(winner_id, x, y, optimizers)

        # Update EWC
        pool.update_after_task(task_loader, num_samples=200)

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
        'forgetting': forgetting
    }


def run_naive_baseline(train_tasks, test_tasks, config):
    """Run Naive Fine-tuning baseline."""
    print("\n" + "="*60)
    print("Running Baseline 1: Naive Fine-tuning")
    print("="*60)

    device = torch.device(config['device'])
    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    baseline = NaiveFineTuning(model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    task_accuracies = []
    final_accuracies = []

    # Training
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training...")

        for x, y in tqdm(task_loader, desc=f"Task {task_id + 1}", leave=False):
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
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training...")

        baseline.train_on_task(task_loader, optimizers)
        baseline.update_after_task(task_loader, num_samples=200)

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
    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    baseline = MonolithicEWC(model, lambda_ewc=config['lambda_ewc'], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    task_accuracies = []
    final_accuracies = []

    # Training
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training...")

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
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{len(train_tasks)}] Training...")

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


def main():
    """Run comprehensive baseline comparison."""
    print("="*60)
    print("MoB Baseline Validation Experiment")
    print("="*60)

    # Configuration
    config = {
        'num_experts': 4,
        'num_tasks': 5,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 5000,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32
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

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/baseline_results.json")

    # Generate visualizations
    plot_comparison(results)

    return results


if __name__ == '__main__':
    results = main()
