"""
Test script for MoB Phase 1 on Split-MNIST.

This script validates the core components of the MoB system:
- VCG auction mechanism
- Expert bidding and training
- EWC forgetting prevention
- Expert specialization on Split-MNIST
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob import (
    PerBatchVCGAuction,
    ExpertPool,
    SimpleCNN
)


def create_split_mnist(num_tasks: int = 5, train: bool = True, batch_size: int = 32, replay_ratio: float = 0.2):
    """
    Create Split-MNIST dataset with replay mechanism to prevent catastrophic forgetting.

    Splits MNIST into multiple tasks, each containing a subset of digits.
    For tasks after the first, includes replay samples from previous tasks.

    Parameters:
    -----------
    num_tasks : int
        Number of tasks to create (default: 5, each with 2 digits).
    train : bool
        Whether to use training or test set.
    batch_size : int
        Batch size for DataLoader (default: 32).
    replay_ratio : float
        Ratio of previous task samples to include in current task (default 0.2 = 20%).

    Returns:
    --------
    tasks : list of DataLoader
        List of DataLoaders, one per task.

    Note:
    -----
    This implementation includes replay from previous tasks to ensure that:
    1. All output neurons get trained throughout learning
    2. Models don't catastrophically forget by overwriting output weights
    3. Fair evaluation of continual learning performance
    """
    import random

    # Download MNIST
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

    # Split into tasks
    tasks = []
    digits_per_task = 10 // num_tasks

    for task_id in range(num_tasks):
        # Get indices for this task's digits
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


def compute_specialization_metrics(win_history: list, num_experts: int):
    """
    Analyzes the auction win history to quantify expert specialization.

    Parameters:
    -----------
    win_history : list
        List of winner IDs from auction history.
    num_experts : int
        Total number of experts.

    Returns:
    --------
    metrics : dict
        Specialization metrics.
    """
    win_counts = np.bincount(win_history, minlength=num_experts)
    win_probs = win_counts / win_counts.sum() if win_counts.sum() > 0 else win_counts

    # Shannon Entropy of Usage (lower = more specialized)
    entropy = -np.sum(win_probs * np.log2(win_probs + 1e-9))
    normalized_entropy = entropy / np.log2(num_experts) if num_experts > 1 else 0

    # Herfindahl-Hirschman Index (higher = more concentrated/specialized)
    hhi = np.sum(win_probs ** 2)

    return {
        'usage_entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'hhi': hhi,
        'expert_usage': win_probs
    }


def plot_specialization(win_history: list, num_experts: int, num_tasks: int, save_path: str = None):
    """
    Visualize expert specialization over time.

    Parameters:
    -----------
    win_history : list
        List of winner IDs.
    num_experts : int
        Number of experts.
    num_tasks : int
        Number of tasks.
    save_path : str, optional
        Path to save the plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Win history over time
    axes[0].plot(win_history, 'o', markersize=1, alpha=0.5)
    axes[0].set_xlabel('Batch Number')
    axes[0].set_ylabel('Winner Expert ID')
    axes[0].set_title('Expert Selection Over Time')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Expert usage distribution
    win_counts = np.bincount(win_history, minlength=num_experts)
    win_probs = win_counts / win_counts.sum()

    axes[1].bar(range(num_experts), win_probs)
    axes[1].set_xlabel('Expert ID')
    axes[1].set_ylabel('Usage Probability')
    axes[1].set_title('Expert Usage Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Specialization plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def run_mob_experiment(
    num_experts: int = 4,
    num_tasks: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    lambda_ewc: float = 5000,
    learning_rate: float = 0.001,
    device: str = 'cpu'
):
    """
    Run the MoB experiment on Split-MNIST.

    Parameters:
    -----------
    num_experts : int
        Number of expert agents.
    num_tasks : int
        Number of tasks in Split-MNIST.
    alpha : float
        Weight for execution cost.
    beta : float
        Weight for forgetting cost.
    lambda_ewc : float
        EWC regularization strength.
    learning_rate : float
        Learning rate for optimizers.
    device : str
        Device to run on ('cpu' or 'cuda').

    Returns:
    --------
    results : dict
        Experiment results.
    """
    device = torch.device(device)
    print(f"Running MoB experiment on {device}")
    print(f"Config: {num_experts} experts, {num_tasks} tasks, α={alpha}, β={beta}, λ={lambda_ewc}")

    # Create Split-MNIST
    print("\nCreating Split-MNIST dataset...")
    train_tasks = create_split_mnist(num_tasks=num_tasks, train=True)
    test_tasks = create_split_mnist(num_tasks=num_tasks, train=False)

    # Initialize expert pool
    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': alpha,
        'beta': beta,
        'lambda_ewc': lambda_ewc,
        'dropout': 0.5
    }

    pool = ExpertPool(num_experts, expert_config, device=device)
    auction = PerBatchVCGAuction(num_experts)

    # Create optimizers
    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=learning_rate)
        for expert in pool.experts
    ]

    # Training
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)

    win_history = []
    task_accuracies = []

    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n[Task {task_id + 1}/{num_tasks}] Training...")

        task_wins = []

        for x, y in tqdm(task_loader, desc=f"Task {task_id + 1}"):
            # Collect bids
            bids, components = pool.collect_bids(x, y)

            # Run auction
            winner_id, payment, metrics = auction.run_auction(bids)

            # Train winner
            pool.train_winner(winner_id, x, y, optimizers)

            # Track
            win_history.append(winner_id)
            task_wins.append(winner_id)

        # Update EWC after task
        print(f"Updating EWC Fisher matrices...")
        pool.update_after_task(task_loader, num_samples=200)

        # Evaluate on current task
        print(f"Evaluating on Task {task_id + 1}...")
        eval_results = pool.evaluate_all(test_tasks[task_id])
        task_accuracies.append(eval_results['ensemble_accuracy'])

        print(f"Task {task_id + 1} Ensemble Accuracy: {eval_results['ensemble_accuracy']:.4f}")

        # Task-specific specialization
        task_spec = compute_specialization_metrics(task_wins, num_experts)
        print(f"Task {task_id + 1} Specialization - HHI: {task_spec['hhi']:.4f}, "
              f"Normalized Entropy: {task_spec['normalized_entropy']:.4f}")

    # Final Evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    final_accuracies = []
    for task_id, test_loader in enumerate(test_tasks):
        results = pool.evaluate_all(test_loader)
        final_accuracies.append(results['ensemble_accuracy'])
        print(f"Task {task_id + 1} Final Accuracy: {results['ensemble_accuracy']:.4f}")

        # Individual expert accuracies
        expert_accs = [results[f'expert_{i}_accuracy'] for i in range(num_experts)]
        print(f"  Expert Accuracies: {[f'{acc:.3f}' for acc in expert_accs]}")

    avg_accuracy = np.mean(final_accuracies)
    print(f"\nAverage Accuracy Across All Tasks: {avg_accuracy:.4f}")

    # Compute forgetting
    if len(final_accuracies) > 1:
        forgetting = np.mean([
            max(0, task_accuracies[i] - final_accuracies[i])
            for i in range(len(final_accuracies) - 1)
        ])
        print(f"Average Forgetting: {forgetting:.4f}")
    else:
        forgetting = 0.0

    # Overall specialization
    overall_spec = compute_specialization_metrics(win_history, num_experts)
    print(f"\nOverall Specialization:")
    print(f"  HHI: {overall_spec['hhi']:.4f}")
    print(f"  Normalized Entropy: {overall_spec['normalized_entropy']:.4f}")
    print(f"  Expert Usage: {[f'{p:.3f}' for p in overall_spec['expert_usage']]}")

    # Expert statistics
    print("\nExpert Statistics:")
    for expert_stats in pool.get_expert_statistics():
        print(f"  Expert {expert_stats['expert_id']}: "
              f"Win Rate={expert_stats['win_rate']:.3f}, "
              f"Batches Won={expert_stats['batches_won']}")

    # Visualize
    print("\nGenerating visualizations...")
    os.makedirs('results', exist_ok=True)
    plot_specialization(win_history, num_experts, num_tasks, save_path='results/specialization.png')

    return {
        'avg_accuracy': avg_accuracy,
        'final_accuracies': final_accuracies,
        'forgetting': forgetting,
        'specialization': overall_spec,
        'win_history': win_history
    }


if __name__ == '__main__':
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run experiment
    results = run_mob_experiment(
        num_experts=4,
        num_tasks=5,
        alpha=0.5,
        beta=0.5,
        lambda_ewc=5000,
        learning_rate=0.001,
        device=device
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Final Results:")
    print(f"  Average Accuracy: {results['avg_accuracy']:.4f}")
    print(f"  Average Forgetting: {results['forgetting']:.4f}")
    print(f"  Specialization HHI: {results['specialization']['hhi']:.4f}")
