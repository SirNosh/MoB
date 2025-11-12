"""
Example script demonstrating bid diagnostics for MoB.

This script shows how to use the BidLogger to diagnose potential issues:
- Alpha (PredictedLoss) signal being ignored
- Beta (ForgettingCost) too high preventing learning
- Bids exploding or vanishing
- Expert specialization patterns

Usage:
    python examples/diagnose_bids.py
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mob import (
    ExpertPool,
    PerBatchVCGAuction,
    BidLogger,
    set_seed,
    get_device
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_split_mnist(num_tasks=5):
    """Create a simple Split-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Split into tasks (2 classes per task)
    classes_per_task = 2
    tasks = []

    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task

        # Filter dataset for this task's classes
        indices = [
            i for i, (_, label) in enumerate(train_dataset)
            if start_class <= label < end_class
        ]

        task_subset = torch.utils.data.Subset(train_dataset, indices)
        task_loader = DataLoader(task_subset, batch_size=32, shuffle=True)

        tasks.append(task_loader)

    return tasks


def main():
    """Run MoB with comprehensive bid logging."""

    print("="*80)
    print("MoB Bid Diagnostics Example")
    print("="*80)

    # Set seed
    set_seed(42)
    device = get_device()

    # Configuration
    config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 5000,
        'dropout': 0.5
    }

    # Create expert pool
    num_experts = 4
    pool = ExpertPool(
        num_experts=num_experts,
        expert_config=config,
        device=device
    )

    # Create auction
    auction = PerBatchVCGAuction()

    # Create bid logger
    bid_logger = BidLogger(
        num_experts=num_experts,
        log_file="bid_logs.json"
    )

    # Create optimizers
    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=0.001)
        for expert in pool.experts
    ]

    # Create Split-MNIST dataset
    print("\nCreating Split-MNIST dataset...")
    tasks = create_split_mnist(num_tasks=2)  # Just 2 tasks for quick demo
    print(f"✓ Created {len(tasks)} tasks")

    # Training loop with bid logging
    print("\n" + "="*80)
    print("Training with Bid Logging")
    print("="*80)

    global_batch_idx = 0

    for task_id, task_loader in enumerate(tasks):
        print(f"\n[Task {task_id}] Training on classes {task_id*2}-{task_id*2+1}")

        for batch_idx, (x, y) in enumerate(task_loader):
            # Limit batches for demo
            if batch_idx >= 20:  # Only first 20 batches per task
                break

            # Collect bids
            bids, components = pool.collect_bids(x, y)

            # Run auction
            winner_id = auction.run_auction(bids)

            # LOG THE BIDS (This is the key!)
            bid_logger.log_batch(
                batch_idx=global_batch_idx,
                bids=bids,
                components=components,
                winner_id=winner_id,
                task_id=task_id
            )

            # Train winner
            pool.train_winner(winner_id, x, y, optimizers)

            # Print occasional updates
            if global_batch_idx % 10 == 0:
                print(f"  Batch {global_batch_idx}: Winner = Expert {winner_id}")

            global_batch_idx += 1

        # Update Fisher after task
        print(f"  Updating Fisher matrices after Task {task_id}...")
        pool.update_after_task(task_loader, num_samples=100)

    # Print comprehensive diagnostics
    print("\n" + "="*80)
    print("BID DIAGNOSTICS ANALYSIS")
    print("="*80)

    bid_logger.print_diagnostics()

    # Print detailed breakdown for first batch
    print("\nDetailed breakdown for first batch:")
    bid_logger.print_batch_details(batch_idx=0)

    # Print detailed breakdown for a later batch (after Fisher updated)
    print("\nDetailed breakdown for batch 25 (Task 1, after Fisher update):")
    bid_logger.print_batch_details(batch_idx=25)

    # Save logs
    bid_logger.save_logs("bid_diagnostics_results.json")

    # Create visualization
    try:
        print("\nCreating bid component visualization...")
        bid_logger.plot_bid_components(save_path="bid_components.png")
    except Exception as e:
        print(f"⚠  Could not create plot: {e}")
        print("  Install matplotlib: pip install matplotlib")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nFiles created:")
    print("  - bid_diagnostics_results.json: Full bid logs")
    print("  - bid_components.png: Visualization (if matplotlib available)")
    print("\nUse these diagnostics to identify:")
    print("  1. If alpha (execution cost) is being ignored")
    print("  2. If beta (forgetting cost) is preventing learning")
    print("  3. If bids are exploding or vanishing")
    print("  4. Expert specialization patterns")


if __name__ == '__main__':
    main()
