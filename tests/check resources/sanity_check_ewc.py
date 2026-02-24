"""
EWC Sanity Check — Single Expert, Two Tasks

Tests the upper limit of what ONE CNN (same architecture/size as a MoB expert,
~421K params) can learn with EWC across two sequential tasks.

Sweeps lambda_ewc to find the best stability-plasticity tradeoff for a single
expert handling 2 tasks.

Usage:
    python tests/check\ resources/sanity_check_ewc.py
    python tests/check\ resources/sanity_check_ewc.py --task1 0 --task2 4   # digits 0,1 then 8,9
    python tests/check\ resources/sanity_check_ewc.py --epochs 6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import gc
import copy
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mob.models import SimpleCNN
from mob.utils import set_seed
from mob.bidding import EWCForgettingEstimator
from tests.test_baselines import create_split_mnist


# =============================================================================
# LOCAL EWC ESTIMATOR — Fixes Fisher normalization for converged models
# =============================================================================
class EWCForgettingEstimatorSanity(EWCForgettingEstimator):
    """
    Subclass that fixes Fisher normalization for converged models.

    When a small CNN converges to 99.8%+ on Task 1, gradients become tiny,
    resulting in Fisher mean < 1e-8. The base class epsilon of 1e-8 then
    causes normalization to be skipped, silently disabling EWC.

    This subclass:
    1. Lowers epsilon to 1e-30 so normalization always happens.
    2. Clamps Fisher minimum to 0.001 so even "unimportant" params get
       basic L2-like protection against drift.
    """
    def _normalize_fisher(self):
        if not self.fisher:
            return

        all_fisher = torch.cat([f.flatten() for f in self.fisher.values()])
        fisher_mean = all_fisher.mean()

        epsilon = 1e-30
        if fisher_mean > 0:
            for n in self.fisher:
                self.fisher[n] = self.fisher[n] / (fisher_mean + epsilon)
                self.fisher[n] = torch.clamp(self.fisher[n], min=0.001)


def train_with_ewc(model, dataloader, optimizer, device, epochs,
                   ewc_estimator=None, task_name="Task"):
    """Train model with optional EWC penalty using the proven estimator."""
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        task_loss_sum = 0
        ewc_loss_sum = 0
        batches = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            task_loss = F.cross_entropy(logits, y)

            # EWC penalty from the estimator
            if ewc_estimator is not None and ewc_estimator.has_fisher():
                ewc_penalty = ewc_estimator.penalty()
            else:
                ewc_penalty = torch.tensor(0.0, device=device)

            loss = task_loss + ewc_penalty
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            task_loss_sum += task_loss.item()
            ewc_loss_sum += ewc_penalty.item()
            batches += 1

    avg_total = total_loss / batches
    avg_task = task_loss_sum / batches
    avg_ewc = ewc_loss_sum / batches
    return avg_total, avg_task, avg_ewc


def evaluate(model, dataloader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def run_single_test(lambda_ewc, train_tasks, test_tasks, task1_idx, task2_idx,
                    device, lr, epochs, seed):
    """Run a single EWC experiment with the given lambda."""
    set_seed(seed)

    # Create fresh model (same as one MoB expert)
    model = SimpleCNN(num_classes=10, input_channels=1, dropout=0.5, width_multiplier=1).to(device)

    # Initialize FC layers by running a dummy forward pass
    dummy = torch.randn(1, 1, 28, 28).to(device)
    with torch.no_grad():
        model(dummy)

    param_count = sum(p.numel() for p in model.parameters())

    # Create the EWC estimator (uses the same proven code as MoB + monolithic baseline)
    ewc_estimator = EWCForgettingEstimatorSanity(
        model,
        lambda_ewc=lambda_ewc,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Task 1 Training ----
    train_with_ewc(model, train_tasks[task1_idx], optimizer, device, epochs,
                   task_name=f"Task 1 (digits {task1_idx*2},{task1_idx*2+1})")

    # Evaluate on Task 1 after training Task 1
    acc_t1_after_t1 = evaluate(model, test_tasks[task1_idx], device)

    # Update Fisher using the estimator (handles normalization + clamping)
    ewc_estimator.update_fisher(train_tasks[task1_idx], num_samples=200)

    # ---- Task 2 Training (with EWC) ----
    # Reset optimizer for new task
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_with_ewc(model, train_tasks[task2_idx], optimizer, device, epochs,
                   ewc_estimator=ewc_estimator,
                   task_name=f"Task 2 (digits {task2_idx*2},{task2_idx*2+1})")

    # Evaluate on both tasks after Task 2
    acc_t1_after_t2 = evaluate(model, test_tasks[task1_idx], device)
    acc_t2_after_t2 = evaluate(model, test_tasks[task2_idx], device)

    forgetting = max(0, acc_t1_after_t1 - acc_t1_after_t2)
    avg_acc = (acc_t1_after_t2 + acc_t2_after_t2) / 2

    return {
        'lambda_ewc': lambda_ewc,
        'param_count': param_count,
        'acc_t1_after_t1': acc_t1_after_t1,
        'acc_t1_after_t2': acc_t1_after_t2,
        'acc_t2_after_t2': acc_t2_after_t2,
        'forgetting': forgetting,
        'avg_accuracy': avg_acc,
    }


def main():
    parser = argparse.ArgumentParser(description='EWC Sanity Check - Single Expert, Two Tasks')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--task1', type=int, default=1, help='Task 1 index (0-4)')
    parser.add_argument('--task2', type=int, default=4, help='Task 2 index (0-4)')
    parser.add_argument('--seeds', type=int, default=3, help='Number of seeds to average over')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Lambda values to sweep (log-spaced from 0 to 10000)
    lambda_values = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    print("=" * 90)
    print("EWC SANITY CHECK — Single Expert (SimpleCNN ~421K params), Two Tasks")
    print("=" * 90)
    print(f"  Task 1: digits {args.task1*2},{args.task1*2+1}")
    print(f"  Task 2: digits {args.task2*2},{args.task2*2+1}")
    print(f"  Epochs per task: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Lambda values: {lambda_values}")
    print("=" * 90)

    # Results table
    all_results = []

    for lam in lambda_values:
        seed_results = []

        for s in range(args.seeds):
            seed = args.seed + s
            set_seed(seed)

            # Create fresh datasets each time
            train_tasks = create_split_mnist(5, train=True, batch_size=args.batch_size)
            test_tasks = create_split_mnist(5, train=False, batch_size=args.batch_size)

            result = run_single_test(
                lambda_ewc=lam,
                train_tasks=train_tasks, test_tasks=test_tasks,
                task1_idx=args.task1, task2_idx=args.task2,
                device=device, lr=args.lr, epochs=args.epochs, seed=seed
            )
            seed_results.append(result)

            # Clean up
            del train_tasks, test_tasks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average across seeds
        avg_result = {
            'lambda_ewc': lam,
            'param_count': seed_results[0]['param_count'],
            'acc_t1_after_t1': np.mean([r['acc_t1_after_t1'] for r in seed_results]),
            'acc_t1_after_t2': np.mean([r['acc_t1_after_t2'] for r in seed_results]),
            'acc_t2_after_t2': np.mean([r['acc_t2_after_t2'] for r in seed_results]),
            'forgetting': np.mean([r['forgetting'] for r in seed_results]),
            'avg_accuracy': np.mean([r['avg_accuracy'] for r in seed_results]),
            'std_accuracy': np.std([r['avg_accuracy'] for r in seed_results]),
        }
        all_results.append(avg_result)

        print(f"  lambda={lam:>8.1f}  |  T1: {avg_result['acc_t1_after_t2']:.4f}  "
              f"T2: {avg_result['acc_t2_after_t2']:.4f}  "
              f"Avg: {avg_result['avg_accuracy']:.4f} ± {avg_result['std_accuracy']:.4f}  "
              f"Forget: {avg_result['forgetting']:.4f}")

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"  Model: SimpleCNN ({all_results[0]['param_count']:,} params)")
    print(f"  Task 1: digits {args.task1*2},{args.task1*2+1}  |  Task 2: digits {args.task2*2},{args.task2*2+1}")
    print("-" * 90)
    print(f"  {'λ_ewc':>10s}  |  {'T1 after T1':>11s}  |  {'T1 after T2':>11s}  |  "
          f"{'T2 after T2':>11s}  |  {'Avg Acc':>10s}  |  {'Forgetting':>10s}")
    print("-" * 90)

    best_avg = max(all_results, key=lambda r: r['avg_accuracy'])

    for r in all_results:
        marker = "  <-- BEST" if r['lambda_ewc'] == best_avg['lambda_ewc'] else ""
        print(f"  {r['lambda_ewc']:>10.1f}  |  {r['acc_t1_after_t1']:>11.4f}  |  "
              f"{r['acc_t1_after_t2']:>11.4f}  |  {r['acc_t2_after_t2']:>11.4f}  |  "
              f"{r['avg_accuracy']:>10.4f}  |  {r['forgetting']:>10.4f}{marker}")

    print("-" * 90)
    print(f"\n  Best lambda_ewc: {best_avg['lambda_ewc']}")
    print(f"  Best avg accuracy: {best_avg['avg_accuracy']:.4f}")
    print(f"  At best: T1={best_avg['acc_t1_after_t2']:.4f}, T2={best_avg['acc_t2_after_t2']:.4f}, "
          f"Forget={best_avg['forgetting']:.4f}")

    # Also find the best "no forgetting" config
    low_forget = [r for r in all_results if r['forgetting'] < 0.05]
    if low_forget:
        best_low_forget = max(low_forget, key=lambda r: r['avg_accuracy'])
        print(f"\n  Best lambda with <5% forgetting: {best_low_forget['lambda_ewc']}")
        print(f"  Accuracy: T1={best_low_forget['acc_t1_after_t2']:.4f}, "
              f"T2={best_low_forget['acc_t2_after_t2']:.4f}, "
              f"Avg={best_low_forget['avg_accuracy']:.4f}")


if __name__ == '__main__':
    main()
