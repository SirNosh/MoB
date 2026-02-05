"""
Standalone MoB experiment runner for iterative development.

ARCHITECTURE NOTES - READ THIS:
===============================
This script is INTENTIONALLY ISOLATED from the baseline comparison code.

To add new features (replay, A-GEM, etc.) to MoB WITHOUT affecting baselines:

1. Create new files:
   - mob/expert_enhanced.py → MoBExpertEnhanced(MoBExpert)
   - mob/pool_enhanced.py → ExpertPoolEnhanced(ExpertPool)

2. Import the enhanced versions ONLY in this script

3. The baseline code (test_baselines.py) continues to use the original
   MoBExpert and ExpertPool classes, completely unaffected.

This script creates its own training loop rather than calling run_mob_experiment()
from test_baselines.py, so you can modify training logic here freely.

Usage:
    python tests/run_mob_only.py                    # Default config
    python tests/run_mob_only.py --lambda_ewc 5000  # Test higher EWC
    python tests/run_mob_only.py --epochs 1         # Quick test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob.models import create_model
from mob.auction import PerBatchVCGAuction
from mob.bidding import ExecutionCostEstimator, EWCForgettingEstimator
from mob.bid_diagnostics import BidLogger
from mob.utils import set_seed

# Import dataset creation (this is safe - just data loading)
from tests.test_baselines import create_split_mnist


# =============================================================================
# LOCAL EXPERT CLASS - Modify this freely without affecting baselines
# =============================================================================

class MoBExpertLocal:
    """
    Local copy of MoBExpert for MoB-only experiments.

    This is a copy of the original MoBExpert that you can modify freely.
    Changes here will NOT affect baselines since they import from mob/expert.py.

    Includes Learning without Forgetting (LwF) support:
    - Li & Hoiem, ECCV 2016: https://arxiv.org/abs/1606.09282
    - Zero memory overhead (no replay buffer)
    - Uses knowledge distillation to preserve old task behavior
    """

    def __init__(
        self,
        expert_id: int,
        model: nn.Module,
        alpha: float,
        beta: float,
        lambda_ewc: float,
        forgetting_cost_scale: float = 1.0,
        device: Optional[torch.device] = None,
        # LwF hyperparameters
        use_lwf: bool = False,
        lwf_temperature: float = 2.0,
        lwf_alpha: float = 0.1,  # Weight for distillation loss (< 0.3 recommended)
    ):
        self.expert_id = expert_id
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.device = device if device is not None else torch.device('cpu')

        self.model.to(self.device)

        self.exec_estimator = ExecutionCostEstimator(model, device=self.device)
        self.forget_estimator = EWCForgettingEstimator(
            model,
            lambda_ewc=lambda_ewc,
            forgetting_cost_scale=forgetting_cost_scale,
            device=self.device
        )

        # Statistics
        self.batches_won = 0
        self.batches_won_this_task = 0
        self.total_batches_seen = 0

        # =====================================================================
        # LwF (Learning without Forgetting) - Li & Hoiem, ECCV 2016
        # =====================================================================
        self.use_lwf = use_lwf
        self.lwf_temperature = lwf_temperature
        self.lwf_alpha = lwf_alpha

        # Storage for soft targets (computed before each task)
        # Maps batch_idx -> soft_targets tensor
        self.lwf_soft_targets: Dict[int, torch.Tensor] = {}
        self.lwf_batch_counter = 0  # Tracks batch position within task
        # =====================================================================

    def record_lwf_soft_targets(self, dataloader, max_batches: int = None):
        """
        Record soft targets BEFORE training on a new task.

        This is the key step in LwF: we capture what the model currently
        outputs on the new task data, then use this as a target during training.

        Args:
            dataloader: DataLoader for the new task
            max_batches: Maximum batches to record (None = all)
        """
        if not self.use_lwf:
            return

        self.model.eval()
        self.lwf_soft_targets = {}
        self.lwf_batch_counter = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                x = x.to(self.device)
                logits = self.model(x)

                # Store soft targets (temperature-scaled softmax)
                # Keep on CPU to save GPU memory
                soft_targets = F.softmax(logits / self.lwf_temperature, dim=1)
                self.lwf_soft_targets[batch_idx] = soft_targets.cpu()

        print(f"[Expert {self.expert_id}] LwF: Recorded {len(self.lwf_soft_targets)} soft target batches")

    def compute_lwf_loss(self, current_logits: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Compute the LwF distillation loss.

        Loss = KL_div(softmax(z_new/T), softmax(z_old/T)) × T²

        The T² scaling ensures gradients are comparable regardless of temperature.

        Args:
            current_logits: Current model outputs
            batch_idx: Index to retrieve stored soft targets

        Returns:
            Distillation loss (0 if no soft targets available)
        """
        if not self.use_lwf or batch_idx not in self.lwf_soft_targets:
            return torch.tensor(0.0, device=self.device)

        # Get stored soft targets
        old_soft_targets = self.lwf_soft_targets[batch_idx].to(self.device)

        # Compute current soft predictions
        current_soft = F.log_softmax(current_logits / self.lwf_temperature, dim=1)

        # KL divergence loss (input should be log-probabilities, target should be probabilities)
        # Note: reduction='batchmean' gives proper gradient scaling
        lwf_loss = F.kl_div(current_soft, old_soft_targets, reduction='batchmean')

        # Scale by T² to maintain gradient magnitude (from Hinton's distillation paper)
        lwf_loss = lwf_loss * (self.lwf_temperature ** 2)

        return lwf_loss

    def reset_lwf_for_new_task(self):
        """Reset LwF batch counter at the start of each task's training."""
        self.lwf_batch_counter = 0

    def compute_bid(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict]:
        """Compute bid using normalized execution + forgetting costs."""
        import math

        self.total_batches_seen += 1

        raw_exec = self.exec_estimator.compute_predicted_loss(x, y)
        raw_forget = self.forget_estimator.compute_forgetting_cost(x, y)

        # Normalize execution cost (cross-entropy ~2.3 untrained, ~0.1 trained)
        norm_exec = raw_exec / 2.5

        # Normalize forgetting cost (log scale for huge range)
        log_forget = math.log1p(raw_forget)
        norm_forget = log_forget / 10.0

        bid = self.alpha * norm_exec + self.beta * norm_forget

        components = {
            'exec_cost': raw_exec,
            'forget_cost': raw_forget,
            'norm_exec_cost': norm_exec,
            'norm_forget_cost': norm_forget,
            'bid': bid,
            'alpha': self.alpha,
            'beta': self.beta
        }
        return bid, components

    def train_on_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict:
        """Train on a batch with EWC regularization and optional LwF distillation."""
        self.model.train()
        self.batches_won += 1
        self.batches_won_this_task += 1

        x = x.to(self.device)
        y = y.to(self.device)

        optimizer.zero_grad()
        logits = self.model(x)
        task_loss = F.cross_entropy(logits, y)
        ewc_penalty = self.forget_estimator.penalty()

        # =====================================================================
        # LwF: Add distillation loss to preserve old task behavior
        # =====================================================================
        lwf_loss = torch.tensor(0.0, device=self.device)
        if self.use_lwf and self.lwf_soft_targets:
            lwf_loss = self.compute_lwf_loss(logits, self.lwf_batch_counter)
            self.lwf_batch_counter += 1

        # Total loss = task loss + EWC penalty + LwF distillation
        total_loss = task_loss + ewc_penalty + self.lwf_alpha * lwf_loss
        # =====================================================================

        # Log first few batches per task
        if self.batches_won_this_task <= 3:
            log_msg = (f"[Expert {self.expert_id}] Batch {self.batches_won_this_task}: "
                      f"task_loss={task_loss.item():.4f}, "
                      f"ewc_penalty={ewc_penalty.item():.4f}")
            if self.use_lwf and lwf_loss.item() > 0:
                log_msg += f", lwf_loss={lwf_loss.item():.4f}"
            log_msg += f", total_loss={total_loss.item():.4f}"
            print(log_msg)

        total_loss.backward()
        optimizer.step()

        result = {
            'task_loss': task_loss.item(),
            'ewc_penalty': ewc_penalty.item(),
            'total_loss': total_loss.item()
        }
        if self.use_lwf:
            result['lwf_loss'] = lwf_loss.item()

        return result

    def update_after_task(self, dataloader, num_samples: int = 200):
        """Update Fisher information after task completion."""
        self.forget_estimator.update_fisher(dataloader, num_samples=num_samples)

    def reset_task_statistics(self):
        """Reset per-task counters."""
        self.batches_won_this_task = 0

    def has_fisher(self) -> bool:
        """Check if expert has Fisher information."""
        return self.forget_estimator.has_fisher()


# =============================================================================
# LOCAL EXPERT POOL - Modify this freely without affecting baselines
# =============================================================================

class ExpertPoolLocal:
    """
    Local copy of ExpertPool for MoB-only experiments.

    Uses MoBExpertLocal instead of MoBExpert.
    Modify freely - changes won't affect baselines.
    """

    def __init__(
        self,
        num_experts: int,
        expert_config: Dict,
        device: Optional[torch.device] = None
    ):
        self.num_experts = num_experts
        self.device = device if device is not None else torch.device('cpu')

        # Create LOCAL experts (not the original MoBExpert)
        self.experts: List[MoBExpertLocal] = []
        for i in range(num_experts):
            model = create_model(
                architecture=expert_config['architecture'],
                num_classes=expert_config['num_classes'],
                input_channels=expert_config.get('input_channels', 1),
                dropout=expert_config.get('dropout', 0.5)
            )
            expert = MoBExpertLocal(
                expert_id=i,
                model=model,
                alpha=expert_config.get('alpha', 0.5),
                beta=expert_config.get('beta', 0.5),
                lambda_ewc=expert_config.get('lambda_ewc', 10.0),
                forgetting_cost_scale=expert_config.get('forgetting_cost_scale', 1.0),
                device=self.device,
                # LwF parameters
                use_lwf=expert_config.get('use_lwf', False),
                lwf_temperature=expert_config.get('lwf_temperature', 2.0),
                lwf_alpha=expert_config.get('lwf_alpha', 0.1),
            )
            self.experts.append(expert)

    def collect_bids(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[np.ndarray, List[Dict]]:
        """Collect bids from all experts."""
        bids = np.zeros(self.num_experts)
        components = []
        for i, expert in enumerate(self.experts):
            bid, comp = expert.compute_bid(x, y)
            bids[i] = bid
            components.append(comp)
        return bids, components

    def train_winner(self, winner_id: int, x: torch.Tensor, y: torch.Tensor,
                     optimizers: List[torch.optim.Optimizer]) -> Dict:
        """Train the winning expert."""
        return self.experts[winner_id].train_on_batch(x, y, optimizers[winner_id])

    def evaluate_all(self, dataloader, verbose: bool = False) -> Dict:
        """Evaluate using forgetting-cost-based routing (highest cost wins = expert knows data)."""
        all_labels = []
        winner_preds = []
        expert_selections = {i: 0 for i in range(self.num_experts)}
        expert_forget_costs_sum = {i: 0.0 for i in range(self.num_experts)}
        num_batches = 0

        for x, y in dataloader:
            x_device = x.to(self.device)
            y_device = y.to(self.device)
            all_labels.append(y_device.cpu())

            # Get forgetting cost from each expert (using pseudo-labels)
            batch_forget_costs = np.zeros(self.num_experts)
            batch_logits = []

            for i, expert in enumerate(self.experts):
                expert.model.eval()
                with torch.no_grad():
                    logits = expert.model(x_device)
                    batch_logits.append(logits)
                
                # Compute forgetting cost using pseudo-labels
                # High cost = expert has learned about this data type (has Fisher info)
                pseudo_labels = logits.argmax(dim=-1).detach()
                forget_cost = expert.forget_estimator.compute_forgetting_cost(x_device, pseudo_labels)
                batch_forget_costs[i] = forget_cost
                expert_forget_costs_sum[i] += forget_cost

            # Route to expert with LOWEST forgetting cost
            winner_id = np.argmin(batch_forget_costs)
            expert_selections[winner_id] += 1
            winning_preds = batch_logits[winner_id].argmax(dim=-1).cpu()
            winner_preds.append(winning_preds)
            num_batches += 1

        all_labels = torch.cat(all_labels)
        winner_preds = torch.cat(winner_preds)
        accuracy = (winner_preds == all_labels).float().mean().item()

        # Diagnostic: which expert was selected most
        total_batches = sum(expert_selections.values())
        primary_eval_expert = max(expert_selections, key=expert_selections.get)
        avg_forget_costs = {k: v/num_batches for k, v in expert_forget_costs_sum.items()}
        
        return {
            'ensemble_accuracy': accuracy,
            'expert_selections': expert_selections,
            'primary_eval_expert': primary_eval_expert,
            'eval_distribution': {k: v/total_batches for k, v in expert_selections.items()},
            'avg_forget_costs': avg_forget_costs
        }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(train_tasks, test_tasks, config):
    """Run MoB experiment with local classes."""

    print("\n" + "="*70)
    print("MoB Experiment (Isolated - won't affect baselines)")
    print("="*70)

    device = torch.device(config['device'])

    # Expert configuration
    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': config['alpha'],
        'beta': config['beta'],
        'lambda_ewc': config['lambda_ewc'],
        'forgetting_cost_scale': config.get('forgetting_cost_scale', 1.0),
        # LwF parameters
        'use_lwf': config.get('use_lwf', False),
        'lwf_temperature': config.get('lwf_temperature', 2.0),
        'lwf_alpha': config.get('lwf_alpha', 0.1),
        'dropout': 0.5
    }

    # Create LOCAL pool (not the original ExpertPool)
    pool = ExpertPoolLocal(config['num_experts'], expert_config, device=device)
    auction = PerBatchVCGAuction(config['num_experts'])

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=config['learning_rate'])
        for expert in pool.experts
    ]

    # Bid logger
    bid_logger = BidLogger(
        num_experts=config['num_experts'],
        alpha=config['alpha'],
        beta=config['beta'],
        log_file=None
    )

    # Metrics
    task_accuracies = []
    final_accuracies = []
    expert_task_wins = {}

    # =========================================================================
    # TRAINING
    # =========================================================================
    global_batch_idx = 0
    epochs_per_task = config.get('epochs_per_task', 4)

    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n{'='*70}")
        print(f"TASK {task_id + 1}/{len(train_tasks)} (Digits {task_id*2}, {task_id*2+1})")
        print(f"{'='*70}")

        # Reset per-task stats
        for expert in pool.experts:
            expert.reset_task_statistics()

        # =====================================================================
        # LwF: Record soft targets BEFORE training on new task
        # ONLY for experts that have Fisher information (already trained)
        # 
        # Key insight: LwF preserves "what the model knows" - but an expert
        # that hasn't been trained on any task has nothing worth preserving.
        # Recording soft targets for untrained experts just constrains them
        # to maintain random outputs, which hurts learning.
        # =====================================================================
        if config.get('use_lwf', False):
            trained_experts = [e for e in pool.experts if e.has_fisher()]
            if trained_experts:
                print(f"  Recording LwF soft targets for {len(trained_experts)} trained experts...")
                for expert in trained_experts:
                    expert.record_lwf_soft_targets(task_loader, max_batches=100)
                    expert.reset_lwf_for_new_task()  # Reset batch counter for training
            else:
                print(f"  LwF: No trained experts yet (Task 1), skipping soft target recording")
        # =====================================================================

        winners_this_task = {}
        task_batch_count = 0

        for epoch in range(epochs_per_task):
            epoch_winners = {}

            pbar = tqdm(task_loader, desc=f"Epoch {epoch+1}/{epochs_per_task}", leave=False)
            for x, y in pbar:
                # Collect bids
                bids, components = pool.collect_bids(x, y)

                # Auction
                winner_id, payment, _ = auction.run_auction(bids)

                # Track
                winners_this_task[winner_id] = winners_this_task.get(winner_id, 0) + 1
                epoch_winners[winner_id] = epoch_winners.get(winner_id, 0) + 1

                # Log
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
                task_batch_count += 1

            print(f"  Epoch {epoch+1} winners: {dict(sorted(epoch_winners.items()))}")

        # Task summary
        print(f"\n  Task {task_id+1} total batches: {task_batch_count}")
        print(f"  Winner distribution: {dict(sorted(winners_this_task.items()))}")

        if winners_this_task:
            primary = max(winners_this_task, key=winners_this_task.get)
            pct = winners_this_task[primary] / task_batch_count * 100
            print(f"  Primary expert: Expert {primary} ({pct:.1f}%)")
            expert_task_wins[task_id] = primary

        # Update Fisher
        winning_experts = list(winners_this_task.keys())
        print(f"\n  Updating Fisher for experts: {winning_experts}")
        for eid in winning_experts:
            pool.experts[eid].update_after_task(task_loader, num_samples=200)

        # Evaluate current task
        results = pool.evaluate_all(test_tasks[task_id])
        task_accuracies.append(results['ensemble_accuracy'])
        print(f"  Task {task_id+1} accuracy: {results['ensemble_accuracy']:.4f}")

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    for task_id, test_loader in enumerate(test_tasks):
        results = pool.evaluate_all(test_loader)
        acc = results['ensemble_accuracy']
        final_accuracies.append(acc)

        trained_expert = expert_task_wins.get(task_id, "?")
        eval_expert = results['primary_eval_expert']
        avg_costs = results['avg_forget_costs']
        status = "OK" if acc > 0.5 else "FAIL"
        
        # Show mismatch between trained expert and eval expert
        match_str = "" if trained_expert == eval_expert else f" ** ROUTED TO Expert {eval_expert}!"
        cost_str = " | ForgetCost: " + ", ".join([f"E{k}:{v:.3f}" for k, v in sorted(avg_costs.items())])
        print(f"  Task {task_id+1} (digits {task_id*2},{task_id*2+1}): "
              f"{acc:.4f} {status} [Trained: Expert {trained_expert}]{match_str}")
        print(f"    {cost_str}")

    # Metrics
    avg_accuracy = np.mean(final_accuracies)
    forgetting_per_task = [
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]
    avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    print(f"  Average Forgetting: {avg_forgetting:.4f}")
    print(f"  Tasks retained (>50%): {sum(1 for a in final_accuracies if a > 0.5)}/{len(final_accuracies)}")

    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': avg_forgetting,
        'expert_task_wins': expert_task_wins,
        'bid_logger': bid_logger
    }


def main():
    parser = argparse.ArgumentParser(description='Run MoB experiment (isolated from baselines)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lambda_ewc', type=float, default=10.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--forgetting_cost_scale', type=float, default=1.0)

    # LwF (Learning without Forgetting) arguments
    parser.add_argument('--use_lwf', action='store_true',
                        help='Enable Learning without Forgetting (knowledge distillation)')
    parser.add_argument('--lwf_temperature', type=float, default=2.0,
                        help='Temperature for LwF soft targets (default: 2.0)')
    parser.add_argument('--lwf_alpha', type=float, default=0.1,
                        help='Weight for LwF distillation loss (recommended < 0.3)')
    parser.add_argument('--save_bids', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)

    config = {
        'num_experts': args.num_experts,
        'num_tasks': 5,
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_ewc': args.lambda_ewc,
        'learning_rate': args.learning_rate,
        'forgetting_cost_scale': args.forgetting_cost_scale,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs,
        # LwF configuration
        'use_lwf': args.use_lwf,
        'lwf_temperature': args.lwf_temperature,
        'lwf_alpha': args.lwf_alpha,
    }

    print("="*70)
    print(f"MoB Standalone Experiment (Seed {args.seed})")
    print("="*70)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nCreating Split-MNIST datasets...")
    train_tasks = create_split_mnist(config['num_tasks'], train=True, batch_size=config['batch_size'])
    test_tasks = create_split_mnist(config['num_tasks'], train=False, batch_size=config['batch_size'])

    results = run_experiment(train_tasks, test_tasks, config)

    # Save
    os.makedirs('results', exist_ok=True)

    if args.save_bids and 'bid_logger' in results:
        print("\n" + "="*70)
        print("BID DIAGNOSTICS")
        print("="*70)
        results['bid_logger'].print_diagnostics()
        results['bid_logger'].save_logs(f"results/mob_bids_seed_{args.seed}.json")

    summary = {
        'seed': args.seed,
        'config': config,
        'task_accuracies': results['task_accuracies'],
        'final_accuracies': results['final_accuracies'],
        'avg_accuracy': results['avg_accuracy'],
        'forgetting': results['forgetting'],
        'expert_task_wins': {str(k): v for k, v in results['expert_task_wins'].items()}
    }
    with open(f"results/mob_results_seed_{args.seed}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved to: results/mob_results_seed_{args.seed}.json")


if __name__ == '__main__':
    main()
