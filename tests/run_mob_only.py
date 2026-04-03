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
from mob.auction import PerBatchAuction
from mob.bidding import ExecutionCostEstimator, EWCForgettingEstimator
from mob.bid_diagnostics import BidLogger
from mob.utils import set_seed
from contibualmob.prototype_store import PrototypeStore

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
        # Bid formula: 'addition' or 'subtraction'
        bid_formula: str = 'addition',
    ):
        self.expert_id = expert_id
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.bid_formula = bid_formula
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
        # Optimizer Reset Tracking (for idle-based reset in continual learning)
        # =====================================================================
        self.last_won_global_batch = -1  # Global batch index when expert last won
        self.last_trained_task = -1  # Task ID when expert last trained (for task-aware)
        # =====================================================================

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

        # Prototype store for feature-distance routing (lazy init)
        self.prototype_store: Optional[PrototypeStore] = None

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
        """
        Compute bid for TRAINING routing.

        Formula options:
        - 'addition': bid = α × exec_cost + β × forget_cost
          (high forget = high bid = LOSES, protects old knowledge)
        - 'subtraction': bid = α × exec_cost - β × forget_cost
          (high forget = low bid = WINS, routes to relevant features)
        """
        import math

        self.total_batches_seen += 1

        raw_exec = self.exec_estimator.compute_predicted_loss(x, y)
        raw_forget = self.forget_estimator.compute_forgetting_cost(x, y)

        # Normalize execution cost (cross-entropy ~2.3 untrained, ~0.1 trained)
        norm_exec = raw_exec / 2.5

        # Normalize forgetting cost (log scale for huge range)
        log_forget = math.log1p(raw_forget)
        norm_forget = log_forget / 10.0

        # Apply formula based on configuration
        if self.bid_formula == 'subtraction':
            bid = self.alpha * norm_exec - self.beta * norm_forget
        else:  # 'addition' (default)
            bid = self.alpha * norm_exec + self.beta * norm_forget

        components = {
            'exec_cost': raw_exec,
            'forget_cost': raw_forget,
            'norm_exec_cost': norm_exec,
            'norm_forget_cost': norm_forget,
            'bid': bid,
            'alpha': self.alpha,
            'beta': self.beta,
            'formula': self.bid_formula
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

        # Use forward_features for prototype accumulation
        if hasattr(self.model, 'forward_features'):
            features, logits = self.model.forward_features(x)

            if self.prototype_store is None:
                self.prototype_store = PrototypeStore(
                    feature_dim=features.shape[1],
                    device=self.device
                )

            self.prototype_store.update(features.detach(), y)
        else:
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
        """Update Fisher information and finalize prototypes after task completion."""
        self.forget_estimator.update_fisher(dataloader, num_samples=num_samples)

        if self.prototype_store is not None:
            self.prototype_store.finalize()

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
        self.load_bias = np.zeros(num_experts)
        self.expert_win_counts = np.zeros(num_experts)
        self.total_auctions = 0
        self.use_conscience = expert_config.get('use_conscience', False)
        self.conscience_rate = expert_config.get('conscience_rate', 0.01)
        self.conscience_decay = expert_config.get('conscience_decay', 0.999)

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
                # Bid formula
                bid_formula=expert_config.get('bid_formula', 'addition'),
            )
            self.experts.append(expert)

    def collect_bids(self, x: torch.Tensor, y: torch.Tensor,
                     train_routing: str = 'label') -> Tuple[np.ndarray, List[Dict]]:
        """Collect bids from all experts.

        Args:
            train_routing: 'label' (default) or 'prototype' (use prototype distance).
        """
        import math

        if train_routing == 'prototype':
            bids = np.zeros(self.num_experts)
            components = []
            for i, expert in enumerate(self.experts):
                expert.total_batches_seen += 1

                has_prototypes = (
                    expert.prototype_store is not None
                    and len(expert.prototype_store.centroids) > 0
                    and hasattr(expert.model, 'forward_features')
                )

                # Get features and logits
                expert.model.eval()
                with torch.no_grad():
                    if hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x.to(self.device))
                    else:
                        logits = expert.model(x.to(self.device))
                        features = None
                expert.model.train()

                if has_prototypes and features is not None:
                    distance_score = expert.prototype_store.compute_routing_score(features)
                    routing_tag = 'prototype'
                else:
                    # No prototypes yet: high default distance so trained experts
                    # are preferred, but this expert can still win if others are worse
                    distance_score = 100.0
                    routing_tag = 'prototype_default'

                pseudo_labels = logits.argmax(dim=-1).detach()
                raw_forget = expert.forget_estimator.compute_forgetting_cost(
                    x.to(self.device), pseudo_labels
                )
                norm_distance = distance_score / 10.0
                norm_forget = math.log1p(raw_forget) / 10.0
                bid = expert.alpha * norm_distance + expert.beta * norm_forget
                if self.use_conscience:
                    bid = bid + self.load_bias[i]

                comp = {
                    'exec_cost': distance_score,
                    'forget_cost': raw_forget,
                    'norm_exec_cost': norm_distance,
                    'norm_forget_cost': norm_forget,
                    'bid': bid,
                    'alpha': expert.alpha,
                    'beta': expert.beta,
                    'routing': routing_tag
                }

                bids[i] = bid
                components.append(comp)
            return bids, components
        else:
            bids = np.zeros(self.num_experts)
            components = []
            for i, expert in enumerate(self.experts):
                bid, comp = expert.compute_bid(x, y)
                if self.use_conscience:
                    bid = bid + self.load_bias[i]
                    comp['bid'] = bid
                bids[i] = bid
                components.append(comp)
            return bids, components

    def train_winner(self, winner_id: int, x: torch.Tensor, y: torch.Tensor,
                     optimizers: List[torch.optim.Optimizer]) -> Dict:
        """Train the winning expert."""
        metrics = self.experts[winner_id].train_on_batch(x, y, optimizers[winner_id])
        self.update_conscience_bias(winner_id)
        return metrics

    def update_conscience_bias(self, winner_id: int):
        """Update conscience load bias based on cumulative win frequencies."""
        if not self.use_conscience:
            return

        self.total_auctions += 1
        self.expert_win_counts[winner_id] += 1
        target_freq = 1.0 / self.num_experts

        for i in range(self.num_experts):
            actual_freq = self.expert_win_counts[i] / self.total_auctions
            self.load_bias[i] += self.conscience_rate * (actual_freq - target_freq)

        self.load_bias *= self.conscience_decay

    def evaluate_all(self, dataloader, verbose: bool = False,
                      routing_strategy: str = 'pseudo_label',
                      bid_logger=None, task_id: int = -1) -> Dict:
        """
        Evaluate using auction routing.

        Args:
            routing_strategy: 'pseudo_label' (forget-cost only) or 'prototype' (feature-distance).
            bid_logger: optional BidLogger to record eval routing decisions.
            task_id: task index for labelling eval batch logs.
        """
        import math

        use_prototype = (routing_strategy == 'prototype')

        all_labels = []
        winner_preds = []
        expert_selections = {i: 0 for i in range(self.num_experts)}
        expert_forget_costs_sum = {i: 0.0 for i in range(self.num_experts)}
        num_batches = 0

        for x, y in dataloader:
            x_device = x.to(self.device)
            y_device = y.to(self.device)
            all_labels.append(y_device.cpu())

            batch_bids = np.zeros(self.num_experts)
            batch_logits = []
            per_expert_log_data = []

            for i, expert in enumerate(self.experts):
                expert.model.eval()
                with torch.no_grad():
                    if use_prototype and hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x_device)
                    else:
                        logits = expert.model(x_device)
                        features = None
                    batch_logits.append(logits)

                pseudo_labels = logits.argmax(dim=-1).detach()
                forget_cost = expert.forget_estimator.compute_forgetting_cost(x_device, pseudo_labels)
                expert_forget_costs_sum[i] += forget_cost

                if use_prototype and expert.prototype_store is not None and features is not None:
                    distance_score = expert.prototype_store.compute_routing_score(features)
                else:
                    # No prototypes: high default distance (same scale as prototype path)
                    distance_score = 100.0

                norm_distance = distance_score / 10.0
                norm_forget = math.log1p(forget_cost) / 10.0
                batch_bids[i] = expert.alpha * norm_distance + expert.beta * norm_forget
                per_expert_log_data.append({
                    'primary_cost': distance_score,
                    'forget_cost': forget_cost,
                    'norm_primary': norm_distance,
                    'norm_forget': norm_forget,
                    'bid': float(batch_bids[i])
                })

            winner_id = np.argmin(batch_bids)
            expert_selections[winner_id] += 1
            winning_preds = batch_logits[winner_id].argmax(dim=-1).cpu()
            winner_preds.append(winning_preds)

            # Log eval batch
            if bid_logger is not None and per_expert_log_data:
                bid_logger.log_eval_batch(
                    batch_idx=num_batches + task_id * 10000,  # offset by task to keep unique
                    per_expert_data=per_expert_log_data,
                    winner_id=int(winner_id),
                    digit_labels=y_device.cpu().tolist(),
                    winner_preds=winning_preds.tolist()
                )
            num_batches += 1

        all_labels = torch.cat(all_labels)
        winner_preds = torch.cat(winner_preds)
        accuracy = (winner_preds == all_labels).float().mean().item()

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

    def evaluate_all_per_sample(
        self,
        dataloader,
        top_k: int = 1,
        temperature: float = 1.0,
        eval_bid_mode: str = 'full',
        routing_strategy: str = 'prototype'
    ) -> Dict:
        """
        Per-sample top-k evaluation using prototype distance routing.

        Each sample is independently routed to its top-k experts by lowest
        prototype distance, and outputs are combined via softmax-weighted sum.
        """
        import math

        use_prototype = (routing_strategy == 'prototype')

        all_labels = []
        all_preds = []
        expert_sample_wins = {i: 0 for i in range(self.num_experts)}
        top2_agreement_count = 0
        top2_total = 0

        for x, y in dataloader:
            x_device = x.to(self.device)
            y_device = y.to(self.device)
            batch_size = x_device.shape[0]
            all_labels.append(y_device.cpu())

            expert_logits = []
            expert_distances = []

            for i, expert in enumerate(self.experts):
                expert.model.eval()
                with torch.no_grad():
                    if use_prototype and hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x_device)
                    else:
                        logits = expert.model(x_device)
                        features = None
                    expert_logits.append(logits)

                if use_prototype and expert.prototype_store is not None and features is not None:
                    per_sample_dist = expert.prototype_store.compute_per_sample_distances(features)
                else:
                    # No prototypes: high default distance (same scale)
                    per_sample_dist = torch.full((batch_size,), 100.0, device=x_device.device)

                if eval_bid_mode == 'full':
                    pseudo_labels = logits.argmax(dim=-1).detach()
                    forget_cost = expert.forget_estimator.compute_forgetting_cost(
                        x_device, pseudo_labels
                    )
                    norm_dist = per_sample_dist / 10.0
                    norm_forget = math.log1p(forget_cost) / 10.0
                    per_sample_bid = expert.alpha * norm_dist + expert.beta * norm_forget
                else:
                    per_sample_bid = per_sample_dist

                expert_distances.append(per_sample_bid)

            distance_matrix = torch.stack(expert_distances, dim=0).T  # (B, N)
            logits_stack = torch.stack(expert_logits, dim=0)  # (N, B, C)

            _, top_k_indices = torch.topk(distance_matrix, k=min(top_k, self.num_experts),
                                          dim=1, largest=False)
            top_k_distances = torch.gather(distance_matrix, 1, top_k_indices)

            if top_k == 1:
                winners = top_k_indices.squeeze(1)
                batch_preds = torch.zeros(batch_size, dtype=torch.long, device=x_device.device)
                for b in range(batch_size):
                    w = winners[b].item()
                    batch_preds[b] = logits_stack[w, b].argmax()
                    expert_sample_wins[w] += 1
            else:
                weights = F.softmax(-top_k_distances / temperature, dim=1)
                num_classes = logits_stack.shape[2]
                combined_logits = torch.zeros(batch_size, num_classes, device=x_device.device)

                for b in range(batch_size):
                    for j in range(min(top_k, self.num_experts)):
                        expert_idx = top_k_indices[b, j].item()
                        combined_logits[b] += weights[b, j] * logits_stack[expert_idx, b]
                    expert_sample_wins[top_k_indices[b, 0].item()] += 1

                batch_preds = combined_logits.argmax(dim=1)

                if top_k >= 2:
                    for b in range(batch_size):
                        e1 = top_k_indices[b, 0].item()
                        e2 = top_k_indices[b, 1].item()
                        if logits_stack[e1, b].argmax().item() == logits_stack[e2, b].argmax().item():
                            top2_agreement_count += 1
                        top2_total += 1

            all_preds.append(batch_preds.cpu())

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        accuracy = (all_preds == all_labels).float().mean().item()

        total_samples = sum(expert_sample_wins.values())
        results = {
            'ensemble_accuracy': accuracy,
            'expert_sample_wins': expert_sample_wins,
            'expert_sample_rates': {k: v / total_samples for k, v in expert_sample_wins.items()},
            'primary_eval_expert': max(expert_sample_wins, key=expert_sample_wins.get),
            'top_k': top_k,
            'eval_bid_mode': eval_bid_mode,
            'temperature': temperature,
        }

        if top2_total > 0:
            results['top2_agreement'] = top2_agreement_count / top2_total

        return results


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

# Note: Optimizer reset now happens at task END, not during training.
# This is simpler and more effective for task-aware continual learning.


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
        # Bid formula
        'bid_formula': config.get('bid_formula', 'addition'),
        'use_conscience': config.get('use_conscience', False),
        'conscience_rate': config.get('conscience_rate', 0.01),
        'conscience_decay': config.get('conscience_decay', 0.999),
        'dropout': 0.5
    }

    # Create LOCAL pool (not the original ExpertPool)
    pool = ExpertPoolLocal(config['num_experts'], expert_config, device=device)
    auction = PerBatchAuction(config['num_experts'])

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=config['learning_rate'])
        for expert in pool.experts
    ]

    # Bid logger
    bid_logger = BidLogger(
        num_experts=config['num_experts'],
        alpha=config['alpha'],
        beta=config['beta'],
        routing_strategy=config.get('routing_strategy', 'pseudo_label'),
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
    train_routing = config.get('train_routing', 'label')
    train_routing_warmup = config.get('train_routing_warmup', 1000)
    prototype_routing_active = False

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
                # Experiment 3: Switch to prototype routing after warmup
                current_routing = 'label'
                if train_routing == 'prototype' and global_batch_idx >= train_routing_warmup:
                    current_routing = 'prototype'
                    if not prototype_routing_active:
                        print(f"\n>>> Switching to PROTOTYPE training routing at batch {global_batch_idx}")
                        prototype_routing_active = True

                # Collect bids
                bids, components = pool.collect_bids(x, y, train_routing=current_routing)

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

        # Update Fisher and finalize prototypes
        winning_experts = list(winners_this_task.keys())
        print(f"\n  Updating Fisher for experts: {winning_experts}")
        for eid in winning_experts:
            pool.experts[eid].update_after_task(task_loader, num_samples=200)
            # Log prototype state after finalization
            expert = pool.experts[eid]
            if expert.prototype_store is not None:
                ps = expert.prototype_store
                bid_logger.log_prototype_finalize(
                    expert_id=eid,
                    event=f'task_{task_id+1}_end',
                    batch_idx=global_batch_idx,
                    classes_seen=list(ps.class_count.keys()),
                    sample_counts=dict(ps.class_count),
                    has_mahalanobis=(ps.inv_cov is not None),
                    feature_dim=ps.feature_dim
                )

        # =====================================================================
        # Optimizer Reset at Task END: Reset ALL winning experts' optimizers
        # Key insight from EWC sanity check: Resetting optimizer between tasks
        # helps because Adam momentum from old task can hurt new task learning.
        # =====================================================================
        if config.get('reset_optimizer', False):
            for eid in winning_experts:
                optimizers[eid] = torch.optim.Adam(
                    pool.experts[eid].model.parameters(),
                    lr=config['learning_rate']
                )
            print(f"  [Optimizer Reset] Reset optimizers for experts: {winning_experts}")
        # =====================================================================

        # Evaluate current task using the PRIMARY expert (not routing)
        primary = max(winners_this_task, key=winners_this_task.get) if winners_this_task else 0
        expert = pool.experts[primary]
        expert.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_eval, y_eval in test_tasks[task_id]:
                x_eval = x_eval.to(device)
                y_eval = y_eval.to(device)
                preds = expert.model(x_eval).argmax(dim=-1)
                correct += (preds == y_eval).sum().item()
                total += y_eval.size(0)
        task_acc = correct / total if total > 0 else 0.0
        task_accuracies.append(task_acc)
        print(f"  Task {task_id+1} accuracy: {task_acc:.4f} (Expert {primary})")

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    # First, show individual expert accuracies on each task (diagnostic)
    print("\n  Individual Expert Accuracies (no routing):")
    print("  " + "-"*60)
    for task_id, test_loader in enumerate(test_tasks):
        accs = []
        for expert in pool.experts:
            expert.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    preds = expert.model(x).argmax(dim=-1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            accs.append(correct / total if total > 0 else 0.0)
        acc_str = " | ".join([f"E{i}:{a:.2%}" for i, a in enumerate(accs)])
        trained_by = expert_task_wins.get(task_id, "?")
        print(f"  Task {task_id+1} (digits {task_id*2},{task_id*2+1}): {acc_str}  [Trained: E{trained_by}]")
    print("  " + "-"*60)
    routing_strategy = config.get('routing_strategy', 'pseudo_label')

    # Finalize prototypes before eval (in case last task didn't trigger finalize)
    if routing_strategy == 'prototype':
        for expert in pool.experts:
            if expert.prototype_store is not None:
                expert.prototype_store.finalize()

    per_sample_mode = config.get('per_sample', False)
    top_k = config.get('top_k', 1)
    eval_bid_mode = config.get('eval_bid_mode', 'full')
    temperature = config.get('temperature', 1.0)

    if per_sample_mode or top_k > 1:
        print(f"\n  Per-Sample Ensemble Accuracy (top_k={top_k}, bid_mode={eval_bid_mode}, "
              f"temp={temperature}, routing={routing_strategy}):")

        for task_id, test_loader in enumerate(test_tasks):
            results = pool.evaluate_all_per_sample(
                test_loader,
                top_k=top_k,
                temperature=temperature,
                eval_bid_mode=eval_bid_mode,
                routing_strategy=routing_strategy
            )
            acc = results['ensemble_accuracy']
            final_accuracies.append(acc)

            trained_expert = expert_task_wins.get(task_id, "?")
            eval_expert = results['primary_eval_expert']
            status = "OK" if acc > 0.5 else "FAIL"

            match_str = "" if trained_expert == eval_expert else f" ** ROUTED TO Expert {eval_expert}!"
            print(f"  Task {task_id+1} (digits {task_id*2},{task_id*2+1}): "
                  f"{acc:.4f} {status} [Trained: Expert {trained_expert}]{match_str}")
            rates = results['expert_sample_rates']
            print(f"    Per-sample routing: " + ", ".join([f"E{k}:{v:.1%}" for k, v in sorted(rates.items())]))
            if 'top2_agreement' in results:
                print(f"    Top-2 agreement: {results['top2_agreement']:.4f}")
    else:
        print(f"\n  Ensemble Accuracy (routing: {routing_strategy}, bid_mode: {eval_bid_mode}):")

        for task_id, test_loader in enumerate(test_tasks):
            results = pool.evaluate_all(
                test_loader,
                routing_strategy=routing_strategy,
                bid_logger=bid_logger,
                task_id=task_id
            )
            acc = results['ensemble_accuracy']
            final_accuracies.append(acc)

            trained_expert = expert_task_wins.get(task_id, "?")
            eval_expert = results['primary_eval_expert']
            avg_costs = results['avg_forget_costs']
            status = "OK" if acc > 0.5 else "FAIL"

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
    parser.add_argument('--lambda_ewc', type=float, default=1000.0)
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
    parser.add_argument('--formula', type=str, default='addition', choices=['addition', 'subtraction'],
                        help='Bid formula: addition (default) or subtraction')
    parser.add_argument('--save_bids', action='store_true')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='Reset optimizer when expert switches tasks or is idle too long')
    parser.add_argument('--idle_threshold', type=int, default=100,
                        help='Batches of inactivity before optimizer reset (default: 100)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Unique name for this experiment run (used in output filenames)')
    parser.add_argument('--routing_strategy', type=str, default='pseudo_label',
                        choices=['pseudo_label', 'prototype'],
                        help='Evaluation routing strategy: pseudo_label (default) or prototype (feature-distance)')
    # Experiment 1: Per-sample top-k routing
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of experts to combine per sample (1=winner-take-all, 2=Mixtral-style)')
    parser.add_argument('--per_sample', action='store_true',
                        help='Enable per-sample routing (each sample routed independently)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for top-k expert combination')
    # Experiment 2: Distance-only bidding
    parser.add_argument('--eval_bid_mode', type=str, default='full',
                        choices=['full', 'distance_only'],
                        help='Eval bid mode: full (distance+forget) or distance_only')
    # Experiment 3: Training-time prototype routing
    parser.add_argument('--train_routing', type=str, default='label',
                        choices=['label', 'prototype'],
                        help='Training routing: label (default) or prototype (after warmup)')
    parser.add_argument('--train_routing_warmup', type=int, default=1000,
                        help='Batches of label-based warmup before switching to prototype routing')
    parser.add_argument('--use_conscience', action='store_true',
                        help='Enable conscience bias load-balancing in bid computation')
    parser.add_argument('--conscience_rate', type=float, default=0.01,
                        help='Conscience bias update rate (default: 0.01)')
    parser.add_argument('--conscience_decay', type=float, default=0.999,
                        help='Conscience bias decay factor (default: 0.999)')

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
        # Bid formula
        'bid_formula': args.formula,
        # Optimizer reset
        'reset_optimizer': args.reset_optimizer,
        'idle_threshold': args.idle_threshold,
        'routing_strategy': args.routing_strategy,
        'top_k': args.top_k,
        'per_sample': args.per_sample,
        'temperature': args.temperature,
        'eval_bid_mode': args.eval_bid_mode,
        'train_routing': args.train_routing,
        'train_routing_warmup': args.train_routing_warmup,
        'use_conscience': args.use_conscience,
        'conscience_rate': args.conscience_rate,
        'conscience_decay': args.conscience_decay,
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

    # Determine output prefix
    exp_name = args.experiment_name
    if exp_name:
        bids_path = f"results/{exp_name}_bids.json"
        results_path = f"results/{exp_name}_results.json"
    else:
        strategy = config.get('routing_strategy', 'pseudo_label')
        bids_path = f"results/mob_bids_{strategy}_seed_{args.seed}.json"
        results_path = f"results/mob_results_seed_{args.seed}.json"

    if args.save_bids and 'bid_logger' in results:
        results['bid_logger'].save_logs(bids_path)
        print("\n" + "="*70)
        print("BID DIAGNOSTICS")
        print("="*70)
        results['bid_logger'].print_diagnostics()

    summary = {
        'seed': args.seed,
        'experiment_name': exp_name,
        'config': config,
        'task_accuracies': results['task_accuracies'],
        'final_accuracies': results['final_accuracies'],
        'avg_accuracy': results['avg_accuracy'],
        'forgetting': results['forgetting'],
        'expert_task_wins': {str(k): v for k, v in results['expert_task_wins'].items()}
    }
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Results saved to: {results_path}")


if __name__ == '__main__':
    main()
