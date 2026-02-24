"""
Expert Pool management for MoB: Mixture of Bidders.

This module provides the ExpertPool class, which manages a collection of
independent MoBExpert agents.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  # Ensure F is imported
from typing import Dict, List, Optional, Tuple

from .expert import MoBExpert
from .models import create_model


class ShiftDetector:
    """
    Detects distribution shifts using Exponential Moving Average (EMA) of execution cost.
    """
    def __init__(self, alpha: float = 0.99, threshold_multiplier: float = 50.0):
        self.alpha = alpha  # Smoothing factor (high = slow adaptation)
        self.threshold_multiplier = threshold_multiplier
        self.ema_cost = None
        self.shift_cooldown = 0

    def update(self, cost: float) -> bool:
        """
        Update the tracker and check for shift.
        Returns True if a significant upward spike is detected.
        """
        # If cooldown is active, decrement and ignore (allow new task to stabilize)
        if self.shift_cooldown > 0:
            self.shift_cooldown -= 1
            # Still update EMA quickly during cooldown to adapting to new normal
            if self.ema_cost is None:
                self.ema_cost = cost
            else:
                self.ema_cost = 0.5 * self.ema_cost + 0.5 * cost
            return False

        if self.ema_cost is None:
            self.ema_cost = cost
            return False

        # Check for spike with a minimum floor to avoid noise at very low losses
        # If ema_cost is tiny (e.g. 0.001), a small jump to 0.05 is 50x but meaningless.
        # We enforce that the cost must also be significantly larger than a baseline (e.g. 0.5)
        # effectively: cost > max(ema_cost, 0.5) * threshold
        baseline = max(self.ema_cost, 0.5) 
        is_shift = cost > (baseline * self.threshold_multiplier)

        # Update EMA
        self.ema_cost = self.alpha * self.ema_cost + (1 - self.alpha) * cost
        
        if is_shift:
            # Reset/Cooldwon to absorb the new distribution
            self.shift_cooldown = 50 
            self.ema_cost = cost # Jump to new level
            
        return is_shift


class ExpertPool:
    """
    A collection of independent MoBExpert agents.

    This class manages the experts but contains NO centralized gater.
    The auction mechanism itself serves as the dynamic routing layer.
    """

    def __init__(
        self,
        num_experts: int,
        expert_config: Dict,
        device: Optional[torch.device] = None,
        use_shift_detection: bool = False,
        reset_optimizer: bool = False,
        idle_threshold: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Initialize the expert pool.
        """
        self.num_experts = num_experts
        self.expert_config = expert_config
        self.device = device if device is not None else torch.device('cpu')
        self.experts: List[MoBExpert] = []

        # Shift Detection
        self.shift_detector = ShiftDetector() if use_shift_detection else None

        # =====================================================================
        # Optimizer Reset Configuration (for idle-based reset in continual learning)
        # =====================================================================
        self.reset_optimizer = reset_optimizer
        self.idle_threshold = idle_threshold
        self.learning_rate = learning_rate
        self.global_batch_idx = 0  # Track global batch index
        # =====================================================================

        # Create experts
        for i in range(num_experts):
            model = self._create_expert_model(expert_config)
            expert = MoBExpert(
                expert_id=i,
                model=model,
                alpha=expert_config.get('alpha', 0.5),
                beta=expert_config.get('beta', 0.5),
                lambda_ewc=expert_config.get('lambda_ewc', 5000),
                forgetting_cost_scale=expert_config.get('forgetting_cost_scale', 1.0),
                device=self.device
            )
            self.experts.append(expert)

    def _create_expert_model(self, config: Dict) -> nn.Module:
        """
        Factory method for creating expert neural networks.
        """
        return create_model(
            architecture=config['architecture'],
            num_classes=config['num_classes'],
            input_channels=config.get('input_channels', 1),
            dropout=config.get('dropout', 0.5),
            input_size=config.get('input_size', 784),
            hidden_sizes=config.get('hidden_sizes', [256, 128])
        )

    def collect_bids(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Gathers bids from all experts for a given batch.
        
        Each expert uses its own Z-Score normalization based on running statistics,
        which preserves VCG independence (each bid depends only on the expert's own history).
        """
        bids = np.zeros(self.num_experts)
        components = []
        for i, expert in enumerate(self.experts):
            bid, comp = expert.compute_bid(x, y)
            bids[i] = bid
            components.append(comp)
        return bids, components

    def _should_reset_optimizer_on_shift(self, expert, shift_detected: bool) -> bool:
        """
        Check if expert's optimizer should be reset based on shift detection.

        For continual learning, we reset when:
        1. reset_optimizer is enabled AND
        2. A distribution shift was detected AND
        3. Expert has trained before (has Fisher - nothing to reset otherwise)

        This provides a natural "task boundary" signal in task-free learning.
        """
        if not self.reset_optimizer:
            return False

        if not shift_detected:
            return False

        # Never reset if expert hasn't trained yet
        if not expert.forget_estimator.has_fisher():
            return False

        return True

    def train_winner(
        self,
        winner_id: int,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizers: List[torch.optim.Optimizer]
    ) -> Dict:
        """
        Train the winning expert on the batch.
        """
        winner = self.experts[winner_id]

        # Check for distribution shift BEFORE training
        shift_detected = False
        if self.shift_detector:
            # We use the current loss as the signal
            # Note: winner.exec_estimator.compute_predicted_loss(x, y) was likely called during bidding
            # so we could optimization this, but recalculating is safer for clean code.
            with torch.no_grad():
                current_loss = winner.exec_estimator.compute_predicted_loss(x, y)
            shift_detected = self.shift_detector.update(current_loss)

        # =====================================================================
        # Optimizer Reset on Shift Detection
        # Key insight: Reset optimizer when distribution shift is detected.
        # This provides a natural "task boundary" signal in task-free learning.
        # =====================================================================
        optimizer_reset = False
        if self._should_reset_optimizer_on_shift(winner, shift_detected):
            # Reset optimizer for this expert
            optimizers[winner_id] = torch.optim.Adam(
                winner.model.parameters(),
                lr=self.learning_rate
            )
            print(f"  [Optimizer Reset] Expert {winner_id} optimizer reset (shift detected)")
            optimizer_reset = True
        # =====================================================================

        metrics = winner.train_on_batch(x, y, optimizers[winner_id])
        metrics['shift_detected'] = shift_detected
        metrics['optimizer_reset'] = optimizer_reset

        # Update tracking for optimizer reset logic
        winner.last_won_global_batch = self.global_batch_idx
        self.global_batch_idx += 1

        return metrics

    def consolidate(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200,
        expert_ids: Optional[List[int]] = None
    ):
        """
        Consolidates knowledge for specific experts by updating their EWC parameters.
        Triggered when a distribution shift is detected.
        
        Args:
            expert_ids: List of expert IDs to update. If None, updates ALL experts.
        """
        targets = expert_ids if expert_ids is not None else range(len(self.experts))
        
        for i in targets:
            self.experts[i].consolidate(dataloader, num_samples=num_samples)

    def evaluate_all(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Evaluates the MoB system using pseudo-label auction routing.

        IMPORTANT: This method does NOT use ground truth labels for routing.
        Instead, it uses pseudo-labels (model's own predictions) to compute
        bids, then routes to the expert with lowest bid.

        Key insight: Expert that's already good at this data will have:
        - Low exec_cost (low loss on its own predictions)
        - Low forget_cost (small gradients = already settled on this data)
        - Therefore LOW bid = WINS the auction
        """
        import math
        results = {}
        all_labels = []
        winner_preds = []

        # 1. Calculate individual expert accuracies (for diagnostics)
        for i, expert in enumerate(self.experts):
            expert.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in dataloader:
                    x_device = x.to(self.device)
                    y_device = y.to(self.device)
                    logits = expert.model(x_device)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y_device).sum().item()
                    total += len(y_device)
            accuracy = correct / total if total > 0 else 0.0
            results[f'expert_{i}_accuracy'] = accuracy

        # 2. Calculate MoB accuracy using PSEUDO-LABEL AUCTION routing (no ground truth labels!)
        for x, y in dataloader:
            x_device = x.to(self.device)
            y_device = y.to(self.device)
            all_labels.append(y_device.cpu())

            batch_bids = np.zeros(self.num_experts)
            batch_logits = []

            # Compute logits and bids for all experts using pseudo-labels
            for i, expert in enumerate(self.experts):
                expert.model.eval()
                with torch.no_grad():
                    logits = expert.model(x_device)
                    batch_logits.append(logits)

                # Use pseudo-labels (model's own predictions) instead of ground truth
                pseudo_labels = logits.argmax(dim=-1).detach()

                # Compute execution cost with pseudo-labels
                raw_exec = F.cross_entropy(logits, pseudo_labels).item()

                # Compute forgetting cost with pseudo-labels
                forget_cost = expert.forget_estimator.compute_forgetting_cost(x_device, pseudo_labels)

                # Same bid formula as training
                norm_exec = raw_exec / 2.5
                norm_forget = math.log1p(forget_cost) / 10.0
                bid = expert.alpha * norm_exec + expert.beta * norm_forget
                batch_bids[i] = bid

            # Select Winner: Lowest Bid = WINS the auction
            auction_winner_id = np.argmin(batch_bids)

            # DEBUG-LOG: Print evaluation decision occasionally
            if len(all_labels) == 1:  # Print only for FIRST batch of each evaluation call
                print(f"[EVAL DEBUG] Batch 0: Bids={np.round(batch_bids, 4)}")
                print(f"             Winner={auction_winner_id} (Min Bid)")

            # Get the winning expert's predictions for this batch
            winning_logits = batch_logits[auction_winner_id]
            winning_preds_batch = winning_logits.argmax(dim=-1).cpu()
            winner_preds.append(winning_preds_batch)

        # Concatenate all predictions and labels
        if all_labels:
            all_labels = torch.cat(all_labels)
            winner_preds = torch.cat(winner_preds)

            # Calculate the final accuracy based on the winners' predictions
            ensemble_accuracy = (winner_preds == all_labels).float().mean().item()
            results['ensemble_accuracy'] = ensemble_accuracy
        else:
            results['ensemble_accuracy'] = 0.0

        return results

    def get_expert_statistics(self) -> List[Dict]:
        """Get statistics for all experts."""
        return [expert.get_statistics() for expert in self.experts]

    def reset_statistics(self):
        """Reset statistics for all experts."""
        for expert in self.experts:
            expert.reset_statistics()

    def save_all(self, directory: str):
        """Save all experts to a directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        for expert in self.experts:
            path = os.path.join(directory, f'expert_{expert.expert_id}.pt')
            expert.save(path)

    def load_all(self, directory: str):
        """Load all experts from a directory."""
        import os
        for expert in self.experts:
            path = os.path.join(directory, f'expert_{expert.expert_id}.pt')
            if os.path.exists(path):
                expert.load(path)

    def __len__(self) -> int:
        return self.num_experts

    def __getitem__(self, idx: int) -> MoBExpert:
        return self.experts[idx]

    def __repr__(self) -> str:
        return f"ExpertPool(num_experts={self.num_experts}, device={self.device})"