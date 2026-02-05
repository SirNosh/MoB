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
        use_shift_detection: bool = False
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

        metrics = winner.train_on_batch(x, y, optimizers[winner_id])
        metrics['shift_detected'] = shift_detected
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
        Evaluates the MoB system using confidence-based expert routing.
        
        IMPORTANT: This method does NOT use ground truth labels for routing.
        Instead, it selects the expert with highest prediction confidence
        (maximum softmax probability), which is the standard, fair approach
        for MoE evaluation in continual learning research.
        
        Reference: This follows the evaluation protocol from:
        - Aljundi et al. (2017) "Expert Gate"
        - Rusu et al. (2016) "Progressive Neural Networks"
        """
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

        # 2. Calculate MoB accuracy using CONFIDENCE-BASED routing (no labels!)
        for x, y in dataloader:
            x_device = x.to(self.device)
            y_device = y.to(self.device)
            all_labels.append(y_device.cpu())

            batch_confidences = np.zeros(self.num_experts)
            batch_logits = []
            
            # 1. Compute logits for all experts (needed for predictions)
            for expert in self.experts:
                expert.model.eval()
                with torch.no_grad():
                     batch_logits.append(expert.model(x_device))

            # 2. Determine winner using MoB AUCTION logic
            # Use 'collect_bids' to get the Bids (Cost) for each expert
            bids, _ = self.collect_bids(x, y) 
            
            # 3. Select Winner: Lowest Bid = Lowest Cost = Winner
            # (Previously we used argmax confidence, which favored overconfident experts on OOD data)
            auction_winner_id = np.argmin(bids)
            
            # DEBUG-LOG: Print evaluation decision occasionally
            if len(all_labels) == 1: # Print only for FIRST batch of each evaluation call
                print(f"[EVAL DEBUG] Batch 0: Bids(Costs)={np.round(bids, 4)}")
                print(f"             Winner={auction_winner_id} (Min Bid) | Labels={y_device[:5].cpu().tolist()}")
            
            # Get the winning expert's predictions for this batch
            winning_logits = batch_logits[auction_winner_id]
            winning_preds_batch = winning_logits.argmax(dim=-1).cpu()
            winner_preds.append(winning_preds_batch)

        # Concatenate all predictions and labels
        if all_labels:
            all_labels = torch.cat(all_labels)
            winner_preds = torch.cat(winner_preds)
            
            # Calculate the final accuracy based on the winners' predictions
            # The key 'ensemble_accuracy' is kept for consistency with the benchmark script
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