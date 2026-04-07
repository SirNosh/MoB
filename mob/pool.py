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
        device: Optional[torch.device] = None
    ):
        """
        Initialize the expert pool.
        """
        self.num_experts = num_experts
        self.expert_config = expert_config
        self.device = device if device is not None else torch.device('cpu')
        self.experts: List[MoBExpert] = []

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
        which preserves bid independence (each bid depends only on the expert's own history).
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
        metrics = winner.train_on_batch(x, y, optimizers[winner_id])
        return metrics

    def update_after_task(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200
    ):
        """
        Updates all experts' EWC parameters after a task is finished.
        """
        for expert in self.experts:
            expert.update_after_task(dataloader, num_samples=num_samples)

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
            winner_id = np.argmin(batch_bids)

            # Get the winning expert's predictions for this batch
            winning_logits = batch_logits[winner_id]
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