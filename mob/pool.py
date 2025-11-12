"""
Expert Pool management for MoB: Mixture of Bidders.

This module provides the ExpertPool class, which manages a collection of
independent MoBExpert agents.
"""

import torch
import torch.nn as nn
import numpy as np
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

        Parameters:
        -----------
        num_experts : int
            Number of expert agents to create.
        expert_config : dict
            Configuration for expert models. Should contain:
            - 'architecture': Model architecture name
            - 'num_classes': Number of output classes
            - 'input_channels': Number of input channels
            - 'alpha': Weight for execution cost
            - 'beta': Weight for forgetting cost
            - 'lambda_ewc': EWC regularization strength
            And any architecture-specific parameters.
        device : torch.device, optional
            Device to run computations on.
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
                device=self.device
            )
            self.experts.append(expert)

    def _create_expert_model(self, config: Dict) -> nn.Module:
        """
        Factory method for creating expert neural networks.

        Parameters:
        -----------
        config : dict
            Configuration dictionary for the model.

        Returns:
        --------
        model : nn.Module
            Initialized neural network model.
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

        Parameters:
        -----------
        x : torch.Tensor
            Input data batch.
        y : torch.Tensor
            Target labels.

        Returns:
        --------
        bids : np.ndarray
            Array of bids from all experts.
        components : list of dict
            Breakdown of bid components for each expert.
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

        Parameters:
        -----------
        winner_id : int
            ID of the winning expert.
        x : torch.Tensor
            Input data batch.
        y : torch.Tensor
            Target labels.
        optimizers : list of torch.optim.Optimizer
            List of optimizers for each expert.

        Returns:
        --------
        metrics : dict
            Training metrics from the winner.
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

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for the completed task.
        num_samples : int
            Number of samples to use for Fisher computation.
        """
        for expert in self.experts:
            expert.update_after_task(dataloader, num_samples=num_samples)

    def evaluate_all(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Evaluates all experts on a given dataset.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for evaluation.

        Returns:
        --------
        results : dict
            Dictionary with individual expert metrics and ensemble metrics.
        """
        results = {}
        all_expert_logits = [[] for _ in range(self.num_experts)]
        all_labels = []

        # Collect predictions from all experts (keep on device for efficiency)
        for x, y in dataloader:
            all_labels.append(y)
            for i, expert in enumerate(self.experts):
                expert.model.eval()
                x_device = x.to(self.device)
                with torch.no_grad():
                    logits = expert.model(x_device)
                    # Keep logits on device during collection for efficiency
                    all_expert_logits[i].append(logits)

        # Concatenate all batches and move to CPU once (more efficient than per-batch)
        all_labels = torch.cat(all_labels)

        # Calculate individual accuracies (concatenate on device, then move to CPU)
        for i in range(self.num_experts):
            if len(all_expert_logits[i]) > 0:
                # Concatenate on device first for efficiency
                expert_logits_concat = torch.cat(all_expert_logits[i])
                expert_preds = expert_logits_concat.argmax(dim=-1).cpu()
                accuracy = (expert_preds == all_labels).float().mean().item()
                results[f'expert_{i}_accuracy'] = accuracy
            else:
                results[f'expert_{i}_accuracy'] = 0.0

        # Calculate ensemble accuracy (average logits)
        if len(all_expert_logits[0]) > 0:
            ensemble_preds = []
            num_batches = len(all_expert_logits[0])

            for j in range(num_batches):
                # Stack and average on device, move to CPU only for final prediction
                batch_logits = torch.stack([all_expert_logits[i][j] for i in range(self.num_experts)])
                avg_logits = batch_logits.mean(dim=0)
                ensemble_preds.append(avg_logits.argmax(dim=-1).cpu())

            ensemble_preds = torch.cat(ensemble_preds)
            ensemble_accuracy = (ensemble_preds == all_labels).float().mean().item()
            results['ensemble_accuracy'] = ensemble_accuracy
        else:
            results['ensemble_accuracy'] = 0.0

        return results

    def get_expert_statistics(self) -> List[Dict]:
        """
        Get statistics for all experts.

        Returns:
        --------
        stats : list of dict
            List of statistics dictionaries, one per expert.
        """
        return [expert.get_statistics() for expert in self.experts]

    def reset_statistics(self):
        """Reset statistics for all experts."""
        for expert in self.experts:
            expert.reset_statistics()

    def save_all(self, directory: str):
        """
        Save all experts to a directory.

        Parameters:
        -----------
        directory : str
            Directory to save expert files.
        """
        import os
        os.makedirs(directory, exist_ok=True)

        for expert in self.experts:
            path = os.path.join(directory, f'expert_{expert.expert_id}.pt')
            expert.save(path)

    def load_all(self, directory: str):
        """
        Load all experts from a directory.

        Parameters:
        -----------
        directory : str
            Directory containing expert files.
        """
        import os

        for expert in self.experts:
            path = os.path.join(directory, f'expert_{expert.expert_id}.pt')
            if os.path.exists(path):
                expert.load(path)

    def __len__(self) -> int:
        """Return the number of experts."""
        return self.num_experts

    def __getitem__(self, idx: int) -> MoBExpert:
        """Get an expert by index."""
        return self.experts[idx]

    def __repr__(self) -> str:
        """String representation of the pool."""
        return f"ExpertPool(num_experts={self.num_experts}, device={self.device})"
