"""
Baseline methods for comparing MoB performance.

This module implements four baseline approaches to validate the MoB architecture:
1. Naive Fine-tuning: Single model, no continual learning
2. Random Assignment: Multi-expert with random routing
3. Monolithic EWC: Single model with EWC
4. Gated MoE: Multi-expert with learned gater
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from .models import create_model
from .expert import MoBExpert


class NaiveFineTuning:
    """
    Baseline 1: Naive Fine-tuning

    A single CNN trained sequentially on tasks without any continual learning
    strategies. This establishes the lower bound performance showing maximum
    catastrophic forgetting.

    Expected: Worst performance, demonstrates the forgetting problem.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize naive fine-tuning baseline.

        Parameters:
        -----------
        model : nn.Module
            The neural network model.
        device : torch.device, optional
            Device to run on.
        """
        self.model = model
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)

    def train_on_task(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Dict:
        """
        Train on a single task without any CL strategies.

        Parameters:
        -----------
        dataloader : DataLoader
            Training data for the task.
        optimizer : torch.optim.Optimizer
            Optimizer for training.

        Returns:
        --------
        metrics : dict
            Training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'avg_loss': avg_loss, 'num_batches': num_batches}

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate the model on a dataset.

        Parameters:
        -----------
        dataloader : DataLoader
            Evaluation data.

        Returns:
        --------
        metrics : dict
            Evaluation metrics including accuracy and loss.
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = F.cross_entropy(logits, y, reduction='sum')

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        return {'accuracy': accuracy, 'loss': avg_loss}


class RandomAssignment:
    """
    Baseline 2: Random Assignment

    Uses the same multi-expert architecture as MoB but assigns batches to
    experts uniformly at random instead of using an auction. This isolates
    the value of the intelligent auction mechanism.

    Expected: Better than naive (due to multi-expert + EWC), but worse than MoB.
    """

    def __init__(
        self,
        num_experts: int,
        expert_config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize random assignment baseline.

        Parameters:
        -----------
        num_experts : int
            Number of expert agents.
        expert_config : dict
            Configuration for experts.
        device : torch.device, optional
            Device to run on.
        """
        self.num_experts = num_experts
        self.device = device if device is not None else torch.device('cpu')

        # Create experts
        self.experts = []
        for i in range(num_experts):
            model = create_model(
                architecture=expert_config['architecture'],
                num_classes=expert_config['num_classes'],
                input_channels=expert_config.get('input_channels', 1),
                dropout=expert_config.get('dropout', 0.5)
            )
            expert = MoBExpert(
                expert_id=i,
                model=model,
                alpha=expert_config.get('alpha', 0.5),
                beta=expert_config.get('beta', 0.5),
                lambda_ewc=expert_config.get('lambda_ewc', 5000),
                device=self.device
            )
            self.experts.append(expert)

        self.assignment_history = []

    def train_on_task(
        self,
        dataloader: DataLoader,
        optimizers: List[torch.optim.Optimizer]
    ) -> Dict:
        """
        Train on a task with random expert assignment.

        Parameters:
        -----------
        dataloader : DataLoader
            Training data.
        optimizers : list of torch.optim.Optimizer
            Optimizers for each expert.

        Returns:
        --------
        metrics : dict
            Training metrics.
        """
        expert_usage = np.zeros(self.num_experts)

        for x, y in dataloader:
            # Randomly select an expert
            winner_id = random.randint(0, self.num_experts - 1)

            # Train the selected expert
            self.experts[winner_id].train_on_batch(x, y, optimizers[winner_id])

            expert_usage[winner_id] += 1
            self.assignment_history.append(winner_id)

        return {'expert_usage': expert_usage / expert_usage.sum()}

    def update_after_task(self, dataloader: DataLoader, num_samples: int = 200):
        """Update EWC for all experts after task completion."""
        for expert in self.experts:
            expert.update_after_task(dataloader, num_samples=num_samples)

    def evaluate_all(self, dataloader: DataLoader) -> Dict:
        """Evaluate all experts and compute ensemble accuracy."""
        all_expert_logits = [[] for _ in range(self.num_experts)]
        all_labels = []

        for x, y in dataloader:
            all_labels.append(y)
            for i, expert in enumerate(self.experts):
                expert.model.eval()
                x_device = x.to(self.device)
                with torch.no_grad():
                    logits = expert.model(x_device)
                    all_expert_logits[i].append(logits.cpu())

        all_labels = torch.cat(all_labels)

        # Individual accuracies
        results = {}
        for i in range(self.num_experts):
            if len(all_expert_logits[i]) > 0:
                expert_preds = torch.cat([logits.argmax(dim=-1) for logits in all_expert_logits[i]])
                accuracy = (expert_preds == all_labels).float().mean().item()
                results[f'expert_{i}_accuracy'] = accuracy

        # Ensemble accuracy
        if len(all_expert_logits[0]) > 0:
            ensemble_preds = []
            num_batches = len(all_expert_logits[0])

            for j in range(num_batches):
                batch_logits = torch.stack([all_expert_logits[i][j] for i in range(self.num_experts)])
                avg_logits = batch_logits.mean(dim=0)
                ensemble_preds.append(avg_logits.argmax(dim=-1))

            ensemble_preds = torch.cat(ensemble_preds)
            results['ensemble_accuracy'] = (ensemble_preds == all_labels).float().mean().item()

        return results


class MonolithicEWC:
    """
    Baseline 3: Monolithic EWC

    A single CNN trained with EWC regularization. This is a strong standard
    continual learning baseline that isolates the architectural benefit of
    multi-expert systems.

    Expected: Strong performance, but MoB should beat it via specialization.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 5000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize monolithic EWC baseline.

        Parameters:
        -----------
        model : nn.Module
            The neural network model.
        lambda_ewc : float
            EWC regularization strength.
        device : torch.device, optional
            Device to run on.
        """
        self.device = device if device is not None else torch.device('cpu')

        # Create a single expert with EWC
        self.expert = MoBExpert(
            expert_id=0,
            model=model,
            alpha=1.0,  # Only execution cost matters for single model
            beta=0.0,
            lambda_ewc=lambda_ewc,
            device=self.device
        )

    def train_on_task(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> Dict:
        """
        Train on a task with EWC regularization.

        Parameters:
        -----------
        dataloader : DataLoader
            Training data.
        optimizer : torch.optim.Optimizer
            Optimizer for training.

        Returns:
        --------
        metrics : dict
            Training metrics.
        """
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            metrics = self.expert.train_on_batch(x, y, optimizer)
            total_loss += metrics['total_loss']
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {'avg_loss': avg_loss, 'num_batches': num_batches}

    def update_after_task(self, dataloader: DataLoader, num_samples: int = 200):
        """Update EWC Fisher matrix after task completion."""
        self.expert.update_after_task(dataloader, num_samples=num_samples)

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate the model on a dataset."""
        return self.expert.evaluate(dataloader)


class GatedMoE:
    """
    Baseline 4: Gated Mixture of Experts

    Multi-expert architecture with a centralized, learnable gater network.
    This is the "knockout test" - it should suffer from gater-level
    catastrophic forgetting, validating MoB's auction-based approach.

    Expected: Worse than MoB due to gater forgetting, possibly worse than random.
    """

    def __init__(
        self,
        num_experts: int,
        expert_config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize gated MoE baseline.

        Parameters:
        -----------
        num_experts : int
            Number of expert agents.
        expert_config : dict
            Configuration for experts.
        device : torch.device, optional
            Device to run on.
        """
        self.num_experts = num_experts
        self.device = device if device is not None else torch.device('cpu')

        # Create experts (without bidding - they're just models)
        self.expert_models = nn.ModuleList([
            create_model(
                architecture=expert_config['architecture'],
                num_classes=expert_config['num_classes'],
                input_channels=expert_config.get('input_channels', 1),
                dropout=expert_config.get('dropout', 0.5)
            ).to(self.device)
            for _ in range(num_experts)
        ])

        # Create the centralized gater network
        # This is the "monolithic dictator" that will suffer from forgetting
        input_size = expert_config.get('input_size', 784)  # 28*28 for MNIST
        self.gater = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_experts)
        ).to(self.device)

        self.routing_history = []
        self.gater_config = expert_config

    def train_on_task(
        self,
        dataloader: DataLoader,
        expert_optimizers: List[torch.optim.Optimizer],
        gater_optimizer: torch.optim.Optimizer
    ) -> Dict:
        """
        Train on a task with learned gating.

        Parameters:
        -----------
        dataloader : DataLoader
            Training data.
        expert_optimizers : list of torch.optim.Optimizer
            Optimizers for each expert.
        gater_optimizer : torch.optim.Optimizer
            Optimizer for the gater network.

        Returns:
        --------
        metrics : dict
            Training metrics.
        """
        expert_usage = np.zeros(self.num_experts)
        total_gater_loss = 0.0
        total_expert_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            # 1. Gater decides routing
            gating_logits = self.gater(x)

            # Top-1 routing: choose expert with highest score for each sample
            winner_ids = torch.argmax(gating_logits, dim=-1)

            # For simplicity, route the whole batch to the most common winner
            # (In a more sophisticated version, you'd handle mixed routing)
            winner_id = winner_ids.mode().values.item()

            # 2. Train the selected expert
            expert = self.expert_models[winner_id]
            optimizer = expert_optimizers[winner_id]

            expert.train()
            optimizer.zero_grad()

            logits = expert(x)
            expert_loss = F.cross_entropy(logits, y)
            expert_loss.backward()
            optimizer.step()

            # 3. Train the gater (this is where forgetting happens!)
            # The gater learns to route based on the current task
            gater_optimizer.zero_grad()

            # Gater loss: encourage routing to the expert that performed well
            # We use the expert's prediction quality as supervision
            with torch.no_grad():
                expert_quality = -F.cross_entropy(logits, y, reduction='none')

            # Compute gater loss: cross-entropy with soft labels based on expert quality
            target_distribution = F.softmax(
                torch.zeros(x.size(0), self.num_experts, device=self.device).scatter_(
                    1, winner_ids.unsqueeze(1), 1.0
                ),
                dim=-1
            )

            gater_loss = F.cross_entropy(gating_logits, winner_ids)
            gater_loss.backward()
            gater_optimizer.step()

            # Track statistics
            expert_usage[winner_id] += 1
            self.routing_history.append(winner_id)
            total_gater_loss += gater_loss.item()
            total_expert_loss += expert_loss.item()
            num_batches += 1

        return {
            'expert_usage': expert_usage / expert_usage.sum() if expert_usage.sum() > 0 else expert_usage,
            'avg_gater_loss': total_gater_loss / num_batches if num_batches > 0 else 0.0,
            'avg_expert_loss': total_expert_loss / num_batches if num_batches > 0 else 0.0
        }

    def evaluate_all(self, dataloader: DataLoader) -> Dict:
        """Evaluate all experts and compute ensemble accuracy."""
        all_expert_logits = [[] for _ in range(self.num_experts)]
        all_labels = []
        gater_routing = []

        for x, y in dataloader:
            x = x.to(self.device)
            all_labels.append(y)

            # Get gater's routing decision
            with torch.no_grad():
                gating_logits = self.gater(x)
                gater_routing.append(gating_logits.argmax(dim=-1).cpu())

            # Collect predictions from all experts
            for i, expert in enumerate(self.expert_models):
                expert.eval()
                with torch.no_grad():
                    logits = expert(x)
                    all_expert_logits[i].append(logits.cpu())

        all_labels = torch.cat(all_labels)
        gater_routing = torch.cat(gater_routing)

        # Individual expert accuracies
        results = {}
        for i in range(self.num_experts):
            if len(all_expert_logits[i]) > 0:
                expert_preds = torch.cat([logits.argmax(dim=-1) for logits in all_expert_logits[i]])
                accuracy = (expert_preds == all_labels).float().mean().item()
                results[f'expert_{i}_accuracy'] = accuracy

        # Ensemble accuracy (average logits)
        if len(all_expert_logits[0]) > 0:
            ensemble_preds = []
            num_batches = len(all_expert_logits[0])

            for j in range(num_batches):
                batch_logits = torch.stack([all_expert_logits[i][j] for i in range(self.num_experts)])
                avg_logits = batch_logits.mean(dim=0)
                ensemble_preds.append(avg_logits.argmax(dim=-1))

            ensemble_preds = torch.cat(ensemble_preds)
            results['ensemble_accuracy'] = (ensemble_preds == all_labels).float().mean().item()

        # Gated accuracy (using gater's routing)
        gated_preds = []
        sample_idx = 0
        for j in range(len(all_expert_logits[0])):
            batch_size = all_expert_logits[0][j].size(0)
            for k in range(batch_size):
                expert_id = gater_routing[sample_idx].item()
                pred = all_expert_logits[expert_id][j][k].argmax().item()
                gated_preds.append(pred)
                sample_idx += 1

        gated_preds = torch.tensor(gated_preds)
        results['gated_accuracy'] = (gated_preds == all_labels).float().mean().item()

        return results
