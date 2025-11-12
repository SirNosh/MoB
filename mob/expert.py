"""
Expert agent implementation for MoB: Mixture of Bidders.

This module defines the MoBExpert class, which encapsulates an expert neural network
along with its bidding and training logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .bidding import ExecutionCostEstimator, EWCForgettingEstimator


class MoBExpert:
    """
    An expert agent in the MoB system.

    It encapsulates a neural network model and the logic for computing bids
    and training. Each expert competes in auctions to process data batches,
    bidding based on their predicted performance and forgetting cost.
    """

    def __init__(
        self,
        expert_id: int,
        model: nn.Module,
        alpha: float = 0.5,
        beta: float = 0.5,
        lambda_ewc: float = 5000,
        forgetting_cost_lr: float = 0.001,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a MoB expert.

        Parameters:
        -----------
        expert_id : int
            Unique identifier for this expert.
        model : nn.Module
            The neural network model for this expert.
        alpha : float
            Weight for execution cost in the bid calculation.
        beta : float
            Weight for forgetting cost in the bid calculation.
        lambda_ewc : float
            EWC regularization strength.
        forgetting_cost_lr : float
            Hypothetical learning rate for forgetting cost approximation.
            Default: 0.001
        device : torch.device, optional
            Device to run computations on.
        """
        self.expert_id = expert_id
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.device = device if device is not None else torch.device('cpu')

        # Move model to device
        self.model.to(self.device)

        # Bidding component estimators
        self.exec_estimator = ExecutionCostEstimator(model, device=self.device)
        self.forget_estimator = EWCForgettingEstimator(
            model,
            lambda_ewc=lambda_ewc,
            forgetting_cost_lr=forgetting_cost_lr,
            device=self.device
        )

        # Statistics tracking
        self.batches_won = 0
        self.total_batches_seen = 0
        self.total_training_loss = 0.0

    def compute_bid(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict]:
        """
        Computes the expert's bid for a batch of data.

        Bid = α * ExecutionCost + β * ForgettingCost

        Parameters:
        -----------
        x : torch.Tensor
            Input data batch.
        y : torch.Tensor
            Target labels.

        Returns:
        --------
        bid : float
            The expert's bid for processing this batch.
        components : dict
            Breakdown of bid components for analysis.
        """
        self.total_batches_seen += 1

        # Compute execution cost (predicted loss)
        exec_cost = self.exec_estimator.compute_predicted_loss(x, y)

        # Compute forgetting cost (EWC cost)
        forget_cost = self.forget_estimator.compute_forgetting_cost(x, y)

        # Calculate final bid
        bid = self.alpha * exec_cost + self.beta * forget_cost

        components = {
            'exec_cost': exec_cost,
            'forget_cost': forget_cost,
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
        """
        Trains the expert on its winning batch, applying the EWC penalty.

        Parameters:
        -----------
        x : torch.Tensor
            Input data batch.
        y : torch.Tensor
            Target labels.
        optimizer : torch.optim.Optimizer
            Optimizer for this expert.

        Returns:
        --------
        metrics : dict
            Training metrics including losses.
        """
        self.model.train()
        self.batches_won += 1

        # Move data to device
        x = x.to(self.device)
        y = y.to(self.device)

        optimizer.zero_grad()

        # Forward pass
        logits = self.model(x)
        task_loss = F.cross_entropy(logits, y)

        # Add EWC penalty
        ewc_penalty = self.forget_estimator.penalty()

        # Total loss
        total_loss = task_loss + ewc_penalty

        # [EWC VERIFICATION] Log first few batches to verify EWC is being applied
        if self.batches_won <= 3:
            print(f"[Expert {self.expert_id}] Batch {self.batches_won}: "
                  f"task_loss={task_loss.item():.4f}, "
                  f"ewc_penalty={ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else ewc_penalty:.4f}, "
                  f"total_loss={total_loss.item():.4f}, "
                  f"has_fisher={self.forget_estimator.has_fisher()}")

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Track statistics
        self.total_training_loss += total_loss.item()

        metrics = {
            'task_loss': task_loss.item(),
            'ewc_penalty': ewc_penalty.item() if isinstance(ewc_penalty, torch.Tensor) else ewc_penalty,
            'total_loss': total_loss.item()
        }

        return metrics

    def update_after_task(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200
    ):
        """
        Updates the expert's EWC parameters after a task is finished.

        This computes the Fisher Information Matrix for the completed task,
        which will be used to prevent forgetting in future tasks.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for the completed task.
        num_samples : int
            Number of samples to use for Fisher computation.
        """
        self.forget_estimator.update_fisher(dataloader, num_samples=num_samples)

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Evaluates the expert on a given dataset.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for evaluation.

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

        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }

        return metrics

    def get_statistics(self) -> Dict:
        """
        Get statistics about this expert's performance.

        Returns:
        --------
        stats : dict
            Dictionary containing expert statistics.
        """
        win_rate = self.batches_won / self.total_batches_seen if self.total_batches_seen > 0 else 0.0
        avg_loss = self.total_training_loss / self.batches_won if self.batches_won > 0 else 0.0

        stats = {
            'expert_id': self.expert_id,
            'batches_won': self.batches_won,
            'total_batches_seen': self.total_batches_seen,
            'win_rate': win_rate,
            'avg_training_loss': avg_loss,
            'has_fisher': self.forget_estimator.has_fisher()
        }

        # Add Fisher statistics if available
        if self.forget_estimator.has_fisher():
            stats.update(self.forget_estimator.get_fisher_stats())

        return stats

    def reset_statistics(self):
        """Reset the expert's statistics counters."""
        self.batches_won = 0
        self.total_batches_seen = 0
        self.total_training_loss = 0.0

    def save(self, path: str):
        """
        Save the expert's model and EWC parameters.

        Parameters:
        -----------
        path : str
            Path to save the expert.
        """
        state = {
            'expert_id': self.expert_id,
            'model_state_dict': self.model.state_dict(),
            'fisher': self.forget_estimator.fisher,
            'optimal_params': self.forget_estimator.optimal_params,
            'alpha': self.alpha,
            'beta': self.beta,
            'lambda_ewc': self.forget_estimator.lambda_ewc,
            'statistics': self.get_statistics()
        }
        torch.save(state, path)

    def load(self, path: str):
        """
        Load the expert's model and EWC parameters.

        Parameters:
        -----------
        path : str
            Path to load the expert from.
        """
        state = torch.load(path, map_location=self.device)

        self.expert_id = state['expert_id']
        self.model.load_state_dict(state['model_state_dict'])
        self.forget_estimator.fisher = state['fisher']
        self.forget_estimator.optimal_params = state['optimal_params']
        self.alpha = state['alpha']
        self.beta = state['beta']
        self.forget_estimator.lambda_ewc = state['lambda_ewc']

    def __repr__(self) -> str:
        """String representation of the expert."""
        return (
            f"MoBExpert(id={self.expert_id}, "
            f"α={self.alpha}, β={self.beta}, "
            f"batches_won={self.batches_won})"
        )
