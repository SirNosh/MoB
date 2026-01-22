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
    """

    def __init__(
        self,
        expert_id: int,
        model: nn.Module,
        alpha: float,
        beta: float,
        lambda_ewc: float,
        forgetting_cost_scale: float, # [NEW] Added to support the pivoted strategy
        device: Optional[torch.device] = None
    ):
        """
        Initialize a MoB expert.
        """
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
            forgetting_cost_scale=forgetting_cost_scale, # [NEW] Pass the new hyperparameter
            device=self.device
        )

        self.batches_won = 0
        self.total_batches_seen = 0
        self.total_training_loss = 0.0

    def compute_bid(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict]:
        """
        Computes the expert's bid for a batch of data.
        """
        self.total_batches_seen += 1
        exec_cost = self.exec_estimator.compute_predicted_loss(x, y)
        forget_cost = self.forget_estimator.compute_forgetting_cost(x, y)
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
        """
        self.model.train()
        self.batches_won += 1
        x = x.to(self.device)
        y = y.to(self.device)
        optimizer.zero_grad()
        logits = self.model(x)
        task_loss = F.cross_entropy(logits, y)
        ewc_penalty = self.forget_estimator.penalty()
        total_loss = task_loss + ewc_penalty
        if self.batches_won <= 3:
            print(f"[Expert {self.expert_id}] Batch {self.batches_won}: "
                  f"task_loss={task_loss.item():.4f}, "
                  f"ewc_penalty={ewc_penalty.item():.4f}, "
                  f"total_loss={total_loss.item():.4f}, "
                  f"has_fisher={self.forget_estimator.has_fisher()}")
        total_loss.backward()
        optimizer.step()
        self.total_training_loss += total_loss.item()
        return {
            'task_loss': task_loss.item(),
            'ewc_penalty': ewc_penalty.item(),
            'total_loss': total_loss.item()
        }

    def update_after_task(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200
    ):
        """
        Updates the expert's EWC parameters after a task is finished.
        """
        self.forget_estimator.update_fisher(dataloader, num_samples=num_samples)

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        # This function is used for individual expert eval, not the main metric
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits, y, reduction='sum')
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return {'accuracy': accuracy, 'loss': avg_loss}

    def get_statistics(self) -> Dict:
        win_rate = self.batches_won / self.total_batches_seen if self.total_batches_seen > 0 else 0.0
        avg_loss = self.total_training_loss / self.batches_won if self.batches_won > 0 else 0.0
        stats = {
            'expert_id': self.expert_id, 'batches_won': self.batches_won,
            'win_rate': win_rate, 'avg_training_loss': avg_loss,
            'has_fisher': self.forget_estimator.has_fisher()
        }
        if self.forget_estimator.has_fisher():
            stats.update(self.forget_estimator.get_fisher_stats())
        return stats

    def reset_statistics(self):
        self.batches_won = 0
        self.total_batches_seen = 0
        self.total_training_loss = 0.0

    def save(self, path: str):
        # Implementation can be simplified or expanded as needed
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def __repr__(self) -> str:
        return f"MoBExpert(id={self.expert_id}, α={self.alpha}, β={self.beta}, batches_won={self.batches_won})"