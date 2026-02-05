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
        self.batches_won_this_task = 0  # Per-task counter for logging
        self.total_batches_seen = 0
        self.total_training_loss = 0.0

    def compute_bid(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict]:
        """
        Computes the expert's bid using RAW SCALED normalization:
        
        1. Execution cost: Raw value scaled (no Z-Score, no per-expert averaging)
        2. Forgetting cost: Log-scale (natural compression of 0 to 500,000+ range)
        
        This approach:
        - Lower exec cost = lower bid = wins (correct incentive!)
        - No per-expert averaging that erases winner's advantage
        - Log forget naturally handles huge forgetting cost ranges
        - Preserves VCG independence (each bid depends only on own costs)
        """
        self.total_batches_seen += 1
        
        # Get raw costs
        raw_exec = self.exec_estimator.compute_predicted_loss(x, y)
        raw_forget = self.forget_estimator.compute_forgetting_cost(x, y)
        
        # === EXECUTION COST: Raw scaled ===
        # Cross-entropy on 10-class starts at ~2.3, drops to ~0.1 when learned
        # Scale to reasonable range (~0 to 1)
        import math
        norm_exec = raw_exec / 2.5
        
        # === FORGETTING COST: Log-scale normalization ===
        # Handles the huge range (0 to 500,000+) naturally
        # log(1 + x) maps: 0 -> 0, 100 -> 4.6, 10000 -> 9.2, 100000 -> 11.5
        log_forget = math.log1p(raw_forget)
        norm_forget = log_forget / 10.0  # Scale to ~0 to 1.5 range
        
        # Compute final bid
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
        """
        Trains the expert on its winning batch, applying the EWC penalty.
        """
        self.model.train()
        self.batches_won += 1
        self.batches_won_this_task += 1  # Track per-task for logging
        x = x.to(self.device)
        y = y.to(self.device)
        optimizer.zero_grad()
        logits = self.model(x)
        task_loss = F.cross_entropy(logits, y)
        ewc_penalty = self.forget_estimator.penalty()
        total_loss = task_loss + ewc_penalty
        # Log first 3 batches per task (not just first 3 ever)
        if self.batches_won_this_task <= 3:
            print(f"[Expert {self.expert_id}] Task Batch {self.batches_won_this_task}: "
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
        """Reset all cumulative statistics."""
        self.batches_won = 0
        self.batches_won_this_task = 0
        self.total_batches_seen = 0
        self.total_training_loss = 0.0

    def reset_task_statistics(self):
        """Reset per-task statistics at the start of a new task."""
        self.batches_won_this_task = 0

    def save(self, path: str):
        # Implementation can be simplified or expanded as needed
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def __repr__(self) -> str:
        return f"MoBExpert(id={self.expert_id}, α={self.alpha}, β={self.beta}, batches_won={self.batches_won})"