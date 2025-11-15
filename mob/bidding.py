"""
Bidding components for MoB: Mixture of Bidders.

This module implements the cost estimators that experts use to compute their bids:
- ExecutionCostEstimator: Predicts loss on the current batch
- EWCForgettingEstimator: Estimates catastrophic forgetting using EWC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ExecutionCostEstimator:
    """
    Estimates execution cost as the predicted loss on a given data batch.
    """
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device if device is not None else torch.device('cpu')

    def compute_predicted_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes the forward-pass cross-entropy loss for the batch.
        """
        self.model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
        return loss.item()


class EWCForgettingEstimator:
    """
    [PIVOTED IMPLEMENTATION] Estimates forgetting cost based on the total
    magnitude of protected knowledge (the Fisher Information). This avoids the
    "Paralyzed Expert" problem of the previous "predicted damage" approach.
    """
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 50000,
        forgetting_cost_scale: float = 1.0, # [NEW] Hyperparameter for the new strategy
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.forgetting_cost_scale = forgetting_cost_scale # Store the new parameter
        self.device = device if device is not None else torch.device('cpu')
        self.fisher = {}
        self.optimal_params = {}

    def update_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200
    ):
        """
        Computes and CUMULATIVELY updates the Fisher Information Matrix.
        This version correctly accumulates knowledge and uses unnormalized
        gradients to ensure a strong EWC signal.
        """
        self.model.train()

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.optimal_params[n] = p.clone().detach()

        current_task_fisher = {n: torch.zeros_like(p, device=self.device)
                               for n, p in self.model.named_parameters() if p.requires_grad}

        samples_seen = 0
        for x, y in dataloader:
            if samples_seen >= num_samples: break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            sampled_y = torch.distributions.Categorical(logits=logits).sample()
            loss = F.nll_loss(log_probs, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    current_task_fisher[n] += p.grad.data.pow(2)
            samples_seen += x.size(0)

        # Accumulate the raw, unnormalized Fisher to maintain a strong signal.
        for n, p_fisher in current_task_fisher.items():
            if n in self.fisher:
                self.fisher[n] += p_fisher.to(self.device)
            else:
                self.fisher[n] = p_fisher.to(self.device)

        fisher_stats = self.get_fisher_stats()
        if fisher_stats and samples_seen > 0:
            print(f"[EWC] Fisher updated: mean={fisher_stats['mean_importance']:.6e}, "
                  f"max={fisher_stats['max_importance']:.6e}, "
                  f"params={fisher_stats['total_params']}, samples={samples_seen}")
            assert fisher_stats['mean_importance'] > 0, "CRITICAL: Fisher matrix is all zeros!"

    def compute_forgetting_cost(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        [NEW STRATEGY] Cost is the total magnitude of the Fisher matrix,
        representing the value of all protected knowledge. This is a static
        measure of the expert's state, not a dynamic prediction.
        """
        if not self.fisher:
            return 0.0

        total_fisher_magnitude = sum(f.sum().item() for f in self.fisher.values())
        
        # Scale the magnitude to be on a comparable scale to the execution loss
        return self.forgetting_cost_scale * total_fisher_magnitude

    def penalty(self, verbose: bool = False) -> torch.Tensor:
        """
        Computes the EWC regularization term for training. This function is correct
        and remains unchanged.
        """
        if not self.fisher:
            return torch.tensor(0.0).to(self.device)
        penalty = torch.tensor(0.0).to(self.device)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                param_diff_sq = (p - self.optimal_params[n].to(self.device)).pow(2)
                penalty += (self.fisher[n] * param_diff_sq).sum()
        final_penalty = (self.lambda_ewc / 2.0) * penalty
        if verbose:
            print(f"[EWC] Penalty computed: {final_penalty.item():.4e} (λ={self.lambda_ewc})")
        return final_penalty

    def has_fisher(self) -> bool:
        return len(self.fisher) > 0

    def get_fisher_stats(self) -> Dict:
        if not self.fisher: return {}
        total_params = sum(f.numel() for f in self.fisher.values())
        mean_importance = sum(f.sum().item() for f in self.fisher.values()) / total_params if total_params > 0 else 0.0
        max_importance = max(f.max().item() for f in self.fisher.values()) if self.fisher else 0.0
        return {'total_params': total_params, 'mean_importance': mean_importance, 'max_importance': max_importance}