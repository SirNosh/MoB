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

    This provides a crucial, data-dependent signal about expert-task fit.
    A lower loss indicates better suitability for the current data, resulting
    in a lower execution cost and thus a more competitive bid.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize the execution cost estimator.

        Parameters:
        -----------
        model : nn.Module
            The neural network model to evaluate.
        device : torch.device, optional
            Device to run computations on. If None, uses CPU.
        """
        self.model = model
        self.device = device if device is not None else torch.device('cpu')

    def compute_predicted_loss(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes the forward-pass cross-entropy loss for the batch.

        A lower loss signifies a better fit and thus a lower execution cost.

        Parameters:
        -----------
        x : torch.Tensor
            Input data batch.
        y : torch.Tensor
            Target labels.

        Returns:
        --------
        loss : float
            The mean cross-entropy loss for the batch.
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
    Estimates forgetting cost using Elastic Weight Consolidation (EWC).

    The Fisher Information Matrix identifies parameters crucial for past tasks.
    This allows quantification of the potential catastrophic forgetting that
    would result from training on new data.

    Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting
    in neural networks"
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 5000,
        forgetting_cost_lr: float = 0.001,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the EWC forgetting estimator.

        Parameters:
        -----------
        model : nn.Module
            The neural network model to protect from forgetting.
        lambda_ewc : float
            EWC regularization strength coefficient.
        forgetting_cost_lr : float
            Hypothetical learning rate for forgetting cost approximation.
            Used to estimate parameter change magnitude when computing forgetting cost.
            Default: 0.001
        device : torch.device, optional
            Device to run computations on. If None, uses CPU.
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.forgetting_cost_lr = forgetting_cost_lr
        self.device = device if device is not None else torch.device('cpu')
        self.fisher = {}
        self.optimal_params = {}

    def update_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200
    ):
        """
        Computes and updates the diagonal Fisher Information Matrix after a task is learned.

        The Fisher matrix approximation uses the gradient of the log-likelihood,
        which captures the importance of each parameter for the current task.

        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for the completed task.
        num_samples : int
            Number of samples to use for Fisher computation (for efficiency).
        """
        self.model.eval()

        # Store the current optimal parameters and initialize Fisher matrix.
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.optimal_params[n] = p.clone().detach()
                self.fisher[n] = torch.zeros_like(p)

        # Accumulate Fisher Information over a subset of the task data.
        samples_seen = 0
        for x, y in dataloader:
            if samples_seen >= num_samples:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            self.model.zero_grad()
            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=-1)

            # Sample from the model's predictive distribution
            sampled_y = torch.distributions.Categorical(logits=logits).sample()
            loss = F.nll_loss(log_probs, sampled_y)
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal approximation)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.data.pow(2)

            samples_seen += x.size(0)

        # Normalize the Fisher matrix.
        for n in self.fisher:
            self.fisher[n] /= samples_seen

        # [EWC VERIFICATION] Log Fisher statistics to verify computation
        fisher_stats = self.get_fisher_stats()
        if fisher_stats:
            mean_imp = fisher_stats['mean_importance']
            max_imp = fisher_stats['max_importance']
            print(f"[EWC] Fisher updated: mean={mean_imp:.6e}, max={max_imp:.6e}, "
                  f"params={fisher_stats['total_params']}, samples={samples_seen}")

            # Assertion to catch Fisher matrix being all zeros (bug indicator)
            assert mean_imp > 0, (
                "CRITICAL: Fisher matrix is all zeros! This indicates EWC is not working. "
                "Possible causes: (1) gradients not computed, (2) samples_seen=0, "
                "(3) model not properly initialized."
            )

    def compute_forgetting_cost(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Estimates the forgetting cost that would be incurred by training on batch (x, y).

        This is approximated by the expected parameter update (gradient) weighted by
        the Fisher importance. It represents how much training on this batch would
        damage knowledge of previous tasks.

        Parameters:
        -----------
        x : torch.Tensor
            Input data batch.
        y : torch.Tensor
            Target labels.

        Returns:
        --------
        cost : float
            Estimated forgetting cost (EWC cost).
        """
        if not self.fisher:
            return 0.0  # No forgetting on the first task.

        x = x.to(self.device)
        y = y.to(self.device)

        self.model.train()  # Set to train to get gradients
        self.model.zero_grad()

        # Calculate the gradient for the current batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=False
        )

        # Calculate the EWC cost: Σ F_i * (Δθ_i)^2
        # We approximate Δθ with the current gradient (scaled by a conceptual learning rate).
        cost = 0.0

        for (n, p), grad in zip(self.model.named_parameters(), grads):
            if n in self.fisher and grad is not None:
                # Use configurable forgetting_cost_lr for parameter change approximation
                cost += (self.fisher[n] * (self.forgetting_cost_lr * grad).pow(2)).sum().item()

        return cost

    def penalty(self, verbose: bool = False) -> torch.Tensor:
        """
        Computes the EWC regularization term to be added to the main loss during training.

        Loss_EWC = (λ/2) Σ F_i * (θ_i - θ*_i)^2

        This penalty term is added to the task loss to prevent the model from
        deviating too much from parameters that were important for previous tasks.

        Parameters:
        -----------
        verbose : bool
            If True, print penalty value for debugging (default: False).

        Returns:
        --------
        penalty : torch.Tensor
            The EWC penalty term (scalar tensor).
        """
        if not self.fisher:
            return torch.tensor(0.0).to(self.device)

        penalty = torch.tensor(0.0).to(self.device)

        for n, p in self.model.named_parameters():
            if n in self.fisher:
                param_diff_sq = (p - self.optimal_params[n].to(self.device)).pow(2)
                penalty += (self.fisher[n] * param_diff_sq).sum()

        final_penalty = (self.lambda_ewc / 2) * penalty

        # [EWC VERIFICATION] Log penalty if requested
        if verbose:
            print(f"[EWC] Penalty computed: {final_penalty.item():.4e} (λ={self.lambda_ewc})")

        return final_penalty

    def has_fisher(self) -> bool:
        """
        Check if Fisher information has been computed.

        Returns:
        --------
        has_fisher : bool
            True if Fisher matrix exists, False otherwise.
        """
        return len(self.fisher) > 0

    def get_fisher_stats(self) -> Dict:
        """
        Get statistics about the Fisher information matrix.

        Returns:
        --------
        stats : dict
            Dictionary containing Fisher matrix statistics.
        """
        if not self.fisher:
            return {}

        total_params = 0
        mean_importance = 0.0
        max_importance = 0.0

        for n, fisher_matrix in self.fisher.items():
            total_params += fisher_matrix.numel()
            mean_importance += fisher_matrix.sum().item()
            max_importance = max(max_importance, fisher_matrix.max().item())

        return {
            'total_params': total_params,
            'mean_importance': mean_importance / total_params if total_params > 0 else 0.0,
            'max_importance': max_importance
        }
