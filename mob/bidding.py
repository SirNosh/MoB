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
    Elastic Weight Consolidation (EWC) Forgetting Estimator.
    
    Implements the EWC regularization method for continual learning, adapted
    for the MoB auction-based expert routing framework.
    
    This implementation follows:
    - Original EWC: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting 
      in neural networks" (PNAS)
    - Online EWC: Schwarz et al. (2018) "Progress & Compress: A scalable framework 
      for continual learning" (ICML)
    
    Key features:
    - Diagonal Fisher Information Matrix approximation
    - Online EWC with exponential moving average for Fisher and optimal params
    - Input-dependent forgetting cost using gradient interference
    """
    
    # Decay factor for Online EWC (gamma in Schwarz et al. 2018)
    FISHER_DECAY = 0.9
    
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 10.0,
        forgetting_cost_scale: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the EWC Forgetting Estimator.
        
        Parameters:
        -----------
        model : nn.Module
            The neural network model to protect.
        lambda_ewc : float
            EWC regularization strength (λ in the original paper).
            Higher values = stronger protection of old knowledge.
        forgetting_cost_scale : float
            Scaling factor for the forgetting cost in bidding.
        device : torch.device, optional
            Device for computations.
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.forgetting_cost_scale = forgetting_cost_scale
        self.device = device if device is not None else torch.device('cpu')
        
        # Fisher Information Matrix (diagonal approximation)
        # F_i = E[(∂log p(y|x)/∂θ_i)²]
        self.fisher: Dict[str, torch.Tensor] = {}
        
        # Optimal parameters at task completion (θ* in the EWC paper)
        self.optimal_params: Dict[str, torch.Tensor] = {}
        
        # Track number of tasks for proper averaging
        self.num_tasks_consolidated = 0

    def update_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200
    ):
        """
        Compute and update the Fisher Information Matrix after task completion.
        
        Uses Online EWC (Schwarz et al. 2018) with exponential moving average
        to avoid unbounded Fisher growth across tasks.
        
        The Fisher Information is computed as:
            F_i = (1/N) * Σ_n (∂L/∂θ_i)²
        
        where L is the log-likelihood (negative cross-entropy).
        
        Parameters:
        -----------
        dataloader : DataLoader
            Data from the completed task for Fisher computation.
        num_samples : int
            Number of samples to use for Fisher estimation (default: 200).
        """
        self.model.train()
        
        # Compute Fisher Information for current task
        current_fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters() if p.requires_grad
        }

        samples_seen = 0
        for x, y in dataloader:
            if samples_seen >= num_samples:
                break
                
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)
            
            self.model.zero_grad()
            logits = self.model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Empirical Fisher: use true labels (standard practice)
            # Alternative: use model's own predictions (original EWC paper)
            loss = F.nll_loss(log_probs, y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    current_fisher[n] += p.grad.data.pow(2) * batch_size
                    
            samples_seen += batch_size

        # Normalize by number of samples
        if samples_seen > 0:
            for n in current_fisher:
                current_fisher[n] /= samples_seen

        # Online EWC: exponential moving average of Fisher matrices
        # This prevents unbounded growth (Issue 4 fix)
        gamma = self.FISHER_DECAY
        for n, f_new in current_fisher.items():
            if n in self.fisher:
                # Weighted combination: γ * F_old + (1-γ) * F_new
                self.fisher[n] = gamma * self.fisher[n] + (1 - gamma) * f_new
            else:
                self.fisher[n] = f_new.clone()

        # CRITICAL: Normalize Fisher to have mean = 1.0
        # This ensures consistent EWC strength across different model scales
        # and allows lambda_ewc to be in a reasonable range (1.0 - 10.0)
        self._normalize_fisher()

        # Online EWC: update optimal parameters with moving average (Issue 3 fix)
        # This maintains a reference point that balances all seen tasks
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p_new = p.clone().detach()
                if n in self.optimal_params:
                    self.optimal_params[n] = gamma * self.optimal_params[n] + (1 - gamma) * p_new
                else:
                    self.optimal_params[n] = p_new

        self.num_tasks_consolidated += 1
        
        # Log Fisher statistics
        fisher_stats = self.get_fisher_stats()
        if fisher_stats and samples_seen > 0:
            print(f"[EWC] Fisher updated (Task {self.num_tasks_consolidated}): "
                  f"mean={fisher_stats['mean_importance']:.6e}, "
                  f"max={fisher_stats['max_importance']:.6e}, "
                  f"params={fisher_stats['total_params']}, samples={samples_seen}")

    def compute_forgetting_cost(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute INPUT-DEPENDENT forgetting cost for a batch.
        
        This estimates the potential damage to previously learned knowledge
        if the model were to train on this batch. The cost is computed as
        the gradient interference with important (Fisher-weighted) parameters:
        
            Cost = Σ_i F_i * (∂L/∂θ_i)²
        
        where F_i is the Fisher information and ∂L/∂θ_i is the gradient
        from the current batch.
        
        This is the key fix for Issue 2: the cost now depends on the input (x, y),
        not just the expert's state.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input batch.
        y : torch.Tensor
            Target labels.
            
        Returns:
        --------
        float
            Estimated forgetting cost (gradient interference).
        """
        if not self.fisher:
            return 0.0

        x = x.to(self.device)
        y = y.to(self.device)

        # Compute gradient for THIS specific batch
        self.model.train()
        self.model.zero_grad()
        
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Measure gradient interference with important parameters
        # This is the expected increase in EWC penalty if we take a gradient step
        interference = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher and p.grad is not None:
                # Weighted squared gradient: how much would this update damage important weights?
                interference += (self.fisher[n] * p.grad.pow(2)).sum().item()

        self.model.zero_grad()  # Clean up gradients
        
        return self.forgetting_cost_scale * interference

    def penalty(self, verbose: bool = False) -> torch.Tensor:
        """
        Compute the EWC regularization penalty for training.
        
        This is the standard EWC loss term:
            L_EWC = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²
        
        where:
        - λ is the regularization strength
        - F_i is the Fisher information for parameter i
        - θ_i is the current parameter value
        - θ*_i is the optimal parameter value from previous tasks
        
        Returns:
        --------
        torch.Tensor
            The EWC penalty to add to the training loss.
        """
        if not self.fisher:
            return torch.tensor(0.0, device=self.device)
            
        penalty = torch.tensor(0.0, device=self.device)
        
        for n, p in self.model.named_parameters():
            if n in self.fisher and n in self.optimal_params:
                # Squared distance from optimal, weighted by Fisher importance
                diff_sq = (p - self.optimal_params[n]).pow(2)
                penalty = penalty + (self.fisher[n] * diff_sq).sum()
        
        final_penalty = (self.lambda_ewc / 2.0) * penalty
        
        if verbose:
            print(f"[EWC] Penalty: {final_penalty.item():.4e} (λ={self.lambda_ewc})")
            
        return final_penalty

    def _normalize_fisher(self):
        """
        Normalize Fisher matrix to have mean = 1.0.
        
        This is CRITICAL for consistent EWC behavior:
        - Ensures lambda_ewc works in a reasonable range (1.0 - 10.0)
        - Makes EWC strength independent of model architecture and batch size
        - Prevents numerical issues from very small/large Fisher values
        """
        if not self.fisher:
            return
        
        # Compute global mean of all Fisher values
        all_fisher = torch.cat([f.flatten() for f in self.fisher.values()])
        fisher_mean = all_fisher.mean()
        
        # Normalize: divide by mean (with epsilon for stability)
        epsilon = 1e-8
        if fisher_mean > epsilon:
            for n in self.fisher:
                self.fisher[n] = self.fisher[n] / (fisher_mean + epsilon)

    def has_fisher(self) -> bool:
        """Check if Fisher information has been computed."""
        return len(self.fisher) > 0 and self.num_tasks_consolidated > 0

    def get_fisher_stats(self) -> Dict:
        """Get statistics about the Fisher Information Matrix."""
        if not self.fisher:
            return {}
            
        all_fisher = torch.cat([f.flatten() for f in self.fisher.values()])
        
        return {
            'total_params': all_fisher.numel(),
            'mean_importance': all_fisher.mean().item(),
            'max_importance': all_fisher.max().item(),
            'std_importance': all_fisher.std().item(),
            'num_tasks': self.num_tasks_consolidated
        }