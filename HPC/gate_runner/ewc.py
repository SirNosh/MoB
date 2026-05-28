"""Elastic Weight Consolidation (online EWC) for the auction's forget-cost term.

Adapted from mob/bidding.py::EWCForgettingEstimator.
Kept minimal: diagonal empirical Fisher, online EMA, Fisher normalization + clamp,
forgetting cost = <F, grad^2>.

Protocol refs:
    docs/protocols/fecam-router-gate.md v1.2.1
    §2.2   bid_i(x) beta-term = <F_i, (theta_i - theta*_i)^2>
    §4.4   Fisher-magnitude match gate (dual criterion: cross-arm <=2x; within-arm CV <=0.5)
    memory/MEMORY.md   fisher_clamp_min = 0.1 (load-bearing)
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """Online EWC (Schwarz 2018) with Fisher clamp = 0.1.

    Scope: a SINGLE expert's trainable parameters. One EWC instance per expert.
    The caller maintains E instances (E = pool.num_experts).
    """

    def __init__(self, fisher_decay: float = 0.9,
                 fisher_clamp_min: float = 0.1,
                 forgetting_cost_scale: float = 1.0):
        self.fisher_decay = fisher_decay
        self.fisher_clamp_min = fisher_clamp_min
        self.forgetting_cost_scale = forgetting_cost_scale
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.num_tasks_consolidated = 0

    # ------------------------------------------------------------------
    # Fisher estimation at end-of-task
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def update_fisher(self, model: nn.Module,
                      dataloader: Iterable,
                      device: torch.device,
                      num_samples: int = 200,
                      forward_fn=None) -> Dict[str, float]:
        """Empirical Fisher on up to num_samples samples, EMA into self.fisher.

        Args:
            model: the expert being consolidated.
            dataloader: iterable of (x, y) over the just-finished task.
            device: cuda or cpu.
            num_samples: sample cap for Fisher estimation (standard: 200).
            forward_fn: optional callable(model, x) -> logits (so callers can
                route through backbone+adapter without polluting this module).
                If None, model(x) is called directly.

        Returns diagnostic dict {"mean_F": ..., "max_F": ..., "log_F_max": ...}
        for §4.4 per-task drift reporting.
        """
        model.train()
        current_fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=device)
            for n, p in model.named_parameters() if p.requires_grad
        }
        samples_seen = 0
        for x, y in dataloader:
            if samples_seen >= num_samples:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            bsz = x.size(0)
            model.zero_grad(set_to_none=True)
            logits = forward_fn(model, x) if forward_fn is not None else model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    current_fisher[n] += p.grad.detach().pow(2) * bsz
            samples_seen += bsz

        if samples_seen > 0:
            for n in current_fisher:
                current_fisher[n] /= samples_seen

        # Online EWC EMA: F_new <- gamma * F_old + (1-gamma) * F_batch
        gamma = self.fisher_decay
        for n, f_new in current_fisher.items():
            if n in self.fisher:
                self.fisher[n] = gamma * self.fisher[n] + (1 - gamma) * f_new
            else:
                self.fisher[n] = f_new.clone()

        self._normalize_and_clamp()

        # Snapshot optimal params (online EMA of theta*)
        for n, p in model.named_parameters():
            if p.requires_grad:
                p_new = p.detach().clone()
                if n in self.optimal_params:
                    self.optimal_params[n] = (gamma * self.optimal_params[n]
                                              + (1 - gamma) * p_new)
                else:
                    self.optimal_params[n] = p_new

        self.num_tasks_consolidated += 1

        diag = self.get_fisher_stats()
        model.zero_grad(set_to_none=True)
        return diag

    # ------------------------------------------------------------------
    # Fisher normalization + clamp (memory/MEMORY.md load-bearing)
    # ------------------------------------------------------------------
    def _normalize_and_clamp(self) -> None:
        """Normalize Fisher to mean 1.0 then clamp to min fisher_clamp_min.

        Port from mob/bidding.py::_normalize_fisher (clamp=0.1, eps=1e-30)
        which is project-memory load-bearing per the MEMORY.md entry.
        """
        if not self.fisher:
            return
        all_f = torch.cat([f.flatten() for f in self.fisher.values()])
        mean = all_f.mean()
        eps = 1e-30
        if mean > 0:
            for n in self.fisher:
                self.fisher[n] = self.fisher[n] / (mean + eps)
                self.fisher[n] = torch.clamp(self.fisher[n], min=self.fisher_clamp_min)

    # ------------------------------------------------------------------
    # Forgetting-cost term for the bid
    # ------------------------------------------------------------------
    def forget_cost(self, model: nn.Module) -> float:
        """Static consolidation cost: sum_l F_l * (theta_l - theta*_l)^2.

        Cheap to compute (no backward). This is the bid's beta-term.
        """
        if not self.fisher:
            return 0.0
        total = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.fisher and n in self.optimal_params:
                    diff_sq = (p - self.optimal_params[n]).pow(2)
                    total += (self.fisher[n] * diff_sq).sum().item()
        return float(self.forgetting_cost_scale * total)

    def ewc_penalty(self, model: nn.Module, lambda_ewc: float = 1.0) -> torch.Tensor:
        """Differentiable EWC penalty for the training loss: (lambda/2) * sum F (theta - theta*)^2."""
        device = next(model.parameters()).device
        if not self.fisher:
            return torch.tensor(0.0, device=device)
        penalty = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.optimal_params:
                F_ = self.fisher[n].to(device)
                opt = self.optimal_params[n].to(device)
                penalty = penalty + (F_ * (p - opt).pow(2)).sum()
        return (lambda_ewc / 2.0) * penalty

    def has_fisher(self) -> bool:
        return len(self.fisher) > 0 and self.num_tasks_consolidated > 0

    def get_fisher_stats(self) -> Dict[str, float]:
        if not self.fisher:
            return {}
        all_f = torch.cat([f.flatten() for f in self.fisher.values()])
        return {
            "mean_F": float(all_f.mean().item()),
            "max_F": float(all_f.max().item()),
            "std_F": float(all_f.std().item()),
            "log_F_max": float(math.log(all_f.max().item() + 1e-30)),
            "num_tasks": int(self.num_tasks_consolidated),
            "total_params": int(all_f.numel()),
        }
