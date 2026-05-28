"""MoB auction routing layer.

Bid formula (§1.3, §2.2):
    b_i(x) = alpha * d_M(x, mu_{i,c}; Sigma_{i,c,s})      # execution cost
           + beta  * <F_i, (theta_i - theta*_i)^2>         # forgetting cost
           + gamma * (f_i / f_bar)                         # conscience

Arm A: alpha, beta, gamma all active.
Arm B: beta = gamma = 0  (HARD-CODED here, not via hyperparam)  per §2.2.

Winner selection: argmin over experts of b_i(x). Deterministic tie-break
by lowest expert index.

Protocol refs:
    docs/protocols/fecam-router-gate.md v1.2.1
    §1.3  bid formula
    §2.2  ablation invariant: Arm B = Arm A with beta=gamma=0, nothing else differs.
    §2.1  "Evaluation routing: Pseudo-label auction" — each expert bids with its
          own argmax prediction, not ground truth.
    §4.4  Fisher clamp; EWC module handles.
    §8.4 test 1  Arm B bid == alpha * d_M bitwise.
    §8.4 test 9  Arm B bid.grad_fn never traces back to Fisher ops.
    §8.4 test 10 Arms disagree on >=1% of pilot bid calls.
    §8.4 test 13 RNG state equality at first bid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .ewc import EWC
from .fecam_core import mahalanobis_per_class, prepare_covs


# ------------------------------------------------------------------
# Conscience: DeSieno EMA win-frequency tracker
# ------------------------------------------------------------------
class Conscience:
    """EMA-of-win-frequency per expert. Bid penalty = gamma * f_i / f_bar.

    f_bar is the mean f across experts; normalizing by it keeps the penalty
    scale-free (doesn't grow unboundedly as training progresses).
    """

    def __init__(self, num_experts: int, ema_decay: float = 0.99):
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        # Uniform prior: f_i = 1/E
        self.freq = torch.full((num_experts,), 1.0 / num_experts)

    def update(self, winner_idx: int) -> None:
        """Update the EMA win-frequency after `winner_idx` took a batch.

        f_i(t+1) = decay * f_i(t) + (1-decay) * 1[i == winner]
        """
        with torch.no_grad():
            one_hot = torch.zeros(self.num_experts)
            one_hot[winner_idx] = 1.0
            self.freq = self.ema_decay * self.freq + (1 - self.ema_decay) * one_hot

    def penalty_vec(self) -> torch.Tensor:
        """Return the conscience penalty vector f_i / f_bar (size E)."""
        f_bar = self.freq.mean().clamp(min=1e-12)
        return self.freq / f_bar

    def as_distribution(self) -> List[float]:
        return self.freq.tolist()


# ------------------------------------------------------------------
# Per-expert prototype bank
# ------------------------------------------------------------------
class PrototypeBank:
    """Holds per-expert, per-class mu, raw Sigma, and the shrunken + normalized
    covariance stack actually used for Mahalanobis.

    Classes are indexed by GLOBAL id (0..num_total_classes-1). A class is
    "known" by the expert once it has accumulated >=1 sample for that class
    during that expert's task(s); unknown classes carry zero mu and cov and
    are skipped during bidding.

    Update cadence: per-class accumulators updated during training (no_grad),
    mu and Sigma recomputed at END OF TASK (canonical FeCAM cadence, §2.0).
    """

    def __init__(self, num_experts: int, num_classes: int, feature_dim: int,
                 device: torch.device,
                 alpha1: float = 1.0, alpha2: float = 1.0,
                 shrink: bool = True, norm_cov: bool = True):
        self.E = num_experts
        self.C = num_classes
        self.d = feature_dim
        self.device = device
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.shrink = shrink
        self.norm_cov = norm_cov

        # Per-(expert, class): count, sum of features, sum of outer products
        self.n = torch.zeros(num_experts, num_classes, dtype=torch.long, device=device)
        self.sum_x = torch.zeros(num_experts, num_classes, feature_dim,
                                 dtype=torch.float64, device=device)
        self.sum_xxT = torch.zeros(num_experts, num_classes, feature_dim, feature_dim,
                                   dtype=torch.float64, device=device)

        # Finalized (after end-of-task): mu, cov_prepared. Kept in float64 for
        # numerical stability during inversion; cast to float32 at use time.
        self.mu = torch.zeros(num_experts, num_classes, feature_dim,
                              dtype=torch.float64, device=device)
        self.cov_prepared = torch.zeros(num_experts, num_classes, feature_dim, feature_dim,
                                        dtype=torch.float64, device=device)
        # Identity prior for unseen classes (inverse is identity -> d_M = ||x-mu||^2)
        for e in range(num_experts):
            for c in range(num_classes):
                self.cov_prepared[e, c] = torch.eye(feature_dim, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, expert_idx: int, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate (features, labels) for expert `expert_idx`. Labels are
        GLOBAL class ids 0..C-1."""
        feats = features.to(torch.float64).to(self.device)
        lbls = labels.to(self.device)
        for c in torch.unique(lbls).tolist():
            mask = (lbls == c)
            Xc = feats[mask]
            self.n[expert_idx, c] += Xc.shape[0]
            self.sum_x[expert_idx, c] += Xc.sum(dim=0)
            self.sum_xxT[expert_idx, c] += Xc.t() @ Xc

    @torch.no_grad()
    def finalize_expert(self, expert_idx: int) -> None:
        """Recompute mu and shrunken+normalized cov for the given expert
        from current accumulators. Called at end-of-task for every expert
        that gained any data this task."""
        for c in range(self.C):
            n_c = int(self.n[expert_idx, c].item())
            if n_c == 0:
                continue
            mu_c = self.sum_x[expert_idx, c] / n_c
            self.mu[expert_idx, c] = mu_c
            if n_c >= 2:
                S = self.sum_xxT[expert_idx, c] - n_c * torch.outer(mu_c, mu_c)
                raw_cov = S / (n_c - 1)
            else:
                raw_cov = torch.eye(self.d, dtype=torch.float64, device=self.device)
            # Shrink + normalize in float64 per §2.0 elements 3, 4
            from .fecam_core import shrink_cov, normalize_cov
            cov_c = raw_cov
            if self.shrink:
                cov_c = shrink_cov(cov_c, alpha1=self.alpha1, alpha2=self.alpha2)
            if self.norm_cov:
                cov_c = normalize_cov(cov_c)
            self.cov_prepared[expert_idx, c] = cov_c

    def known_classes(self, expert_idx: int) -> List[int]:
        return (self.n[expert_idx] > 0).nonzero(as_tuple=True)[0].tolist()

    @torch.no_grad()
    def expert_min_dM(self, expert_idx: int, features: torch.Tensor,
                      candidate_classes: Optional[Sequence[int]] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (min_d_M_per_sample, argmin_class) for a given expert.

        min_d_M_per_sample: (B,) tensor — the expert's best (lowest) d_M
        across all classes it knows about (or the specified candidate list).
        argmin_class: (B,) global class id corresponding to the argmin.

        If the expert knows no classes, returns +inf and -1 so it always
        loses the auction against any expert that has seen data.
        """
        B = features.shape[0]
        if candidate_classes is None:
            classes = self.known_classes(expert_idx)
        else:
            classes = [c for c in candidate_classes
                       if int(self.n[expert_idx, c].item()) > 0]
        if not classes:
            return (torch.full((B,), float("inf"), device=features.device),
                    torch.full((B,), -1, dtype=torch.long, device=features.device))

        C_sel = len(classes)
        mus = self.mu[expert_idx, classes]                          # (C_sel, d)
        covs = self.cov_prepared[expert_idx, classes]               # (C_sel, d, d)
        feats64 = features.to(torch.float64)
        dists = mahalanobis_per_class(feats64, mus, covs,
                                      tukey_enabled=False)          # (B, C_sel)
        min_vals, argmin_sel = dists.min(dim=-1)
        # Map back to global class ids
        classes_tensor = torch.tensor(classes, dtype=torch.long, device=features.device)
        argmin_class = classes_tensor[argmin_sel]
        return min_vals.to(features.dtype), argmin_class


# ------------------------------------------------------------------
# Bid composition + winner selection
# ------------------------------------------------------------------
def compose_bid(d_M: torch.Tensor, forget: torch.Tensor, conscience: torch.Tensor,
                *, alpha: float, beta: float, gamma: float) -> torch.Tensor:
    """Scalar bid per expert: alpha*d_M + beta*forget + gamma*conscience.

    Shapes:
        d_M:         (E,)  execution cost per expert (we reduce over batch by mean
                           before calling this; see collect_bids).
        forget:      (E,)  static EWC cost per expert.
        conscience:  (E,)  f_i / f_bar per expert.

    For Arm B callers pass beta=0, gamma=0 — the bid collapses to alpha*d_M
    bitwise. Acceptance test 1 verifies this.
    """
    bid = alpha * d_M
    if beta != 0.0:
        bid = bid + beta * forget
    if gamma != 0.0:
        bid = bid + gamma * conscience
    return bid


def pick_winner(bids: torch.Tensor) -> int:
    """Argmin with deterministic tie-break (lowest expert index wins)."""
    # torch.argmin picks the first occurrence on ties for a 1-D contiguous tensor,
    # which already is "lowest index" deterministic. Cast to int for indexing.
    return int(torch.argmin(bids).item())


# ------------------------------------------------------------------
# Auction orchestration
# ------------------------------------------------------------------
class Auction:
    """Stateless-per-batch auction runner.

    Usage during training:
        auction = Auction(num_experts=4, arm="A", alpha=..., beta=..., gamma=...)
        for batch in loader:
            cls_feats = backbone(x)   # (B, d)  frozen
            winner, diag = auction.route(pool, bank, ewcs, conscience,
                                          cls_feats, y_global,
                                          candidate_classes=seen_classes)
            # train ONLY pool[winner] on this batch

    Bidding uses pseudo-labels at EVAL (§2.1), ground-truth-agnostic bids
    at TRAIN (match: we use the expert's best-d_M-across-its-known-classes,
    same formula in both paths — "pseudo-label auction" literally means each
    expert uses its own argmax, §2.1).
    """

    def __init__(self, arm: str, num_experts: int,
                 alpha: float, beta_a: float, gamma_a: float):
        assert arm in ("A", "B"), f"arm must be A or B, got {arm!r}"
        self.arm = arm
        self.num_experts = num_experts
        self.alpha = alpha
        if arm == "A":
            self.beta = beta_a
            self.gamma = gamma_a
        else:
            # §2.2 implementation discipline: HARD-CODED zeros, not flags.
            self.beta = 0.0
            self.gamma = 0.0

    def collect_dM(self, bank: PrototypeBank, features: torch.Tensor,
                   candidate_classes: Optional[Sequence[int]] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For each expert, take the batch-mean of the per-sample min_d_M
        over candidate_classes. Returns (dM_vec (E,), pseudo_labels (E, B))."""
        B = features.shape[0]
        E = self.num_experts
        dm = torch.zeros(E, device=features.device, dtype=features.dtype)
        pseudo = torch.full((E, B), -1, dtype=torch.long, device=features.device)
        for e in range(E):
            min_vals, argmin_class = bank.expert_min_dM(e, features, candidate_classes)
            dm[e] = min_vals.mean()
            pseudo[e] = argmin_class
        return dm, pseudo

    def collect_forget_costs(self, pool, ewcs: List[EWC]) -> torch.Tensor:
        """Per-expert static EWC forget cost. Arm B: NEVER call this (its
        beta=0 so the product zeros out). We short-circuit to zero for Arm B
        to make test 9 pass (no Fisher ops traced in Arm B's bid graph).
        """
        E = self.num_experts
        if self.arm == "B":
            return torch.zeros(E, device=next(pool[0].parameters()).device)
        out = torch.zeros(E, device=next(pool[0].parameters()).device)
        for e in range(E):
            out[e] = ewcs[e].forget_cost(pool[e])
        return out

    def collect_conscience(self, conscience: Conscience,
                           device: torch.device) -> torch.Tensor:
        """Per-expert conscience penalty. Arm B zeros it out (same rationale
        as forget cost)."""
        if self.arm == "B":
            return torch.zeros(self.num_experts, device=device)
        return conscience.penalty_vec().to(device)

    def route(self, pool, bank: PrototypeBank, ewcs: List[EWC],
              conscience: Conscience, features: torch.Tensor,
              candidate_classes: Optional[Sequence[int]] = None
              ) -> Tuple[int, Dict]:
        """Run one auction. Returns (winner_idx, diag_dict)."""
        device = features.device
        dM, pseudo = self.collect_dM(bank, features, candidate_classes)
        forget = self.collect_forget_costs(pool, ewcs)
        consc = self.collect_conscience(conscience, device)
        bids = compose_bid(dM, forget, consc,
                           alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        winner = pick_winner(bids)
        diag = {
            "alpha_dM_per_expert": dM.detach().cpu().tolist(),
            "beta_forget_per_expert": forget.detach().cpu().tolist(),
            "gamma_conscience_per_expert": consc.detach().cpu().tolist(),
            "bid_per_expert": bids.detach().cpu().tolist(),
            "winner": winner,
        }
        return winner, diag
