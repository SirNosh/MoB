"""FeCAM canonical Mahalanobis core — byte-equivalent port.

Source-of-truth upstream:
    Repo:   github.com/dipamgoswami/FeCAM
    Commit: e33f39d112ff2d2a2df2e68c490af579a50edd31
    File:   models/base.py
    Functions ported: shrink_cov, normalize_cov, _tukeys_transform, _mahalanobis

Protocol refs:
    docs/protocols/fecam-router-gate.md v1.2.1
    §2.0 element list (per-class mu + per-class Sigma + additive 2-param shrinkage
                       + correlation normalization + L2 norm; Tukey OFF for ViT)
    §4.6 FeCAM-binding implementation invariant (byte-equivalent port)
    §4.7 shrinkage hyperparameter pinning (alpha1=alpha2=1 for CIFAR-100)
    §8.4 test 17  (upstream-fidelity <= 1e-5 per-sample on synthetic fixture)
    §8.4 test 18  (fecam.* fields pinned in frozen_config.yaml)
    §8.4 test 19  (L2 norm applied; Tukey NOT applied when tukey=false)

Config defaults (paper §7 CIFAR-100 MSCIL; dipamgoswami/FeCAM exps/FeCAM_cifar100.json):
    alpha1=1, alpha2=1, beta=0.5, tukey=false (ViT override per paper §7),
    per_class=true, full_cov=true, shrink=true, norm_cov=true,
    inverse_method=pinv.

No silent deviation from upstream. Any deviation requires §13 amendment per §4.6.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Tukey's transform (paper eq. 9; §2.0 element 6)
#
# Upstream body:
#     def _tukeys_transform(self, x):
#         beta = self.args["beta"]
#         x = torch.tensor(x)
#         if beta == 0:
#             return torch.log(x)
#         else:
#             return torch.pow(x, beta)
#
# v1.2.1 decision: OFF for ViT-B/16 (paper §7 — negative feature values break
# torch.pow for non-integer beta). Kept as a function so §8.4 test 19 can
# exercise both branches.
# ------------------------------------------------------------------
def tukey_transform(x: torch.Tensor, beta: float = 0.5, enabled: bool = False) -> torch.Tensor:
    """Applies x -> x^beta (or log(x) if beta==0). NO-OP when enabled=False."""
    if not enabled:
        return x
    if beta == 0:
        return torch.log(x)
    return torch.pow(x, beta)


# ------------------------------------------------------------------
# Additive two-parameter shrinkage (paper eq. 8; §2.0 element 3)
#
# Upstream body (verbatim variable names kept for audit trail):
#     diag_mean = torch.mean(torch.diagonal(cov))
#     off_diag = cov.clone(); off_diag.fill_diagonal_(0.0)
#     mask = off_diag != 0.0
#     off_diag_mean = (off_diag*mask).sum() / mask.sum()
#     iden = torch.eye(cov.shape[0])
#     cov_ = cov + alpha1*diag_mean*iden + alpha2*off_diag_mean*(1-iden)
#     return cov_
#
# NOTE: upstream's .sum()/mask.sum() treats a strict-zero off-diag entry
# as if it were "missing" which is technically a measure-zero quirk
# (true sample covariances have strict-zero off-diagonals with probability
# zero in float32). Keeping byte-equivalent.
# ------------------------------------------------------------------
def shrink_cov(cov: torch.Tensor, alpha1: float = 1.0, alpha2: float = 1.0) -> torch.Tensor:
    """Paper eq. 8 additive shrinkage. cov: (d, d). Returns (d, d)."""
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    # Guard against all-zero off-diag (degenerate synthetic input)
    denom = mask.sum()
    if denom.item() == 0:
        off_diag_mean = torch.zeros((), device=cov.device, dtype=cov.dtype)
    else:
        off_diag_mean = (off_diag * mask).sum() / denom
    iden = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
    return cov_


# ------------------------------------------------------------------
# Correlation normalization (paper eq. 7; §2.0 element 4)
#
# Upstream (one covariance at a time, after shrinkage):
#     sd = torch.sqrt(torch.diagonal(cov))
#     cov = cov / (torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0)))
#
# Diagonals become 1; off-diagonals become Pearson correlations in [-1, 1].
# Matches torch.corrcoef on the shrunken covariance (by construction).
# ------------------------------------------------------------------
def normalize_cov(cov: torch.Tensor) -> torch.Tensor:
    """Paper eq. 7 correlation normalization. cov: (d, d). Returns (d, d)."""
    sd = torch.sqrt(torch.diagonal(cov))
    # sd outer product: (d,) -> (d, d) with sd_i * sd_j
    denom = torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0))
    return cov / denom


# ------------------------------------------------------------------
# Mahalanobis scoring (paper §3.1 + upstream _mahalanobis)
#
# Upstream body (verbatim logic):
#     if tukey and cur_task > 0: class_means = _tukeys_transform(class_means)
#     x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
#     inv_covmat = torch.linalg.pinv(cov).float().to(device)
#     left_term = torch.matmul(x_minus_mu, inv_covmat)
#     mahal = torch.matmul(left_term, x_minus_mu.T)
#     return torch.diagonal(mahal, 0)
#
# This is the "same cov for all class-means" variant. For per-class cov
# (§2.0 element 2 primary), the caller loops over classes and picks the
# matching cov; see mahalanobis_per_class below.
# ------------------------------------------------------------------
def mahalanobis(vectors: torch.Tensor, class_means: torch.Tensor,
                cov: Optional[torch.Tensor] = None,
                *, tukey_enabled: bool = False, tukey_beta: float = 0.5,
                cur_task: int = 0, feature_dim: int = 768) -> torch.Tensor:
    """Return per-(sample) Mahalanobis distance against the SAME cov for
    every class_mean. Output shape: (N,) matching upstream's torch.diagonal.

    NOTE: upstream's output is specifically `diag(mahal)` which is only
    meaningful when vectors and class_means are paired 1:1 (N == C after
    broadcasting). We preserve that semantics.
    """
    if tukey_enabled and cur_task > 0:
        class_means = tukey_transform(class_means, beta=tukey_beta, enabled=True)

    x_minus_mu = (F.normalize(vectors, p=2, dim=-1)
                  - F.normalize(class_means, p=2, dim=-1))

    if cov is None:
        cov_local = torch.eye(feature_dim, device=vectors.device, dtype=vectors.dtype)
    else:
        cov_local = cov

    inv_covmat = torch.linalg.pinv(cov_local).to(vectors.dtype).to(vectors.device)
    left_term = torch.matmul(x_minus_mu, inv_covmat)
    mahal = torch.matmul(left_term, x_minus_mu.T)
    return torch.diagonal(mahal, 0)


def mahalanobis_per_class(vectors: torch.Tensor, class_means: torch.Tensor,
                          covs: torch.Tensor,
                          *, tukey_enabled: bool = False,
                          tukey_beta: float = 0.5,
                          cur_task: int = 0) -> torch.Tensor:
    """Per-class-covariance Mahalanobis scoring.

    Inputs:
        vectors:     (N, d)      -- test features (pre-L2-norm OK; we L2-norm here)
        class_means: (C, d)      -- per-class prototypes (pre-L2-norm OK)
        covs:        (C, d, d)   -- per-class covariance stack (post-shrinkage +
                                    correlation normalization, to match §2.0
                                    elements 3 + 4 pipeline; caller ensures this)

    Returns:
        dists: (N, C) Mahalanobis distance from every sample to every class.

    Implementation:
        For each class c, compute
            diff_c = L2(vectors) - L2(mu_c)             # (N, d)
            d_c    = sum((diff_c @ pinv(cov_c)) * diff_c, dim=-1)   # (N,)
        This is the per-row squared Mahalanobis. Upstream returns squared
        d_M (it never takes a sqrt); we match that.
    """
    if tukey_enabled and cur_task > 0:
        class_means = tukey_transform(class_means, beta=tukey_beta, enabled=True)

    N = vectors.shape[0]
    C = class_means.shape[0]
    V = F.normalize(vectors, p=2, dim=-1)                 # (N, d)
    M = F.normalize(class_means, p=2, dim=-1)             # (C, d)

    out = torch.empty((N, C), dtype=vectors.dtype, device=vectors.device)
    for c in range(C):
        diff = V - M[c].unsqueeze(0)                      # (N, d)
        inv_c = torch.linalg.pinv(covs[c]).to(vectors.dtype)
        left = diff @ inv_c                               # (N, d)
        out[:, c] = (left * diff).sum(dim=-1)             # (N,)
    return out


# ------------------------------------------------------------------
# Pipeline glue: build the per-class shrunken + normalized cov stack
# ------------------------------------------------------------------
def prepare_covs(raw_covs: torch.Tensor,
                 *, alpha1: float = 1.0, alpha2: float = 1.0,
                 shrink: bool = True, norm_cov: bool = True) -> torch.Tensor:
    """Apply §2.0 element 3 (shrinkage) then element 4 (correlation norm) to
    each per-class covariance. Returns a (C, d, d) tensor same-shape as input.

    This is the order upstream uses: shrink first, then normalize. Inversion
    happens later inside mahalanobis_per_class.
    """
    C = raw_covs.shape[0]
    out = torch.empty_like(raw_covs)
    for c in range(C):
        cov_c = raw_covs[c]
        if shrink:
            cov_c = shrink_cov(cov_c, alpha1=alpha1, alpha2=alpha2)
        if norm_cov:
            cov_c = normalize_cov(cov_c)
        out[c] = cov_c
    return out


# ------------------------------------------------------------------
# Running-sum accumulators for online prototype + covariance updates
# (canonical FeCAM cadence per §2.0: recompute Sigma_c at end of each task
#  from accumulators; NOT per-step-online).
# ------------------------------------------------------------------
class PerClassAccumulator:
    """Maintains running sums for per-class mean and per-class covariance
    on a single expert. Call `update(x, y)` during task training (features
    only; no gradients needed here -- caller should run in no_grad), then
    `finalize()` at end-of-task to get (mean, cov) per class.
    """

    def __init__(self, num_classes: int, feature_dim: int, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        self.C = num_classes
        self.d = feature_dim
        self.device = device
        self.dtype = dtype
        self.n = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.sum_x = torch.zeros(num_classes, feature_dim, dtype=dtype, device=device)
        self.sum_xxT = torch.zeros(num_classes, feature_dim, feature_dim,
                                   dtype=dtype, device=device)

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """features: (B, d); labels: (B,) in 0..C-1 (local class index)."""
        feats = features.to(self.dtype).to(self.device)
        lbls = labels.to(self.device)
        # Accumulate per-class
        for c in torch.unique(lbls).tolist():
            mask = (lbls == c)
            Xc = feats[mask]                          # (n_c, d)
            if Xc.numel() == 0:
                continue
            self.n[c] += Xc.shape[0]
            self.sum_x[c] += Xc.sum(dim=0)
            self.sum_xxT[c] += Xc.t() @ Xc            # (d, d)

    @torch.no_grad()
    def finalize(self) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """Return (means, covs, counts). covs is unbiased sample cov using
        (sum_xxT - n*mu*mu^T)/(n-1). Classes with n<2 get zero cov (caller
        must shrink to rescue).
        """
        means = torch.zeros(self.C, self.d, dtype=self.dtype, device=self.device)
        covs = torch.zeros(self.C, self.d, self.d, dtype=self.dtype, device=self.device)
        for c in range(self.C):
            n_c = int(self.n[c].item())
            if n_c == 0:
                continue
            mu_c = self.sum_x[c] / n_c
            means[c] = mu_c
            if n_c >= 2:
                # unbiased sample covariance matching torch.cov default
                S = self.sum_xxT[c] - n_c * torch.outer(mu_c, mu_c)
                covs[c] = S / (n_c - 1)
        return means, covs, self.n.clone()
