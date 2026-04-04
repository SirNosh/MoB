"""
Prototype-based routing for MoB: Mixture of Bidders.

Manages per-expert class centroids in feature space for distance-based routing.
Built incrementally during training, used at eval time to route inputs to the
expert whose learned feature space is closest to the input.
"""

import torch
from typing import Dict, Optional


class PrototypeStore:
    """
    Stores per-class feature centroids for a single expert.

    During training, accumulates running sums of features per class.
    At consolidation, finalizes centroids and computes inverse covariance
    for Mahalanobis distance.
    """

    # Minimum samples before using Mahalanobis (need enough for covariance)
    MIN_SAMPLES_FOR_MAHALANOBIS = 256

    def __init__(
        self,
        feature_dim: int,
        device: torch.device,
        online_mahalanobis: bool = False,
        inv_cov_update_interval: int = 50
    ):
        self.feature_dim = feature_dim
        self.device = device

        # Running accumulators (keyed by class label)
        self.class_sum: Dict[int, torch.Tensor] = {}
        self.class_count: Dict[int, int] = {}

        # Finalized centroids
        self.centroids: Dict[int, torch.Tensor] = {}

        # Shared covariance across all classes (for Mahalanobis)
        self.cov_sum = torch.zeros(feature_dim, feature_dim, device=device)
        self.cov_count = 0
        self.inv_cov: Optional[torch.Tensor] = None
        self.online_mahalanobis = online_mahalanobis
        self.inv_cov_update_interval = max(1, int(inv_cov_update_interval))
        self.update_count = 0

    def _recompute_inv_cov(self, allow_pinv: bool):
        """
        Recompute inverse covariance from running accumulators.

        Args:
            allow_pinv: If True, use pseudo-inverse on failure (finalize path).
                        If False, fall back to Euclidean (set inv_cov=None).
        """
        if self.cov_count < self.MIN_SAMPLES_FOR_MAHALANOBIS or len(self.centroids) == 0:
            self.inv_cov = None
            return

        # Global mean
        total_sum = torch.zeros(self.feature_dim, device=self.device)
        total_count = 0
        for cls in self.class_sum:
            total_sum += self.class_sum[cls]
            total_count += self.class_count[cls]
        global_mean = total_sum / total_count

        # Covariance = E[x x^T] - mean mean^T
        cov = self.cov_sum / self.cov_count - global_mean.unsqueeze(1) * global_mean.unsqueeze(0)

        # Regularize for numerical stability
        eps = 1e-4
        cov += eps * torch.eye(self.feature_dim, device=self.device)

        try:
            self.inv_cov = torch.linalg.inv(cov)
        except torch.linalg.LinAlgError:
            if allow_pinv:
                self.inv_cov = torch.linalg.pinv(cov)
            else:
                self.inv_cov = None

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Accumulate feature statistics from a training batch.

        Args:
            features: (batch_size, feature_dim) detached feature tensor.
            labels: (batch_size,) ground-truth labels.
        """
        features = features.detach()
        labels = labels.detach()
        self.update_count += 1

        for cls in labels.unique().tolist():
            mask = labels == cls
            cls_features = features[mask]  # (n, feature_dim)

            if cls not in self.class_sum:
                self.class_sum[cls] = torch.zeros(self.feature_dim, device=self.device)
                self.class_count[cls] = 0

            self.class_sum[cls] += cls_features.sum(dim=0)
            self.class_count[cls] += cls_features.shape[0]

        # Accumulate shared covariance (subtract global mean later at finalize)
        self.cov_sum += features.T @ features  # (D, D)
        self.cov_count += features.shape[0]

        # Keep centroids up to date incrementally so they're available
        # during training (not just after finalize at task boundaries).
        # inv_cov remains None unless finalize() or online_mahalanobis updates it.
        for cls in labels.unique().tolist():
            self.centroids[cls] = self.class_sum[cls] / self.class_count[cls]

        # Optional online Mahalanobis: periodically recompute inverse covariance
        # during training so prototype routing can use Mahalanobis before finalize.
        if (
            self.online_mahalanobis
            and self.cov_count >= self.MIN_SAMPLES_FOR_MAHALANOBIS
            and self.update_count % self.inv_cov_update_interval == 0
        ):
            # On inversion failure, keep inv_cov=None so routing falls back to Euclidean.
            had_inv_cov = self.inv_cov is not None
            self._recompute_inv_cov(allow_pinv=False)
            has_inv_cov = self.inv_cov is not None
            if not had_inv_cov and has_inv_cov:
                print(
                    f"[PrototypeStore] Online Mahalanobis active "
                    f"(samples={self.cov_count}, updates={self.update_count})"
                )
            elif had_inv_cov and not has_inv_cov:
                print(
                    f"[PrototypeStore] Online Mahalanobis disabled due to inversion failure; "
                    f"falling back to Euclidean (samples={self.cov_count}, updates={self.update_count})"
                )

    def finalize(self):
        """
        Compute centroids from accumulated sums and inverse covariance.

        Called at consolidation time (shift detection or end of task).
        Does NOT reset accumulators — prototypes grow across shifts.
        """
        # Compute centroids
        for cls in self.class_sum:
            self.centroids[cls] = self.class_sum[cls] / self.class_count[cls]

        # Compute inverse covariance for Mahalanobis distance
        self._recompute_inv_cov(allow_pinv=True)

    def _compute_distance_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from each sample to each centroid.

        Args:
            features: (B, D) feature tensor (already detached).

        Returns:
            Distance matrix of shape (B, C) where C = number of centroids.
            Returns None if no centroids exist.
        """
        centroid_labels = sorted(self.centroids.keys())
        centroid_matrix = torch.stack([self.centroids[c] for c in centroid_labels])

        # features: (B, D), centroid_matrix: (C, D)
        diff = features.unsqueeze(1) - centroid_matrix.unsqueeze(0)  # (B, C, D)

        if self.inv_cov is not None:
            # Mahalanobis: d = sqrt(diff @ inv_cov @ diff^T)
            transformed = torch.matmul(diff, self.inv_cov)
            distances = (transformed * diff).sum(dim=-1)  # (B, C)
            distances = distances.clamp(min=0).sqrt()
        else:
            # Euclidean fallback
            distances = diff.norm(dim=-1)  # (B, C)

        return distances

    def compute_routing_score(self, features: torch.Tensor) -> float:
        """
        Compute mean distance from batch features to nearest class centroid.

        Uses Mahalanobis distance if inverse covariance is available,
        otherwise falls back to Euclidean distance.

        Args:
            features: (batch_size, feature_dim) feature tensor.

        Returns:
            Mean minimum distance across the batch (lower = better match).
        """
        if not self.centroids:
            return 1e6

        features = features.detach()
        distances = self._compute_distance_matrix(features)  # (B, C)
        min_distances = distances.min(dim=1).values  # (B,)
        return min_distances.mean().item()

    def compute_per_sample_distances(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample minimum distance to nearest centroid.

        Unlike compute_routing_score() which returns a scalar mean,
        this returns the raw per-sample distances — needed for per-sample
        (per-token) routing in MoE-style top-k expert combination.

        Args:
            features: (batch_size, feature_dim) feature tensor.

        Returns:
            Per-sample min-distances of shape (B,). Returns 1e6 tensor if
            no centroids exist.
        """
        if not self.centroids:
            return torch.full((features.shape[0],), 1e6, device=features.device)

        features = features.detach()
        distances = self._compute_distance_matrix(features)  # (B, C)
        return distances.min(dim=1).values  # (B,)
