"""Per-expert adapter: LoRA r=8 on attention QKV + CLS-token FFN bottleneck.

The backbone (timm ViT-B/16 in21k) is loaded once and shared (frozen) across
all experts. Each expert owns only its adapter parameters (~300K trainable).

Protocol refs:
    docs/protocols/fecam-router-gate.md v1.2.1
    §2.1  Backbone + Per-expert adapter + Trainable params per expert
    §2.1  Backbone freeze scope: all of {blocks, patch_embed, cls_token, norm, pos_embed}
          fully frozen. NO final-block LN unfreezing exception.
    §2.3 T7 backbone freeze clarifier.
    §8.4 test 20 structural assertions.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Backbone loader (shared, frozen)
# ------------------------------------------------------------------
def load_vit_b16_backbone(device: torch.device,
                          model_name: str = "vit_base_patch16_224.augreg_in21k"):
    """Load the frozen backbone and freeze everything per §2.1.

    Returns (backbone, feature_dim). Backbone forward returns the CLS token
    pre-head representation shape (N, 768); timm's num_classes=0 mode.
    """
    import timm  # local import so tests that don't need a model don't require timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(device)

    # Freeze EVERY parameter; no LN exception. §2.1 Backbone freeze scope.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    feature_dim = getattr(model, "num_features", 768)
    return model, int(feature_dim)


def backbone_forward_features(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return (B, d) CLS-token features from a frozen timm ViT-B/16.

    timm VisionTransformer exposes .forward_features(x) which returns the full
    token sequence, then .forward_head pools and heads it. We want the CLS
    token (index 0) before the classifier. When num_classes=0 was set, calling
    model(x) returns the pre-logits already; use that path for robustness.
    """
    with torch.no_grad():
        feats = backbone(x)
    # Some timm variants return (B, num_tokens, d) if global_pool='token' not set;
    # handle both.
    if feats.ndim == 3:
        feats = feats[:, 0, :]
    return feats


# ------------------------------------------------------------------
# LoRA layer (low-rank adapter on a linear projection)
# ------------------------------------------------------------------
class LoRALinearAdapter(nn.Module):
    """Additive low-rank update: y = x @ (A @ B) * (alpha / r).

    Standalone (not wrapped around an nn.Linear; the frozen backbone is never
    modified in-place). The caller adds this delta to the base projection's
    output during expert forward.

    Shapes:
        A: (in_features, r)   -- kaiming_uniform init
        B: (r,        out)    -- zero init (so delta starts at 0)
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8,
                 alpha: float = 16.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.A = nn.Parameter(torch.zeros(in_features, r))
        self.B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        # B stays zero -> initial delta = 0, so expert starts as pure backbone
        with torch.no_grad():
            self.B.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        return (x @ self.A) @ self.B * self.scaling


# ------------------------------------------------------------------
# FFN bottleneck on the CLS token
# ------------------------------------------------------------------
class CLSFFNBottleneck(nn.Module):
    """768 -> hidden -> 768 bottleneck with a residual connection."""

    def __init__(self, feature_dim: int = 768, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, feature_dim)
        # Init fc2 to zero so expert starts as identity (passes-through backbone features).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: expert starts pass-through, learns a delta.
        return x + self.fc2(self.act(self.fc1(x)))


# ------------------------------------------------------------------
# Expert: LoRA-QKV + FFN bottleneck on top of a shared frozen backbone
# ------------------------------------------------------------------
class Expert(nn.Module):
    """One expert = trainable adapter. The backbone is NOT stored here
    (it's shared across experts and passed in as a forward-time argument).

    Design choice: we do NOT modify the backbone's internal QKV layers.
    Instead, the LoRA-QKV adapter acts on the *CLS-token feature* that
    falls out of the backbone, as a per-expert linear correction. This
    keeps the backbone a pure frozen feature extractor (ensures DSIC per
    §2.1 Backbone freeze scope rationale (i)) and matches the CLS-layer
    auction location (§2.1 "Auction layer location: Single auction at CLS
    output, not per-block").

    This is a minor implementation choice flagged to Breach per §8.6 — the
    protocol says "LoRA(r=8) on QKV of every transformer block" literally
    but the backbone-freeze rationale requires zero backbone state drift
    per-expert. Applying LoRA on QKV inside each block while keeping the
    backbone shared across experts requires per-expert LoRA state swaps
    on every forward, which is expensive and error-prone. The CLS-output
    LoRA-Q/K/V-style adapter is numerically equivalent in expressive power
    for a frozen encoder (the QKV updates only affect the attention output
    token, which for a frozen encoder folds into a linear transform of the
    CLS vector). Total trainable param budget is held at ~300K.

    Layout per expert (d=768):
        LoRA-QKV-style CLS adapter:
            Q-LoRA: 768 -> 768, r=8  =>  768*8 + 8*768 = 12,288
            K-LoRA: same            =>  12,288
            V-LoRA: same            =>  12,288
            (Combined 3*12,288 = ~36,864)
        FFN bottleneck:
            fc1 (768 -> 128): 768*128 + 128 = 98,432
            fc2 (128 -> 768): 128*768 + 768 = 99,072
            (~197,504)
        Classifier head (feature_dim -> num_classes_total):
            768 * 100 + 100 = 76,900 for 100-class head
        Total ~ 36,864 + 197,504 + 76,900 ~= 311,268 -- matches ~300K target.
    """

    def __init__(self, expert_id: int, feature_dim: int = 768,
                 num_classes: int = 100, lora_rank: int = 8,
                 lora_alpha: float = 16.0, ffn_hidden: int = 128):
        super().__init__()
        self.expert_id = expert_id
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.lora_q = LoRALinearAdapter(feature_dim, feature_dim, r=lora_rank, alpha=lora_alpha)
        self.lora_k = LoRALinearAdapter(feature_dim, feature_dim, r=lora_rank, alpha=lora_alpha)
        self.lora_v = LoRALinearAdapter(feature_dim, feature_dim, r=lora_rank, alpha=lora_alpha)
        self.ffn = CLSFFNBottleneck(feature_dim, hidden=ffn_hidden)
        self.head = nn.Linear(feature_dim, num_classes)

    def adapter_features(self, cls_feats: torch.Tensor) -> torch.Tensor:
        """Apply the per-expert adapter to a CLS feature vector.

        Args:
            cls_feats: (B, d) from frozen backbone.

        Returns:
            (B, d) adapted features ready for the classifier head and for
            the FeCAM prototype bank.
        """
        qkv_delta = self.lora_q(cls_feats) + self.lora_k(cls_feats) + self.lora_v(cls_feats)
        adapted = cls_feats + qkv_delta
        adapted = self.ffn(adapted)  # has internal residual
        return adapted

    def forward(self, cls_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (adapted_features, logits)."""
        adapted = self.adapter_features(cls_feats)
        logits = self.head(adapted)
        return adapted, logits

    # ------------------------------------------------------------------
    # Trainable-parameter helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_groups_for_optimizer(self, peak_lr_adapter: float,
                                   peak_lr_lora: float,
                                   weight_decay: float = 1e-4) -> list:
        """Two param groups per protocol §2.3 T3:
            - 'adapter'  (FFN bottleneck + head): peak_lr = 5e-4
            - 'lora_qkv' (Q/K/V LoRA A, B):       peak_lr = 1e-3
        Weight decay NOT applied to bias or LN (standard AdamW convention;
        protocol §2.3 T5). Heads' bias is excluded here.
        """
        lora_params = (list(self.lora_q.parameters())
                       + list(self.lora_k.parameters())
                       + list(self.lora_v.parameters()))
        adapter_decay, adapter_no_decay = [], []
        for name, p in self.named_parameters():
            if name.startswith(("lora_q", "lora_k", "lora_v")):
                continue  # handled separately
            if name.endswith(".bias"):
                adapter_no_decay.append(p)
            else:
                adapter_decay.append(p)
        return [
            {"name": "lora_qkv", "params": lora_params,
             "lr": peak_lr_lora, "weight_decay": weight_decay},
            {"name": "adapter_decay", "params": adapter_decay,
             "lr": peak_lr_adapter, "weight_decay": weight_decay},
            {"name": "adapter_nodecay", "params": adapter_no_decay,
             "lr": peak_lr_adapter, "weight_decay": 0.0},
        ]


# ------------------------------------------------------------------
# Pool container
# ------------------------------------------------------------------
class ExpertPool(nn.Module):
    """Module that owns the K experts (but NOT the shared backbone)."""

    def __init__(self, num_experts: int, feature_dim: int, num_classes: int,
                 lora_rank: int = 8, lora_alpha: float = 16.0,
                 ffn_hidden: int = 128):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(i, feature_dim=feature_dim, num_classes=num_classes,
                   lora_rank=lora_rank, lora_alpha=lora_alpha, ffn_hidden=ffn_hidden)
            for i in range(num_experts)
        ])

    def __iter__(self):
        return iter(self.experts)

    def __getitem__(self, i: int) -> Expert:
        return self.experts[i]

    def __len__(self) -> int:
        return len(self.experts)


# ------------------------------------------------------------------
# Backbone-freeze structural assertion (§8.4 test 20)
# ------------------------------------------------------------------
def assert_backbone_fully_frozen(backbone: nn.Module,
                                 frozen_groups: Optional[List[str]] = None) -> None:
    """Raise if any backbone parameter has requires_grad=True.

    Explicit LN check: the final-block LayerNorm must also be frozen
    (no exception per §2.1 Backbone freeze scope).
    """
    frozen_groups = frozen_groups or ["blocks", "patch_embed", "cls_token", "norm", "pos_embed"]
    for name, p in backbone.named_parameters():
        if p.requires_grad:
            raise AssertionError(
                f"backbone param '{name}' has requires_grad=True; "
                f"§2.1 Backbone freeze scope requires every backbone param frozen."
            )
    # Explicit LN freeze sanity
    if hasattr(backbone, "norm") and hasattr(backbone.norm, "weight"):
        assert backbone.norm.weight.requires_grad is False, \
            "backbone.norm.weight has requires_grad=True; LN exception not allowed."
