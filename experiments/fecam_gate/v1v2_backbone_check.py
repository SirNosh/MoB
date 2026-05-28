"""
FeCAM shrinkage V1/V2 empirical sanity check (Phase-1 gate §4.7 pre-freeze override window).

Purpose
-------
Measure V1 = mean(diag(Sigma_c)) and V2 = mean(off-diag(Sigma_c)) for per-class
sample covariances computed on CIFAR-100 task-0 training features, under TWO
frozen backbones:
  - ViT-B/16 in21k pretrained (LAMDA-PILOT canonical; d=768)
  - ResNet-18 ImageNet1k pretrained (FeCAM paper reference; d=512)

Decision rule (per docs/protocols/fecam-router-gate.md §4.7):
  ratio = max(V1_vit / V1_rn18, V2_vit / V2_rn18, V1_rn18 / V1_vit, V2_rn18 / V2_vit)
  ratio >= 10  -> recommend gamma1 = gamma2 = 10 (paper Split-ImageNet-R ViT value)
  ratio <  10  -> keep    gamma1 = gamma2 =  1 (paper CIFAR-100 MSCIL ResNet-18 value, current default)

FeCAM paper: Goswami et al., NeurIPS 2023, arXiv 2309.14062 v3.
Reference code: github.com/dipamgoswami/FeCAM, models/base.py::shrink_cov.

This script is READ-ONLY w.r.t. the MoB codebase:
  - does NOT import mob/ or contibualmob/
  - does NOT touch contibualmob/prototype_store.py
  - does NOT write mob/gate/fecam_core.py (post-freeze sprint)

Usage
-----
    python experiments/fecam_gate/v1v2_backbone_check.py \
        --data_root ./data \
        --out results/fecam_gate/v1v2_empirical_2026-04-19.json \
        --seed 42 \
        --batch_size 128

Dependencies
------------
torch >= 2.0, torchvision >= 0.15, numpy.
timm is optional and preferred for ViT in21k weights (LAMDA-PILOT canonical).
If timm unavailable, falls back to torchvision ViT_B_16 IN1K weights and records
the substitution in the output JSON. IN1K vs in21k weights are NOT identical;
the finding's applicability to the gate is caveated in that case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Class ordering
# ------------------------------------------------------------------
# LAMDA-PILOT's canonical 'benchmark' ordering for CIFAR-100 is a fixed shuffle
# computed with numpy RandomState(1993) on np.arange(100). This is the order
# shipped in their exps/cifar100_b0_inc10.json. We reproduce it here so the
# measurement is comparable to any downstream LAMDA-PILOT gate run.
#
# If you want the 'identity' ordering (classes 0..9 as task-0), pass
# --class_order identity. For the gate, we pin to LAMDA-PILOT's convention.
def lamda_pilot_class_order() -> List[int]:
    rng = np.random.RandomState(1993)
    order = np.arange(100)
    rng.shuffle(order)
    return order.tolist()


# ------------------------------------------------------------------
# Backbone loaders
# ------------------------------------------------------------------
def load_vit_b16(device: torch.device) -> Tuple[nn.Module, int, str, str]:
    """
    Returns (model, feature_dim, weight_source_tag, fingerprint).
    Prefers timm's vit_base_patch16_224_in21k (LAMDA-PILOT canonical).
    Falls back to torchvision's vit_b_16 IN1K weights; records substitution.
    The returned model, when called on a (N, 3, 224, 224) batch, must return
    the penultimate CLS token representation of shape (N, feature_dim).
    """
    try:
        import timm  # type: ignore
        model = timm.create_model(
            "vit_base_patch16_224_in21k", pretrained=True, num_classes=0
        )
        model.eval().to(device)
        feature_dim = model.num_features  # 768
        tag = "timm:vit_base_patch16_224_in21k"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] timm ViT in21k unavailable ({e!r}); falling back to torchvision IN1K ViT-B/16")
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        # Strip classification head; keep pre-logits (CLS token after encoder).
        model.heads = nn.Identity()
        model.eval().to(device)
        feature_dim = 768
        tag = "torchvision:vit_b_16:IMAGENET1K_V1(FALLBACK_NOT_IN21K)"
    fingerprint = _fingerprint_model(model)
    return model, feature_dim, tag, fingerprint


def load_resnet18(device: torch.device) -> Tuple[nn.Module, int, str, str]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    # Penultimate: replace fc with Identity -> output of avgpool, shape (N, 512)
    model.fc = nn.Identity()
    model.eval().to(device)
    feature_dim = 512
    tag = "torchvision:resnet18:IMAGENET1K_V1"
    fingerprint = _fingerprint_model(model)
    return model, feature_dim, tag, fingerprint


def _fingerprint_model(model: nn.Module) -> str:
    """Hash of the first 4096 bytes of the model's concatenated state_dict values.
    Lets us prove later which backbone weights we used without shipping the checkpoint."""
    h = hashlib.sha256()
    n_bytes = 0
    for _, v in model.state_dict().items():
        b = v.detach().cpu().contiguous().view(-1).float().numpy().tobytes()
        h.update(b[: max(0, 4096 - n_bytes)])
        n_bytes = min(4096, n_bytes + len(b))
        if n_bytes >= 4096:
            break
    return h.hexdigest()[:16]


# ------------------------------------------------------------------
# CIFAR-100 task-0 dataset
# ------------------------------------------------------------------
def build_cifar100_task0_loader(
    data_root: str,
    class_order: List[int],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, List[int]]:
    # Bicubic 32 -> 224 per protocol §5.3. No augmentation (measurement only).
    tfm = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tfm)
    task0_classes = class_order[:10]
    mask = np.isin(np.array(ds.targets), task0_classes)
    indices = np.nonzero(mask)[0].tolist()
    # Deterministic order (no shuffle, covariance is order-invariant).
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )
    return loader, task0_classes


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------
@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    l2_normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, lbls = [], []
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = model(x)
        if f.ndim > 2:
            # Defensive reshape in case a backbone returns (N, d, 1, 1)
            f = f.flatten(1)
        if l2_normalize:
            f = F.normalize(f, p=2, dim=-1)
        feats.append(f.cpu())
        lbls.append(y)
    X = torch.cat(feats, dim=0)
    Y = torch.cat(lbls, dim=0)
    dt = time.time() - t0
    print(f"  extracted {X.shape[0]} features, d={X.shape[1]}, {dt:.1f}s")
    return X, Y


# ------------------------------------------------------------------
# Per-class Sigma + V1 + V2
# ------------------------------------------------------------------
def per_class_v1_v2(
    X: torch.Tensor,
    Y: torch.Tensor,
    class_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    """Compute Sigma_c per class and V1_c, V2_c per class.
    Sigma_c = (X_c - mu_c)^T (X_c - mu_c) / (n_c - 1)  (unbiased sample cov, matches torch.cov default)
    V1_c = mean(diag(Sigma_c))
    V2_c = mean of off-diagonal entries  = (sum(Sigma_c) - trace(Sigma_c)) / (d*(d-1))
    """
    out: Dict[int, Dict[str, float]] = {}
    d = X.shape[1]
    n_off = d * (d - 1)
    for c in class_ids:
        idx = (Y == c).nonzero(as_tuple=True)[0]
        Xc = X[idx].double()  # float64 for numerical stability
        n_c = Xc.shape[0]
        if n_c < 2:
            raise RuntimeError(f"class {c} has n_c={n_c} < 2; cannot form covariance")
        mu_c = Xc.mean(dim=0, keepdim=True)
        Xc0 = Xc - mu_c
        Sigma_c = (Xc0.t() @ Xc0) / (n_c - 1)
        trace_c = torch.diagonal(Sigma_c).sum().item()
        sum_c = Sigma_c.sum().item()
        V1_c = trace_c / d
        V2_c = (sum_c - trace_c) / n_off
        # Condition diagnostic: smallest / largest eigenvalue (catch near-singular)
        eigvals = torch.linalg.eigvalsh(Sigma_c)
        lam_min = float(eigvals.min().item())
        lam_max = float(eigvals.max().item())
        out[int(c)] = {
            "n": int(n_c),
            "V1": float(V1_c),
            "V2": float(V2_c),
            "trace": float(trace_c),
            "sum": float(sum_c),
            "lam_min": lam_min,
            "lam_max": lam_max,
            "cond_ratio": float(lam_max / max(abs(lam_min), 1e-30)),
        }
    return out


def summarize(per_class: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    v1 = np.array([v["V1"] for v in per_class.values()])
    v2 = np.array([v["V2"] for v in per_class.values()])
    return {
        "V1_mean": float(v1.mean()),
        "V1_std": float(v1.std(ddof=1)) if len(v1) > 1 else 0.0,
        "V2_mean": float(v2.mean()),
        "V2_std": float(v2.std(ddof=1)) if len(v2) > 1 else 0.0,
        "V1_min": float(v1.min()),
        "V1_max": float(v1.max()),
        "V2_min": float(v2.min()),
        "V2_max": float(v2.max()),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out", type=str,
                   default="results/fecam_gate/v1v2_empirical_2026-04-19.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--class_order", type=str, default="lamda_pilot",
                   choices=["lamda_pilot", "identity"])
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    print(f"device={device}")

    if args.class_order == "lamda_pilot":
        class_order = lamda_pilot_class_order()
    else:
        class_order = list(range(100))

    print("Loading CIFAR-100 task-0 train loader...")
    loader, task0_classes = build_cifar100_task0_loader(
        args.data_root, class_order, args.batch_size, args.num_workers, args.seed
    )
    print(f"  task0 classes: {task0_classes}")

    results: Dict = {
        "schema_version": 1,
        "date": "2026-04-19",
        "protocol_ref": "docs/protocols/fecam-router-gate.md §4.7 (v1.2 pre-freeze override window)",
        "seed": args.seed,
        "class_order_strategy": args.class_order,
        "task0_class_ids": task0_classes,
        "device": str(device),
        "batch_size": args.batch_size,
        "backbones": {},
    }

    # ViT-B/16
    print("\n[backbone 1/2] ViT-B/16")
    vit, d_vit, vit_tag, vit_fp = load_vit_b16(device)
    X_vit, Y_vit = extract_features(vit, loader, device, l2_normalize=True)
    pc_vit = per_class_v1_v2(X_vit, Y_vit, task0_classes)
    sum_vit = summarize(pc_vit)
    results["backbones"]["vit_b16"] = {
        "tag": vit_tag,
        "fingerprint": vit_fp,
        "feature_dim": int(d_vit),
        "n_samples_total": int(X_vit.shape[0]),
        "per_class": pc_vit,
        "summary": sum_vit,
    }
    del vit, X_vit, Y_vit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ResNet-18
    print("\n[backbone 2/2] ResNet-18")
    rn, d_rn, rn_tag, rn_fp = load_resnet18(device)
    X_rn, Y_rn = extract_features(rn, loader, device, l2_normalize=True)
    pc_rn = per_class_v1_v2(X_rn, Y_rn, task0_classes)
    sum_rn = summarize(pc_rn)
    results["backbones"]["resnet18"] = {
        "tag": rn_tag,
        "fingerprint": rn_fp,
        "feature_dim": int(d_rn),
        "n_samples_total": int(X_rn.shape[0]),
        "per_class": pc_rn,
        "summary": sum_rn,
    }
    del rn, X_rn, Y_rn

    # Divergence test per §4.7
    eps = 1e-30
    v1_vit, v1_rn = sum_vit["V1_mean"], sum_rn["V1_mean"]
    v2_vit, v2_rn = sum_vit["V2_mean"], sum_rn["V2_mean"]
    # For V2, use absolute value for ratio (V2 can be tiny negative after L2-normalization centering).
    a_v2_vit, a_v2_rn = abs(v2_vit), abs(v2_rn)
    ratios = {
        "V1_vit_over_rn18": v1_vit / max(v1_rn, eps),
        "V1_rn18_over_vit": v1_rn / max(v1_vit, eps),
        "V2_vit_over_rn18": a_v2_vit / max(a_v2_rn, eps),
        "V2_rn18_over_vit": a_v2_rn / max(a_v2_vit, eps),
    }
    max_ratio = max(ratios.values())
    recommendation = "gamma1=gamma2=10" if max_ratio >= 10.0 else "gamma1=gamma2=1"
    results["divergence_test"] = {
        "rule": "ratio >= 10 -> flip to gamma=10; else keep gamma=1",
        "ratios": ratios,
        "max_ratio": float(max_ratio),
        "V1_vit_mean": v1_vit,
        "V1_rn18_mean": v1_rn,
        "V2_vit_mean": v2_vit,
        "V2_rn18_mean": v2_rn,
        "recommendation": recommendation,
    }

    # Caveats pre-populated; script adds runtime-detected ones.
    caveats = []
    if "FALLBACK_NOT_IN21K" in vit_tag:
        caveats.append(
            "ViT weights are torchvision IN1K, NOT timm in21k. LAMDA-PILOT canonical "
            "is in21k. Install timm and re-run for a gate-binding number."
        )
    if a_v2_vit < 1e-8 or a_v2_rn < 1e-8:
        caveats.append(
            "|V2| < 1e-8 for at least one backbone. Additive two-parameter shrinkage's "
            "off-diagonal term becomes numerically degenerate (gamma2*V2 ~ 0). "
            "Escalate to Breach: shrinkage formula may need a floor."
        )
    any_ill = False
    for bk_name in ("vit_b16", "resnet18"):
        for _, rec in results["backbones"][bk_name]["per_class"].items():
            if rec["lam_min"] <= 0:
                any_ill = True
                break
    if any_ill:
        caveats.append(
            "At least one per-class Sigma has non-positive smallest eigenvalue (n_c=500, d=768 "
            "for ViT -> rank-deficient). Shrinkage is what fixes this; V1/V2 are still well-defined "
            "as (mean of diag) and (mean of off-diag) regardless of rank."
        )
    results["caveats"] = caveats

    # Write out
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
    print("\n=== DECISION ===")
    print(f"V1 ViT={v1_vit:.6g}  ResNet18={v1_rn:.6g}")
    print(f"V2 ViT={v2_vit:.6g}  ResNet18={v2_rn:.6g}")
    print(f"max ratio = {max_ratio:.3f}")
    print(f"recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
