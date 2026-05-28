"""Single-seed, single-arm entry point for the MoB Phase-1 gate.

Usage:
    python -m gate_runner.run_arm \
        --arm A --seed 42 \
        --config gate_runner/config.yaml \
        --out results/gate/seed_42_arm_A.json

Protocol: docs/protocols/fecam-router-gate.md v1.2.1

Flow per §2.3 + §3:
    1. Seed everything (§4.3).
    2. Load frozen ViT-B/16 in21k backbone (§2.1); assert fully frozen.
    3. Build K experts (LoRA-QKV + FFN), shared backbone (§2.1).
    4. Run alpha pre-pass on task-0 backbone features (§2.1 "alpha row" +
       §8.4 test 15). Both arms use the same alpha.
    5. For task t in 0..T-1:
         a. For epoch in 0..19 (§2.3 T1):
            - For each batch:
                * Backbone forward (no_grad, frozen features).
                * Auction (§1.3 bid formula; Arm B: beta=gamma=0 hard-coded).
                * Train ONLY the winning expert on this batch.
                * Update conscience EMA (Arm A).
                * Log per-step bid decomposition.
         b. End-of-epoch: warmup countdown on task 0.
       c. End-of-task:
         - Compute Fisher for each expert that won any batch this task.
         - Update per-(expert, class) prototype accumulators for the winning
           expert on the task's training set (in no_grad) and recompute
           mu + shrunken+normalized cov.
         - §2.3 T7 Optimizer reset: clear state for EVERY expert (all K).
    6. Final FeCAM Mahalanobis evaluation on 100-class test set (all seen).
    7. Write results JSON.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .auction import Auction, Conscience, PrototypeBank
from .data import (build_all_tests, build_task_loaders, lamda_pilot_class_order,
                   task_class_splits)
from .ewc import EWC
from .experts import (ExpertPool, assert_backbone_fully_frozen,
                      backbone_forward_features, load_vit_b16_backbone)
from .fecam_core import mahalanobis_per_class
from .utils import (canonical_yaml_hash, compute_aia, compute_forgetting,
                    get_logger, new_run_record, read_json, set_all_seeds,
                    sha256_bytes, sha256_state_dict, sha256_tensor, write_json)


# ------------------------------------------------------------------
# Warmup + cosine scheduler for task 0; cosine only for tasks 1..T-1
# ------------------------------------------------------------------
def build_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int,
                    warmup_steps: int, do_warmup: bool,
                    start_lr_frac: float, min_lr: float):
    """Warmup (linear) -> cosine to min_lr. Applied to param_group lrs.

    - do_warmup=True (task 0): linear 0 -> peak over warmup_steps, then
      cosine peak -> min_lr over (total_steps - warmup_steps).
    - do_warmup=False (tasks 1+): cosine peak -> min_lr over total_steps.

    Implementation notes:
        - PyTorch's CosineAnnealingLR uses per-group peak LR at step 0 and
          decays toward eta_min=min_lr. We set eta_min to the SMALLEST
          min_lr across groups; groups with larger peaks still end at min_lr
          (approximate; the protocol pins min_lr as a scalar).
    """
    if do_warmup and warmup_steps > 0:
        def lr_lambda(step):
            if step < warmup_steps:
                # linear from start_lr_frac -> 1.0
                return start_lr_frac + (1 - start_lr_frac) * (step / warmup_steps)
            return 1.0
        warmup_sched = LambdaLR(optimizer, lr_lambda=lr_lambda)
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=min_lr)
        return SequentialLR(optimizer, [warmup_sched, cosine_sched],
                            milestones=[warmup_steps])
    else:
        return CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=min_lr)


# ------------------------------------------------------------------
# Alpha pre-pass (§2.1 "alpha row" + §8.4 test 15)
# ------------------------------------------------------------------
@torch.no_grad()
def alpha_prepass(backbone: nn.Module, train_loader_0, bank: PrototypeBank,
                  device: torch.device, num_experts: int,
                  num_classes_task0: int, logger) -> float:
    """Compute alpha = median d_M across all (x, i, c) tuples on task-0.

    Since no expert has trained yet, all expert prototype banks are
    initialized identically (all-zero mu; identity cov). We therefore run
    ONE forward pass through the backbone and accumulate per-(class) mean
    features to build a bootstrap prototype per class, then compute
    Mahalanobis distances from every sample to every prototype under
    identity covariance. This yields a deterministic alpha that is
    independent of expert index (all experts share the same initial
    prototypes; the per-(expert, class) distinction kicks in after the
    first task's training + end-of-task FeCAM recomputation).

    Per §4.6 this is a bootstrap ONLY; real prototypes are per-expert and
    arrive after task-end FeCAM updates.
    """
    logger.info("Starting alpha pre-pass on task-0 backbone features...")
    feats_buf = []
    lbls_buf = []
    for x, y in train_loader_0:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        f = backbone_forward_features(backbone, x)
        feats_buf.append(f)
        lbls_buf.append(y)
    feats = torch.cat(feats_buf, dim=0)     # (N, d)
    lbls = torch.cat(lbls_buf, dim=0)       # (N,)
    classes_present = torch.unique(lbls).tolist()

    # Per-class mean -> bootstrap mu
    d = feats.shape[1]
    mus = torch.zeros(len(classes_present), d, dtype=torch.float64, device=device)
    for i, c in enumerate(classes_present):
        mask = (lbls == c)
        mus[i] = feats[mask].to(torch.float64).mean(dim=0)

    # Identity-covariance d_M (Euclidean on L2-normalized features)
    covs = torch.stack([torch.eye(d, dtype=torch.float64, device=device)
                        for _ in classes_present])
    dists = mahalanobis_per_class(feats.to(torch.float64), mus, covs)  # (N, C)
    alpha = float(np.median(dists.detach().cpu().numpy()))
    logger.info(f"alpha pre-pass complete: alpha = median(d_M) = {alpha:.6f} "
                f"(N={feats.shape[0]}, C={len(classes_present)})")
    return alpha


# ------------------------------------------------------------------
# Build optimizer + scheduler for a single expert
# ------------------------------------------------------------------
def build_optimizer_for_expert(expert, cfg):
    t = cfg["training"]
    peak_lr_adapter = t["schedule"]["peak_lr"]["adapter"]
    peak_lr_lora = t["schedule"]["peak_lr"]["lora_qkv"]
    wd = t["optimizer"]["weight_decay"]
    betas = tuple(t["optimizer"]["betas"])
    eps = t["optimizer"]["eps"]
    groups = expert.param_groups_for_optimizer(peak_lr_adapter, peak_lr_lora, weight_decay=wd)
    opt = torch.optim.AdamW(groups, betas=betas, eps=eps)
    return opt


def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """§2.3 T7 per-expert optimizer reset: clear Adam m_t, v_t buffers."""
    from collections import defaultdict
    optimizer.state = defaultdict(dict)


# ------------------------------------------------------------------
# Training step: only the winner is updated
# ------------------------------------------------------------------
def train_one_batch(winner, backbone, cls_feats, y, optimizers, schedulers,
                    warmup_active: bool, device: torch.device) -> Tuple[float, torch.Tensor]:
    """Forward through winner's adapter + head, backward, step.

    Returns (loss_value, logits_detached).
    """
    opt = optimizers[winner.expert_id]
    sch = schedulers[winner.expert_id]
    opt.zero_grad(set_to_none=True)
    winner.train()
    adapted, logits = winner(cls_feats)     # cls_feats is detached from backbone (no_grad)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    sch.step()
    return float(loss.item()), logits.detach()


# ------------------------------------------------------------------
# Evaluation: pseudo-label auction over all seen classes; FeCAM d_M
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(backbone, pool, bank: PrototypeBank, auction: Auction,
             conscience: Conscience, loader, seen_classes: List[int],
             device: torch.device) -> float:
    """Return accuracy (%) over `loader`. Uses pseudo-label auction to pick
    winner per batch, then the winner's own argmin_class as the prediction.
    """
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cls = backbone_forward_features(backbone, x)
        winner_idx, _ = auction.route(pool, bank, [], conscience, cls,
                                      candidate_classes=seen_classes)
        _, pred_class = bank.expert_min_dM(winner_idx, cls, seen_classes)
        correct += (pred_class == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_per_task(backbone, pool, bank, auction, conscience,
                      data_root, class_order, classes_per_task, num_tasks_done,
                      cfg, device, seed) -> List[float]:
    """Evaluate on each task 0..num_tasks_done-1 separately using their own
    test-split loader. Returns list of accuracies [acc_0, acc_1, ..., acc_{t-1}]."""
    accs = []
    seen_classes = list(class_order[:num_tasks_done * classes_per_task])
    for t in range(num_tasks_done):
        _, test_loader_t, _, _ = build_task_loaders(
            data_root=data_root, task_t=t, class_order=class_order,
            classes_per_task=classes_per_task,
            batch_size=cfg["training"]["batch_size"],
            eval_batch_size=cfg["training"]["eval_batch_size"],
            num_workers=cfg["training"]["num_workers"],
            pin_memory=cfg["training"]["pin_memory"],
            seed=seed, image_size=cfg["data"]["image_size"],
            download=False,  # already downloaded by now
        )
        acc = evaluate(backbone, pool, bank, auction, conscience,
                       test_loader_t, seen_classes, device)
        accs.append(acc)
    return accs


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MoB Phase-1 FeCAM-Router gate runner")
    parser.add_argument("--arm", required=True, choices=["A", "B"],
                        help="A = MoB-Full; B = FeCAM-Router (beta=gamma=0)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, default="gate_runner/config.yaml")
    parser.add_argument("--out", type=str, required=True,
                        help="Output JSON path (will create parent dirs)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    logger = get_logger("gate_runner.run_arm")
    started_at = datetime.datetime.utcnow().isoformat()
    t_wall0 = time.time()

    # 1) Load + hash config BEFORE seeding so the hash is deterministic
    config_path = Path(args.config)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    config_hash = canonical_yaml_hash(config_path)
    logger.info(f"config_hash = {config_hash}")

    # 2) Seed sources (§4.3) — must precede ANY CUDA work
    set_all_seeds(
        seed=args.seed,
        cudnn_deterministic=cfg["determinism"]["cudnn_deterministic"],
        cudnn_benchmark=cfg["determinism"]["cudnn_benchmark"],
        use_deterministic_algorithms=cfg["determinism"]["use_deterministic_algorithms"],
        warn_only=cfg["determinism"]["use_deterministic_warn_only"],
    )
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    logger.info(f"device={device} gpu_name={gpu_name} arm={args.arm} seed={args.seed}")

    # 3) Build run record
    record = new_run_record(
        arm=args.arm, seed=args.seed,
        protocol_version=cfg["protocol"]["version"],
        config_hash=config_hash, gpu=gpu_name,
    )
    record["started_at"] = started_at

    # 4) Backbone
    logger.info("Loading frozen ViT-B/16 backbone (timm in21k)...")
    backbone, feature_dim = load_vit_b16_backbone(
        device=device, model_name=cfg["model"]["backbone"]["name"])
    assert_backbone_fully_frozen(backbone)
    backbone_hash = sha256_state_dict(backbone.state_dict())
    record["backbone_hash"] = backbone_hash
    logger.info(f"backbone feature_dim={feature_dim} hash={backbone_hash[:16]}...")

    # 5) Class ordering + task splits
    num_tasks = cfg["data"]["num_tasks"]
    classes_per_task = cfg["data"]["classes_per_task"]
    total_classes = num_tasks * classes_per_task
    class_order = lamda_pilot_class_order(cfg["data"]["class_order_seed"])
    task_splits = task_class_splits(class_order, num_tasks, classes_per_task)
    logger.info(f"num_tasks={num_tasks} classes_per_task={classes_per_task} "
                f"class_order[:10]={class_order[:10]}")

    # 6) Expert pool
    num_experts = cfg["pool"]["num_experts"]
    pool = ExpertPool(
        num_experts=num_experts, feature_dim=feature_dim, num_classes=total_classes,
        lora_rank=cfg["model"]["expert"]["lora_rank"],
        lora_alpha=cfg["model"]["expert"]["lora_alpha"],
        ffn_hidden=cfg["model"]["expert"]["ffn_bottleneck_hidden"],
    ).to(device)
    trainable_counts = [pool[e].count_trainable() for e in range(num_experts)]
    logger.info(f"pool K={num_experts} trainable_per_expert={trainable_counts}")

    # 7) Prototype bank + EWC per expert + conscience
    bank = PrototypeBank(
        num_experts=num_experts, num_classes=total_classes, feature_dim=feature_dim,
        device=device,
        alpha1=cfg["fecam"]["alpha1"], alpha2=cfg["fecam"]["alpha2"],
        shrink=cfg["fecam"]["shrink"], norm_cov=cfg["fecam"]["norm_cov"],
    )
    ewcs = [EWC(fisher_decay=cfg["ewc"]["fisher_decay"],
                fisher_clamp_min=cfg["ewc"]["fisher_clamp_min"])
            for _ in range(num_experts)]
    conscience = Conscience(num_experts, ema_decay=cfg["conscience"]["ema_decay"])

    # 8) Alpha pre-pass on task-0 training set (before any training/auction)
    train_loader_0, _, _, _ = build_task_loaders(
        data_root=args.data_root, task_t=0, class_order=class_order,
        classes_per_task=classes_per_task,
        batch_size=cfg["training"]["batch_size"],
        eval_batch_size=cfg["training"]["eval_batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        seed=args.seed, image_size=cfg["data"]["image_size"],
    )
    alpha = alpha_prepass(backbone, train_loader_0, bank, device,
                          num_experts, classes_per_task, logger)
    record["alpha_calibrated"] = alpha

    # 9) Construct auction
    auction = Auction(
        arm=args.arm, num_experts=num_experts,
        alpha=alpha, beta_a=cfg["auction"]["beta_arm_a"],
        gamma_a=cfg["auction"]["gamma_arm_a"],
    )
    logger.info(f"auction arm={auction.arm} alpha={auction.alpha:.4f} "
                f"beta={auction.beta} gamma={auction.gamma}")

    # 10) Optimizers (one per expert)
    optimizers = [build_optimizer_for_expert(pool[e], cfg) for e in range(num_experts)]

    # 11) Training loop
    per_task_acc_rows: List[List[float]] = []
    routing_winner_counts = [0] * num_experts
    first_bid_hash = None
    loss_at_t0_hash = None

    for t in range(num_tasks):
        logger.info(f"=== TASK {t}/{num_tasks - 1} (classes {task_splits[t]}) ===")
        train_loader, _, _, task_classes = build_task_loaders(
            data_root=args.data_root, task_t=t, class_order=class_order,
            classes_per_task=classes_per_task,
            batch_size=cfg["training"]["batch_size"],
            eval_batch_size=cfg["training"]["eval_batch_size"],
            num_workers=cfg["training"]["num_workers"],
            pin_memory=cfg["training"]["pin_memory"],
            seed=args.seed, image_size=cfg["data"]["image_size"],
            download=False,
        )
        # Estimate total steps for scheduler
        steps_per_epoch = max(1, len(train_loader))
        total_steps = steps_per_epoch * cfg["training"]["epochs_per_task"]
        do_warmup = (t in cfg["training"]["warmup"]["applied_to_tasks"])
        warmup_steps = cfg["training"]["warmup"]["steps"] if do_warmup else 0
        schedulers = [
            build_scheduler(
                optimizers[e],
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                do_warmup=do_warmup,
                start_lr_frac=(cfg["training"]["warmup"]["start_lr"] /
                               cfg["training"]["schedule"]["peak_lr"]["lora_qkv"]),
                min_lr=cfg["training"]["schedule"]["min_lr"],
            )
            for e in range(num_experts)
        ]

        winners_this_task = set()
        candidate_classes = list(class_order[: (t + 1) * classes_per_task])

        for epoch in range(cfg["training"]["epochs_per_task"]):
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                cls_feats = backbone_forward_features(backbone, x)

                # === Auction ===
                winner_idx, diag = auction.route(
                    pool, bank, ewcs, conscience, cls_feats,
                    candidate_classes=candidate_classes,
                )
                winners_this_task.add(winner_idx)
                routing_winner_counts[winner_idx] += 1

                # Capture test-13 / test-11 artifacts on the very first step
                if first_bid_hash is None:
                    bid_vec = torch.tensor(diag["bid_per_expert"])
                    first_bid_hash = sha256_tensor(bid_vec)
                    record["first_bid_hash"] = first_bid_hash
                if loss_at_t0_hash is None:
                    # Loss hash BEFORE any update this run
                    with torch.no_grad():
                        _, l0 = pool[winner_idx](cls_feats)
                        ce0 = F.cross_entropy(l0, y, reduction="none")
                        loss_at_t0_hash = sha256_tensor(ce0)
                        record["loss_at_t0_hash"] = loss_at_t0_hash

                # === Train the winner only ===
                loss_val, _ = train_one_batch(
                    winner=pool[winner_idx], backbone=backbone,
                    cls_feats=cls_feats, y=y,
                    optimizers=optimizers, schedulers=schedulers,
                    warmup_active=do_warmup, device=device,
                )

                # Update conscience EMA (Arm B: harmless no-op, value not read)
                conscience.update(winner_idx)

                if batch_idx % 50 == 0:
                    logger.info(
                        f"  t={t} ep={epoch} b={batch_idx}/{steps_per_epoch} "
                        f"winner={winner_idx} loss={loss_val:.4f} "
                        f"a*dM={diag['alpha_dM_per_expert'][winner_idx]:.4f} "
                        f"b*F={diag['beta_forget_per_expert'][winner_idx]:.4f} "
                        f"g*f={diag['gamma_conscience_per_expert'][winner_idx]:.4f}"
                    )

        # === End-of-task: compute Fisher + update prototype bank ===
        logger.info(f"[task {t}] winners: {sorted(winners_this_task)}; "
                    f"updating Fisher + prototypes...")
        # Build a small dataloader copy over the task training set for
        # Fisher sampling (reuses existing loader)
        for e in winners_this_task:
            diag_f = ewcs[e].update_fisher(
                pool[e], train_loader, device,
                num_samples=cfg["ewc"]["fisher_num_samples"],
                forward_fn=lambda m, x: m(backbone_forward_features(backbone, x))[1],
            )
            record["fisher_diagnostics"][f"expert_{e}_task_{t}"] = diag_f

        # Update prototype accumulators for the winners using backbone+adapter features
        # We re-iterate the train loader (deterministic order because shuffle generator
        # has been exhausted). For fair per-(expert, class) accounting, each sample is
        # attributed to ITS WINNING expert at training time. Without per-sample routing
        # history, we attribute task-end prototype features to the expert that won
        # the MOST batches this task (proxy), which matches §2.0's cadence rule
        # "winning expert's features for each class".
        dominant_winner = max(winners_this_task,
                              key=lambda e: routing_winner_counts[e])
        logger.info(f"[task {t}] dominant winner for FeCAM: expert_{dominant_winner}")
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                cls_f = backbone_forward_features(backbone, x)
                adapted, _ = pool[dominant_winner](cls_f)
                bank.update(dominant_winner, adapted, y)
        bank.finalize_expert(dominant_winner)

        # === §2.3 T7 Optimizer reset for EVERY expert ===
        for e in range(num_experts):
            reset_optimizer_state(optimizers[e])
        logger.info(f"[task {t}] optimizer state reset for all {num_experts} experts")

        # === Per-task eval on tasks 0..t (triangular) ===
        accs_row = evaluate_per_task(
            backbone, pool, bank, auction, conscience,
            data_root=args.data_root, class_order=class_order,
            classes_per_task=classes_per_task, num_tasks_done=t + 1,
            cfg=cfg, device=device, seed=args.seed,
        )
        per_task_acc_rows.append(accs_row)
        logger.info(f"[task {t}] per-task accs: {[f'{a:.2f}' for a in accs_row]} "
                    f"mean={np.mean(accs_row):.2f}")

    # 12) Final eval on all 100 classes
    logger.info("Running final FeCAM evaluation on all 100 classes...")
    all_test = build_all_tests(
        data_root=args.data_root, class_order=class_order,
        eval_batch_size=cfg["training"]["eval_batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        image_size=cfg["data"]["image_size"], download=False,
    )
    all_seen_classes = list(class_order)
    final_acc = evaluate(backbone, pool, bank, auction, conscience,
                         all_test, all_seen_classes, device)
    logger.info(f"FINAL ACC (all {total_classes} classes): {final_acc:.3f}")

    # 13) Populate record + save
    record["per_task_acc"] = per_task_acc_rows
    record["final_acc"] = final_acc
    record["aia"] = compute_aia(per_task_acc_rows)
    record["forgetting"] = compute_forgetting(per_task_acc_rows)
    record["routing_distribution"] = conscience.as_distribution()
    record["routing_winner_counts"] = routing_winner_counts
    record["completed_at"] = datetime.datetime.utcnow().isoformat()
    record["wall_clock_sec"] = time.time() - t_wall0

    write_json(args.out, record)
    logger.info(f"wrote {args.out}")
    logger.info(f"final_acc={final_acc:.3f} aia={record['aia']:.3f} "
                f"forgetting={record['forgetting']:.3f} "
                f"wall_clock_sec={record['wall_clock_sec']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
