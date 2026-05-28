# Phase-1 Killer-Gate Protocol: MoB vs FeCAM-Router on Split-CIFAR-100

**Protocol ID:** `MOB-GATE-001`
**Version:** 1.2.1 (training-spec pin over v1.2 per Chamber pre-Phase-2 commission 2026-04-19 ~14:30)
**Date:** 2026-04-19 (v1.0) / 2026-04-19 (v1.1 amendment) / 2026-04-19 (v1.2 amendment) / 2026-04-19 (v1.2.1 amendment)
**Owner (PI):** Dev Vyas
**Author (Methodology):** Breach
**Status:** Pre-data. v1.2.1 supersedes v1.2. Awaiting freeze signature from Nosh (Lab Director) and technical ACK from Jett (implementation).
**Freeze date:** 2026-04-26. After this date, any change requires an explicit, logged amendment (see §13) and invalidates prior pilot results if the change affects an arm definition, metric, or analysis test.

> **v1.2.1 changelog is in §15.15–§15.18 at the end of this document.** v1.2.1 is a training-specification pin pass triggered by Chamber's pre-Phase-2 commission review (2026-04-19 ~14:30), which surfaced that v1.2's per-task epoch budget was carrying Jett's wall-clock-estimate value (10 epochs/task) rather than the LAMDA-PILOT canonical value (`exps/fecam.json: tuned_epoch=20` at the pinned SHA `7a6e904c`). v1.2.1 pins the epoch count at 20 (matching LAMDA-PILOT canonical for every adapter/LoRA-based method in the `cifar224` pack) and adds a dedicated training-spec section (§2.3) that pins AdamW, cosine schedule, batch size 128, weight decay, linear warmup policy, optimizer-reset discipline, and backbone-freeze policy (no LN exception). v1.2.1 touches only §1 (version header), §2.1 (invariant rows upgraded to pinned values), §2.3 (NEW), §8.4 (one acceptance test added, total 20), §12 (wall-clock estimate updated), §15 (changelog). All v1.1/v1.2 text — statistical gate framework, seed plan, Fisher-match gate, FeCAM-canonical Mahalanobis core, §6 power analysis, §9 outcome branches, §4.6 repo pins, §4.7 shrinkage hyperparameters — is unchanged. The v1.1 changelog is at §15.1–§15.7; v1.2 at §15.8–§15.14; v1.2.1 at §15.15–§15.18.

> **Scope.** This document specifies the Phase-1 killer gate only — the single head-to-head comparison between MoB-Full and FeCAM-Router on Split-CIFAR-100, whose outcome determines the program's survival per `docs/research-party/synthesis.md` §0. It does not specify the Phase-3 legibility-baseline suite (L2P / DualPrompt / CODA-Prompt / SLCA / HiDe-Prompt / RanPAC / EASE / LoRA-MoE-CL). The document reserves experimental slots for those baselines but their full protocol is deferred.

---

## 1. Preregistration

### 1.1 Hypotheses (registered; cannot be amended post-data without annotation)

Let `A_T^{MoB}` and `A_T^{FeCAM}` denote the Final Average Accuracy at task T=20 on Split-CIFAR-100-20T, per-seed, paired across arms.

Let `Δ_s = A_T^{MoB}(s) - A_T^{FeCAM}(s)` for seed `s ∈ S`.

- **H_null**: `E[Δ] = 0`. MoB and FeCAM-Router have identical Final Avg Acc in expectation.
- **H_alt (one-sided)**: `E[Δ] > 0`. MoB has strictly higher Final Avg Acc than FeCAM-Router.
- **Practical-significance threshold**: `Δ_practical = 1.0 percentage points` (absolute). A statistically significant but sub-threshold win does **not** pass the gate. Rationale: Cypher's CIFAR-100 audit (`docs/lit-review/04-cifar100-benchmark-audit.md` §6.6) documents 1.5–3pp variance across class-order seeds on this benchmark, so an effect smaller than 1pp is not robustly separable from pre-existing protocol noise.

### 1.2 Gate decision rule (registered; cannot be amended post-data) — v1.1 symmetric one-sided BCa framework

**v1.1 amendment (resolves KAY/O D1 / D7).** The v1.0 rule mixed a one-sided primary superiority test with a two-sided FAIL criterion that had no registered test statistic and was operationally near-unreachable at n=10. v1.1 replaces this with a **symmetric one-sided BCa framework** so that PASS and FAIL are mirror-image decisions under matched-severity tests, and TIE is strictly the residual.

Let `L_95` be the 95% BCa one-sided **lower** confidence bound on `E[Δ]` computed from `{Δ_s : s ∈ S}` (10,000 resamples; see §6.2).
Let `U_95` be the 95% BCa one-sided **upper** confidence bound on `E[Δ]` computed from the same resample ensemble.

**PASS**: `L_95 ≥ +Δ_practical` (i.e., `L_95 ≥ +1.0pp`).
**FAIL**: `U_95 ≤ −Δ_practical` (i.e., `U_95 ≤ −1.0pp`).
**TIE**: neither of the above (the default residual).

All three regions are mutually exclusive and exhaustive by construction. Both the PASS and the FAIL tests run at α=0.05 one-sided on their own tail. The **familywise Type-I error rate for the composite "PASS or FAIL" decision** is bounded above by α_PASS + α_FAIL = 0.10 under the worst-case null. Breach preregisters this composite α=0.10; if Nosh requires strict FWER=0.05 across the two decisions, split α=0.025 per side (tightening both bounds equivalently). The composite α is listed explicitly in the gate memo.

The paired one-sided t-test (§6.1) and Wilcoxon signed-rank test (§6.3) are retained as secondary robustness reports for PASS and FAIL, both evaluated symmetrically (one-sided on each tail). They do not drive the gate decision; the BCa bounds do.

No decision rule may be relaxed after data collection begins. If the pilot (§7) reveals that `Δ_practical = 1.0pp` is statistically undetectable at n=10, this is reported as an **underpowered gate** and escalated to Nosh per §13 — the bar is not moved.

**Equivalent framing for reader legibility (not a second test).** The symmetric one-sided bounds can be read as a TOST-style equivalence region of width `2·Δ_practical = 2.0pp` centered at zero: PASS requires the CI to fall entirely to the right of the region; FAIL requires it to fall entirely to the left; TIE is everything inside or straddling either boundary. v1.1 commits to the symmetric one-sided BCa framing as primary; the TOST reading is an expositional equivalent, not a separate analysis.

### 1.3 Registered arms

- **Arm A — MoB-Full**: bid = α·d_M(x, μ_{i,c}, Σ_i) + β·c_forget,i + γ·(f_i / f̄)
- **Arm B — FeCAM-Router**: bid = α·d_M(x, μ_{i,c}, Σ_i), with β≡0, γ≡0

Exact definitions in §2. **v1.2 binding (see §2.0 and §4.6):** the Mahalanobis core `d_M(x, μ_{i,c}, Σ_i)` used in BOTH arms is the LAMDA-PILOT + FeCAM-paper canonical implementation described in §2.0, pinned to repo commit SHAs in §4.6. Arm B = Arm A with `β=γ=0`; this is the ONLY code-path difference between arms.

### 1.4 Registered primary metric

Final Average Accuracy `A_T` on the 20-task split after task 20 completes, computed as `A_T = (1/T) Σ_i a_{i,T}` per Lopez-Paz-Ranzato convention (Cypher §4.1). Evaluated on the CIFAR-100 standard test set AND the ciFAIR-100 duplicate-free retest set (Cypher §2.2, §8). Headline gate decision uses the CIFAR-100 standard test set for leaderboard legibility; ciFAIR is reported as a robustness check.

### 1.5 Registered secondary metrics (do not drive the gate decision)

- Average Forgetting `F_T` (Chaudhry 2018).
- Backward Transfer `BWT` (Lopez-Paz-Ranzato 2017).
- Per-task accuracy trajectory matrix `a_{i,j}` for all `i ≤ j ≤ T`.
- Routing entropy per task (Shannon entropy over expert-win distribution on task-i eval set).
- Expert utilization Gini coefficient.
- FLOP-adjusted accuracy (Avg Acc per 10^15 total training FLOPs).
- Per-step Fisher magnitude distribution (§4.4 diagnostic).
- Per-step bid decomposition: `α·d_M`, `β·c_forget`, `γ·conscience` magnitudes.

### 1.6 Registered preregistration amendment log

Reserved section. All post-freeze amendments logged here with date, signer, diff, and justification. v1.1 amendment pass (2026-04-19, pre-freeze, pre-data) is summarized in §15 and cross-referenced to KAY/O's red-team defects D1–D8.

### 1.7 Paper metric primacy (v1.1; resolves KAY/O minor defect #9 / §7.5)

Any downstream paper, preprint, technical report, blog post, or external presentation describing the results of this gate MUST report the §1.2 primary gate test statistic — the 95% BCa one-sided bounds `L_95` and `U_95` on `E[Δ]` at 20T CIFAR-100 — as the headline comparison between MoB-Full and FeCAM-Router. Secondary metrics (§1.5) may be reported alongside but **may not substitute** for or be reframed as the primary comparison. In particular, `F_T`, `BWT`, routing entropy, Gini, FLOP-adjusted accuracy, and per-task trajectories are permissible supplements; the headline Phase-1 outcome claim must cite `L_95` (for a PASS) or `U_95` (for a FAIL) or the residual TIE classification.

Violation of this clause invalidates the preregistration claim on any publication derived from this gate. This constraint is inherited by any follow-up paper that cites the gate outcome as evidence.

### 1.8 Wall-clock disclosure on PASS (v1.1; resolves KAY/O minor defect #8 / §4.4)

If the gate PASSES, the published paper MUST disclose, as a prominent adjunct to the headline BCa bound, the per-seed-mean **wall-clock ratio `wallclock^A / wallclock^B`** alongside the FLOP-matched test statistic. Rationale: a PASS at 1.2× Arm-A wall-clock is a materially different research claim than a PASS at matched wall-clock. The reader must be able to evaluate both the compute-normalized and the latency-normalized claim from the headline table. Failure to disclose the wall-clock ratio is treated as a preregistration violation under §1.7.

---

## 2. Two-Arm Experimental Design

The two arms are **strict nested ablations**: Arm B's bid is Arm A's bid with `β=γ=0`. Nothing else differs. Any observed Arm A > Arm B delta must be attributable to the `β·c_forget + γ·conscience` terms — since every other architectural, training, and data-handling choice is held identical by this protocol.

### 2.0 FeCAM-canonical Mahalanobis core (v1.2; resolves gap analysis `docs/lit-review/05-fecam-code-comparison.md`)

The Mahalanobis term `d_M(x, μ_{i,c}, Σ_i)` that appears in both arms' bids is defined as the **paper-canonical FeCAM recipe** (Goswami et al., NeurIPS 2023, arXiv 2309.14062, §3 and §7), ported byte-equivalent from the upstream reference implementation `dipamgoswami/FeCAM` at commit SHA pinned in §4.6. Arm A and Arm B share this core identically; the only code-path difference between arms is the hard-coded `β=γ=0` in Arm B's bid-composition step, per §2.2.

**The five load-bearing elements, all four covariance elements mandatory:**

1. **Per-class prototype means** `μ_{i,c}` — one mean per (expert, class) pair, maintained by incremental update as classes are seen by each expert.
2. **Per-class covariance `Σ_{i,c}`** (NOT shared per expert) — one covariance matrix per (expert, class) pair. This is FeCAM's primary configuration (paper Table 1: per-class beats "common covariance" by 2.1pp on CIFAR-100 T=5 under ResNet-18). The v1.1 §2.1 row "Covariance structure: tied low-rank shared `U ∈ ℝ^(768×32)` + per-expert diagonal scaling" is RE-SCOPED in §2.1 (see v1.2 note) to cover only the internal numerical representation of `Σ_{i,c}`, not a substitute for the per-class grain.
3. **Additive two-parameter shrinkage** (paper eq. 8, `dipamgoswami/FeCAM:models/base.py::shrink_cov`):
   ```
   Σ_s = Σ + γ₁·V₁·I + γ₂·V₂·(1−I)
   ```
   where `V₁ = mean(diag(Σ))`, `V₂ = mean(off-diag(Σ))`, and `γ₁, γ₂` are fixed hyperparameters per §4.7. Single-parameter ridge `Σ + ε·I` is explicitly NOT canonical FeCAM and is a v1.2-binding violation; see §4.6 escalation note on LAMDA-PILOT's deviation.
4. **Correlation normalization** (paper eq. 7, `dipamgoswami/FeCAM:models/base.py::normalize_cov`):
   ```
   Σ̂(i,j) = Σ(i,j) / (σ(i) · σ(j))     where σ(i) = √Σ(i,i)
   ```
   Applied after shrinkage, before inversion. Diagonals become 1; off-diagonals become Pearson correlations in [−1, 1]. Matches `torch.corrcoef` applied to the shrunken covariance.
5. **L2 normalization of features and class means** (paper §3.1, `_mahalanobis`):
   ```
   x̃ = x / ‖x‖₂,   μ̃ = μ / ‖μ‖₂,   d_M² = (x̃ − μ̃)ᵀ Σ̂⁻¹ (x̃ − μ̃)
   ```
   The L2 normalization is applied to features AND prototypes consistently, before the subtraction.
6. **Tukey β transform** (paper eq. 9, §7) — **v1.2 decision: OFF for the gate.** The paper §7 explicitly states: *"when using the ViT encoder pre-trained on ImageNet-21K, we also have negative values in the feature representations, hence we do not apply the tukey's transformation on the features for those experiments."* Since §2.1 pins the backbone to ViT-B/16 ImageNet-21k, Tukey is disabled in BOTH arms to match the paper's own ViT configuration. The Tukey code path remains present (per `frozen_config.yaml: tukey: false`) and its correct operation is verified by §8.4 test 19.

**Inversion** is via `torch.linalg.pinv` (Moore-Penrose pseudoinverse) on the correlation-normalized shrunken covariance. This matches both upstream repos and handles the near-singular case without silent failure. `torch.linalg.inv` is NOT substituted at the primary path (it is used as a debug-only assertion that the same result emerges when the matrix is well-conditioned).

**Prototype and covariance update cadence.** Running-sum / running-outer-product accumulators per (expert, class), with `Σ_{i,c}` recomputed from accumulators at the end of each task (canonical FeCAM cadence per `dipamgoswami/FeCAM` upstream). Incremental-every-step updates are not canonical and are not used in either arm. The `contibualmob/prototype_store.py` "online Mahalanobis" code path is EXPLICITLY OUT OF SCOPE for this gate; the gate runs entirely through the v1.2-bound FeCAM pathway.

**What is out of scope.** This protocol does NOT modify `contibualmob/prototype_store.py` (the existing MoB prototype store). That module stays in place for S1 / MNIST / non-gate experiments. The gate implementation lives in a new module (Jett-specified; `mob/gate/fecam_core.py` or equivalent) that imports / ports from LAMDA-PILOT + `dipamgoswami/FeCAM` per §4.6.

### 2.1 What is held IDENTICAL across arms (protocol invariants)

| Invariant | Specification | Source of truth |
|---|---|---|
| Backbone | Frozen ViT-B/16, ImageNet-21k pretrained (timm `vit_base_patch16_224.augreg_in21k`) | Chamber R2; synthesis §2.2 |
| Backbone freeze scope | **v1.2.1 pinned:** parameter groups `["blocks", "patch_embed", "cls_token", "norm", "pos_embed"]` all fully frozen (`requires_grad=False`). **No final-block LayerNorm unfreezing exception.** Rationale: (i) unfreezing final-block LN makes LN scale/shift parameters private per-expert bid-time-hidden state, which breaks DSIC (an expert could bias its own `d_M` score downward by tuning LN scale without declaring it as a bid component); (ii) LoRA QKV already supplies direction-aware reweighting of attention features, so LN-unfreezing adds negligible headroom; (iii) LAMDA-PILOT's `dualprompt.json` pack confirms full backbone freeze is the convention for pre-trained-ViT CIL baselines. Verified at init by the backbone-hash check and by §8.4 test 20's structural check. | §2.3; Chamber pre-Phase-2 commission 2026-04-19 |
| Backbone frozen weights hash | SHA-256 of state_dict, logged per run; verifies no backbone parameter (including LN) drifted post-init | Jett acceptance test (§8.4) |
| Per-expert adapter | LoRA(r=8) on QKV of every transformer block + per-expert FFN bottleneck | Chamber R2 §2.2; synthesis §2.2 |
| Trainable params per expert | ~300K (as Chamber specced; exact number logged) | Chamber R2 |
| Expert count | 4 experts (per S1/S2 commitment; `memory/MEMORY.md` and `feedback_4experts.md`) | project.md §13 |
| Routing feature | CLS token after frozen backbone, projected through shared `W_route` tied across experts | synthesis §2.2 |
| Covariance structure | **v1.2 re-scoped:** paper-canonical **per-class** covariance `Σ_{i,c}` per §2.0 element (2); the v1.1 "tied low-rank shared `U ∈ ℝ^(768×32)` + per-expert diagonal scaling (r=32)" is an internal numerical representation strategy (Woodbury-identity-friendly storage of `Σ_{i,c}`) and MAY be used as long as the final `d_M` values are byte-equivalent (≤1e-5 per-sample) to the direct per-class `Σ_{i,c}` path verified by §8.4 test 17. If the tied-low-rank representation cannot deliver per-class grain within this tolerance, the direct per-class path is used and the tied-low-rank is dropped. | §2.0; synthesis §1 item 4; Chamber R2 §4.2 |
| Mahalanobis formulation | **v1.2 bound:** LAMDA-PILOT + FeCAM-paper canonical pathway per §2.0 (per-class Σ + additive two-parameter shrinkage `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)` + correlation normalization + L2-normalized features; Tukey OFF for ViT-B/16 per paper §7). Repo pins in §4.6. | §2.0; §4.6; paper arXiv 2309.14062 |
| Shrinkage hyperparameters | **v1.2:** `γ₁, γ₂` fixed before any seed launches per §4.7 (default `γ₁=γ₂=1` per paper §7 CIFAR-100 MSCIL; pre-registered override window closes at freeze). Public, fixed, identical across arms. NOT data-dependent post-freeze. Single-parameter ridge (`cov + α·I`) is explicitly NON-canonical and is a binding violation. | §2.0 element (3); §4.6 paper §7 |
| L2 normalization of features and prototypes | **v1.2:** applied to both features and class means before the subtraction `(x̃ − μ̃)`, per §2.0 element (5) / paper §3.1. Identical across arms. | §2.0; paper §3.1 |
| Tukey β transform | **v1.2:** OFF for the gate. `config.tukey = false`. Paper §7 disables Tukey for ViT features due to negative values; this protocol follows that recipe. Code path present; §8.4 test 19 verifies it is NOT applied when disabled. | §2.0 element (6); paper §7 |
| Prototype class granularity | **v1.2:** per-class means `μ_{i,c}` AND per-class covariances `Σ_{i,c}` (NOT shared-within-expert). Supersedes the v1.1 "shared-within-expert covariance" row. | §2.0 element (1)–(2); paper Table 1 |
| α (execution-cost weight) | **v1.1 (D4):** Calibrated once on a **shared pre-routing pre-pass**: a backbone-only forward on the entire task-1 training set is run *before* any training in either arm; for each `(x, i, c)` compute `d_M(x, μ_{i,c})` using initialized prototypes; α is set to the median `d_M` across all `(x, i, c)` tuples. The single resulting scalar is written to `frozen_config.yaml` and both arms load this identical α at launch. Not per-arm. Not task-1 features post-routing. Verified by §8.4 test 10. | Astra posted-price convention; v1.1 resolves KAY/O D4 |
| Auction layer location | Single auction at CLS output, not per-block | synthesis §2.2 |
| Task partition | Three configurations: 5T×20C, 10T×10C, 20T×5C | synthesis §2.2 |
| Headline config | 20T×5C (the gate config; 20T is the synthesis commitment) | synthesis §0 |
| Class ordering | LAMDA-PILOT default random permutation, seeded by run seed | synthesis §6.2 deferred; this protocol defers to harness default |
| Test set | Standard CIFAR-100 test (10K) | Cypher §2 |
| Robustness test set | ciFAIR-100 duplicate-free retest | Cypher §2.2 |
| Upsample 32→224 | Bicubic interpolation, recorded in log | Cypher §2.3 |
| Data augmentation | LAMDA-PILOT default (random crop with padding=4, horizontal flip, per-channel normalization). Identical across arms | harness default |
| Optimizer | **v1.2.1 pinned:** AdamW (`torch.optim.AdamW`, `β₁=0.9, β₂=0.999, ε=1e-8`), `weight_decay=1e-4`. Chosen per LoRA-CL convention (InfLoRA, SLCA body head, CODA-Prompt). LAMDA-PILOT's own `fecam.json` uses SGD+MultiStepLR, but that config targets the APER adapter family; for the LoRA-QKV path (our §2.1 trainable-params choice) AdamW+cosine is the dominant and paper-reproducible convention. Identical across arms. | §2.3; LoRA-CL convention; Chamber 2026-04-19 |
| LR schedule | **v1.2.1 pinned:** cosine decay from peak to `min_lr=1e-5` over each task's step budget, no warm-restart between tasks. Peak LR: `5e-4` for adapter path, `1e-3` for LoRA-QKV path (InfLoRA/SLCA canonical). **Linear warmup over 100 steps on task 1 only**; subsequent tasks skip warmup (adapter arrives pre-warmed from task 0). See §2.3 for rationale. Identical across arms. | §2.3; Chamber 2026-04-19 |
| Per-task epochs | **v1.2.1 pinned: 20 epochs/task**, matched across arms, same count for every task index 0..T-1. Citation: LAMDA-PILOT `exps/fecam.json: tuned_epoch=20` at commit `7a6e904c` (the §4.6 harness binding), corroborated by the entire LAMDA-PILOT `cifar224` adapter/LoRA config pack (`aper_aperpter.json`, `ranpac.json`, `mos.json`, `coda_prompt.json`, `slca.json: epochs=20`, `ease.json: init_epochs=20` all use 20). The v1.2 row's "reserved for Jett" deferment is superseded; the 20-epoch pin closes Jett's wall-clock-estimate tension against LAMDA-PILOT canonical in favor of canonical. Running undertime would render any PASS outcome dismissible as "you beat an undertrained FeCAM." | §2.3; §4.6; LAMDA-PILOT `exps/fecam.json` @ `7a6e904c` |
| Batch size | **v1.2.1 pinned: 128.** With 20 epochs and ~40 steps/epoch (CIFAR-100 5-class task at bs=128), yields ~800 optimizer steps per task — enough step budget for cosine decay to have non-trivial effect and for the 100-step task-1 warmup to settle within the first ~2.5 epochs. Note: LAMDA-PILOT `fecam.json` uses `batch_size=48`; Chamber R2 selects 128 as the LoRA-QKV convention (InfLoRA, SLCA) — divergence from LAMDA-PILOT's FeCAM config is documented as intentional in §2.3 / §15.15. Identical across arms. | §2.3; Chamber R2 §2.2; LoRA-CL convention |
| Random seed domain | `{42, 43, 44, 45, 46, 47, 48, 49, 50, 51}` (contiguous, preregistered). Paired across arms | §4 |
| Determinism | `torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`, full 5-source seeding (Py/NumPy/torch-CPU/torch-CUDA/cuBLAS-deterministic env var) | §4.3 |
| DataLoader worker seeding | **v1.1 (minor #6):** `DataLoader(..., worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(s))`, where `seed_worker(worker_id)` seeds `random`, `numpy.random`, `torch` from `base_seed + worker_id` (standard PyTorch idiom). `num_workers=0` is permitted as an equally-acceptable alternative; Jett picks one and freezes it in `frozen_config.yaml` before the pilot. Verified by §8.4 test 14 (`test_dataloader_worker_rng_pinned`). | §4.3 / §8.4 |
| EWC Fisher estimator | Projected-gradient Fisher with `r_f=32` shared backward (Killjoy R2 §2.4). Computed identically, but Arm B multiplies its contribution to the bid by zero | synthesis §1 item 5 |
| Fisher clamp | `min=0.1` per `memory/MEMORY.md` (load-bearing in MoB; also applied in Arm B to preserve implementation symmetry — but has no effect because β=0) | project.md §9 |
| Optimizer reset | **v1.2.1 tightened:** on task end, after Fisher update, before the next task's first optimizer step, `optimizer.state = defaultdict(dict)` invoked **for every expert (all 4), not just the task-winner(s)**. Rationale: Adam's second-moment estimate `v_t` carries task-specific gradient magnitude memory; leaving stale `v_t` on a non-winner expert means that when a future task does route to that expert, its first few steps take miscalibrated step sizes. Resetting all experts keeps the momentum state task-local across the board. Applied identically in both arms. Verified by §8.4 test 20 (`test_optimizer_reset_applies_to_every_expert`). | project.md §9; `memory/MEMORY.md`; Chamber 2026-04-19 |
| Evaluation routing | Pseudo-label auction: each expert's bid uses its own argmax predictions (not ground truth) | project.md §9 |
| Evaluation frequency, logging cadence | Identical across arms. Eval: end-of-task only (standard CIL; `a_{i,j}` for `j` = task-end). Logging: per-step routing/bid/Fisher to parquet; per-epoch accuracy/loss to json. Exact cadence and buffer-flush policy frozen in `frozen_config.yaml`. (Per-task epoch count is pinned separately above at 20.) | Jett freezes once |

### 2.2 What is VARIED between arms (the ablation)

| Variant | Arm A (MoB-Full) | Arm B (FeCAM-Router) |
|---|---|---|
| β (forget-cost weight) | Calibrated against empirical median `c_forget` | **0 (hard-coded)** |
| γ (conscience weight) | 0.1 × α scale (synthesis §2.1 default) | **0 (hard-coded)** |
| `c_forget` computed? | Yes (projected-gradient EWC) | Yes (computed but not used in bid — for logging only) |
| Conscience EMA updated? | Yes (DeSieno `f_i` win-frequency EMA) | Yes (updated but not used in bid — for logging only) |

> **Implementation discipline (§8.4 acceptance test):** β and γ in Arm B must be enforced as hard-coded constants in the bid computation, not set via a hyperparameter flag. A passing Arm B run must still log `c_forget` and `conscience` values (they are computed) to prove the code path is exercised and allow post-hoc inspection that the Arm A mechanism would have worked identically. This prevents silent divergence via code-path asymmetry.

### 2.3 Training specification (v1.2.1; pinned by Chamber pre-Phase-2 commission 2026-04-19)

This subsection pins the per-task training recipe. v1.2 left these items deferred to Jett per §2.1 row "reserved for Jett" placeholders; Chamber's pre-Phase-2 commission surfaced that this deferment (in particular the 10-epoch wall-clock estimate) risked training FeCAM undertime relative to LAMDA-PILOT canonical, rendering any PASS outcome dismissible as "you beat an undertrained FeCAM." v1.2.1 closes this by pinning all seven items below. Both arms, matched.

**T1. Per-task epoch count.** **20 epochs per task.** Binding citation: `sun-hailong/LAMDA-PILOT/exps/fecam.json` at the §4.6-pinned commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63` specifies `tuned_epoch: 20`. Corroborating LAMDA-PILOT `cifar224` adapter/LoRA pack (`aper_aperpter.json: 20`, `ranpac.json: 20`, `mos.json: 20`, `coda_prompt.json: 20`, `slca.json: epochs=20`, `ease.json: init_epochs=20`) — every adapter/LoRA-based method in the canonical config pack uses 20 epochs. Running undertime at 10 epochs would train FeCAM to a state materially weaker than the published reference and invalidate the headline claim. 20 epochs × 20 tasks × ~40 steps/epoch ≈ 16,000 optimizer steps per full run per arm; per-task step budget ≈ 800 steps (enough for cosine decay and 100-step task-1 warmup to matter).

**T2. Optimizer.** **AdamW** (`torch.optim.AdamW`) with `β₁=0.9, β₂=0.999, ε=1e-8`. Rationale: LoRA-QKV continual-learning literature (InfLoRA, SLCA body-lr head, CODA-Prompt) uses Adam/AdamW as the dominant convention. LAMDA-PILOT itself splits: SGD+MultiStepLR for the FeCAM/APER/SLCA-backbone paths (ResNet-style finetune) and Adam+constant for L2P/DualPrompt/CODA-Prompt (prompt-tuning). The gate's §2.1 trainable-params choice (LoRA on QKV of every transformer block + per-expert FFN bottleneck) sits in the LoRA-QKV regime where AdamW+cosine is paper-reproducible and has the strongest Phase-1 baseline behavior. **Intentional divergence from `fecam.json: optimizer=sgd`** — this is NOT a §4.6 binding violation because §4.6 binds Arm B's *Mahalanobis scoring recipe* against `dipamgoswami/FeCAM` upstream and its *harness/dataloader/backbone loading* against LAMDA-PILOT; the optimizer choice is a §2.1 protocol-layer invariant and v1.1 already declared AdamW in the §2.1 row. v1.2.1 merely pins the numeric values.

**T3. Learning-rate schedule.** Cosine decay from peak to `min_lr=1e-5` over each task's step budget (800 steps), **no warm-restart at task boundary** — each task's cosine is a fresh full half-period from peak → min_lr. Peak LR: **5e-4 for the adapter path (per-expert FFN bottleneck), 1e-3 for the LoRA-QKV path** — separate parameter groups with separate peak LRs. These are the InfLoRA/SLCA canonical values. Single cosine scheduler, one `LambdaLR` or `CosineAnnealingLR` with `T_max = steps_per_task` instantiated fresh at each task boundary.

**T4. Batch size.** **128.** Yields ~40 steps/epoch × 20 epochs = ~800 steps/task, enough step budget for cosine to have non-trivial effect. CIFAR-100 at 5 classes/task × 500 train samples/class = 2500 samples/task, so epoch size ≈ 2500/128 ≈ 19.5 steps (closer to ~20 than 40; the 800-step estimate is for the 20T config where each task has 2500 samples). **Divergence from `fecam.json: batch_size=48`** — Chamber R2 selects 128 per LoRA-QKV convention; this is documented as intentional.

**T5. Weight decay.** **1e-4** on LoRA and FFN-bottleneck parameters. Not applied to LayerNorm or bias parameters (standard AdamW decoupled-weight-decay convention). **Divergence from `fecam.json: weight_decay=5e-4`** — again, intentional; Chamber R2's LoRA-QKV convention uses 1e-4. Both divergences (T4, T5) are from LAMDA-PILOT's FeCAM-specific config, not from its LoRA-path conventions; they reflect that the gate trains with LoRA adapters where the FeCAM paper trained with a fully-finetuned classifier head. Documented explicitly in §15.15.

**T6. Warmup.** **Linear warmup over 100 optimizer steps, on task 1 only.** Tasks 2..T skip warmup — the adapter arrives pre-warmed from task 0's trained state, and re-warmuping each task resets learning to near-zero and wastes the first ~2.5 epochs of the cosine's effective budget. Implementation: warmup is a linear ramp from `1e-6` to the peak LR (5e-4 or 1e-3 depending on parameter group) over the first 100 steps of task 1's total 800, after which the cosine takes over for the remaining 700 steps of task 1. Tasks 2..T begin directly at peak LR and cosine-decay over their full 800 steps. Verified by §8.4 test 20's schedule inspection.

**T7. Optimizer reset at task boundary.** At the END of each task (after the final gradient step of that task and after the post-task Fisher update), `optimizer.state = defaultdict(dict)` is invoked **for every expert (all 4 experts)**, not just the task's winner(s). This clears Adam's first- and second-moment buffers (`m_t`, `v_t`) so that when the next task routes to any expert (winner or previously-cold), its first few steps take well-calibrated step sizes rather than Adam step sizes biased by stale gradient-magnitude memory from a prior task's training trajectory. This is per `memory/MEMORY.md` plus Chamber's v1.2.1 tightening from "winner only" → "every expert." Applied identically in both arms. Verified by §8.4 test 20.

**Freeze-policy clarifier (cross-reference, not a separate item).** §2.1's "Backbone freeze scope" row (v1.2.1) pins the frozen parameter groups as `["blocks", "patch_embed", "cls_token", "norm", "pos_embed"]` with **no final-block LN unfreezing exception**. This belongs conceptually in the training spec but is enforced at init time via the backbone-hash check and §8.4 test 20's structural assertion; the row lives in §2.1 because the backbone itself is a §2.1 invariant. Do not conflate: "freeze policy" is §2.1; "training spec" (what is trained, how) is §2.3 T1–T7.

**What is NOT pinned here (intentional).** Gradient accumulation factor, DataLoader worker count, mixed-precision dtype, `torch.compile` mode, logging cadence — all remain in §8.5 as Jett-tunable as long as the numerical invariants above hold (verified by §8.4 tests 3, 5, 11, 13).

### 2.4 Why this is a KAY/O-invariant strict ablation

Per `docs/research-party/round1/kayo-position.md` §2, KAY/O's threat is that MoB is epiphenomenal: the prototype-argmin (FeCAM) does the work; the β·forget term only matters in a narrow strip where Fisher-magnitude seed variance dominates the signal. This protocol's Arm B is **exactly** the null KAY/O proposes. Arm B is FeCAM applied at the routing layer (not the classifier layer) with the MoB backbone/adapter architecture held invariant. Any Arm A advantage therefore isolates the contribution of `β·c_forget + γ·conscience`. Any Arm A tie or loss confirms KAY/O's null.

There is no room for protocol-shopping here: both arms use the same harness, the same backbone, the same adapters, the same covariance, the same schedule, the same seeds. The only axis of variation is the two terms under test.

---

## 3. Matched-Compute Definition

### 3.1 Operational definition (primary)

**Matched total training FLOPs (within ±5%).** Per Killjoy R2, MoB's projected-gradient EWC costs one shared backward per step plus O(E·r_f) per-expert Fisher dot products — this is non-zero overhead relative to Arm B, which does the shared backward and the dot products but contributes zero to the bid. In practice the compute overhead of Arm A over Arm B is `<5%` (Killjoy §4 crux 2), so matching to ±5% in FLOP budget is a light constraint that Arm B auto-satisfies by construction if both arms run the same per-task epoch budget.

FLOP count per run is computed via `fvcore.nn.FlopCountAnalysis` on the forward graph and multiplied by the measured number of forward+backward passes per task. The per-step Fisher-projection overhead is counted explicitly (see §3.3).

### 3.2 Secondary definitions (reported, not used for matching)

Both are reported per run to make the primary-matching claim falsifiable.

- **Matched gradient-step budget**: both arms use the same number of optimizer steps per task × same number of tasks. This is the natural consequence of §2.1 (matched batch size + matched per-task epochs + same dataset partition).
- **Matched wall-clock**: wall-clock per run on identical hardware. The expected Arm A overhead of 2–5% (Killjoy R2 §2.1) should manifest as slightly longer wall-clock for Arm A; this is not a matching constraint, but if Arm A wall-clock falls **below** Arm B's, the run is suspect and is rejected pending investigation.

### 3.3 FLOP accounting recipe (Jett-facing)

Per run, log the following FLOP decomposition:

```
F_total = F_forward_training + F_backward_training + F_eval + F_bid_mechanism + F_fisher_projection
```

Where:
- `F_forward_training = (n_steps × tokens_per_step × F_forward_per_token)` per `fvcore`.
- `F_backward_training ≈ 2 × F_forward_training` (standard convention).
- `F_eval = (n_eval_passes × n_eval_samples × F_forward_per_sample)`.
- `F_bid_mechanism = n_steps × B × (2·r·d + E·r + E·r_f)` — bid compute per step, per Killjoy R2 §2.1 formula.
- `F_fisher_projection = n_fisher_updates × (r_f × n_params + E · r_f)` — projected-gradient EWC.

Report each term. Arm-B `F_bid_mechanism` and `F_fisher_projection` must be computed even though the values do not enter the bid (the code path runs, see §2.2 implementation discipline).

**Matching criterion:** `| F_total^A - F_total^B | / F_total^B ≤ 0.05`. If the pilot (§7) violates this, the per-task epoch budget is adjusted uniformly until it holds, then the protocol is frozen.

### 3.4 Hardware normalization

All gate runs executed on the same GPU model (specified by Jett before pilot launch; candidates: 1×A100 80GB or 1×H100 80GB). The specific GPU model, CUDA version, PyTorch version, and torch.compile status are recorded per run. A run on a different GPU is permitted for seed-parallelization efficiency **only if** the specific hardware is logged AND a single-seed sanity rerun on the primary hardware confirms accuracy matches within 0.5pp.

---

## 4. Seed Budget and Variance Control

### 4.1 Seed allocation (preregistered)

- **Number of seeds per arm per task-count configuration**: 10.
- **Seed values**: `S = {42, 43, 44, 45, 46, 47, 48, 49, 50, 51}`. Contiguous. Preregistered.
- **Pairing**: Arm A seed `s` and Arm B seed `s` share every stochastic choice they can share — class ordering, mini-batch ordering, LoRA init, classifier-head init, data augmentation RNG. This is a **paired-difference design**; statistical power increases substantially over an independent-samples design, at the cost of requiring the paired implementation discipline (§8.4).
- **Configurations**: three task counts × two arms × 10 seeds = **60 runs**. This is the full gate budget.

### 4.2 Configuration prioritization if compute limits bind

If total compute is insufficient for 60 runs, run in this order:

1. **20T×5C, both arms, 10 seeds each** (20 runs) — this is the gate-defining config per synthesis §0.
2. **5T×20C, both arms, 10 seeds each** (20 runs) — design-faithful config per synthesis §2.2.
3. **10T×10C, both arms, 10 seeds each** (20 runs) — community-legibility config.

Running (1) alone is sufficient to make the gate decision. (2) and (3) are for paper completeness and do not modify the gate.

### 4.3 Determinism protocol

Each seed `s` controls all of:

```python
random.seed(s)
numpy.random.seed(s)
torch.manual_seed(s)
torch.cuda.manual_seed_all(s)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # deterministic cuBLAS
torch.use_deterministic_algorithms(True, warn_only=True)
```

Any non-determinism source that `warn_only=True` surfaces is logged. If a non-deterministic op is load-bearing, Jett reports it to Breach before pilot launch; the protocol is amended to either eliminate it (preferred) or accept the added seed-variance in the power analysis (fallback).

### 4.4 Fisher-magnitude matching (KAY/O 18× concern) — v1.1 dual-criterion gate

Per `memory/MEMORY.md`: Fisher magnitude varies 18× across initializations. KAY/O R1 §2 and KAY/O red-team §1.3 argue the entire MoB-vs-FeCAM delta could be explained by within-arm across-seed Fisher variance even under paired design — the paired difference `Δ_s` becomes a random-slope-in-Fisher-magnitude estimator rather than a constant-in-expectation treatment effect, and the BCa CI on `{Δ_s}` becomes a CI conditional on whatever Fisher draws the sampled seeds happened to produce.

**v1.1 amendment (resolves KAY/O D2).** The v1.0 inclusion criterion checked only across-arm-within-seed Fisher agreement, which is near-trivially satisfied by construction (same init, same data order, same Fisher estimator) and does NOT test the 18× within-arm across-seed threat. v1.1 adds a second, binding inclusion criterion that directly bounds the R1-relevant quantity.

**Mitigation:**

1. Every run logs per-task, per-expert, per-step Fisher magnitude (L2 norm of the Fisher diagonal after clamping). Call this `||F_{i,t}(s)||` for expert i, task t, seed s.
2. Before running the primary statistical test, compute the per-seed average Fisher magnitude `F̄^A(s) = mean over (i, t)` (and symmetric `F̄^B(s)`).
3. **First inclusion criterion (cross-arm within-seed, v1.0 retained as implementation sanity check).** Because arms are paired on seed and share init/data ordering, `F̄^A(s) ≈ F̄^B(s)` must hold within 2× for every seed. A seed is included iff `max(F̄^A(s), F̄^B(s)) / min(F̄^A(s), F̄^B(s)) ≤ 2`. This check catches implementation drift (e.g., a divergent Fisher buffer allocation), not the R1 threat. If fewer than 8 of 10 seeds satisfy this, the implementation has silently diverged — investigate and rerun with Fisher-clamp recalibration per criterion (5) below.
4. **Second inclusion criterion (within-arm across-seed, v1.1 NEW — resolves D2).** Compute `CV(log F̄^A) = std_s(log F̄^A(s)) / |mean_s(log F̄^A(s))|` across the 10 full-run seeds within Arm A, and symmetrically `CV(log F̄^B)`. The pilot gate (§7.2) requires `CV(log F̄^A(s)) ≤ 0.5` across the 3 pilot seeds; the full-run post-hoc check requires the same bound across all included seeds. **Pre-registered branch if violated:**
   - **Branch (a), preferred.** Tighten the Fisher clamp (try values `{0.1, 0.3, 1.0, 3.0}` in that order per KAY/O red-team §5.5 amendment) and rerun the pilot. The first clamp value that produces `CV(log F̄^A) ≤ 0.5` is adopted; its value is logged as a §13 amendment. No other clamp values may be tried post-hoc.
   - **Branch (b), fallback.** If no clamp value in `{0.1, 0.3, 1.0, 3.0}` brings `CV(log F̄^A)` under 0.5, the protocol proceeds to the full run at the clamp value that minimized `CV(log F̄^A)`, BUT the gate memo and any downstream paper MUST carry a prominent limitation annotation: "The paired difference `Δ_s` reflects the within-arm Fisher-magnitude distribution of seeds {42..51} at clamp=X; generalization of the BCa bound to a fresh seed draw is weaker than the nominal CI suggests because Arm A's `β·c_forget` contribution is heteroscedastic across Fisher draws at this configuration."
   - **v1.1 pre-registers branch (a) as the default.** Branch (b) fires only if (a) exhausts the clamp ladder without bringing CV under 0.5. This decision tree is binding; Breach may not freelance a new clamp value or shift to a different estimator post-hoc.
5. **Clamp recalibration ladder (v1.1, resolves KAY/O red-team §5.5).** If fewer than 8 of 10 seeds satisfy criterion (3), OR if `CV(log F̄^A) > 0.5` at the pilot stage, rerun with clamp values `{0.1, 0.3, 1.0, 3.0}` in that exact order; the first value that produces both (3) pass and criterion (4) `CV ≤ 0.5` is adopted. The adopted value is logged as a §13 amendment. No other clamp values may be tried. Post-hoc values outside this ladder are a preregistration violation.
6. **Per-task Fisher drift report (v1.1, resolves KAY/O red-team §5.3).** Report `max_t ratio(F̄^A_t, F̄^B_t)` per seed (not just the (i,t)-mean). If any task's per-task ratio exceeds 3×, flag the seed for inspection. If ≥2 seeds flag, rerun them.
7. Across-seed variance diagnostic: report `σ_{log F̄}` per arm. Under protocol invariants this should be ≈identical between arms; large divergence is reported in the gate memo.

### 4.5 Why paired-on-seed matters (and why unpaired inflates variance)

Under `memory/MEMORY.md`'s documented 18× Fisher seed-variance, the MoB-vs-FeCAM unpaired two-sample test is heteroscedastic and low-power. With paired seeds, the Fisher-magnitude variability cancels to first order (both arms see the same Fisher), and the paired-difference variance is the variance of `β·c_forget`'s contribution, which is exactly what we want to test. This is why the paired design is mandatory, not optional.

### 4.6 FeCAM-binding implementation invariant (v1.2; resolves `docs/lit-review/05-fecam-code-comparison.md`)

This subsection is an **implementation invariant**, binding on both Jett (§8) and any downstream reviewer. Violation of any clause here means the run is **invalid** — even if all numerics look fine and all statistical gates pass. The gate decision rule (§1.2) cannot rescue a protocol violation here; such a run is not FeCAM by name and the headline claim "MoB vs FeCAM" loses its referent.

**Repo pins (immutable at freeze 2026-04-26; recorded in `frozen_config.yaml`):**

| Role | Repo | Commit SHA | Files (SHAs) |
|---|---|---|---|
| Harness / trainer / data pipeline | `sun-hailong/LAMDA-PILOT` | `7a6e904c5bc5cb7a4e1823b3434020be27469b63` (main, 2026-01-29) | `models/fecam.py` (blob `2a41d5d50841…`), `models/base.py` (blob `bd25ae352ed1…`), `exps/fecam.json` |
| Mahalanobis recipe source of truth (paper-canonical) | `dipamgoswami/FeCAM` | `e33f39d112ff2d2a2df2e68c490af579a50edd31` (main, 2024-11-12) | `models/base.py` (blob `358ed8a1214c…`) — functions `_tukeys_transform`, `shrink_cov`, `normalize_cov`, `_mahalanobis`; `exps/FeCAM_cifar100.json` |
| Paper (prose canonical source) | arXiv 2309.14062 v3 | N/A | §3, §7 (hyperparameters), Table 1 (per-class ablation), Table 2 (element ablation), eq. (7)/(8)/(9) |

**Why two repo pins (and not just LAMDA-PILOT).** Inline audit during v1.2 drafting discovered that LAMDA-PILOT's `models/fecam.py` at the pinned SHA is a simplified re-implementation that **deviates from the paper in one load-bearing way**: its `shrink_cov(cov, alpha)` applies single-parameter ridge `cov + 100·I`, not the paper's two-parameter additive `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)`. Per the gap analysis (`docs/lit-review/05-fecam-code-comparison.md` §2.2), this is a paper-Table-2 load-bearing element. LAMDA-PILOT does correctly omit Tukey for ViT features (matching paper §7), and correctly implements per-class Σ, correlation normalization (`torch.corrcoef`), and L2-normalized Mahalanobis — on those four elements it is equivalent to the paper. To bind against the paper-canonical recipe, v1.2 pins the **covariance-shrinkage code path specifically** against `dipamgoswami/FeCAM:models/base.py::shrink_cov` (paper-canonical two-parameter additive), while using LAMDA-PILOT for everything else (harness, dataloaders, backbone wiring, ViT-specific Tukey=off decision, config schema). This composite binding is the only way to bind against both "canonical FeCAM per paper" and "LAMDA-PILOT harness" simultaneously without a contradiction. The escalation is logged in §15.9.

**Implementation contract on Jett:**

1. **Byte-equivalent port.** Jett implements `mob/gate/fecam_core.py` (or equivalent path) that ports verbatim from `dipamgoswami/FeCAM:models/base.py` the functions `shrink_cov` (two-parameter additive, paper eq. 8), `normalize_cov`, and the L2-normalized `_mahalanobis` scoring function. The port may be line-for-line or refactored for vectorization, BUT must produce per-sample `d_M` values within **1e-5 absolute tolerance** of a reference run of the upstream function on a fixed synthetic tensor fixture (§8.4 test 17).
2. **Harness wiring.** Jett wires this `fecam_core` into the LAMDA-PILOT harness at the routing-layer auction point. The trainer, optimizer, scheduler, dataloader, data augmentation, and ViT-B/16 backbone loading follow LAMDA-PILOT's `models/fecam.py` path but MUST NOT call LAMDA-PILOT's `shrink_cov` (which is the non-canonical single-parameter version). A code-level lint check enforces this: `ripgrep` on the gate module must return zero hits for any call into LAMDA-PILOT's `Learner.shrink_cov` in the gate code path.
3. **No silent deviation.** Any deviation from the upstream ports (even ones that appear numerically equivalent) requires a §13 amendment with written justification and a comparative numerical experiment showing ≤1e-5 per-sample equivalence across ≥1000 random synthetic inputs. Jett is not authorized to "improve" the FeCAM recipe; optimizations are held until post-gate.
4. **Arm B's Mahalanobis core is identical to Arm A's.** Both arms call the same `fecam_core` module, with the same `γ₁, γ₂` values, same L2 normalization, same `Tukey=off` config. The only difference is that Arm B's bid-composition step hard-codes `β=γ=0` per §2.2.
5. **Failure mode declaration.** If any of (1)–(4) is violated — in particular if Arm B silently runs LAMDA-PILOT's `cov + 100·I` shrinkage, or if the tied-low-rank internal representation (v1.1 §2.1) produces numerical drift >1e-5 from the canonical path — the run is declared **invalid under §4.6** and the gate decision is blocked until a compliant re-run completes. This is independent of whether the statistical numbers "look right."

**Verification path.** §8.4 test 17 is the primary verifier for clause (1); §8.4 test 18 verifies clause (3) / §4.7 hyperparameters; §8.4 test 19 verifies clause (1) + clause (4) on the feature-preprocessing step (L2 normalization applied symmetrically; Tukey NOT applied given `config.tukey=false`). These three tests are CI-enforced on every commit touching the gate module, same policy as tests 1–16.

### 4.7 Shrinkage hyperparameter pinning (v1.2; `γ₁, γ₂` values)

The paper's §7 documents `γ₁ = γ₂ = 1` for **many-shot CIL on CIFAR-100 with ResNet-18** (eq. 8, Figure 9 heatmap). The paper does NOT publish a CIFAR-100 + ViT-B/16 configuration; for Split-ImageNet-R with ViT-B/16 the paper uses `γ₁ = γ₂ = 10`. The gap analysis (`docs/lit-review/05-fecam-code-comparison.md` §2.2) corroborates the ResNet-18 values.

**v1.2 pre-freeze decision:** adopt `γ₁ = γ₂ = 1` (paper CIFAR-100 MSCIL values) as the gate defaults. This is the tightest paper-published match to our benchmark (CIFAR-100). The ViT-B/16 substitution is an architectural swap that does not, per the paper's own ablation structure (Table 2, Figure 9), trigger a known γ shift — but is acknowledged as a minor out-of-paper extrapolation. This is a registered item in §15.10 (v1.2 escalations).

**Pre-registered override window (pilot-only, binding):** Jett runs a single backbone-only forward pass on the task-1 training set at seed 42, computes the empirical `V₁ = mean(diag(Σ))` and `V₂ = mean(off-diag(Σ))` from per-class sample covariances, and reports the values in the pilot report. If either `V₁` or `V₂` is ≥10× the ResNet-18 reference magnitude (rough reference from paper Figure 9 heatmap), Jett flags to Breach for a pre-freeze override to `γ₁ = γ₂ = 10` (the ViT-B/16 ImageNet-R value). The override is approved iff Breach signs off **before** the freeze date; post-freeze, the values are locked in `frozen_config.yaml` and cannot be changed without a §13 post-data amendment (which invalidates prior runs).

**Post-freeze immutability:** once `frozen_config.yaml` is signed, `γ₁` and `γ₂` are two of the fields covered by §8.4 test 18 (`test_fecam_hyperparameters_pinned`). Any drift from the frozen values blocks the pilot and full run.

---

## 5. Dataset Integrity Checks

From Cypher's CIFAR-100 audit (`docs/lit-review/04-cifar100-benchmark-audit.md`), the CIFAR-100 family has documented data-quality issues that inflate or deflate reported accuracy. This protocol addresses each.

### 5.1 ciFAIR retest

- Load CIFAR-100 training set as-is.
- Evaluate on BOTH `CIFAR-100-test` (standard, 10K images) AND `ciFAIR-100-test` (duplicate-free, Barz & Denzler 2020, cvjena.github.io/cifair).
- Report `A_T^{CIFAR-100}` and `A_T^{ciFAIR-100}` per seed per arm.
- **Gate decision uses `A_T^{CIFAR-100}`** (community-standard; comparable to L2P/DualPrompt/FeCAM leaderboards).
- `A_T^{ciFAIR-100}` is a secondary robustness metric. If the MoB-vs-FeCAM delta flips sign between the two test sets, the gate decision is escalated to Nosh before either outcome is accepted.

### 5.2 Noise robustness

- Do **not** use CIFAR-100N labels (instance-dependent human noise from Wei et al. 2021). This protocol tests clean-label continual learning; noise robustness is a separate question.
- Use CIFAR-100's original fine labels as released.

### 5.3 Upsampling

- 32×32 → 224×224 via **bicubic** interpolation (`torchvision.transforms.functional.resize(..., interpolation=InterpolationMode.BICUBIC)`), antialias=True.
- Record the interpolation function (including antialias flag) in run metadata.
- **Rationale:** bicubic is the ViT-B/16 fine-tuning convention in the LAMDA-PILOT ecosystem; bilinear (also common) causes small but non-zero accuracy shifts and would be a confound if the pilot reveals one arm is more sensitive to the interpolation method.

### 5.4 Class ordering

- Use LAMDA-PILOT's default random-permutation class ordering, seeded by the run seed.
- **Explicitly not used**: superclass-aware ordering, semantic-adversarial ordering. These are deferred (synthesis §6.2, open PI question; `project.md` §6). If at any point the gate outcome turns on class ordering, this is a separate confound and the protocol is amended to include superclass-aware runs.

### 5.5 Pretraining leakage acknowledgment

Per Cypher §6.2 and KAY/O §3.2, ViT-B/16 ImageNet-21k pretraining shares semantic concepts with CIFAR-100. This leakage is constant across arms (same backbone) and does not bias the Arm A vs Arm B comparison. It is acknowledged in the limitations section of any downstream paper but does not affect the gate.

---

## 6. Statistical Analysis Plan

### 6.1 Primary test — v1.1 symmetric one-sided BCa (resolves D1)

The gate decision is governed by §1.2 (BCa bounds `L_95`, `U_95`). The primary test **statistic** is the BCa bootstrap on the paired-difference vector `{Δ_s : s ∈ S}`; see §6.2 for the bootstrap specification.

**Secondary parametric tests (reported; not decision-driving):**

- **Paired one-sided t-test for superiority**, one-sided at α=0.05: test statistic `t_sup = (Δ̄ - Δ_practical) / (s_Δ / √n)` with df=n-1. Reject H_null: `E[Δ] = Δ_practical` in favor of H_alt: `E[Δ] > Δ_practical` iff `p_sup < 0.05`.
- **Paired one-sided t-test for inferiority**, one-sided at α=0.05: test statistic `t_inf = (Δ̄ + Δ_practical) / (s_Δ / √n)` with df=n-1. Reject H_null: `E[Δ] = -Δ_practical` in favor of H_alt: `E[Δ] < -Δ_practical` iff `p_inf < 0.05`.
- Both t-tests are reported symmetrically. The Gaussianity assumption is checked via Shapiro-Wilk at α=0.05; if rejected, the BCa bounds in §6.2 remain authoritative and the t-tests are annotated as having a failed normality check.
- **v1.1 NOTE:** the v1.0 rule "PASS requires t-test significant AND `Δ̄ ≥ Δ_practical`" is superseded by §1.2's BCa-bound rule. The t-test is retained only as a robustness check.

### 6.2 Confidence interval (primary for gate decision) — v1.1 symmetric bounds

**BCa bootstrap** (bias-corrected accelerated) on the paired-difference vector `{Δ_s : s ∈ S}`, 10,000 resamples, `scipy.stats.bootstrap(method='BCa')`. A single resample ensemble is generated; both the one-sided lower bound `L_95` and the one-sided upper bound `U_95` are computed from it (NOT from two independent resample draws).

- Report **`L_95`**: the 95% BCa one-sided lower confidence bound on `E[Δ]`.
- Report **`U_95`**: the 95% BCa one-sided upper confidence bound on `E[Δ]`.
- Report the 95% two-sided BCa CI for expositional completeness.
- **Gate decision (per §1.2):**
  - PASS iff `L_95 ≥ +Δ_practical`.
  - FAIL iff `U_95 ≤ −Δ_practical`.
  - TIE otherwise.
- Resample seed: fixed, logged in `frozen_config.yaml` (e.g., `bootstrap_seed=20260426`). Non-negotiable post-freeze.

BCa is chosen over percentile bootstrap because n=10 is small enough that the bias correction and acceleration are non-trivial (DiCiccio & Efron 1996). Resample count is 10,000 per statistical-bootstrap convention for ≤2 decimal places in the CI bounds.

### 6.3 Secondary tests — v1.1 symmetric

- **Wilcoxon signed-rank test** on paired differences (`scipy.stats.wilcoxon`) run symmetrically: one-sided at α=0.05 on the superiority tail AND one-sided at α=0.05 on the inferiority tail. Reported as non-parametric robustness check. Not used for the gate.
- **Per-task-count subgroup**: paired BCa bootstrap (§6.2) on each task count separately (5T, 10T, 20T), reporting `L_95` and `U_95` for each. Reported but not used for the gate (the gate is defined on 20T per synthesis §0).
- **ciFAIR cross-check**: same BCa bounds on `A_T^{ciFAIR-100}` paired differences. Reported. Gate decision uses CIFAR-100 only.

### 6.4 Multiple comparison policy

- **Phase-1 gate is a single a-priori comparison** (MoB-Full vs FeCAM-Router at 20T). No multiple-comparison correction is applied to the primary test.
- The secondary subgroup tests (5T, 10T) are exploratory and reported as such.
- When Phase-3 legibility baselines are added (L2P, DualPrompt, CODA-Prompt, SLCA, HiDe-Prompt, RanPAC, EASE, LoRA-MoE-CL — 8 additional methods), **Holm-Bonferroni correction with α=0.05** is applied across those 8 tests. The Phase-1 gate result is not re-tested under correction; it stands on its pre-registered single-comparison basis.
- `F_T`, `BWT`, routing entropy, Gini: reported without inferential testing; descriptive only.

### 6.5 Power analysis (to be performed after pilot; §7) — v1.1 tightened bands (resolves D3)

**v1.1 amendment rationale (resolves KAY/O D3).** v1.0 labeled σ_Δ ∈ (1.16, 1.75] as "marginally powered; proceed with annotation." KAY/O's actual power calculation at σ_Δ=1.75pp, n=10, Δ=1.0pp, α=0.05 one-sided: non-centrality δ = 1.0/(1.75/√10) ≈ 1.807; critical t at df=9 = 1.833; power ≈ **46%**. That is not marginally powered; that is a coin flip. v1.1 removes the 46%-power band from the "proceed" region and pre-registers n=20 as a deterministic backup rather than a Nosh-approved case-by-case escalation.

Given `Δ_practical = 1.0pp` and `n=10`, the minimum detectable effect at 80% power under paired-t with `σ_Δ` (paired-difference std) is:

```
MDE_80 = (t_{0.95, n-1} + t_{0.80, n-1}) · σ_Δ / √n
       ≈ (1.833 + 0.883) · σ_Δ / √10
       ≈ 0.859 · σ_Δ
```

so `σ_Δ ≤ 1.16pp` is required for 80% power at n=10 detecting Δ=1.0pp.

**v1.1 Decision rule after pilot (binding, replaces v1.0 bands):**

Let `σ_upper` denote the 80%-confidence chi-squared upper bound on `σ_Δ` computed from the pilot paired-difference sample.

- **`σ_upper ≤ 1.30pp`** → **GO at n=10.** At σ_Δ=1.30pp, power ≈ 80% (δ = 1.0/(1.30/√10) = 2.434; critical t df=9 = 1.833; P(T_9(2.434) > 1.833) ≈ 0.80). Proceed with the preregistered 10-seed full run on the remaining 7 seeds {45..51}.
- **`1.30 < σ_upper ≤ 1.75pp`** → **GO at n=20 (pre-registered backup, v1.1 NEW).** Extend the seed domain to `S_extended = {42..61}` (doubling the budget; 20 paired seeds). At σ_Δ=1.75pp, n=20 gives δ = 1.0/(1.75/√20) ≈ 2.556, and power ≈ 0.78; σ_Δ=1.30pp, n=20 gives power ≈ 0.96. The pilot seeds {42,43,44} count toward the 20; the full run then executes seeds {45..61}, 17 additional seeds per arm at 20T. This backup is **pre-committed** and does NOT require Nosh escalation. Budget increase: roughly +7 GPU-days at 20T (per §12). Budget approval from Nosh is required only at the point of dispatch (to confirm resource availability), not at the decision-rule level.
- **`σ_upper > 1.75pp`** → **ESCALATE to Nosh** per §13 (this is the v1.0 outer-guard, retained). Do not silently increase n beyond 20 or lower Δ_practical. Options then available to Nosh:
  - (a) approve a further n-increase (cost/benefit analysis);
  - (b) accept a larger Δ_practical with documented amendment and external reader sign-off;
  - (c) program escalates to a pivot decision per synthesis §5.

**Why n=20 backup rather than a tighter σ_upper cap at 1.16:** KAY/O offered both as valid D3 fixes. Breach v1.1 selects the n=20 backup because (i) σ_upper estimates from a pilot of n=3 or n=5 are themselves noisy (KAY/O §6.6), so a cap-based rule makes the whole program sensitive to a small stochastic fluctuation in σ_upper; (ii) the paper is stronger with n=20 if σ_Δ lands in that band than with a terminated-gate escalation; (iii) the additional compute cost (+7 GPU-days) is affordable inside the Phase-1 timeline per §12 and does not compress §5 Phase-2 or §5 Phase-3 on the critical path. The outer-guard at 1.75pp is preserved because beyond that band, even n=20 is underpowered and the pilot is telling us something real about the noise floor.

**Pilot `σ_Δ` estimator choice.** The pilot's point estimate at n=3 is extremely noisy (KAY/O §6.6: tiny movements in `σ_pilot` push the decision across bands). v1.1 retains n=3 as the pilot size (consistent with §7 wall-clock budget) but the decision is based on the **80% chi-squared upper bound `σ_upper`**, not on the point estimate. The multiplier at df=2 (n=3) is sqrt(2/χ²_{0.20,2}) ≈ 2.12 — so `σ_upper ≈ 2.12 · σ_pilot`. Breach explicitly accepts this conservatism: a borderline pilot that produces `σ_pilot=0.6pp` yields `σ_upper ≈ 1.27pp` (GO at n=10 under v1.1), and a pilot with `σ_pilot=0.8pp` yields `σ_upper ≈ 1.70pp` (GO at n=20 under v1.1 backup). Both paths are deterministic and pre-registered; no post-hoc analyst choice remains.

---

## 7. Pilot Protocol (pre-full-run gate)

Before committing to the 10-seed full run, execute a **3-seed pilot** on both arms on the 20T configuration only. Budget: 2–3 days on a single A100/H100.

### 7.1 Pilot seeds

`S_pilot = {42, 43, 44}` (first 3 seeds of `S`).

### 7.2 Pilot success criteria (all must hold before full run launches) — v1.1

1. **Compute match**: `|F_total^A - F_total^B| / F_total^B ≤ 0.05` on each of 3 paired runs.
2. **Fisher magnitude match (cross-arm within-seed)**: `max(F̄^A(s), F̄^B(s)) / min(F̄^A(s), F̄^B(s)) ≤ 2` for every `s ∈ S_pilot`.
3. **Fisher CV inclusion (within-arm across-seed; v1.1 NEW, resolves D2)**: `CV(log F̄^A(s)) = std_s(log F̄^A(s)) / |mean_s(log F̄^A(s))| ≤ 0.5` across `s ∈ S_pilot`, and symmetrically for Arm B. If violated at the pilot, execute the §4.4 branch-(a) clamp-ladder rerun BEFORE launching the full run. Branch (b) is only reachable if the full clamp ladder `{0.1, 0.3, 1.0, 3.0}` exhausts without satisfying the bound; in that case the pilot emits status `"CV_LIMITATION"` and the full run proceeds at the best clamp with the §4.4 limitation annotation inherited into the gate memo.
4. **Implementation integrity** (§8.4 acceptance tests all pass): **all 20 tests in §8.4 are green** (v1.2.1 count; §8.4 header is the single source of truth). This includes v1.1 tests on RNG-state equality at first bid, DataLoader worker determinism, bitwise-loss equality at t=0, α shared-pre-pass, Fisher CV inclusion, FLOP accounting unit test, arms-disagree-on-pilot-inputs, gradient-graph-excludes-cforget-and-conscience; v1.2 tests on FeCAM-port fidelity, hyperparameter pinning, and L2/Tukey preprocessing; **and the v1.2.1 training-spec-pinning test (test 20)**. See §8.4 for the full enumerated list.
5. **Power adequacy (v1.1 refined, resolves D3)**: pilot `σ_Δ` 80%-confidence chi-squared upper bound `σ_upper` falls in one of §6.5's three preregistered bands. `σ_upper ≤ 1.30pp` → GO at n=10; `1.30 < σ_upper ≤ 1.75pp` → GO at n=20 (pre-committed backup, not Nosh-approved case-by-case); `σ_upper > 1.75pp` → ESCALATE to Nosh per §13.
6. **No silent divergence**: the per-step bid diagnostics from Arm A show non-zero β·c_forget contribution (proving the forget term is load-bearing in Arm A); Arm B shows zero contribution.
7. **Arms disagree on at least one pilot input (v1.1 NEW, resolves KAY/O D8)**: over the pilot runs, Arm A and Arm B must produce different winning experts on ≥1% of bid calls. If the arms never disagree, β and γ are not mechanistically active, and the gate is uninformative regardless of outcome — escalate to Nosh per §13 before full-run launch.

### 7.3 Pilot output artifact

`results/gate/pilot/pilot-report.json` containing (v1.1 schema):
```json
{
  "protocol_version": "1.2.1",
  "n_seeds": 3,
  "compute_match_ratio_per_seed": [...],
  "fisher_match_ratio_per_seed_per_task": [...],
  "fisher_cv_log_arm_a": <float>,
  "fisher_cv_log_arm_b": <float>,
  "fisher_cv_inclusion_passed": true,
  "fisher_clamp_adopted": <float>,
  "fisher_clamp_ladder_history": [<float>, ...],
  "arms_disagree_fraction": <float>,
  "arms_disagree_check_passed": true,
  "acceptance_tests_passed": true,
  "acceptance_tests_enumerated": [{"name": "...", "passed": true}, ...],
  "pilot_sigma_delta": <float>,
  "pilot_sigma_delta_80pct_upper": <float>,
  "power_band": "GO_N10" | "GO_N20_BACKUP" | "ESCALATE",
  "recommended_n": 10 | 20,
  "go_nogo": "GO_N10" | "GO_N20" | "NOGO" | "ESCALATE" | "CV_LIMITATION"
}
```

Pilot status:
- `"GO_N10"` → launch full 10-seed run with the remaining 7 seeds {45..51}.
- `"GO_N20"` → launch full 20-seed run with 17 additional seeds {45..61} per arm (v1.1 D3 backup).
- `"NOGO"` → investigate, fix, rerun pilot (e.g., acceptance tests failed, implementation bug).
- `"ESCALATE"` → bring to Nosh (σ_upper > 1.75pp, or arms-disagree fraction < 1%).
- `"CV_LIMITATION"` → §4.4 branch (b) fired; proceed with the §4.4 limitation annotation inherited into the gate memo.

---

## 8. Implementation Interface (Jett-facing Contract)

This section specifies the minimum Jett must implement and the acceptance tests that must pass before any pilot seed launches.

### 8.1 CLI

```
python -m mob.gate.run \
  --arm {mob_full, fecam_router} \
  --n_tasks {5, 10, 20} \
  --seed <int> \
  --config <path_to_frozen_config.yaml> \
  --output_dir results/gate/<arm>/<n_tasks>T/<seed>/
```

The `<frozen_config.yaml>` file is version-controlled and stores every non-seed hyperparameter referenced in §2.1. Jett freezes it once before the pilot. Changes require amendment per §13.

### 8.2 Required output per run

Under `results/gate/<arm>/<n_tasks>T/<seed>/`:

- `run.json`: top-level summary (see §8.3 schema).
- `accuracy_matrix.npy`: `a_{i,j}` for all `i ≤ j ≤ T` (the full CIL matrix).
- `accuracy_matrix_cifair.npy`: same, evaluated on ciFAIR-100.
- `routing_log.parquet`: per-step routing decisions (winning expert, bid components α·d_M, β·c_forget, γ·conscience, pseudo-label used).
- `fisher_log.parquet`: per-task per-expert per-step Fisher L2 magnitude.
- `flop_log.json`: FLOP accounting per §3.3.
- `wallclock.json`: per-task wall-clock, total wall-clock, per-step latency percentiles.
- `gpu_memory.json`: peak VRAM per task.
- `config_snapshot.yaml`: the frozen config used for this run.
- `backbone_hash.txt`: SHA-256 of frozen backbone state_dict (§8.4 invariant).
- `env.json`: PyTorch version, CUDA version, GPU model, OS, commit SHA of the MoB repo.

### 8.3 `run.json` schema (top-level summary)

```json
{
  "protocol_id": "MOB-GATE-001",
  "protocol_version": "1.2.1",
  "arm": "mob_full" | "fecam_router",
  "n_tasks": 5 | 10 | 20,
  "seed": 42..51,
  "final_avg_acc_cifar100": <float, 0-100>,
  "final_avg_acc_cifair100": <float, 0-100>,
  "forgetting_F_T": <float>,
  "BWT": <float>,
  "routing_entropy_per_task": [<float>, ...],
  "utilization_gini_per_task": [<float>, ...],
  "fisher_L2_mean_per_task": [<float>, ...],
  "bid_component_means_per_task": {
    "alpha_dM": [<float>, ...],
    "beta_cforget": [<float>, ...],
    "gamma_conscience": [<float>, ...]
  },
  "flops_total": <float>,
  "flops_decomposition": {
    "forward_training": <float>,
    "backward_training": <float>,
    "eval": <float>,
    "bid_mechanism": <float>,
    "fisher_projection": <float>
  },
  "wallclock_seconds": <float>,
  "gpu_memory_peak_gb": <float>,
  "acceptance_tests_passed": true,
  "env": {...},
  "backbone_hash": "<sha256>",
  "git_commit": "<sha>"
}
```

### 8.4 Acceptance tests (must pass before any pilot seed runs) — v1.2.1 expanded to 20 tests

All 20 tests live in `tests/gate/test_arms.py`, `tests/gate/test_fecam_binding.py`, and (v1.2.1 NEW) `tests/gate/test_training_spec.py` and are CI-enforced on every commit touching `mob/gate/`, `mob/bidding.py`, `contibualmob/bidding.py`, `contibualmob/pool.py`, or `mob/gate/fecam_core.py`. Tests 1–8 are v1.0; tests 9–16 are v1.1 NEW; tests 17–19 are v1.2 NEW; test 20 is v1.2.1 NEW (training-spec-pinning structural check).

1. **`test_arm_b_bid_has_no_beta_or_gamma`**: instantiate Arm B with a synthetic feature vector; verify that the returned bid equals `α·d_M` exactly. Assertion: `bid == α * d_M` (bitwise on deterministic CPU).
2. **`test_arm_b_still_computes_cforget_and_conscience`**: instantiate Arm B; run one step; verify `c_forget` and `conscience` logs are populated with non-trivial (non-zero, varying) values.
3. **`test_arm_a_and_b_share_identical_backbone_at_init`**: instantiate Arm A and Arm B with the same seed; `hash(model_A.state_dict())` at step 0 must equal `hash(model_B.state_dict())` at step 0, for backbone + LoRA + classifier head.
4. **`test_arm_a_and_b_share_identical_batch_order`**: instantiate a paired dataloader run under both arms with the same seed; first 20 batches must be bitwise identical across arms.
5. **`test_determinism_single_seed`**: run Arm A on seed 42 twice; outputs (final accuracy, per-task accuracy, Fisher magnitudes) must be bitwise identical.
6. **`test_flop_accounting_covers_bid_and_fisher`**: FLOP counts from `flop_log.json` are strictly positive for `bid_mechanism` and `fisher_projection` in both arms (not just Arm A).
7. **`test_cifair_eval_runs`**: the ciFAIR test set loader loads successfully and produces an accuracy number in [0, 100].
8. **`test_protocol_invariants_frozen`**: the config file SHA-256 matches the preregistered hash; if not, test fails with a message pointing to §13.

**v1.1 NEW tests (9–16):**

9. **`test_arm_b_code_path_never_invokes_ewc_for_forget_cost`** (v1.1, KAY/O new test a): structural check that Arm B's control-flow graph for the bid never invokes the EWC forward-backward path to contribute to the bid. `c_forget` may still be *computed* (required by test 2 and §2.2's log-path parity) but must NOT influence the returned `bid` tensor. Implemented as an assertion that Arm B's `bid.grad_fn` does not trace back to any Fisher-projection op. (See also test 12 for the closely related gradient-graph check.)
10. **`test_arms_disagree_on_at_least_one_percent_of_pilot_inputs`** (v1.1, KAY/O new test b / D8): run both arms on the pilot seed 42 for the first 1000 bid calls. Assert that the winning expert differs between Arm A and Arm B on at least 1% of those calls (≥10 disagreements). If this test fails, β and γ are mechanistically inactive and the gate is uninformative regardless of outcome — blocks pilot launch.
11. **`test_losses_bitwise_identical_at_t0`** (v1.1, KAY/O new test c / minor #7): at step 0 (before any gradient step in either arm, before any bid is submitted), compute the per-sample cross-entropy loss on a fixed batch of 32 CIFAR-100 training images at seed=42 under both arms. Assertion: `torch.equal(loss_A, loss_B)` (bitwise). Stronger than test 3 (weights) because it also checks all intermediate buffer states and RNG-consumption-order-dependent ops.
12. **`test_flop_accounting_unit_test`** (v1.1, KAY/O new test d / minor #15): construct a toy workload with known analytical FLOP count (e.g., a 2-layer MLP with fixed dimensions on a 4-sample batch). Run the FLOP accounting pipeline (fvcore + §3.3 analytical overhead formula) and assert the returned total matches the analytical answer within 1%. Protects against fvcore counting bugs and against drift in the §3.3 formula when `{r, d, E, r_f, B}` are modified.
13. **`test_rng_state_equality_at_first_bid`** (v1.1, KAY/O new test e / minor #5): after instantiating Arm A and Arm B at the same seed and running the data pipeline up to the first bid call, assert that `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `numpy.random.get_state()`, and Python's `random.getstate()` are byte-identical across arms. (cuBLAS state is enforced via `CUBLAS_WORKSPACE_CONFIG=":4096:8"` at process start; if any per-process cuBLAS state handle is exposed by the running PyTorch version, its serialization is also included.) This is stronger than test 3's weight-hash check, which only covers step-0 state_dict values.
14. **`test_dataloader_worker_rng_pinned`** (v1.1, KAY/O new test f / minor #6): construct the gate's `DataLoader` with the v1.1 §2.1 `worker_init_fn=seed_worker` and a seeded generator, request `num_workers > 0`, and assert that the first 100 augmented-batch outputs are bitwise identical across a second instantiation at the same seed. If `num_workers=0` is the configured default, this test instead asserts that no `worker_init_fn` path is exercised and that `num_workers == 0` in the loaded config.
15. **`test_alpha_calibration_shared_prepass`** (v1.1, KAY/O new test g / D4): assert that `frozen_config.yaml` contains a single scalar `alpha` value and a field `alpha_source == "shared_prepass"`, that both arms load this identical α at launch (`config_A.alpha == config_B.alpha`), and that no per-arm α computation is reachable in the arm-launch code path. Sanity-check that the logged pre-pass produced the expected median-`d_M` value on a deterministic backbone-only replay of task-1 at seed 42.
16. **`test_fisher_cv_inclusion`** (v1.1, KAY/O new test h / D2): on a synthetic 3-seed micro-run (toy task, ~60s wall-clock), verify that the §4.4 per-arm `CV(log F̄)` statistic is correctly computed and that the clamp-ladder branch logic (branch (a) preferred, branch (b) fallback) is exercised end-to-end. This is a *logic test on the gate decision mechanism*, not a data test; it verifies that if `CV > 0.5` is observed, the §4.4 ladder executes deterministically and the pilot-report JSON is populated correctly. Passes deterministically.

**v1.2 NEW tests (17–19) — FeCAM-canonical binding (see §4.6):**

17. **`test_fecam_port_fidelity_against_upstream`** (v1.2): load a fixed synthetic tensor fixture under `tests/gate/fixtures/fecam_reference/` consisting of (a) a `d=768` feature matrix `X ∈ ℝ^{128×768}` drawn from `torch.Generator().manual_seed(20260426)` standard-normal, (b) a set of per-class prototypes `μ ∈ ℝ^{10×768}` from the same generator, (c) per-class sample covariances `Σ_c ∈ ℝ^{10×768×768}` built from held-out normal samples. Run the upstream paper-canonical pipeline (`dipamgoswami/FeCAM:models/base.py` functions — `shrink_cov` with `γ₁=γ₂=1`, `normalize_cov`, `_mahalanobis` with L2-normalized inputs, Tukey OFF) and compute the reference `d_M^{upstream}` per sample per class. Run the gate's `mob/gate/fecam_core.py` (or Jett's chosen path) on the same fixture and compute `d_M^{ours}`. **Assertion:** `max(|d_M^{upstream} − d_M^{ours}|) ≤ 1e-5` per sample. Stronger than a spot-check: tests all four covariance elements (shrinkage, correlation-norm, L2-norm, per-class) and their composition order in one go. Tukey is tested separately by test 19. Fixture and reference values are deterministic and version-controlled at the frozen config SHA; any change to the fixture file is a §13 amendment.
18. **`test_fecam_hyperparameters_pinned`** (v1.2): assert that `frozen_config.yaml` contains explicit fields `fecam.gamma1`, `fecam.gamma2`, `fecam.tukey`, `fecam.tukey_beta`, `fecam.per_class_cov`, `fecam.l2_normalize`, `fecam.inverse_method` with exactly the values `{gamma1: 1.0, gamma2: 1.0, tukey: false, tukey_beta: 0.5, per_class_cov: true, l2_normalize: true, inverse_method: "pinv"}` (or the §4.7 override values if a pre-freeze override was signed). Assert that both arms load the same values (`config_A.fecam == config_B.fecam` deep-equal). Assert that the `gamma1`/`gamma2` defaults match the FeCAM paper §7 CIFAR-100 MSCIL values (`γ₁=γ₂=1` per arXiv 2309.14062 §7, corroborated by `dipamgoswami/FeCAM:exps/FeCAM_cifar100.json` at the pinned SHA: `alpha1=1, alpha2=1, beta=0.5, tukey=true, per_class=true, shrink=true, norm_cov=true, full_cov=true`). **Tukey discrepancy note:** the pinned upstream config has `tukey=true` for ResNet-18; the gate uses `tukey=false` per paper §7 ViT-B/16 guidance. This is recorded as an intentional, paper-traceable deviation from the ResNet-18 config; the test asserts this specific (tukey=false) value matches the protocol-pinned value and emits an informational log line documenting the paper-§7 justification.
19. **`test_fecam_feature_preprocessing_l2_and_tukey`** (v1.2): on a fixed synthetic feature batch `X ∈ ℝ^{32×768}` with mixed-sign entries (simulating ViT features), verify that (a) the L2 normalization step produces unit vectors — `torch.allclose(X̃.norm(dim=-1), torch.ones(32), atol=1e-6)` after `fecam_core.preprocess(X)` — for both features and class means on the same call; (b) the Tukey transform is NOT applied when `config.tukey=false` — assert `X̃` has the same sign pattern as `F.normalize(X, p=2, dim=-1)` (Tukey β=0.5 on negative-valued features would raise or produce complex values; absence of such is the test); (c) if a future protocol amendment sets `config.tukey=true` AND features are non-negative (e.g., for a ResNet-18 variant), the Tukey transform IS applied in the correct order (Tukey → L2 normalization → subtract, per paper §7 / `dipamgoswami/FeCAM:models/base.py::_maha_dist` control flow at line 120 of the pinned blob). Passes deterministically.

**v1.2.1 NEW test (20) — training-spec pinning:**

20. **`test_training_spec_pinned_v1_2_1`** (v1.2.1, §2.3 T1–T7 + §2.1 backbone-freeze row): a single structural test that reads `frozen_config.yaml` and asserts each of the following: (a) `training.epochs_per_task == 20` (T1; LAMDA-PILOT `fecam.json: tuned_epoch=20` at `7a6e904c`); (b) `training.optimizer.name == "AdamW"` with `β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=1e-4` (T2, T5); (c) `training.schedule.type == "cosine"` with `min_lr=1e-5`, peak LR field present as a mapping `{adapter: 5e-4, lora_qkv: 1e-3}` (T3); (d) `training.batch_size == 128` (T4); (e) `training.warmup.type == "linear"`, `training.warmup.steps == 100`, `training.warmup.applied_to_tasks == [0]` (T6, task-1-only); (f) `training.optimizer_reset.cadence == "task_end_after_fisher"` and `training.optimizer_reset.scope == "every_expert"` (T7); (g) `model.backbone.frozen_param_groups` is a superset of `["blocks", "patch_embed", "cls_token", "norm", "pos_embed"]` and `model.backbone.ln_unfreeze_exception` is absent or explicitly `false` (§2.1 Backbone freeze scope row); (h) both arms load an identical training block (`config_A.training == config_B.training` deep-equal). Also performs a runtime structural check: instantiate Arm A on a toy 2-task schedule, assert `model.backbone.norm.weight.requires_grad is False` (the LN-scale parameter is frozen), assert that after task 0 ends the optimizer state is empty for all 4 experts (not just the winner), and assert that task 1's first optimizer step uses the peak LR for its parameter group without a warmup ramp. Passes deterministically on a 60-second toy run.

**All 20 tests must pass before pilot seed 42 launches. Any regression in any test blocks the full run until resolved. Tests 17–19 are CI-enforced on every commit touching `mob/gate/fecam_core.py`, `mob/gate/run.py`, or any file that provides the Mahalanobis core to the gate auction. Test 20 is CI-enforced on every commit touching `frozen_config.yaml`, the training loop, the scheduler construction, the optimizer-reset helper, or the backbone-freeze initializer.**

### 8.5 Items Jett may tune (implementation details, not invariants)

- Gradient accumulation factor (as long as effective batch size matches)
- DataLoader `num_workers`, `prefetch_factor`, `pin_memory`
- Mixed-precision autocast (fp16/bf16 — but chosen once, held across arms)
- `torch.compile` mode (default/reduce-overhead/max-autotune — chosen once, held across arms)
- Logging cadence within a task (as long as the 1.2.2 invariants are logged)
- CPU-side dataset caching

### 8.6 Items Jett must flag to Breach rather than tune

- Any per-task epoch choice different from the LAMDA-PILOT ViT-CIL default.
- Any deviation in α calibration procedure from "match empirical median of `d_M` on task-1 features" (Astra posted-price convention).
- Any deviation in β or γ calibration procedure from the defaults in `project.md` §2.2.
- Any change that causes the Arm-A/Arm-B acceptance tests (§8.4) to fail.

---

## 9. Outcome Branches (mapped to synthesis §5 Phase 1) — v1.1 aligned with §1.2 symmetric BCa framework

### 9.1 PASS (`L_95 ≥ +Δ_practical = +1.0pp`, per §1.2)

- Protocol deliverable: `results/gate/summary.json` with `gate_status: "PASS"`, headline fields `L_95` and `U_95` both reported, wall-clock ratio `wallclock^A / wallclock^B` reported per §1.8.
- **Scope of the PASS claim (v1.1 explicit, per KAY/O red-team §11.3):** a PASS at these 10 (or 20) seeds at 20T CIFAR-100 with ViT-B/16 resolves the R1 threat ONLY at this configuration and ONLY under this sample of the Fisher-magnitude distribution. Stronger R1 refutation (β-only ablation, task-count-scaling trend, Fisher-stratified conditional Δ̄) is deferred to Phase 2/3 and is NOT preregistered as a Phase-1 gate condition. The gate memo and any downstream paper MUST state this scope explicitly.
- Trigger **Phase 2** per synthesis §5:
  - Coordinate with Sage — the Pólya-urn convergence theorem writeup is in-flight and should cite this gate's β-contribution measurement as the empirical grounding.
  - Jett implements the conscience term (if not already active in Arm A) at full scale on MNIST for stability validation.
- Trigger **Phase 3** per synthesis §5:
  - Full legibility-baseline suite (L2P, DualPrompt, CODA-Prompt, SLCA, HiDe-Prompt, RanPAC, EASE, plus one LoRA-MoE-CL) runs under the same harness on the same task splits with Holm-Bonferroni correction (per §6.4). This is a separate Phase-3 protocol.

### 9.2 TIE (neither `L_95 ≥ +1.0pp` nor `U_95 ≤ −1.0pp`, per §1.2)

- Protocol deliverable: `results/gate/summary.json` with `gate_status: "TIE"`, headline fields `L_95` and `U_95` reported.
- **Scope of the TIE outcome (v1.1 explicit):** a TIE is *consistent* with KAY/O's R1 null but does not *confirm* it. The thesis "auction is the load-bearing contribution" is not supported by these data.
- Synthesis §5 Phase 1 pivot branch: reposition program to continual-fine-tuning-only framing. Rerun the gate on a **sequential-domain benchmark**, not CIL. Scope the alternative benchmark with Fade's recommendation (synthesis §6.3 — continual instruction tuning, sequential domain adaptation on math/code/science/biomedical domains, etc.) before starting.
- Do not collect more CIFAR-100 seeds to "push" the CI. Moving the bar post-hoc is a §1.2 violation.
- **Config-cross-check rule (v1.1 NEW, resolves KAY/O red-team §7.1):** if the 20T gate returns TIE or FAIL but the 5T or 10T configurations (secondary) return a PASS-equivalent result, this is reported as a scale-dependent effect in the gate memo but does NOT override the 20T gate. The 20T result is binding for the synthesis §0 Phase-1 decision. The 5T/10T results may motivate a follow-up study at that task count but do not modify the Phase-1 branch.

### 9.3 FAIL (`U_95 ≤ −Δ_practical = −1.0pp`, per §1.2)

- Protocol deliverable: `results/gate/summary.json` with `gate_status: "FAIL"`, headline fields `L_95` and `U_95` reported.
- **Scope of the FAIL outcome (v1.1 explicit):** a FAIL under the v1.1 symmetric framework means FeCAM-Router materially beats MoB-Full at this configuration at these seeds. This is stronger than KAY/O's R1 prediction (R1 predicted tie-or-narrow-loss); a FAIL confirms the auction is net-negative at this config, which exceeds R1's null.
- Synthesis §5 Phase 1 terminate branch: the routing-mechanism thesis does not survive. Extract the prototype-store engineering (tied low-rank Σ via Woodbury, online Mahalanobis updates, EWC+prototype integration at the CLS layer) as a standalone contribution.
- The auction-mechanism thesis is not rescuable via more CIFAR seeds; the program does not retry on CIFAR.

---

## 10. Threats to Validity and Mitigations

| # | Threat | Description | Mitigation |
|---|---|---|---|
| T1 | Selection bias in class ordering | Favorable random class orderings could make MoB look better by coincidence. | Paired design: both arms see identical class orderings per seed. Variance cancels. |
| T2 | Compute-matching bias | Arm A could benefit from extra compute that isn't counted. | §3.3 explicit FLOP decomposition + ±5% matching + wall-clock sanity check. |
| T3 | Implementation asymmetry | Arm B could silently exercise the β/γ path via a bug. | §8.4 acceptance tests 1, 2, 6 enforce β=0/γ=0 and log-path parity. |
| T4 | Fisher-magnitude seed dominance | Per `memory/MEMORY.md`, Fisher varies 18× across seeds; could dominate the `β·c_forget` signal. | §4.4 per-seed Fisher-match criterion + Fisher clamp min=0.1 + paired design cancels Fisher variance across arms. |
| T5 | ciFAIR duplicate inflation | 10% of CIFAR-100 test set is near-duplicate of training; inflates both arms. | §5.1 ciFAIR cross-check. Delta should survive both test sets. |
| T6 | Pretraining leakage | ViT-B/16 ImageNet-21k overlaps CIFAR-100 semantically. | Constant across arms; does not bias A vs B. Acknowledged in paper limitations. |
| T7 | Underpowered n=10 | If `σ_Δ` is large, the one-sided lower bound may be too wide to exclude zero even when MoB has a real ~1pp advantage. | §7.2 pilot power analysis. Escalate to Nosh per §13 rather than silently modify protocol. |
| T8 | α calibration asymmetry | If α is calibrated on task-1 features in Arm A but separately in Arm B, the scales diverge. | α calibration done once on the shared backbone before either arm's bid mechanism activates. Identical α value used in both arms. |
| T9 | Early-stopping / best-epoch cheese | Reporting best-epoch accuracy rather than final-task accuracy biases toward whichever arm has less stable training. | Primary metric is `a_{i,T}` at task T=20 exactly (no best-over-trajectory). |
| T10 | Interpolation sensitivity | Bicubic vs bilinear 32→224 upsampling changes accuracy by 0.5–1pp. | §5.3 fixed to bicubic + antialias=True, logged per run. |
| T11 | Optimizer-reset timing | Timing of optimizer reset at task boundary is load-bearing per `memory/MEMORY.md`. | Applied identically in both arms (§2.1). |
| T12 | Fisher-clamp calibration drift | Clamp at min=0.1 was tuned on MNIST; CIFAR-100 LoRA adapters have different Fisher scales. | Pilot (§7) verifies Fisher-magnitude distributions; if clamp needs recalibration, do it once and apply to both arms. |
| T13 | Task-boundary signal asymmetry | If Arm A uses task-boundary information more aggressively than Arm B, an apparent advantage is task-ID leakage. | Both arms use identical task-boundary handling (§2.1 "Optimizer reset"); neither gets ground-truth task IDs at eval. |
| T14 | Publication-bias-style early termination | Stopping after pilot because results "look good" and reporting pilot as final. | Pilot is for power/compute/implementation verification only. Full-run N=10 is committed to regardless of pilot pointing. Pilot seeds {42,43,44} are reused in the full run. |
| T15 | Preregistration drift | Amendments made mid-data-collection that conveniently favor one arm. | §13 amendment log + freeze date 2026-04-26 + any amendment post-freeze invalidates prior runs affected by the amendment. |

---

## 11. Checklists

### 11.1 Jett pre-launch checklist (before first pilot seed)

- [ ] Frozen config YAML committed to repo and its SHA-256 recorded in `MOB-GATE-001.lock`.
- [ ] `tests/gate/test_arms.py` implements all 8 acceptance tests in §8.4.
- [ ] CI runs `test_arms.py` on every commit touching `mob/gate/` or `contibualmob/bidding.py`.
- [ ] Output directory skeleton `results/gate/{mob_full,fecam_router}/{5,10,20}T/<seed>/` exists and is writable.
- [ ] LAMDA-PILOT harness is checked out at a specific commit; commit SHA recorded in `env.json`.
- [ ] Frozen ViT-B/16 weights downloaded and SHA-256 recorded in `backbone_hash.txt`.
- [ ] ciFAIR-100 test set downloaded and verified (SHA-256 match against cvjena.github.io/cifair published hash).
- [ ] `fvcore` installed and FLOP-counting on a toy forward pass matches a back-of-envelope calculation within 5%.
- [ ] Determinism flags verified (§4.3) by running seed 42 twice and diffing `run.json`.
- [ ] α calibration procedure run once on task-1 features; resulting α value recorded in frozen config.
- [ ] β and γ (for Arm A) calibrated per `project.md` §2.2; recorded in frozen config.
- [ ] Fisher clamp min=0.1 confirmed active in `contibualmob/bidding.py:_normalize_fisher()`.
- [ ] Optimizer-reset policy confirmed active (task-aware: all winning experts at task end).

### 11.2 Breach pre-test checklist (before running the primary t-test)

- [ ] All 60 runs complete (or the 20 runs at 20T if compute-limited).
- [ ] All runs pass §8.4 acceptance tests post-hoc (tests re-run on saved artifacts).
- [ ] Per-seed Fisher-magnitude ratio check (§4.4) passes for ≥8 of 10 seeds; if <8, recalibrate and rerun.
- [ ] Compute-match (§3) passes within ±5% on all 20T seeds.
- [ ] Shapiro-Wilk on paired `Δ_s` computed; if p<0.05, BCa bootstrap is authoritative.
- [ ] BCa bootstrap run with 10,000 resamples; seed for resampling recorded.
- [ ] Per-arm summary (mean, std, min, max) computed on `A_T` at 20T.
- [ ] ciFAIR cross-check run; sign of `Δ̄^{CIFAR-100}` matches sign of `Δ̄^{ciFAIR-100}`. If not, escalate.
- [ ] `results/gate/summary.json` written with gate verdict.

### 11.3 KAY/O pre-data red-team checklist (adversarial review before any seed runs)

KAY/O reviews this document for the following attacks; any successful attack blocks the full run:

- [ ] **Is β=0 in Arm B a sufficient null for "auction is epiphenomenal"?** Can KAY/O construct a scenario where Arm B as defined still benefits from MoB-specific mechanism? (Protocol answer: conscience γ is also zeroed, and α calibration is shared — if KAY/O finds an additional MoB-specific hidden channel, protocol is amended.)
- [ ] **Is matched-FLOP the right compute definition?** Can KAY/O argue for matched-wall-clock or matched-steps as strictly fairer? (Protocol answer: all three reported; primary is FLOP per §3.1.)
- [ ] **Does 10 seeds at `Δ_practical=1.0pp` constitute a meaningful test vs FeCAM's 1–3pp reported variance?** (Protocol answer: pilot power analysis in §7.2 resolves.)
- [ ] **Is the Fisher-magnitude matching criterion (2×) tight enough?** KAY/O's R1 cited 18× raw variance; protocol clamps to 0.1 and requires 2× matching. Is there a seed path where F̄^A and F̄^B diverge within 2× but produce materially different `c_forget` distributions?
- [ ] **Can the Arm-B implementation still carry β·c_forget signal via the α·d_M term if the prototype updates depend on which expert wins, and winning changes between arms?** (Protocol answer: yes, but this is an intended consequence — if Arm A's β term changes who wins and that changes prototype evolution and that produces the gain, this IS the mechanism working. The gate measures the end-to-end effect of including β and γ in the bid. The §1.1 preregistration is explicit that `E[Δ]` is the total effect, not a decomposed effect.)
- [ ] **Are there pre-data analysis degrees of freedom not yet frozen?** E.g., test-set choice, exclusion criteria, transformation of A_T. (Protocol answer: all specified in §1, §5, §6.)
- [ ] **Does the paired design handle the case where Arm A's routing puts different data in front of each expert than Arm B does, so the "pairing" of initial state is lost after step 1?** (Protocol answer: acknowledged. The pairing is on *pre-training initialization*, not on mid-training state. This is the correct paired-design semantics — the paired-difference test tests the total effect of the β,γ intervention applied from step 1.)
- [ ] **Is the gate decision rule stable to pilot-observed `σ_Δ`?** (Protocol answer: §6.5 and §7.2 pre-commit escalation paths.)
- [ ] **Is there a protocol path to rescue the program from a FAIL outcome that is not preregistered?** (Protocol answer: no. §9.3 terminates the routing-mechanism thesis; no CIFAR reruns.)

KAY/O should produce a written red-team memo (target: 500-1000 words) citing this checklist before pilot seed 42 launches. Successful attacks update the protocol via amendment (§13) pre-data. Attacks raised post-data are logged but do not retroactively modify the gate decision.

---

## 12. Timeline and Budget

| Phase | Duration | Compute (1× A100/H100 80GB) | Deliverable |
|---|---|---|---|
| Jett implementation + acceptance tests | 5–7 days | Test-only (<0.5 GPU-day) | §8.4 tests pass in CI |
| Breach + KAY/O pre-data protocol review | 2 days | None | Protocol v1.0 frozen |
| Pilot (3 seeds × 2 arms × 20T only) | 3–4 days | ~2 GPU-days (v1.2.1: ~8 hours/seed at 20T × 6 runs under 20-epoch pin) | `results/gate/pilot/pilot-report.json` |
| Pilot adjudication | 1 day | None | GO/NOGO/ESCALATE decision |
| Full 10-seed run, 20T×5C (the gate) | 5–7 days | ~6 GPU-days (v1.2.1: 7 remaining seeds × 2 arms × ~8h each under 20-epoch pin) | `results/gate/mob_full/20T/` + `results/gate/fecam_router/20T/` |
| Full 10-seed run, 5T×20C | 2–3 days | ~2 GPU-days (v1.2.1 update) | 5T results |
| Full 10-seed run, 10T×10C | 3–4 days | ~3 GPU-days (v1.2.1 update) | 10T results |
| Statistical analysis + ciFAIR cross-check | 2 days | <0.2 GPU-day | `results/gate/summary.json` |
| Writeup | 3–5 days | None | Gate-decision memo |
| **Total gate decision** | **~3–4 weeks** | **~13–14 GPU-days on 1×A100/H100 (v1.2.1 re-costed under 20-epoch pin)** | synthesis §5 Phase-1 branch trigger |

Wall-clock estimates assume Chamber R2's ~300K-params-per-expert adapter and the **v1.2.1-pinned 20 epochs/task** for 20T (LAMDA-PILOT canonical `tuned_epoch=20` from `exps/fecam.json` at commit `7a6e904c`, per §2.3 T1). The v1.2 timeline was built against Jett's initial 10-epoch wall-clock estimate; the 20-epoch pin roughly doubles per-seed training time, so the pilot (3 seeds × 2 arms) grows from ~1 GPU-day to ~2 GPU-days, and the full 20T 10-seed run grows from ~3 GPU-days to ~6 GPU-days. **Total gate compute at 20T×10 seeds: ~10–12 GPU-days (up from ~8).** If the pilot's §6.5 D3 backup triggers n=20, the full-run cost doubles further. This revised budget has been confirmed feasible within the scheduled Phase-1 window (§5 synthesis Phase-1) and does not compress Phase-2 on the critical path; timeline row totals below have been updated in-place.

---

## 13. Amendment Protocol

1. Any change to §1 (preregistration), §2.1 (protocol invariants), §3 (matched compute), §4 (seed budget), §5 (dataset integrity), §6 (statistical analysis plan), §7 (pilot criteria), or §9 (outcome branches) after the freeze date (2026-04-26) is an **amendment** and requires:
   - Written justification (≥200 words) logged in §1.6.
   - Signatures from Nosh and KAY/O.
   - A classification: (a) **pre-data amendment** — no runs contaminated, freeze date updates; (b) **post-data amendment** — prior runs may be invalidated; any amendment that tightens or loosens the gate rule (§1.2) invalidates all completed runs on the affected configuration.
2. Changes to §8 (implementation interface) are **implementation-level** and do not require amendment unless they break an §8.4 acceptance test.
3. Changes to §10 (threats), §11 (checklists), or §12 (timeline) are **editorial** and logged in git history only.

---

## 14. References

- Synthesis: `docs/research-party/synthesis.md`
- KAY/O Round 1: `docs/research-party/round1/kayo-position.md`
- Cypher CIFAR-100 audit: `docs/lit-review/04-cifar100-benchmark-audit.md`
- Chamber Round 2: `docs/research-party/round2/chamber-r2.md`
- Killjoy Round 2: `docs/research-party/round2/killjoy-r2.md`
- Project blueprint: `project.md`
- Memory (Fisher clamp, optimizer reset): `memory/MEMORY.md`
- LAMDA-PILOT: https://github.com/sun-hailong/LAMDA-PILOT
- FeCAM: Goswami et al., NeurIPS 2023, arXiv 2309.14062
- ciFAIR: Barz & Denzler 2020, cvjena.github.io/cifair
- BCa bootstrap: DiCiccio & Efron, Statistical Science 1996
- Chaudhry forgetting: arXiv 1801.10112
- Lopez-Paz & Ranzato BWT: GEM, NeurIPS 2017

---

*End of protocol v1.0. Breach. 2026-04-19.*

---

## 15. v1.1 Changelog (amendment pass over v1.0)

**Author:** Breach. **Date:** 2026-04-19 (same day as v1.0; pre-data, pre-freeze). **Trigger:** KAY/O red-team memo `docs/protocols/fecam-router-gate-redteam.md` (APPROVE WITH AMENDMENTS, 2026-04-19). **Scope:** amendment pass only, not a rewrite. No changes to scope, seed values, task configs, arm definitions, primary metric, or `Δ_practical`.

All v1.1 amendments are pre-data and pre-freeze; under §13 they are classified as pre-data amendments that update the freeze content without invalidating any prior runs (because no runs exist yet).

### 15.1 Critical amendments (KAY/O D1–D3, binding)

**v1.1-A1 (resolves KAY/O D1 / §6.3 / §9).** §1.2 gate decision rule rewritten to a **symmetric one-sided BCa framework**: PASS iff `L_95 ≥ +Δ_practical`; FAIL iff `U_95 ≤ −Δ_practical`; TIE otherwise. All three regions mutually exclusive and exhaustive. §6.1, §6.2, §6.3, §9.1, §9.2, §9.3 updated to match. The v1.0 one-sided-primary-plus-two-sided-FAIL hybrid (which had no registered FAIL test statistic and an operationally near-unreachable FAIL condition at n=10) is fully superseded. The t-test and Wilcoxon are retained as secondary symmetric robustness checks.

**v1.1-A2 (resolves KAY/O D2 / §5.2 / §5.4).** §4.4 Fisher-magnitude gate extended from a single cross-arm-within-seed criterion to a **dual criterion** that also enforces `CV(log F̄^A(s)) ≤ 0.5` across seeds within Arm A (and symmetrically for B). If violated at pilot, branch (a) runs the clamp ladder `{0.1, 0.3, 1.0, 3.0}` and adopts the first value that satisfies the bound (pre-registered as the default branch); branch (b) fallback is only reachable if the clamp ladder exhausts, in which case the full run proceeds at the best clamp with a binding limitation annotation inherited into the gate memo and any downstream paper. The clamp ladder is taken verbatim from KAY/O red-team §5.5. A per-task drift report (`max_t ratio(F̄^A_t, F̄^B_t)`) is added per KAY/O red-team §5.3.

**v1.1-A3 (resolves KAY/O D3 / §6.7).** §6.5 power bands tightened. v1.0 labeled σ_Δ ∈ (1.16, 1.75] as "marginally powered; proceed with annotation" — KAY/O's direct power calculation showed this band drops to ~46% power, which is a coin flip. v1.1 removes the 46%-power proceed-region and pre-registers **n=20 as a deterministic backup** in the σ_upper ∈ (1.30, 1.75] band (seeds extend to {42..61}; +17 seeds per arm past pilot). The outer-guard at σ_upper > 1.75pp → Nosh escalation is retained. The n=20 backup is pre-committed and does NOT require Nosh approval at the decision-rule level (only resource-availability check at dispatch).

### 15.2 Minor defect amendments (KAY/O #4–#9, all landed)

**v1.1-A4 (resolves KAY/O minor #4 / §3.5).** §2.1 α calibration row updated: α is calibrated on a **shared pre-routing pre-pass** across the entire task-1 training set (backbone-only forward with initialized prototypes; median `d_M` across all (x, i, c) tuples). Single scalar written to `frozen_config.yaml`; both arms load it at launch. Verified by §8.4 test 15.

**v1.1-A5 (resolves KAY/O minor #5 / §3.1–§3.2 LoRA & RNG state).** New §8.4 acceptance test 13 (`test_rng_state_equality_at_first_bid`) asserts byte-identical Python `random`, NumPy, Torch-CPU, and Torch-CUDA RNG states between Arm A and Arm B at the first bid step (not just step-0 weights). cuBLAS state enforced via `CUBLAS_WORKSPACE_CONFIG` at process start.

**v1.1-A6 (resolves KAY/O minor #6 / §3.1 DataLoader).** §2.1 adds a DataLoader-worker-seeding invariant: `worker_init_fn=seed_worker` with a seeded generator, OR `num_workers=0`. Jett picks one and freezes in `frozen_config.yaml`. Verified by §8.4 test 14 (`test_dataloader_worker_rng_pinned`).

**v1.1-A7 (resolves KAY/O minor #7 / §3.1 loss equality at t=0).** New §8.4 acceptance test 11 (`test_losses_bitwise_identical_at_t0`) asserts that Arm A and Arm B produce bitwise-identical per-sample cross-entropy losses on a fixed seed-42 batch at step 0 before any gradient step or bid. Stronger than the weight-hash test because it also catches intermediate-buffer and RNG-consumption-order asymmetries.

**v1.1-A8 (resolves KAY/O minor #8 / §4.4 wall-clock trap).** New §1.8 mandates that any PASS outcome in a downstream paper discloses the wall-clock ratio `wallclock^A / wallclock^B` as a prominent adjunct to the headline BCa statistic. Rationale: a PASS at 1.2× Arm-A wall-clock is a materially different research claim than a PASS at matched wall-clock; the reader must evaluate both.

**v1.1-A9 (resolves KAY/O minor #9 / §7.5 paper metric primacy).** New §1.7 mandates that the primary gate test statistic (`L_95`/`U_95` at 20T CIFAR-100) is the headline comparison in any downstream paper/preprint/presentation. Secondary metrics may be reported alongside but cannot substitute. Violation invalidates the preregistration claim.

### 15.3 New acceptance tests (KAY/O 8 new; §8.4 grew 8 → 16)

**v1.1-A10.** §8.4 acceptance tests 9–16 added:
- 9: `test_arm_b_code_path_never_invokes_ewc_for_forget_cost` (KAY/O new test a).
- 10: `test_arms_disagree_on_at_least_one_percent_of_pilot_inputs` (KAY/O new test b / D8).
- 11: `test_losses_bitwise_identical_at_t0` (KAY/O new test c / minor #7; see A7).
- 12: `test_flop_accounting_unit_test` (KAY/O new test d / minor #15-style).
- 13: `test_rng_state_equality_at_first_bid` (KAY/O new test e / minor #5; see A5).
- 14: `test_dataloader_worker_rng_pinned` (KAY/O new test f / minor #6; see A6).
- 15: `test_alpha_calibration_shared_prepass` (KAY/O new test g / D4; see A4).
- 16: `test_fisher_cv_inclusion` (KAY/O new test h / D2; see A2).

All 16 tests are CI-enforced on every commit touching gate-relevant modules and must pass before pilot seed 42 launches.

### 15.4 Pilot protocol updates

**v1.1-A11.** §7.2 pilot success criteria expanded from 5 to 7: added criterion 3 (`CV(log F̄) ≤ 0.5` inclusion, resolves D2), criterion 7 (arms-disagree-on-at-least-1%-of-pilot-inputs, resolves D8). Criterion 5 reworked to reference the v1.1 §6.5 three-band power rule (resolves D3). §7.3 pilot-report JSON schema expanded with `fisher_cv_log_arm_*`, `fisher_clamp_adopted`, `arms_disagree_fraction`, `power_band`, `recommended_n` fields, and `go_nogo` enum expanded to `{GO_N10, GO_N20, NOGO, ESCALATE, CV_LIMITATION}`.

### 15.5 Out-of-scope (deliberately NOT changed in v1.1)

v1.1 is an amendment pass, not a rewrite. The following were preserved from v1.0 per KAY/O's approve-with-amendments disposition:
- Δ_practical = 1.0pp (KAY/O red-team §6.5 rationale-rewrite is a framing suggestion, not a binding defect; rationale can be refined in the gate memo if needed).
- Seed set `{42..51}` as the n=10 primary; the n=20 extension `{42..61}` is conditional on pilot σ_upper landing in the D3 backup band.
- Three-task-config design (5T, 10T, 20T) with 20T as the binding gate config.
- Two-arm strict nested ablation (β=γ=0 as the null).
- All of §10 threats table, §11 checklists, §12 timeline, §13 amendment protocol, §14 references.

### 15.6 Escalations to Nosh (none)

v1.1 accepted all three critical defects (D1, D2, D3) and all six minor defects as binding amendments. No D-defect or minor defect was escalated as over-constraining, theoretically suspect, or infeasible. KAY/O's red-team prescription was implemented as specified for D1, D2, minor #4, #5, #6, #7, #8, #9, and the 8 new tests. For D3, Breach selected KAY/O's **second** prescribed option (pre-commit n=20 backup) rather than his first (tighten σ_upper cap to ≤1.30pp) for the reasons stated in §6.5 ("Why n=20 backup rather than a tighter σ_upper cap"); both options were offered by KAY/O as valid and the implementation selects one — this is not an escalation but a pre-offered choice. For D2, Breach implemented KAY/O's branch-(a) preferred + branch-(b) fallback decision tree verbatim.

### 15.7 Freeze-readiness assessment

v1.1 addresses every defect KAY/O flagged as blocking (D1, D2, D3) and every minor defect. The protocol as of v1.1 can be frozen on **2026-04-26** as originally scheduled. No timeline impact: the v1.1 amendment pass consumed <1 day of Breach + 0 days of Jett (tests 9–16 ship as part of Jett's original §8.4 implementation sprint, which already runs 5–7 days per §12 and now includes 8 additional tests but is scoped the same).

Any round-2 review from KAY/O should focus on two implementation-choice points where v1.1's resolution modestly differs from KAY/O's prescription:
- D3 option-selection (n=20 backup rather than σ_upper ≤ 1.30pp cap). v1.1 justifies this in §6.5.
- D2 branch-(b) limitation-annotation language. v1.1 specifies the annotation text in §4.4 item 4b; KAY/O may wish to review it for adequacy.

All other amendments land as-specified.

---

### 15.8 v1.2 Changelog (FeCAM-canonical binding pass)

**Author:** Breach. **Date:** 2026-04-19 (same day as v1.0/v1.1; pre-data, pre-freeze). **Trigger:** Inline FeCAM-code comparison `docs/lit-review/05-fecam-code-comparison.md` (2026-04-19) established that `contibualmob/prototype_store.py` implements 1 of 4 FeCAM recipe elements (shared Σ with ridge regularization only), so Arm B under the v1.1 pathway would not be paper-canonical FeCAM and the headline claim "MoB beats FeCAM" would lose its referent. **Scope:** implementation-binding pass only. No change to the statistical gate framework (§1.2, §6), Fisher-match gate (§4.4), seed plan (§4.1), tests 1–16 (§8.4), power analysis (§6.5), outcome branches (§9), or freeze date (2026-04-26). **Classification (§13):** pre-data, pre-freeze amendment — no runs contaminated.

### 15.9 v1.2-B1 — Mahalanobis core bound to LAMDA-PILOT + `dipamgoswami/FeCAM` upstream (paper-canonical)

- **§1.3** amended: both arms' `d_M` is explicitly the v1.2-bound core per §2.0 and §4.6; β=γ=0 in Arm B is the ONLY code-path difference.
- **§2.0 (NEW)** inserted before §2.1: defines the five load-bearing FeCAM elements (per-class μ, per-class Σ, additive two-parameter shrinkage `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)`, correlation normalization, L2 normalization) and the v1.2 decision to disable Tukey for ViT-B/16 per paper §7. Out-of-scope note: `contibualmob/prototype_store.py` is unchanged; the gate runs through a new module `mob/gate/fecam_core.py` that ports from the upstream pins.
- **§2.1** amended: the "Covariance structure" row is re-scoped — v1.1's tied-low-rank `U ∈ ℝ^(768×32)` is now an internal numerical-representation option gated by §8.4 test 17 (per-sample ≤1e-5 equivalence to the direct per-class path). The "Mahalanobis formulation" row now names the paper-canonical recipe explicitly. The "Shrinkage parameter λ" row is replaced with a "Shrinkage hyperparameters" row pinning `γ₁, γ₂` per §4.7. New rows: "L2 normalization of features and prototypes" (paper §3.1), "Tukey β transform" (OFF for ViT per paper §7). "Prototype class granularity" row is upgraded from shared-within-expert Σ to per-class `μ_{i,c}` AND per-class `Σ_{i,c}` (the paper's primary, not the "common covariance" ablation).

### 15.10 v1.2-B2 — §4.6 FeCAM-binding implementation invariant (Jett contract)

- **§4.6 (NEW):** implementation invariant with repo pins (LAMDA-PILOT `7a6e904c5bc5cb7a4e1823b3434020be27469b63`; `dipamgoswami/FeCAM` `e33f39d112ff2d2a2df2e68c490af579a50edd31`) and a five-clause Jett contract (byte-equivalent port, harness wiring, no silent deviation, identical Mahalanobis core across arms, invalidity-on-violation declaration). **Escalation on record:** LAMDA-PILOT's `models/fecam.py::shrink_cov` at the pinned SHA applies single-parameter ridge `cov + 100·I`, NOT the paper's two-parameter additive shrinkage — this is a paper-Table-2 load-bearing deviation. v1.2 resolves this by binding the shrinkage code path specifically against `dipamgoswami/FeCAM:models/base.py::shrink_cov` (paper-canonical) while using LAMDA-PILOT for harness/trainer/data/backbone and its correct ViT-specific `tukey=off` decision. This composite binding is documented in §4.6 as the only way to satisfy both "paper-canonical" and "LAMDA-PILOT harness" simultaneously.
- **§4.7 (NEW):** shrinkage hyperparameter pinning. `γ₁ = γ₂ = 1` adopted as default (paper §7 CIFAR-100 MSCIL). The paper does NOT publish CIFAR-100 + ViT-B/16 γ values; this is the tightest paper-published match to the gate benchmark. A pre-freeze override window is defined: Jett runs a backbone-only forward pass at seed 42, reports empirical `V₁, V₂`, and if either is ≥10× the ResNet-18 reference magnitude, Breach may approve an override to `γ₁=γ₂=10` (the paper's Split-ImageNet-R + ViT value) before freeze. Post-freeze values are locked by §8.4 test 18.

### 15.11 v1.2-B3 — §8.4 expanded 16 → 19 tests (FeCAM-port fidelity, hyperparameter pinning, preprocessing)

- **§8.4** header updated ("16 tests" → "19 tests") and gate-trigger file list extended to include `mob/gate/fecam_core.py`.
- **Test 17 `test_fecam_port_fidelity_against_upstream` (v1.2, §4.6 clause 1):** ≤1e-5 per-sample `d_M` equivalence between Jett's port and the upstream `dipamgoswami/FeCAM` reference on a deterministic synthetic tensor fixture at seed 20260426 (`d=768, N=128, C=10`). Verifies all four covariance elements jointly.
- **Test 18 `test_fecam_hyperparameters_pinned` (v1.2, §4.7):** asserts `frozen_config.yaml.fecam` fields match paper-canonical values (`γ₁=γ₂=1, tukey=false, tukey_beta=0.5, per_class_cov=true, l2_normalize=true, inverse_method=pinv`) and both arms load identical values. Cites paper §7 CIFAR-100 MSCIL reference and the `dipamgoswami/FeCAM:exps/FeCAM_cifar100.json` pinned blob for corroboration. Flags the intentional ResNet-18 → ViT-B/16 Tukey deviation as a paper-§7-traceable decision.
- **Test 19 `test_fecam_feature_preprocessing_l2_and_tukey` (v1.2, §4.6 clause 1 + 4):** verifies L2 normalization produces unit vectors, Tukey is NOT applied when `config.tukey=false` (no complex-valued or exception-raising paths on mixed-sign ViT features), and IF Tukey is enabled on a future variant, the ordering Tukey → L2 → subtract matches the upstream control flow.

All three tests are CI-enforced and must pass before pilot seed 42 launches. They do not replace any existing test; they are additive.

### 15.12 Out-of-scope (deliberately NOT changed in v1.2)

- §1.2 gate decision rule (symmetric one-sided BCa framework).
- §4.1 seed plan (`{42..51}` → `{42..61}` conditional on D3 backup).
- §4.4 Fisher-match gate (dual-criterion).
- §6 statistical analysis plan.
- §6.5 power bands and pilot decision rule.
- §7 pilot protocol and §7.2 success criteria (tests 17–19 ARE part of criterion 4 "implementation integrity"; §7.2 text "all 16 tests in §8.4 are green" should be read as "all 19 tests" under v1.2, but the explicit numeric count in §7.2 criterion 4 is deliberately left as the v1.1 wording to minimize edit surface — §8.4 header is the single source of truth for the test count).
- §8.4 tests 1–16 (no existing test modified).
- §9 outcome branches.
- §10 threats, §11 checklists, §12 timeline, §13 amendment protocol, §14 references.
- Freeze date 2026-04-26.

### 15.13 v1.2 escalations to Nosh (ONE on record, flagged; does not block freeze)

1. **LAMDA-PILOT `shrink_cov` deviates from FeCAM paper eq. 8 (load-bearing).** LAMDA-PILOT at the pinned SHA uses `cov + 100·I` (single-parameter ridge), not `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)`. Per the directive's escalation criterion, v1.2 surfaces this explicitly and binds the shrinkage-code path against `dipamgoswami/FeCAM` upstream (paper-canonical) rather than LAMDA-PILOT. Harness binding remains LAMDA-PILOT. Rationale and binding choice are fully documented in §4.6. **Disposition: accepted as a composite binding; no timeline impact; freeze date 2026-04-26 holds.**
2. **Paper does not publish CIFAR-100 + ViT-B/16 γ values.** v1.2 adopts `γ₁=γ₂=1` (paper §7 CIFAR-100 MSCIL ResNet-18 values) as the tightest paper-published match to the gate benchmark, with a pre-freeze override window for ViT-specific empirical checks (§4.7). This is a minor out-of-paper extrapolation explicitly acknowledged. **Disposition: accepted with pre-freeze override window; no escalation required unless Jett's empirical `V₁, V₂` diverges ≥10× from ResNet-18 reference, in which case Breach pre-approves override to `γ₁=γ₂=10` per paper Split-ImageNet-R + ViT values.**

Neither escalation blocks the 2026-04-26 freeze; both are pre-data and pre-freeze. Nosh is notified but does not need to adjudicate — the dispositions are within Breach's preregistered authority per §13 for pre-freeze implementation bindings.

### 15.14 Freeze-readiness assessment (v1.2)

v1.2 is a pure binding pass; no v1.1 statistical/methodological text was modified, only added to (§4.6, §4.7, §8.4 tests 17–19) or re-scoped with explicit superseding language (§2.1 Covariance row / Mahalanobis row / Prototype granularity row, with v1.1 phrasing preserved as historical reference where possible). The protocol as of v1.2 can be frozen on **2026-04-26** as originally scheduled. No timeline impact: the v1.2 amendment pass consumed <1 day of Breach + adds ~1 day of Jett work (three additional tests with one synthetic fixture file) to the existing §8.4 implementation sprint, which already runs 5–7 days per §12 and now includes 19 tests but is scoped the same.

Optional round-2 review is NOT commissioned on v1.2 per Nosh's pre-agreed disposition (this is an implementation binding, not a statistical change). The gap analysis (`docs/lit-review/05-fecam-code-comparison.md`) is the sole external justification; the paper and the two pinned repos are the sole external references.

---

*End of protocol v1.2. Breach. 2026-04-19. Supersedes v1.1. Freeze target: 2026-04-26.*

---

### 15.15 v1.2.1 Changelog (training-spec pin pass over v1.2)

**Author:** Breach. **Date:** 2026-04-19 (same day as v1.0/v1.1/v1.2; pre-data, pre-freeze). **Trigger:** Chamber's pre-Phase-2 commission review (2026-04-19 ~14:30) surfaced that v1.2's per-task epoch count was carrying Jett's wall-clock-estimate value (10 epochs/task) rather than the LAMDA-PILOT canonical value (`exps/fecam.json: tuned_epoch=20` at the pinned SHA `7a6e904c`), and that v1.2's §2.1 training-related rows left optimizer/LR/batch-size/warmup/reset scope as "reserved for Jett" deferments that could drift unsupervised between Jett's hands and the frozen config. Chamber specified a full 7-item training spec to pin before freeze. **Scope:** training-spec pin pass only. **Classification (§13):** pre-data, pre-freeze amendment — no runs contaminated. Edits touch: §1 version header, §2.1 invariant rows (Backbone freeze scope, Optimizer, LR schedule, Per-task epochs, Batch size, Optimizer reset, Training epoch count row retired), §2.3 (NEW training-spec block; former §2.3 KAY/O-invariant-ablation renamed to §2.4), §7.2 pilot criterion 4 (test count 16 → 20 via §8.4), §7.3 pilot report schema `protocol_version` → `1.2.1`, §8.4 (test 20 added; header 19 → 20), §12 (wall-clock re-cost under 20-epoch pin), §15 (new §15.15–§15.18). No change to §1.2 gate decision rule, §4.1 seed plan, §4.4 Fisher-match gate, §4.6 FeCAM-binding invariant, §4.7 shrinkage hyperparameters, §6 statistical analysis plan, §6.5 power analysis, §9 outcome branches, or the freeze date 2026-04-26.

### 15.16 v1.2.1-C1 — Per-task epochs pinned at 20 (Chamber primary fix)

- **§2.1 "Per-task epochs" row** upgraded from "reserved for Jett" deferment to **pinned at 20 epochs/task**, with binding citation to `sun-hailong/LAMDA-PILOT/exps/fecam.json: tuned_epoch=20` at commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63` (the §4.6 harness binding SHA). Corroborating citations to six additional LAMDA-PILOT `cifar224` adapter/LoRA pack configs (`aper_aperpter.json`, `ranpac.json`, `mos.json`, `coda_prompt.json`, `slca.json`, `ease.json`), all of which use 20 epochs.
- **§2.3 T1 (NEW)** documents the pin, the binding citation, the corroborating pack, and the reasoning ("Running undertime at 10 epochs would train FeCAM to a state materially weaker than the published reference and invalidate the headline claim").
- **§12 timeline** updated: wall-clock estimate doubles (pilot ~1 → ~2 GPU-days; full 10-seed 20T ~3 → ~6 GPU-days; total ~8 → ~13–14 GPU-days under 20-epoch pin). Confirmed feasible within Phase-1 window.
- **§8.4 test 20 (NEW)** `test_training_spec_pinned_v1_2_1` asserts `frozen_config.yaml.training.epochs_per_task == 20` at load time, blocking any drift from the pin.

### 15.17 v1.2.1-C2 — Training-spec audit (Chamber items T2–T7 pinned in §2.3)

All seven Chamber items pinned in the NEW §2.3 training specification block. Summary of pin values and provenance:

- **T2 Optimizer:** AdamW `β₁=0.9, β₂=0.999, ε=1e-8`, `weight_decay=1e-4`. §2.1 Optimizer row upgraded from "as per LAMDA-PILOT default" to explicit numeric values. Intentional divergence from `fecam.json: optimizer=sgd` annotated in §2.3 T2; not a §4.6 violation because §4.6 binds Mahalanobis scoring, not optimizer choice.
- **T3 LR schedule:** cosine decay to `min_lr=1e-5`, peak LR `5e-4` for adapter path / `1e-3` for LoRA-QKV path (InfLoRA/SLCA canonical), no warm-restart at task boundary. §2.1 LR schedule row upgraded with explicit values.
- **T4 Batch size:** 128. §2.1 Batch size row upgraded from "matched across arms" to explicit 128. Intentional divergence from `fecam.json: batch_size=48` annotated.
- **T5 Weight decay:** 1e-4 on LoRA + FFN-bottleneck params; not on LN/bias (standard AdamW decoupled-WD). Pinned in T2 row.
- **T6 Warmup:** linear over 100 steps on task 1 only; tasks 2..T skip warmup (adapter arrives pre-warmed). Rationale: re-warmuping each task resets learning to near-zero and wastes ~2.5 epochs of cosine budget.
- **T7 Optimizer reset at task boundary:** on task end, after Fisher update, **for every expert (all 4), not just task-winner(s)**. v1.2 §2.1 Optimizer reset row said "Applied identically in both arms" but did NOT explicitly state "every expert" — Chamber's audit caught this as ambiguous. v1.2.1 tightens to "every expert" with explicit rationale (stale Adam `v_t` on a non-winner expert biases first-step magnitudes when a future task routes to it). Verified by §8.4 test 20.
- **Backbone freeze clarifier (§2.1 Backbone freeze scope row, NEW):** fully-frozen parameter groups `["blocks", "patch_embed", "cls_token", "norm", "pos_embed"]`, **no final-block LN unfreezing exception**. Rationale: LN-unfreeze makes LN scale a private bid-time-hidden parameter, breaking DSIC; LoRA QKV already supplies direction-aware reweighting, so LN-unfreezing adds negligible headroom. §2.1 never had an explicit LN exception, but Chamber asked for the defensive pin; the NEW row closes the ambiguity.

### 15.18 v1.2.1 escalations and freeze-readiness

**Escalations to Nosh: NONE.** All seven Chamber items land cleanly. The three intentional divergences from LAMDA-PILOT's `fecam.json` (AdamW vs SGD, batch 128 vs 48, weight_decay 1e-4 vs 5e-4) are all consistent with the gate's §2.1 LoRA-QKV trainable-params choice and with the LoRA-CL literature convention (InfLoRA, SLCA, CODA-Prompt); they are NOT §4.6 binding violations because §4.6 binds the Mahalanobis scoring recipe and harness wiring (dataloader, trainer loop structure, ViT-B/16 backbone loading), not the optimizer/LR/batch-size/WD choice. The binding is internally consistent: Arm B runs paper-canonical FeCAM Mahalanobis inside a LoRA-QKV + AdamW + cosine training loop; this is the natural transposition of FeCAM to the LoRA-CL regime and is consistent with LAMDA-PILOT's own LoRA-path configs (e.g., `slca.json` optimizer behavior).

**Verification against escalation criteria per Breach directive:**
1. No Chamber item conflicts with v1.0/1.1/1.2 text in a way requiring structural modification — all items slot into §2.1 row upgrades + §2.3 addition.
2. **One second pre-freeze defect identified and closed in this pass**: v1.2 §2.1 "Optimizer reset" row did NOT explicitly state "for every expert" despite the `memory/MEMORY.md` rationale requiring it; v1.2.1 tightens to explicit "every expert." This is annotated in §15.17 T7 as a Chamber-adjacent discovery, not a separate escalation; no timeline impact.
3. LAMDA-PILOT's `exps/fecam.json` at the pinned commit `7a6e904c` was verified directly via `raw.githubusercontent.com`: `{tuned_epoch: 20, init_lr: 0.01, optimizer: "sgd", batch_size: 48, weight_decay: 0.0005, min_lr: 0}`. `tuned_epoch=20` is confirmed. The optimizer/batch/WD values in `fecam.json` (SGD, 48, 5e-4) are NOT adopted because the gate uses LoRA-QKV trainable params, for which AdamW+cosine+128+1e-4 is the published convention (Chamber's T2–T5 selections). This divergence is documented in §2.3 T2, T4, T5.

**Freeze-readiness:** v1.2.1 is ready for Nosh freeze signature on **2026-04-26** as originally scheduled. No timeline slip on the freeze-signing side. Budget impact: total gate compute re-cost from ~8 GPU-days to ~13–14 GPU-days under the 20-epoch pin (§12). Jett's implementation sprint absorbs test 20 plus the frozen-config schema extensions (training block with all 7 T-items, backbone freeze scope with `ln_unfreeze_exception: false`); the added work is ≤1 day against the existing 5–7-day sprint that already includes 19 tests (v1.2). No round-2 review is commissioned on v1.2.1 per Nosh's pre-agreed disposition for pre-freeze implementation pins.

---

**frozen_config.yaml fragment (v1.2.1 reference; Jett renders canonical file):**

```yaml
# v1.2.1 training specification (§2.3 T1–T7)
training:
  epochs_per_task: 20                       # T1: LAMDA-PILOT fecam.json @ 7a6e904c
  batch_size: 128                           # T4
  optimizer:
    name: AdamW
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 1.0e-4                    # T2, T5
  schedule:
    type: cosine
    min_lr: 1.0e-5                          # T3
    peak_lr:
      adapter: 5.0e-4
      lora_qkv: 1.0e-3                      # T3
    warm_restart_at_task_boundary: false    # T3
  warmup:
    type: linear
    steps: 100
    start_lr: 1.0e-6
    applied_to_tasks: [0]                   # T6: task-1 only
  optimizer_reset:
    cadence: task_end_after_fisher
    scope: every_expert                     # T7: all 4 experts, not just winner

model:
  backbone:
    frozen_param_groups:                    # §2.1 Backbone freeze scope (v1.2.1)
      - blocks
      - patch_embed
      - cls_token
      - norm
      - pos_embed
    ln_unfreeze_exception: false            # explicit; DSIC safety
```

Full `frozen_config.yaml` fields for FeCAM (`fecam.gamma1`, `fecam.gamma2`, `fecam.tukey`, etc.) are per §4.7 / §8.4 test 18 and are unchanged from v1.2.

---

*End of protocol v1.2.1. Breach. 2026-04-19. Supersedes v1.2. Freeze target: 2026-04-26.*
