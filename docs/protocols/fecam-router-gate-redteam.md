# KAY/O Red-Team: Phase-1 Killer-Gate Protocol (`MOB-GATE-001` v1.0)

**Target:** `docs/protocols/fecam-router-gate.md` v1.0 (Breach, 2026-04-19).
**Freeze date under review:** 2026-04-26.
**Reviewer:** KAY/O (adversarial).
**Date:** 2026-04-19.
**Stance:** Pre-data red-team. Not peer review of results — review of the rules that will decide whether MoB lives or dies.

---

## 1. Executive Summary

### 1.1 Verdict

**APPROVE WITH AMENDMENTS.** The protocol's skeleton is defensible: two-arm strict nested ablation with β=γ=0 as the null, paired-on-seed design, preregistered primary test, preregistered outcome branches, explicit threats table. Breach did the work. But there are eight defects that can each independently bias the gate decision or render it uninformative, and at least one (D1 below) is a structural flaw in the decision rule itself that Breach must amend before 2026-04-26.

Freeze can proceed **on 2026-04-26 only if D1–D3 are amended.** D4–D8 are strongly advised patches; a responsible PI would amend them as well but the gate can legally run without.

### 1.2 Top-5 ranked defects

| # | Defect | Severity | Patchability | Blocking for freeze? |
|---|---|---|---|---|
| **D1** | One-sided primary test has no decision region that triggers §9.3 FAIL. "FeCAM beats MoB" cannot be concluded by the preregistered test. | **CRITICAL** | Trivial (add TOST or switch to two-sided) | **YES** |
| **D2** | Fisher-match gate (`2×` intra-seed cross-arm) does not address KAY/O's R1 threat. The R1 threat was Fisher variance *across seeds within Arm A*; the gate checks Fisher agreement *across arms within a seed*, which is near-trivially satisfied by design and does not bound the actual confound. | **CRITICAL** | Medium (add per-seed Fisher exclusion or Fisher-magnitude stratification) | **YES** |
| **D3** | Power analysis at the "marginal" band (σ_Δ ∈ (1.16, 1.75]) drops below 50% power at Δ=1.0. Protocol proceeds in this band while labeling it "marginally powered." A 45%-powered gate is not a gate; it is a coin flip. | **HIGH** | Trivial (tighten escalation threshold to σ_Δ ≤ 1.16pp) | **YES** |
| **D4** | α is calibrated "once on task-1 features" but §2.1 does not specify *whose* task-1 features — Arm A's, Arm B's, or a pre-run shared pass. If per-arm, α differs across arms silently. If Arm A's, Arm B is tuned against Arm A's prototype geometry. | HIGH | Trivial (specify: calibrated once on a shared backbone-only pre-pass, logged in frozen config) | Advised |
| **D5** | LoRA/classifier-head init equality across arms is asserted by §8.4 test 3 but the test only hashes `state_dict()` at step 0. Between Arm A and Arm B instantiation, intervening allocations (Fisher buffer, prototype buffer, bid-logger buffer) advance the Torch-CUDA RNG stream differently, so step-0 weights can match while step-0 *RNG state* differs — surfacing at first dropout mask or first dataloader worker fork. | HIGH | Medium (explicit RNG-fork-point pinning before LoRA init, plus hash the RNG states too) | Advised |
| **D6** | Bicubic interpolation specified but bicubic implementation not pinned. `torchvision.transforms.functional.resize(..., InterpolationMode.BICUBIC, antialias=True)` vs PIL `Image.BICUBIC` vs cv2 `INTER_CUBIC` produce pixel-level differences. Not a cross-arm confound (same impl in both), but a cross-run reproducibility gap. Minor because the paired design absorbs it; still specify. | LOW | Trivial (lock one impl in frozen config) | Advised |
| **D7** | "TIE" outcome is defined under a one-sided test framing where tie is operationally ambiguous. Either run a proper TOST equivalence test at ±Δ_practical, or reframe as two-sided with three rejection regions (PASS, FAIL, TIE = neither bound excludes their threshold). | HIGH | Medium (add TOST block to §6) | Yes — together with D1, these are the same family of bug |
| **D8** | No acceptance test or preregistered check that Arm A and Arm B actually produce *different* expert selections on any real input. If β and γ don't change the argmin for any x in any seed, the gate is uninformative regardless of outcome — a no-op produces a PASS by noise or a TIE by determinism. | MEDIUM | Trivial (add `test_arms_disagree_on_pilot_inputs` to §8.4) | Advised |

### 1.3 On the 18× Fisher-variance threat (the R1 crux)

**The protocol does NOT handle the 18× threat. It handles a different threat by the same name.**

R1 threat: Fisher magnitude varies 18× **across seeds within a single arm** (`memory/MEMORY.md`). This means the `β·c_forget` signal that Arm A uses to differentiate experts is itself varying 18× across seeds. In a paired design, Arm A's favorable seed and Arm B's matched seed share the *same* Fisher draw — which means Arm A's β·c_forget benefits from that draw in a way Arm B (β=0) cannot benefit from, and cannot suffer from. This is not a confound in the paired difference; it is a **selection effect on the effect size itself**: the paired difference `Δ_s = A^A_s − A^B_s` is not a constant-in-expectation treatment effect, it is a random-slope-in-Fisher-magnitude treatment effect. Across 10 seeds with 18× Fisher spread, the empirical Δ̄ reflects a conditional average over whatever Fisher draws the 10 seeds happened to produce — and the bootstrap CI on `{Δ_s}` is a CI on *this sample's Fisher distribution*, not on the population treatment effect.

What the protocol checks (§4.4): `max(F̄^A(s), F̄^B(s)) / min(F̄^A(s), F̄^B(s)) ≤ 2` *within each seed, across arms*. Given that §2.1 pins arm invariants so tightly that Arm A and Arm B should have *near-identical* Fisher magnitudes at the same seed by construction (same init, same data order, same Fisher estimator, differing only in whether Fisher enters the bid), this check is near-trivially satisfied — the 2× bound is 18× weaker than the within-seed-across-arm agreement that the protocol's own construction enforces.

What the protocol does NOT check: whether `F̄^A(s)` itself varies by more than some bounded ratio across the 10 seeds `s ∈ S`. This is the R1-threat-relevant quantity. If it ranges 18× (per project memory), then Arm A's advantage will be concentrated in the favorable-Fisher seeds, and the paired test will detect that concentrated advantage, and Breach will correctly conclude "Arm A > Arm B on these seeds" — but the generalization to "MoB > FeCAM-Router in expectation" will not hold, because a fresh seed from the 18× Fisher distribution could produce either sign of `Δ`.

**Fix:** amend §4.4 to add a second inclusion criterion: the ratio `max_s F̄^A(s) / min_s F̄^A(s)` (and same for B) must be ≤ some preregistered bound (e.g., ≤ 4× after clamp; if it exceeds, either (a) document in the result memo as a limitation and still report, or (b) re-run with higher clamp until within-arm spread is bounded). OR add a Fisher-stratified analysis: report `Δ̄` conditional on Fisher quartile within the seed set.

Without this fix, a "PASS" at n=10 leaves your R1 threat **unfalsified**. See §11 (verdict).

---

## 2. Threat Model Recap

### 2.1 Restatement of R1 threat

`kayo-position.md` §2: The MoB auction is epiphenomenal. Given prototypes, `argmin_i [α·d_M + β·c_forget]` reduces to FeCAM's prototype-argmin unless `β·c_forget` straddles the Mahalanobis gap between the top-two experts. The useful regime is a narrow strip. Inside that strip, MoB is betting the *fine structure* of EWC Fisher matrices is informative. Project memory: Fisher varies 18× across initializations before a manual clamp. At that variance, the straddle-regime signal is dominated by noise. R1 prediction: on CIFAR-100 20T with 10 seeds, paired design, `Δ̄ < 1pp` with CIs that cross zero.

### 2.2 Attack surfaces that survive a perfect execution of this protocol

Even if Jett implements §8 flawlessly, Breach runs the pilot cleanly, the full 10 seeds complete, and the primary test yields a specific outcome, the following is true about what that outcome *means* for the R1 threat:

**A PASS tells you:** `E[Δ] ≥ 1pp` at the sampled seed distribution (a 10-element sample from N(42..51) under 18× Fisher spread) under the specific {backbone, adapter, shrinkage, class-order-harness, epoch-schedule} configuration. It tells you β and γ are jointly load-bearing at that config at those seeds.

**A PASS does NOT tell you:** (i) whether β alone would have sufficed (γ is bundled in; need β-only ablation); (ii) whether the effect survives a different 10-seed draw from the Fisher distribution; (iii) whether the auction is doing work *independent of the prototype geometry*, because Arm B is FeCAM-on-prototypes at the routing layer, which is a *weaker* null than the stronger FeCAM-at-classifier null that Cypher flagged (§5 row "FeCAM"); (iv) whether the effect scales with #experts (locked at 4); (v) whether the effect is robust to the task-boundary signal (task-aware optimizer reset fires in both arms, but Arm A's β term uses task-boundary Fisher updates while Arm B's β=0; if task-boundary info leakage helps Arm A more than Arm B, that's an observed effect but not a routing-mechanism effect).

**A TIE tells you:** the R1 null is *consistent* with the data. It does not confirm R1. It is the weaker version of R1 confirmation.

**A FAIL tells you:** FeCAM-Router beats MoB. This is stronger than R1 — R1 only predicts tie-or-narrow-win. But see D1: under the preregistered one-sided test, FAIL is not actually detectable.

**Defect surface:** the gate as designed answers "does adding β and γ to the bid, with everything else invariant, increase Avg Acc at 20T on CIFAR-100 at these seeds under this config by ≥1pp?" That is a narrower question than "is MoB's auction mechanism irreducible?" The 4-expert, 20T, ViT-B/16, single-auction-at-CLS config is a specific point in the design space, and the R1 threat was articulated at the *program* level. Breach's protocol is scoped to this point correctly; the scope limitation must be stated explicitly in §9.1 (PASS deliverable) so that a PASS is not over-interpreted.

---

## 3. Strict-Ablation Integrity Audit

Arm B = Arm A with β=γ=0. Hold "identical" every other quantity. I find the following asymmetries or underspecifications.

### 3.1 RNG identity across arms

§4.3 specifies 5-source seeding (Py/NumPy/Torch-CPU/Torch-CUDA/cuBLAS) plus `cudnn.deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True, warn_only=True)`. §8.4 test 5 verifies determinism of Arm A across two runs with the same seed.

**Defect:** `warn_only=True` (§4.3). This means `torch.use_deterministic_algorithms` will *warn* on a non-deterministic op rather than error out. If a non-deterministic op fires (e.g., `index_add_` on CUDA, bilinear upsample with `align_corners=False` in some torch versions, certain `scatter_add` paths), the arms will silently diverge even under identical seeds. §4.3 does say "if non-deterministic ops surface, Jett reports to Breach," which is the right triage, but the protocol does not hard-require the fix.

- **Severity:** MEDIUM. Impact: 0.1–0.3pp drift across arms at the same seed, concentrated in seeds that touch the non-deterministic kernel. Over 10 seeds this could inflate or deflate Δ̄ by ~0.1–0.3pp.
- **Amendment:** Change `warn_only=True` to `warn_only=False` *after* Jett verifies no ops fire during a full gate run. If an op fires, it is fixed or the run is rejected. Enforced in `test_determinism_single_seed` across both arms.

**Defect:** DataLoader worker seeds. §8.5 lists `num_workers` as Jett-tunable. But if `num_workers > 0` and `worker_init_fn` is not pinned to `seed_worker(worker_id)`, each worker's augmentation RNG starts differently per-process, and Arm A vs Arm B at the same seed see *different* augmentation sequences.

- **Severity:** HIGH. Impact: on CIFAR-100 with random crop + flip, augmentation-RNG differences can shift Avg Acc by 0.3–0.7pp. This is a direct cross-arm confound.
- **Amendment:** Add to §2.1 invariants: `DataLoader(worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(s))`, `num_workers=0` *or* `num_workers>0 with worker_init_fn that seeds numpy, random, torch from base_seed+worker_id`. Add a `test_dataloader_worker_rng_pinned` acceptance test.

### 3.2 LoRA initialization identity

§8.4 test 3: `hash(model_A.state_dict()) == hash(model_B.state_dict())` at step 0 for backbone + LoRA + classifier head.

**Defect:** This test passes if the *values* match, but RNG *state* can differ. If Arm A's `__init__` allocates {Fisher buffer, prototype buffer, bid logger} *before* LoRA init, and Arm B allocates a different subset (or same allocations with `β=0` short-circuits), the allocation sequence consumes different numbers of RNG draws, and the LoRA-init draw for Arm B in a *subsequent* task (task 2 re-init, classifier-head re-init per task, LoRA re-adaptation) starts from a different RNG state.

- **Severity:** HIGH. Impact: 0.2–0.5pp drift cumulative over 20 tasks. Hard to detect because step-0 hash matches.
- **Amendment:** §8.4 add test 9: `test_rng_state_equality_at_first_bid` — assert `torch.cuda.get_rng_state_all()` and `torch.get_rng_state()` are byte-identical at the first bid step in Arm A and Arm B. If they diverge, Arm B must be refactored to allocate the same buffers (β=0 doesn't mean don't allocate). Since §2.2 already requires Arm B to "compute but not use" c_forget and conscience, this is consistent.

### 3.3 Per-expert tied low-rank Σ initialization

§2.1 specifies tied low-rank shared `U ∈ R^(768×32)` + per-expert diagonal scaling r=32. The diagonal scaling vectors have initialization.

**Defect:** Initialization of the diagonal scaling vector is not specified in §2.1. FeCAM's paper initializes from unit-diagonal or from a pre-pass; the protocol neither pins it nor inherits it from a named implementation. If Jett ships with one default and Breach later evaluates under another, amendment is triggered after freeze.

- **Severity:** MEDIUM. Impact: 0.1–0.3pp.
- **Amendment:** §2.1 row "Covariance structure" — add: "Diagonal scaling initialized to 1.0 (unit). `U` initialized via PCA of a pre-pass on backbone features from 1000 held-out training images, seed `42` (fixed across all runs)." OR name the FeCAM implementation exactly and defer to it.

### 3.4 Shrinkage λ calibration

§2.1 row "Shrinkage parameter λ": "Public, fixed, calibrated once on pretraining features; NOT data-dependent." Crux-2 resolution (Astra/Sage).

**Defect:** "Calibrated once on pretraining features" — whose pretraining features? The ImageNet-21k features? The CIFAR-100 pre-task-1 features? "Pretraining" is ambiguous.

- **Severity:** LOW. Not an Arm A vs Arm B asymmetry because both arms use the same λ. But it's a reproducibility gap and a crack where future amendments can slip in data-dependent calibration.
- **Amendment:** §2.1 row specify: "λ computed once from backbone features of 1000 held-out ImageNet-21k validation images (NOT CIFAR-100), via FeCAM's shrinkage estimator formula. Value logged in `frozen_config.yaml`." If implementation uses CIFAR-100 features for calibration (plausible default), that is data-dependent in the DSIC sense and Astra crux 2 re-opens.

### 3.5 α calibration asymmetry

§2.1 row "α (execution-cost weight)": "Calibrated once against empirical median `d_M` on task-1 features. Same α used in both arms."

**Defect:** "Task-1 features" — in **which arm's** task-1? Arm A and Arm B will, after step 1, route differently (that's the point), so their task-1 feature distributions at any non-zero step diverge. If α is calibrated from Arm A's task-1 features (routed under Arm A's bid), Arm B inherits α tuned against a distribution Arm B doesn't actually see. If calibrated from Arm B's, symmetric problem. If calibrated from a *shared pre-routing pass* (e.g., all data passes through shared backbone, no routing, compute median d_M across all experts' prototypes), that is symmetric — but the protocol does not say that.

- **Severity:** HIGH. Impact: 0.3–1.0pp. If α is tuned to Arm A's data, Arm B is systematically mis-weighted.
- **Amendment:** §2.1 row: "α calibrated on a shared pre-routing pass: backbone-only forward on the entire task-1 training set, compute per-expert `d_M(x, μ_{i,c})` using initialized prototypes, take the median across all (x, i, c). This pre-pass occurs *before* any training in either arm and produces one α value written to `frozen_config.yaml`. Both arms load this α." Acceptance test 10: `test_alpha_calibration_shared_prepass` verifies this.

### 3.6 Per-task epochs chosen once, held across arms

§2.1 specifies "matched across arms" but "Jett selects once and freezes." §8.6 says deviations from LAMDA-PILOT default require Jett→Breach escalation.

**Defect:** "Jett selects once" is fine, but the rationale and value must be logged *before* pilot, not justified post-hoc. If Jett picks 10 epochs/task and the Arm-A-favorable pilot fails at 10 and passes at 20, a re-selection is a preregistration violation even if the per-task epoch count is "implementation-level."

- **Severity:** MEDIUM. This is a preregistration-discipline issue more than an asymmetry issue.
- **Amendment:** §8.5/§8.6 change: "Jett selects per-task epochs and records the value in `frozen_config.yaml` before the pilot. This value is not changed after pilot without an amendment per §13." Add to §11.1: "[ ] Per-task epoch count chosen and committed to `frozen_config.yaml` before pilot seed 42 runs."

### 3.7 Optimizer reset under ablated bid dynamics

§2.1 row "Optimizer reset": "On task end, after Fisher update, per `memory/MEMORY.md`. Applied identically in both arms."

**Defect:** Task-aware mode. Reset applies to "winning experts at task end." But *who won* differs between arms (that's the ablation). So "winning experts" is an arm-dependent set. The rule is identical (reset winners), but the *effect* is not identical across arms. This is correct for a strict ablation (the downstream consequence of the β,γ intervention includes which experts reset), but the protocol says "applied identically" which is ambiguous.

- **Severity:** LOW. Not a bug; a presentation issue.
- **Amendment:** §2.1 row: change "Applied identically in both arms" to "Rule applied identically (reset all experts that won at least one step of the just-finished task); the *set* of experts reset may differ across arms as a downstream consequence of β,γ — this is correct ablation behavior." Prevents reviewer confusion.

### 3.8 β and γ code paths

§2.2: "β and γ in Arm B must be enforced as hard-coded constants in the bid computation, not set via a hyperparameter flag." §8.4 test 1 verifies.

**Defect (already flagged in D5):** Hard-coding β=0 is correct. But Arm B still computes `c_forget` and `conscience` (§2.2) "for logging only." If the code path is `bid = α * d_M + β * c_forget + γ * conscience` with `β=γ=0`, the multiply-by-zero still evaluates `c_forget` and `conscience`, which may run CUDA kernels whose RNG state consumption is implementation-defined. If Arm A's `c_forget` path and Arm B's `c_forget` path differ by as much as a single kernel launch (e.g., Arm A caches a buffer, Arm B re-computes), the RNG streams drift.

- **Severity:** MEDIUM. Covered partly by D5's RNG-state test if added.
- **Amendment:** Add §8.4 test 11: `test_arm_b_c_forget_computation_bitwise_identical_to_arm_a` — run one forward step, capture `c_forget` tensor bit-exact and compare. If not identical at the same seed, the paths have diverged.

### 3.9 Gradient graph and memory allocation

If `β * c_forget` with `β=0` is implemented as `torch.zeros_like(c_forget)`, Arm B never builds gradient edges from `c_forget` to the bid. If implemented as `0.0 * c_forget`, Arm B does build those edges. This affects memory allocator pattern and therefore CUDA kernel selection (heuristics depend on allocator state).

- **Severity:** LOW (likely within Δ=1.0pp noise) but non-zero.
- **Amendment:** §8.4 test 12: `test_arm_b_gradient_graph_excludes_cforget_and_conscience` — verify that `bid.grad_fn` in Arm B does not reference `c_forget` or `conscience`. Enforced as a structural constraint: Arm B's `bid = α * d_M` literally, not `α * d_M + 0.0 * c_forget + 0.0 * conscience`.

---

## 4. Matched-Compute Audit

### 4.1 FLOP-match is near-trivially satisfied

§3.1 defines matched FLOP within ±5%. Per §2.1, Arm B **computes** projected-gradient EWC (one shared backward + per-expert dot products) but multiplies by zero in the bid. So `F_bid_mechanism` and `F_fisher_projection` are strictly equal across arms by construction (same tensors computed, same kernels launched). Then `F_total^A − F_total^B` is zero in expectation, and the ±5% matching is automatic.

**Defect:** This makes the matching criterion decorative, not informative. A reader reasonably concludes "matched FLOPs" means "I've compared under equal compute budget in a meaningful sense." What it actually means here is "I've computed the same FLOPs and then thrown away the bid contribution in one arm." Arm A and Arm B have matched FLOPs but Arm A *uses* more of them.

- **Severity:** MEDIUM. Not a confound — it's actually the right design for a strict ablation — but the *claim* "matched-FLOP" is misleading. If Breach publishes "MoB wins under matched FLOPs," the reader infers a stronger statement than the protocol delivers.
- **Amendment:** §3.1 reframe: "Both arms execute identical forward+backward+Fisher computation; Arm B sets β=γ=0 in the bid formula only. FLOP equality across arms is therefore definitional. The ±5% criterion is a sanity check against implementation-level drift, not a matching claim." Update any downstream paper language accordingly.

### 4.2 fvcore FLOP accounting validity

§3.3 uses `fvcore.nn.FlopCountAnalysis` on the forward graph × measured steps.

**Defect:** fvcore does not count the bid-mechanism or Fisher-projection ops, because those are outside the standard forward graph. §3.3 adds them analytically via the formula `n_steps × B × (2·r·d + E·r + E·r_f)` from Killjoy R2 §2.1. This formula is not independently verified in the protocol, and it assumes tied low-rank with the exact dimensions Jett is specifying. If Jett's implementation diverges (e.g., r=64 in a sensitivity sweep), the analytical formula is stale.

- **Severity:** LOW. The FLOP numbers are not used for any binding decision (per §4.1 above).
- **Amendment:** §3.3 add: "If any of {r, d, E, r_f, B} change from the frozen-config values, FLOP accounting formula must be re-derived and committed in the amendment."

### 4.3 Wall-clock as secondary

§3.2: wall-clock "reported as secondary." §3.2 also says "if Arm A wall-clock falls *below* Arm B's, the run is suspect."

**Defect:** §3.2 does not specify what "suspect" triggers — investigate, rerun, escalate? This is an unspecified branch in outcome handling.

- **Severity:** LOW.
- **Amendment:** §3.2: "If `wallclock^A < wallclock^B × 0.95` (i.e., Arm A more than 5% faster wall-clock than Arm B), investigate — expected asymmetry is `wallclock^A ≥ wallclock^B` because Arm A has non-trivially used `c_forget` and `conscience`. Investigation: check GPU-time profile per op; if no clear cause, rerun the affected seed. If repeatable, escalate to Breach + Nosh."

### 4.4 The wall-clock research-claim trap

If Arm A is (expected) 2–5% slower wall-clock at matched FLOPs (per Killjoy R2), and Avg Acc Δ̄ = 1.1pp in favor of Arm A, the *FLOP-adjusted* claim is "MoB uses same FLOPs for 1.1pp gain," but the *wall-clock-adjusted* claim is "MoB uses 2–5% more wall-clock for 1.1pp gain." A reviewer can reframe this as a modest improvement under realistic latency budget.

- **Severity:** LOW. Framing concern, not a bug.
- **Amendment:** §6.3 list "FLOP-adjusted accuracy" *and* "wall-clock-adjusted accuracy" as secondary. Pre-commit to reporting both in any paper so the reviewer doesn't catch this post-hoc.

---

## 5. Fisher-Match Gate Audit

### 5.1 What the protocol measures (§4.4)

"max(F̄^A(s), F̄^B(s)) / min(F̄^A(s), F̄^B(s)) ≤ 2" per seed. Seeds failing this are excluded. If fewer than 8 of 10 pass, rerun with clamp recalibration.

### 5.2 This does not handle the R1 threat

See §1.3 above. The check is *across arms within a seed*, not *across seeds within an arm*. By §2.1 protocol invariants (same seed → same init → same Fisher computation path), arms should agree to much better than 2× by construction. The check is detecting implementation bugs (e.g., an accidental `Fisher.to(device_A_only)`), not the 18× R1 threat.

### 5.3 Fisher magnitude temporal drift

Fisher at task 1 ≠ Fisher at task 20. §4.4 logs "per-task, per-expert, per-step" Fisher magnitude but the inclusion criterion collapses across (i, t) into `F̄(s) = mean over (i, t)`. A seed where Fisher ratio is 1.5× at task 1 but 5× at task 15 — concentrated drift in one arm — is averaged away.

- **Severity:** MEDIUM. Impact: if one arm is Fisher-unstable at late tasks, the late-task Avg Acc contribution to `A_T = (1/T) Σ a_{i,T}` is biased. Expected 0.2–0.5pp.
- **Amendment:** §4.4 also report `max_t ratio(F̄^A_t, F̄^B_t)` per seed. If *any* task's per-task ratio exceeds 3×, flag the seed for inspection. Trigger rerun if ≥2 seeds flag.

### 5.4 Paired design's partial inoculation

Paired-on-seed cancels Arm-A-vs-Arm-B Fisher correlation. It does NOT cancel within-arm-across-seed Fisher variance. If seed 42 yields Fisher magnitude 10× the clamp floor and seed 49 yields 0.5× the clamp floor, Arm A's β·c_forget contribution is scaled differently across those seeds — the treatment effect is heteroscedastic across seeds, violating the i.i.d. Gaussian assumption of the paired t-test (§6.1) and informing the BCa bootstrap's acceleration term.

- **Severity:** HIGH. Impact: could either inflate or deflate the BCa lower bound depending on which seeds land where in the Fisher distribution. The Shapiro-Wilk check in §6.1 will not catch this if n=10 is too small to detect non-Gaussianity.
- **Amendment:** §4.4 add a final check: `std(log F̄^A(s)) / mean(log F̄^A(s))` (CV of log-Fisher across seeds) ≤ preregistered threshold (propose 0.5). If CV exceeds threshold, the paired Δ̄ is heteroscedastic-in-Fisher-draw and the BCa CI is reported with an annotation: "Δ̄ reflects the Fisher-magnitude distribution of seeds {42..51}; generalization to a fresh seed pull is weaker than the nominal CI suggests."

### 5.5 Clamp recalibration is analyst-discretion

§4.4: "If fewer than 8 of 10 seeds satisfy this, rerun with Fisher-clamp recalibration per `memory/MEMORY.md` §EWC-Fisher-Clamping."

**Defect:** "Fisher-clamp recalibration" is not a preregistered rule. `memory/MEMORY.md` says `min=0.1` is the fix, but doesn't prescribe how to pick a new value if 0.1 fails at CIFAR-100 scale. Analyst discretion here is a degree of freedom.

- **Severity:** HIGH. Impact: Breach could pick the clamp that makes the pilot succeed, which is post-hoc tuning dressed as preregistration.
- **Amendment:** §4.4 add: "Clamp recalibration: try clamp values {0.1, 0.3, 1.0, 3.0} in that order; the first value that produces ≥8 of 10 seeds satisfying the 2× criterion is adopted. The adopted value is logged as an amendment per §13. No other values may be tried."

---

## 6. Statistical Analysis Plan Audit

### 6.1 What is Δ_s?

§6.1 defines `Δ_s = A_T^A(s) − A_T^B(s)`. `A_T` per §1.4 is "Final Average Accuracy on the 20-task split after task 20." The paired design produces `{Δ_s : s ∈ {42..51}}`, n=10.

**Ambiguity:** `A_T = (1/T) Σ_i a_{i,T}` per §1.4. This is the Lopez-Paz-Ranzato Avg Acc. Is it computed from a class-balanced micro-accuracy (pooled test set), or an arithmetic average of per-task accuracies (macro-accuracy over tasks)? §1.4 cites L-P-R convention, which is the latter. Confirm.

- **Severity:** LOW if Jett implements correctly; MEDIUM if Jett implements pooled micro-accuracy and calls it "Avg Acc."
- **Amendment:** §1.4 add the formula explicitly: "A_T = (1/T) * Σ_{i=1}^{T} a_{i,T}, where a_{i,T} is the accuracy on task i's test partition evaluated at end of task T. Macro-average over tasks."

### 6.2 Resample count

§6.2: "10,000 resamples." Consistent with §1.2. Confirmed.

### 6.3 Primary test is one-sided; FAIL region requires two-sided

This is the **D1 critical defect**.

§1.2:
- PASS = "one-sided 95% lower CI on E[Δ] ≥ Δ_practical" AND "paired one-sided t-test rejects H_null at α=0.05."
- TIE = "95% two-sided CI contains zero" OR "one-sided lower bound < Δ_practical but ≥ −Δ_practical."
- FAIL = "95% two-sided CI strictly below zero."

The PASS test is one-sided: H_null: E[Δ] = 0 vs H_alt: E[Δ] > 0. This test has **no rejection region for E[Δ] < 0**. A one-sided test that fails to reject H_null says "insufficient evidence that MoB > FeCAM"; it does not say anything about whether FeCAM > MoB.

§1.2's FAIL clause quietly switches to a **two-sided** CI to detect "FeCAM beats MoB." §6.2 says BCa reports "95% one-sided lower bound" for the gate AND "95% two-sided CI for robustness." So the two-sided CI is computed. Fine — but:

**Defect 1 (D1):** The FAIL condition ("95% two-sided CI strictly below zero") is operationally almost unreachable at n=10. With σ_Δ ≈ 1pp (the pilot's optimistic target), the 95% two-sided CI has half-width ≈ 2.26·σ_Δ/√10 ≈ 0.71pp. For the upper bound of the two-sided CI to fall strictly below zero, we need Δ̄ < -0.71pp. This requires FeCAM to beat MoB by ≥0.71pp mean, which is plausible but not guaranteed even when FeCAM is genuinely better. At σ_Δ = 1.75pp (the "marginal" band in §6.5), half-width is ~1.25pp, so FAIL requires Δ̄ < -1.25pp — which means MoB has to lose by more than 1.25pp before FAIL fires. FAIL is a high-bar condition.

The deeper structural issue: **Breach has conflated two test frameworks.** The primary test is a one-sided superiority test. The preregistered FAIL outcome requires an inferiority test. These are not the same test, and Breach has not preregistered the inferiority test explicitly. A clean preregistration would use one of:

**Option A (two-sided with equivalence region):**
- H_null: E[Δ] = 0 vs H_alt: E[Δ] ≠ 0, α=0.05 two-sided.
- PASS: 95% two-sided CI lower bound ≥ Δ_practical.
- FAIL: 95% two-sided CI upper bound ≤ −Δ_practical.
- TIE: everything else.
- All three regions are mutually exclusive and exhaustive.

**Option B (one-sided primary + TOST inferiority):**
- Primary one-sided t-test for PASS (as already specified).
- Add a TOST (two one-sided tests) equivalence test at bounds ±Δ_practical for TIE determination.
- FAIL: separate one-sided t-test at α=0.05 on H_null': E[Δ] = 0 vs H_alt': E[Δ] < −Δ_practical. Explicit.

**Option C (preregister all three as separate tests):** three one-sided tests (superiority, inferiority, equivalence-boundary). Report all three; decision rule mutually exclusive.

Current §1.2 is a hybrid that sort-of-does Option A without naming it and sort-of-does Option B without specifying the FAIL test statistic. Worse, the one-sided test in §6.1 is the only test the protocol actually specifies a p-value for; the FAIL branch has no p-value at all, only a CI criterion. A CI-only FAIL with no corresponding hypothesis test is non-standard.

- **Severity:** **CRITICAL**. Impact: the FAIL branch as written will almost never fire. The gate effectively has two outcomes — PASS and NOT-PASS — not three. That's fine if Breach is willing to fold FAIL into TIE and treat both as "auction not demonstrated." But §9.3 says FAIL triggers "terminate the routing-mechanism thesis; no CIFAR reruns" while TIE pivots to continual-FT. These are different project outcomes and the gate cannot operationally distinguish them.
- **Amendment (BINDING, Option B recommended):** §1.2 and §6 rewrite:
  - PASS: 95% BCa one-sided lower bound on E[Δ] ≥ Δ_practical = 1.0pp.
  - FAIL: 95% BCa one-sided *upper* bound on E[Δ] ≤ −Δ_practical. (i.e., use the symmetric inferiority test.)
  - TIE: neither above.
  - Shapiro-Wilk, Wilcoxon: as secondary, per §6.1/§6.3.
  - §9.3 updated to match.

### 6.4 Multiple-comparison bookkeeping

§6.4: "Phase-1 gate is a single a-priori comparison. No correction."

**Defect:** §1.5 lists 8 secondary metrics. §6.4 says "F_T, BWT, routing entropy, Gini: reported without inferential testing; descriptive only." But §5.1 says ciFAIR is a "robustness check" and the protocol pre-commits to escalating if the sign flips between CIFAR-100 and ciFAIR. Escalation is a decision rule. Is the ciFAIR test an inferential test? If so, it is a second comparison, and the family-wise error rate is now 1 − (0.95)² = 9.75%, not 5%.

- **Severity:** MEDIUM.
- **Amendment:** §6.4 clarify: "The ciFAIR cross-check is descriptive, not inferential. A sign flip between A_T^{CIFAR-100} and A_T^{ciFAIR-100} *does not* modify the gate decision; it triggers a post-hoc investigation whose outcome is reported in the gate memo but does not override the primary test result." OR: conduct a joint test with Bonferroni-2 (α=0.025 each); accept the 25% loss of power in exchange for FWER control.

### 6.5 Δ_practical = 1.0pp provenance

§1.1 justifies Δ_practical by reference to Cypher's audit §6.6 ("1.5–3pp variance across class-order seeds"). 1.0pp is under the low end of that range.

**Defect:** If class-order seed variance alone is 1.5pp, a paired-difference-CI lower-bound-above-1.0pp criterion can be met by an effect that does not exceed class-order seed variance. The rationale "1.0pp is robustly separable from pre-existing protocol noise" is backwards — an effect under the noise floor is not robustly separable from it.

- **Severity:** MEDIUM. But defensible if Breach argues the *paired* variance is lower than the *unpaired* class-order variance (which it is, by design). Breach should say so explicitly.
- **Amendment:** §1.1 rewrite rationale: "1.0pp is smaller than the 1.5–3pp unpaired class-order variance reported in Cypher §6.6. The paired design cancels class-order variance (arms share class order per seed), so paired σ_Δ is expected to be substantially smaller. The 1.0pp threshold is chosen to be detectable under a paired design with σ_Δ ≤ 1.16pp (per §6.5), not to exceed unpaired class-order variance. Unpaired robustness is reported as secondary."

### 6.6 Power analysis at n=3 pilot

§6.5 uses a χ² upper bound for σ_Δ from n=3 (df=2). The 80% upper CI multiplier at df=2 is approximately sqrt(2/χ²_{0.20,2}) = sqrt(2/0.446) ≈ 2.12. So a pilot point-estimate σ_pilot = 0.5pp has 80% upper bound ≈ 1.06pp. Pilot σ_pilot = 0.8pp has 80% upper bound ≈ 1.70pp.

**Defect:** The pilot σ upper bound is extremely noisy. The go/no-go decision at §7.2 is therefore extremely sensitive to pilot draws. A pilot σ_pilot = 0.6pp → upper bound 1.27pp → "marginal" band → proceed with marginal annotation. A pilot σ_pilot = 0.55pp → upper bound 1.17pp → "marginal" band. A pilot σ_pilot = 0.82pp → upper bound 1.74pp → "marginal." Tiny movement in pilot σ pushes the decision across bands.

- **Severity:** HIGH. Impact: the pilot's σ estimate is a coin-flip input to a binding decision.
- **Amendment:** §6.5/§7.2: require n_pilot = 5 minimum (not 3). At df=4 the χ² upper-80% multiplier is ~1.62, much better-behaved. Cost: 2 extra pilot runs × ~4hr/run = +~8 GPU-hours.

### 6.7 The σ_Δ = 1.75pp "marginal" threshold

§6.5: `1.16 < σ_Δ ≤ 1.75pp` → "marginally powered; proceed but annotate."

**Defect (D3):** Compute actual power at σ_Δ = 1.75pp, n=10, Δ=1.0, α=0.05 one-sided:
- Non-centrality parameter δ = 1.0 / (1.75/√10) ≈ 1.807.
- Critical t at df=9: t_crit = 1.833.
- Power = P(T_9(δ=1.807) > 1.833) ≈ **0.46** (via noncentral t CDF).

That is 46% power. Not 80%, not even 60%. The "marginal" band explicitly proceeds under <50% power, which is strictly worse than a coin flip at detecting Δ=1.0pp.

At σ_Δ = 1.16pp (the PASS band upper edge):
- δ = 1.0 / (1.16/√10) ≈ 2.726.
- Power = P(T_9(2.726) > 1.833) ≈ **0.78**. Confirmed ≈80%.

So the "marginal" band is actually the "below 80% power all the way down to 46%" band. It should not be proceed-with-annotation; it should be escalate.

- **Severity:** **HIGH.** Impact: fraction of gate-pass probability. If true σ_Δ lands in this band and true Δ = 1.0pp, the gate has 46–78% chance of correctly passing and 22–54% chance of a Type-II error — a hidden fail when MoB is actually winning. The protocol then terminates the routing-mechanism thesis under §9.3 FAIL, or pivots under §9.2 TIE, on false evidence.
- **Amendment:** §6.5 retime the bands:
  - σ_Δ upper-80% ≤ 1.16pp → GO (≥80% power).
  - 1.16 < σ_Δ upper-80% ≤ 1.30pp → marginally powered (~70% power); proceed only with Nosh sign-off.
  - σ_Δ upper-80% > 1.30pp → ESCALATE.

Or alternatively: pre-commit to n=20 (double seed budget to {42..61}) as the backup, not Nosh-approved case-by-case. At σ_Δ = 1.75pp, n=20 gives δ = 1.0/(1.75/√20) = 2.556, and power ≈ 0.78. n=20 recovers 80% power up to σ_Δ ≈ 1.75. This is the clean fix.

---

## 7. Analyst Degrees of Freedom Audit

### 7.1 Task-config choice

§4.1 commits to 3 configs (5T, 10T, 20T). §1.4 says gate decision uses 20T. §4.2 says 20T alone is sufficient for the gate.

**Defect:** If 20T ties and 10T passes, does Breach pivot under §9.2 (20T is the gate) or declare PASS on the 10T result? §4.2 prioritization is clear (20T first, then 5T, then 10T), but the outcome-branch sections (§9.1–9.3) are all defined at 20T only. They do not say "if 10T passes and 20T doesn't, what happens."

- **Severity:** MEDIUM. A reviewer could later accuse Breach of cherry-picking 10T if it were the only config that passed.
- **Amendment:** §9 add: "If 20T yields TIE or FAIL but 5T or 10T yields PASS, this is reported as a scale-dependent effect; the 20T gate result governs the routing-mechanism-thesis decision per synthesis §0. The 5T/10T results inform a follow-up study at that task count but do not override the 20T gate." Bind the 20T result as primary.

### 7.2 Seed re-rolls

§4.1: seeds `{42..51}`, contiguous, preregistered.

**Defect:** If a run fails due to hardware (OOM, CUDA error, preemption), is the re-run a new seed or the same seed with the same hardware? §4.1 doesn't say.

- **Severity:** LOW. Likely intended: same seed, rerun. But a cheese path exists: "seed 47 failed; we substituted seed 52" is defensible-sounding but breaks pairing with Arm B at seed 47.
- **Amendment:** §4.1 add: "If a seed fails (hardware OOM, preemption, bug), the same seed is rerun on the same (or equivalent, per §3.4) hardware. Seed values are never substituted. If a seed cannot be completed after 3 attempts, it is reported as 'failed' in the summary and the primary analysis is run on the completed subset with the n reduced accordingly; the failure is disclosed in the gate memo."

### 7.3 Compute-match tolerance slippage

§3.3 criterion: `| F_total^A − F_total^B | / F_total^B ≤ 0.05`.

**Defect:** Per §4.1 above, FLOP match is auto-satisfied. But if a pilot shows 6% (due to counting method error, not implementation difference), does the gate-pass require an adjustment, and in which direction? "Adjust per-task epoch budget uniformly until it holds, then the protocol is frozen" per §3.3. If both arms have the same epoch budget (they do), adjusting it uniformly does not change the cross-arm ratio. This is an incoherent fix.

- **Severity:** LOW. Artifact of §4.1's matched-FLOP-is-definitional issue.
- **Amendment:** §3.3 fix: "If the pilot shows > 5% ratio, diagnose the cause (likely accounting bug, since compute is definitionally matched per §2.1). Fix the accounting or exclude the miscounted term from the ratio. Do not adjust epoch budget."

### 7.4 Freeze-date amendment loophole

§13: "Any amendment post-freeze that tightens or loosens the gate rule invalidates all completed runs on the affected configuration."

**Defect:** "Pre-data amendment" vs "post-data amendment" distinction — if an amendment is made AFTER the pilot but BEFORE the full run, that's post-pilot-data but pre-full-run-data. §13 does not explicitly classify this case.

- **Severity:** LOW. But a cheese path: "the pilot showed X, we amended to account for X, we rerun." If X is a bug, this is correct. If X is a σ draw in the wrong band, this is post-hoc adjustment.
- **Amendment:** §13 add: "Amendments between pilot completion and full-run launch are classified as post-data amendments. They invalidate any full-run seed already in progress. Amendments that alter the primary test (§6.1, §6.2) or the gate decision rule (§1.2) at this stage are not permitted except via Nosh escalation with explicit KAY/O countersign."

### 7.5 Paper metric primacy

**Defect:** §1.4 specifies the primary metric; §1.5 lists 8 secondaries. But the downstream paper is not scoped in this document. A PASS at 20T with Δ̄ = 1.2pp could be de-emphasized in the paper in favor of, e.g., a 3pp BWT improvement if that frames better. This is not preregistered.

- **Severity:** MEDIUM.
- **Amendment:** §1.4 add: "Any downstream paper, preprint, or presentation describing the results of this gate MUST report the primary test (Avg Acc BCa one-sided lower bound at 20T CIFAR-100) as the headline result. Secondary metrics may be added but may not substitute. This is a preregistration condition and violation invalidates the preregistration claim."

---

## 8. Dataset-Integrity Challenges

### 8.1 ciFAIR

§5.1 pre-commits to both CIFAR-100 and ciFAIR-100 evaluation. §11.1 pre-launch checklist requires ciFAIR download and hash verification. §8.4 test 7 verifies the loader runs.

**Verdict:** Clean. Breach did not use the "if harness supports" loophole I was prepared to attack. Approve as-is.

- **Minor residual:** §5.1 escalates on sign-flip. This is a descriptive escalation (not changing the gate) per §6.4 amendment above.

### 8.2 Upsample implementation

§5.3: "bicubic interpolation (`torchvision.transforms.functional.resize(..., interpolation=InterpolationMode.BICUBIC)`), antialias=True."

**Verdict:** Mostly clean. D6 flagged that different bicubic implementations (PIL, cv2, torchvision) differ. §5.3 pins torchvision explicitly, which is adequate.

- **Minor:** `antialias=True` is default in recent torchvision but changed defaults across versions. Pin the torchvision version in `env.json` (§8.2 lists env.json fields; torchvision version should be there explicitly).
- **Amendment:** §8.2 add "torchvision_version" to env.json fields.

### 8.3 Class ordering

§5.4: "LAMDA-PILOT default random-permutation class ordering, seeded by the run seed."

**Defect:** "Seeded by the run seed" — does this mean LAMDA-PILOT's class-order function is deterministic given the same seed? I cannot verify without reading LAMDA-PILOT code (KAY/O scope prohibits loading impl). If LAMDA-PILOT uses its own internal seed that drifts across versions, the class order varies across LAMDA-PILOT commits.

- **Severity:** MEDIUM. Cross-arm: same as long as both arms share LAMDA-PILOT version, which §11.1 checklist requires. Cross-reproducibility: high risk.
- **Amendment:** §5.4 add: "The class-order permutation for each seed is logged explicitly in `run.json` under `class_order: [int, ...]` (the permutation of 0..99 used by that run). A post-hoc consistency check verifies that Arm A seed `s` and Arm B seed `s` share `class_order`."

---

## 9. Outcome-Branch Integrity

Covered in §6.3 / D1 above. This is the single most amendment-forcing defect.

**Summary of the defect:** §9.2 TIE and §9.3 FAIL are defined under a two-sided CI framing while the primary test in §6.1 is one-sided. The FAIL branch requires a test that the protocol does not operationalize (it uses a CI criterion, not a test statistic, and the CI it uses is "for robustness" per §6.2). In practice FAIL is almost unreachable at n=10 unless FeCAM wins by ≥ ~0.7–1.25pp depending on σ_Δ.

**Proposed fix (binding amendment per D1):** convert the entire §1.2/§9 framework to symmetric one-sided BCa tests:
- PASS: one-sided BCa lower bound on E[Δ] ≥ +Δ_practical.
- FAIL: one-sided BCa upper bound on E[Δ] ≤ −Δ_practical.
- TIE: otherwise.

All three regions mutually exclusive. Both superiority and inferiority tested at α=0.05 per side; the Type-I error for the *composite* decision is 0.05 + 0.05 = 0.10 if Breach wants to be strict, which can be tightened to α=0.025 per side for FWER 0.05. This FWER concern is worth naming in the amendment.

---

## 10. Jett-Facing Contract Audit

§8.4 lists 8 acceptance tests. I add the following (numbering continues):

- **9. `test_rng_state_equality_at_first_bid`** (per D5 in §3.1): Torch-CPU and Torch-CUDA RNG states byte-identical at the first bid call in Arm A vs Arm B at the same seed.
- **10. `test_alpha_calibration_shared_prepass`** (per §3.5): α is computed once via a shared pre-routing pass and both arms load the same α value from `frozen_config.yaml`. Assertion: `config_A.alpha == config_B.alpha` and `alpha_source == "shared_prepass"`.
- **11. `test_arm_b_c_forget_computation_bitwise_identical_to_arm_a`** (per §3.8): after one forward pass at identical seed, `c_forget` tensors in Arm A and Arm B are bit-identical.
- **12. `test_arm_b_gradient_graph_excludes_cforget_and_conscience`** (per §3.9): `Arm B.bid.grad_fn` does not trace back to `c_forget` or `conscience` tensors.
- **13. `test_arms_disagree_on_at_least_one_input`** (per D8): run both arms for 10 training steps on pilot seed 42. At least one step must have a different winning expert in Arm A vs Arm B. (If they never disagree, β and γ are not doing anything and the gate is uninformative regardless of outcome.)
- **14. `test_losses_bitwise_identical_at_t0`** (per §10 in prompt): at step 0 (before any gradient step in either arm), the per-sample loss on a fixed batch must be bitwise identical across Arm A and Arm B. (Stronger than the weight-hash test #3, which only compares weights.)
- **15. `test_flop_accounting_unit_test`** (per §10 in prompt): a unit test that runs a known-FLOP synthetic workload and verifies that `flop_log.json`'s computation matches the known answer within 1%. Protects against fvcore counting bugs.
- **16. `test_class_order_logged_per_run`** (per §8.3): `run.json` contains `class_order: List[int]` of length 100, and both arms at the same seed have identical `class_order`.

All 16 tests must pass before pilot seed 42 launches.

---

## 11. Verdict

### 11.1 Decision

**APPROVE WITH AMENDMENTS.**

### 11.2 Binding amendments (must be applied before 2026-04-26 freeze)

In ranked order by severity:

1. **D1/§9: Fix the outcome-branch test framework.** Convert §1.2/§9 to symmetric one-sided BCa tests (PASS: lower bound ≥ +Δ_practical; FAIL: upper bound ≤ −Δ_practical). Without this, the FAIL branch is operationally unreachable and the protocol has two outcomes where it claims three.
2. **D2/§4.4: Add within-arm-across-seed Fisher-stratification check.** The current 2× cross-arm check does not address the R1 18× threat. Add `std(log F̄^A(s)) / mean(log F̄^A(s)) ≤ 0.5` as an inclusion criterion. Without this, a PASS leaves R1 unfalsified.
3. **D3/§6.5: Tighten the σ_Δ escalation threshold.** Current "marginal" band drops power to 46%. Either cap at σ_Δ ≤ 1.30pp or pre-commit n=20 as the backup. Without this, the gate can fail on 50%-powered evidence while branding as "marginally powered."
4. **D4/§2.1 α calibration:** specify α is calibrated on a shared pre-routing pass, not per-arm task-1.
5. **D5/§3.1+§8.4:** pin RNG state equality (not just weight equality) at first bid, add test 9; pin DataLoader worker seeds (§3.1 amendment).
6. **D7/§1.2+§6:** if Breach chooses Option B (not Option A), add the explicit TOST / inferiority-test specification. D1 amendment subsumes this; list for completeness.

Advised (not strictly blocking freeze but strongly recommended):

7. D6/§5.3: pin torchvision version in env.json.
8. D8/§8.4: add `test_arms_disagree_on_at_least_one_input` (test 13).
9. §3.2: specify "suspect" wall-clock response.
10. §5.4: log class_order per run.
11. §6.6: increase pilot n from 3 to 5.
12. §7.2: specify seed re-roll rule.
13. §7.5: paper metric primacy clause.

### 11.3 What a PASS actually resolves for R1

Even with all binding amendments applied, a PASS (Δ̄ ≥ +1pp lower bound) on these 10 seeds at 20T CIFAR-100 with ViT-B/16 resolves the R1 threat **only at this configuration, only under this sample of the Fisher-magnitude distribution**. The stronger R1-threat refutation — "the auction is mechanistically irreducible" — requires:

- β-only ablation (set γ=0 but β active) to isolate which term is load-bearing.
- A 5T vs 10T vs 20T trend showing the auction's advantage grows with task count (predicts a mechanism effect, not a config artifact).
- A Fisher-stratified analysis showing Δ̄ > 1pp holds conditional on favorable, median, and unfavorable Fisher draws (rules out Fisher-variance-driven outcome).

The Phase-1 gate does NOT include β-only and α-only ablations. Those are promised in the synthesis §5 Phase 2/3 path but not preregistered as Phase-1 gate conditions. A PASS therefore **licenses proceeding to Phase 2/3 but does not refute R1**. Breach should state this explicitly in §9.1.

### 11.4 What a TIE actually resolves for R1

A TIE (neither CI bound excludes its threshold) is *consistent* with R1 but not proof of R1. The protocol's §9.2 response (pivot to continual-FT framing) is the correct operational response because the thesis "auction is the load-bearing contribution" is not supported. R1 is rewarded but not confirmed.

### 11.5 What a FAIL actually resolves for R1

Under the D1 amendment, a FAIL (upper bound ≤ −1pp) is **stronger** than R1 predicted. R1 predicted tie-or-narrow-loss; FAIL says FeCAM-Router materially beats MoB. This confirms the auction is net-negative at this config, which is beyond the R1 null. The §9.3 termination response is correct.

### 11.6 Can freeze proceed on 2026-04-26?

**Yes, conditional on amendments D1/D2/D3 being applied.** D1 is a clean diff to §1.2/§6/§9 that Breach can author in under 2 hours. D2 is an additional §4.4 clause and a post-hoc analysis step; no implementation change. D3 is a single-line adjustment to §6.5 thresholds. Total amendment effort: <1 day for Breach + Nosh sign-off.

**No, if these amendments are not applied.** The protocol as written:
- cannot operationalize its own FAIL branch (D1);
- declares the R1 threat "handled" while checking a different quantity (D2);
- proceeds with <50% power in a band it labels "proceed" (D3).

Each of these individually is sufficient to disqualify the gate from deciding a program-survival question. Running the protocol in its v1.0 form and declaring PASS/FAIL on its output is not an honest preregistration.

### 11.7 One-line bottom line

Breach wrote a tight protocol. The scaffolding is right; the hypothesis-test framework is broken at its joints, and the Fisher-match gate is a cosmetic check misbranded as the R1 mitigation. Three small amendments and this is a publishable preregistration. Without them, the gate is a theater of rigor.

---

*End of red-team memo. KAY/O. 2026-04-19.*

*Claims must hold under confounders and stronger baselines. Gates must distinguish their three outcomes. Mitigations must mitigate the threat they name.*
