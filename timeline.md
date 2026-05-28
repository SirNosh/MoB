# timeline.md — MoB Research Ledger

> **Private state file. Governed by global directive in `C:\Users\devya\.claude\CLAUDE.md`:
> this file is in `.gitignore` and MUST NEVER be pushed to GitHub or any public repo.**

> **Purpose**: the chronological log of experiments, executions, ablations, file creation,
> and cognitive reasoning. Append a new entry after every significant execution, test
> run, new file creation, or debugging session.
> For the stable blueprint, see `project.md`.

---

## Format for new entries

```
## [YYYY-MM-DD HH:MM] <title>

**What**: <one-line description>
**Why**: <motivating hypothesis / what it tests>
**Config / command**: <exact invocation if a run>
**Outcome**: <result, metrics, link to artifact>
**Interpretation**: <what it means in the research context>
**Follow-up**: <next step, open question, or "closed">
**Files**:
  - created: <path> — <purpose, why new file needed>
  - modified: <path> — <change>
```

---

## 2026-04-19 — Phase 0 → Phase 1/2 transition

### [2026-04-19 ~04:15] Jett HPC sprint return + Nosh post-sprint config correction

**What**: Jett delivered the complete `HPC/` gate-runner package (13 files,
2,629 LOC). Nosh reviewed on return, caught one pre-freeze config defect,
and fixed inline without requiring a second Jett round.

**Jett deliverables (verified file-by-file)**:
  - `HPC/README.md` (108 LOC), `HPCguide.md` (253 LOC), `requirements.txt`
  - `HPC/slurm/run_gate.sh` (96 LOC) — array 0–39, `(seed, arm) = (42+N//2, A|B)`
  - `HPC/gate_runner/` 9 modules:
    - `run_arm.py` (524 LOC) — full task loop: seed → frozen ViT-B/16 →
      K=4 experts → α pre-pass on task-0 → T tasks × 20 epochs × auction →
      train winner only → end-of-task Fisher + prototype + per-expert optimizer
      reset → final 100-class FeCAM eval → run.json emit.
    - `fecam_core.py` (275 LOC) — byte-equivalent port of
      `dipamgoswami/FeCAM @ e33f39d1:models/base.py` `shrink_cov`,
      `normalize_cov`, `_tukeys_transform`, `_mahalanobis`. Upstream bodies
      preserved verbatim in docstrings for audit. Uses `torch.linalg.pinv`.
    - `auction.py` (317 LOC) — bid `α·d_M + β·forget + γ·conscience`;
      Arm B hard-codes β=γ=0 and short-circuits forget-cost collection to
      keep Fisher ops out of Arm B's bid graph (test 9 verifies).
    - `experts.py` (279 LOC) — LoRA(r=8) Q/K/V applied as CLS-token linear
      correction + FFN bottleneck 768→128→768 + head, ~310K trainable ≈ §2.1.
      `assert_backbone_fully_frozen` structural guard.
    - `data.py` (195 LOC) — LAMDA-PILOT `RandomState(1993)` class ordering,
      bicubic 32→224, `seed_worker` + seeded `Generator`.
    - `ewc.py` (179 LOC) — online EWC with Fisher `clamp_min=0.1` load-bearing.
    - `utils.py` (221 LOC) — 5-source seeding, canonical YAML hash,
      state-dict hash, tensor hash, run.json schema helper.
    - `config.yaml` (162 LOC) — every §2.1 invariant + §2.3 T1-T7 + §4.6 + §4.7
      pinned; SHA-256 emitted in every run.json.
    - `tests/test_acceptance.py` (614 LOC) — **all 20 tests from §8.4**,
      classified `[PRE-RUN]` (synthetic-fixture, pass before pilot) or
      `[POST-RUN]` (needs `results/gate/*.json`; skips cleanly pre-pilot).

**Jett-flagged divergences from sprint brief (all correct calls)**:
  - **K=4, not K=8** — Dev's sprint brief said K=8; protocol §2.1 pins K=4
    load-bearing per `memory/MEMORY.md` + `feedback_4experts.md`. Jett
    correctly followed protocol. Side-effect: project.md §2 S2 row was
    ambiguous on K (the "Experts" column was occupied by adapter
    architecture, not count) — Nosh added explicit `K=4` tag to that row
    post-sprint so there's no residual doc drift.
  - **Arm B = Arm A with β=γ=0** (not single-expert FeCAM) — protocol
    §1.3 + §2.2 unambiguous. Resolved per protocol.
  - **LoRA-QKV location** — applied as CLS-token linear correction (frozen
    backbone) rather than inside each block's QKV proj. Expressively
    equivalent for a frozen encoder, keeps DSIC-safety of backbone invariance.
    Acceptable under §2.1 "~300K trainable" row.
  - **No data aug** per sprint brief vs §2.1 LAMDA-PILOT-default (crop+flip)
    — matched across arms either way so does not bias A-vs-B. Flagged only.

**Nosh post-sprint fix (one defect caught on review)**:
  - **Task partition default**: Jett defaulted `data.num_tasks=10,
    classes_per_task=10` per Dev's sprint brief. Protocol §2.1 "Headline
    config: 20T×5C" + §1.4 primary metric "A_T on the **20-task split**" +
    §1.7 paper metric primacy make 20T×5C load-bearing for the preregistered
    gate. Running 10T would invalidate §1.4's registered statistic because
    A_10 ≠ A_20. Fix: flipped `config.yaml` `num_tasks: 10→20`,
    `classes_per_task: 10→5`. Code is partition-neutral (both values are
    config-driven throughout `data.py` and `run_arm.py`, no hardcoded 10s),
    so the flip is a clean 2-line edit. No Jett round-trip needed. Updated
    `HPC/README.md` compliance-flag #2 + `HPC/gate_runner/config.yaml` note
    to match. Wall-clock per seed is comparable (20T×5C ≈ 4000 optimizer
    steps, 10T×10C ≈ 8000 — but 20T×5C has half the samples/task so the
    SGD-step counts are close).

**Jett deferred to post-freeze (documented, out of scope)**:
  - fvcore FLOP decomposition emitted into run.json (test 12 passes;
    test 6 `flop_log.json` emission is §8.5 Jett-tunable).
  - ciFAIR-100 eval (Breach owns the gate memo, §5.1 / §6.3 cross-check).
  - `routing_log.parquet` per-step emission (stdout + slurm .out is
    sufficient for gate adjudication).

**Seeds plan (as shipped)**: slurm array 0-39 correctly implements the
protocol's n=10 primary (seeds 42-51, indices 0-19) + D3-conditional
backup n=20 (seeds 52-61, indices 20-39). Pilot is indices 0-5 (3 seeds
× 2 arms). Dev runs pilot first; σ_upper test per §6.5 decides whether
indices 20-39 actually execute. **Correction to my 2026-04-19 ~04:00
timeline entry below**: that entry said "n=20 upfront" — actually Jett
correctly kept the protocol's n=10 primary with conditional n=20 D3
backup, which is the right design per §4.3 / §6.5. The 40-job array
just prestages both so no separate submission is needed if D3 triggers.

**Freeze-readiness**: ✅ v1.2.1 signs as planned on 2026-04-26. Jett's
deliverable is protocol-compliant; the one config-default defect is
fixed inline. No v1.2.2 amendment needed.

**Outcome**: HPC deployment package ready for Dev upload. Status moves
from "awaiting Jett" → "awaiting Dev cluster upload + pilot".

**Follow-up (Dev actions)**:
  1. Upload `HPC/` → `/home/users/dvyas4/mob-gate/` (rsync or scp per
     README's "Upload from local" section).
  2. Cluster-side venv setup per `HPCguide.md` steps 1–4.
  3. Run pre-pilot acceptance tests: `pytest gate_runner/tests/test_acceptance.py -v`
     (PRE-RUN subset must all pass; POST-RUN subset skips cleanly).
  4. Launch pilot: `sbatch --array=0-5 slurm/run_gate.sh` — 3 seeds × 2 arms.
  5. Verify 6 `results/gate/seed_{42,43,44}_arm_{A,B}.json` produced,
     download and ping Nosh.
  6. On Nosh approval of pilot numerics: launch remainder. Default is
     `sbatch --array=6-19 slurm/run_gate.sh` for the n=10 protocol primary;
     §6.5 D3 backup `--array=20-39` only if pilot σ_upper ∈ (1.30, 1.75].

**Follow-up (Nosh actions)**:
  - Freeze signature on v1.2.1, 2026-04-26.
  - On pilot results back: fire KAY/O for pre-adjudication sanity check
    on 3 seeds (prevent a silent data issue from burning the full array).
  - On full-array results back: fire KAY/O for post-experiment red-team,
    then BCa bootstrap, then pass/tie/fail verdict.

**Files**:
  - created (Jett): full `HPC/` tree (13 files, 2,629 LOC).
  - modified (Nosh, post-sprint): `HPC/gate_runner/config.yaml` (task
    partition 10T→20T), `HPC/README.md` (compliance flag #2 + operating
    ranges comment), `project.md` §2 S2 row (K=4 explicit).

---

### [2026-04-19 ~04:00] HPC pivot — Jett commissioned for complete gate-runner sprint targeting ACIDSDB

**What**: Dev revealed access to the ACIDSDB HPC cluster (V100 32GB,
partitions qGPU24/48/120, cuda/12.1, python/3.10.14, home `/home/users/dvyas4/`)
and asked for a self-contained `HPC/` deliverable he can upload and run at
full scale. This collapses the n=10-vs-n=20 seeds debate — with cluster
access the gate runs at **n=20 seeds upfront** as a slurm array (40 jobs =
20 seeds × 2 arms). No hold-out regime needed.

**Why now**: Before Dev's HPC reveal, Jett's gate-runner was on the
post-freeze contract (write code → review → pilot on local GPU →
incrementally collect seeds). HPC access means we can ship the full gate
directly to a cluster array and recover the `wall_clock_per_seed → n_seeds`
tradeoff we were making. Net compute capacity is now the binding constraint,
not Dev's local RTX 4070. Bringing Jett's contract pre-freeze is the
right acceleration because (a) v1.2.1 protocol text is frozen-ready, (b)
Jett's deliverable doesn't touch protocol/theory docs (Nosh-owned), and
(c) the slurm scaffolding needs the protocol's hyperparameter pins anyway.

**Correction (added on Jett-return review at ~04:15)**: My initial framing
below said "n=20 upfront" — that was wrong. The correct design is
**protocol n=10 primary + D3-conditional n=20 backup pre-staged in the
same 40-job array**. Protocol §4.3 / §6.5 D3 specifies a backup extension
to n=20 ONLY if pilot σ_upper lands in (1.30, 1.75]. Jett correctly
implemented this: slurm array indices 0-19 = n=10 primary (seeds 42-51);
indices 20-39 = n=20 D3 backup (seeds 52-61), submitted conditionally.
HPC just makes both branches pre-stageable as a single submission.

**Commission scope**: Jett builds a complete, minimal `HPC/` directory:
  - `README.md`, `HPCguide.md` (env setup + modules), `requirements.txt`
  - `slurm/run_gate.sh` — slurm array 0–39, `(seed, arm) = (42 + N//2, A|B)`
  - `gate_runner/` package:
    - `run_arm.py` — single-seed single-arm entry point
    - `fecam_core.py` — byte-equivalent port of
      `dipamgoswami/FeCAM @ e33f39d112ff2d2a2df2e68c490af579a50edd31`
      `models/base.py` (`shrink_cov`, `normalize_cov`, `_mahalanobis`,
      `_tukeys_transform` — OFF for ViT per §7). **Uses `pinv`, not `inv`.**
    - `auction.py` — port from `mob/` / `contibualmob/` with Fisher
      clamp min=0.1 and optimizer reset every expert at task boundary
      (v1.2.1 §2.3 T7)
    - `experts.py` — ViT-B/16 frozen backbone + LoRA(r=8) QKV +
      per-expert FFN bottleneck (~300K trainable/expert)
    - `data.py` — CIFAR-100 via LAMDA-PILOT class ordering
      `np.random.RandomState(1993).shuffle(np.arange(100))`
    - `ewc.py` — Fisher estimator with load-bearing clamp
    - `config.yaml` — mirrors v1.2.1 §15.18 `frozen_config.yaml`
    - `utils.py` — seeding, hashing, logging helpers
    - `tests/test_acceptance.py` — all 20 tests from v1.2.1 §15
  - Result path: `./results/gate/seed_${SEED}_arm_${ARM}.json`
  - JSON schema: `{arm, seed, protocol_version, config_hash, per_task_acc,
    final_acc, aia, forgetting, fisher_diagnostics, routing_distribution,
    first_bid_hash, loss_at_t0_hash, caveats}`

**Constraints pinned into commission**:
  - Do **NOT** run locally (sandbox blocks; also wastes cycles Jett should
    spend on correctness).
  - Do **NOT** modify timeline/project/protocol/theory docs — Nosh-owned
    files, edits would collide with this entry.
  - Escalate if v1.2.1 §1.3 vs §3 contradict on Arm B definition.
  - Escalate if any §2.3 training-spec or §15.18 hyperparameter is unpinned.
  - FeCAM port byte-equivalent to commit `e33f39d1` — no modernization.
  - Slurm params: `--partition=qGPU48 --gres=gpu:V100:1 --mem=32G
    --cpus-per-task=4 --time=06:00:00 --array=0-39`.

**Compute envelope under HPC**: 20 seeds × 2 arms × ~13–14 GPU-days per
arm → ~280 V100-hours total. At qGPU48's 48-hour limit per job that's
40 × ~6hr jobs fitting inside a single array submission. Fully parallel:
wall-clock ≈ 6hr to first full n=20 gate if the cluster is not contended.

**Outcome**: _Pending_. Jett running as background agent
`a3b32a8c6b23c0103`. Nosh using the window to update project state
(this entry + project.md §3/§11 amendment below).

**Interpretation**: This is the right move if and only if Jett's
deliverable reproduces byte-equivalent FeCAM and passes all 20 acceptance
tests on the cluster. If Jett surfaces a defect requiring a v1.2.2
protocol amendment (e.g., §1.3 vs §3 Arm B contradiction, or an unpinned
hyperparameter in §15.18), Breach is fired immediately. Freeze-date for
v1.2.1 remains 2026-04-26; if v1.2.2 is needed the freeze slides to
2026-04-27.

**Follow-up**:
  - When Jett returns: verify deliverables, confirm 20/20 acceptance
    tests can run under slurm, tell Dev upload instructions.
  - Post-upload on cluster: Dev runs pilot (array indices 0–5 = 3 seeds
    × 2 arms), confirms numerics match local expectations, submits
    remainder (indices 6–39).
  - Results flow back: Dev downloads `./results/gate/*.json` → local
    repo → Nosh fires KAY/O for post-experiment red-team review →
    BCa bootstrap on paired Δ → pass/tie/fail decision.
  - Freeze v1.2.1 signature on 2026-04-26 happens on schedule; HPC
    deployment is a freeze-independent engineering track.

**Files**: will be created by Jett under `HPC/`. This entry documents
intent only; actual file manifest appended on Jett return.

**Ledger note**: I told Dev the Jett commission includes the scaffolding
files (README, HPCguide, requirements) so Nosh will NOT pre-create them —
duplicating work and violating the agent-task "don't touch same files"
guidance. Nosh owns timeline/project/protocol/theory updates only.

---

### [2026-04-19 00:50] arxiv 2512.10969 authorship confirmed + Phase 1/2 commissions fired

**What**: Dev confirmed arxiv 2512.10969 is his own preprint. The last Phase 0
blocker is cleared. Nosh commissioned Track B (Sage — Pólya-urn + DeSieno
conscience proof) and Track A (Breach — FeCAM-Router ablation protocol) in
parallel.

**Why**: Both tracks are arxiv-independent theoretical/methodological work that
load-bear the paper regardless of the Phase-1 gate outcome. Running them in
parallel collapses sequential latency. Track B is the paper's theoretical
contribution; Track A is the protocol document Jett will implement against
the moment he's commissioned.

**Outcome**: _Pending_. Both specialists dispatched as Agent calls. See return
summaries in this ledger under subsequent 2026-04-19 entries.

**Interpretation**: Research program enters Phase 1 + Phase 2. Phase 3 (CIFAR-100
full suite), Phase 4 (LLM MoE port), Phase 5 (paper) remain downstream of Phase 1
gate outcome.

**Follow-up**:
  - On Sage return: commission Omen to review the proof for mathematical soundness
    before paper integration.
  - On Breach return: commission Jett (IE) to implement the protocol against the
    LAMDA-PILOT harness.
  - Weekly arxiv watchlist (Dealbreaker #4 — Fade) still ambient.

**Files**: will be updated by specialists.

---

### [2026-04-19 ~01:05] Sage return — Pólya-urn + DeSieno proof writeup

**What**: Sage returned a ~4,200-word publishable-substrate theory writeup.

**Outcome**: `docs/theory/polya-urn-conscience-proof.md`. Three numbered theorems
with explicit A1–A9 assumptions:
  - **T1 (Pólya-urn collapse without conscience)** — proved via Benaïm SA + Pemantle
    urn framework. Strictly-increasing reinforcement established via Bodnar-Okhrin
    Mahalanobis-variance bound. Collapse with positive probability is the
    load-bearing negative result.
  - **T2 (ergodicity under γ > γ_min)** — proved via Borkar two-timescale SA with
    explicit `γ_min = 2αL·diam(X)/K` and entropy Lyapunov function.
  - **T3 (local convergence + selection-bias TV bound)** — sketched; full TV bound
    chain deferred as proof gap G2.
  - **Prop 5.1 (DSIC under conscience)** — short clean proof invoking Astra's
    R2 fixed-public-λ resolution.

Seven proof gaps enumerated (G1–G7): three structural (G3 global convergence,
G4 learned bidders, G7 cross-task), three technical (G1 γ_min tightness, G2 TV
bound, G5 shared-γ), one empirical (G6 Lipschitz L).

**Interpretation**: Theory substrate is in place. Paper section will be distilled
from this. The load-bearing ask is **Chamber's empirical bid-surface Lipschitz
constant L at each scale** — every γ_min estimate in §6 is qualitative without it.
C6.1 (shared-γ-across-layers at S3) is a high-priority conjecture blocking Phase 4.

**Follow-up**:
  - Commission Chamber to measure empirical L at S1 (and commit to a measurement
    protocol for S2/S3) — addresses G6.
  - Myerson monotonicity for learned-bidder variants (G4) stays deferred unless
    learned-α,β variants enter scope.
  - Before paper integration: commission Omen to review proofs for mathematical
    soundness.

**Files**:
  - created: `docs/theory/polya-urn-conscience-proof.md`
  - appended: `_bmad/.session/sage.md`

---

### [2026-04-19 ~01:05] Breach return — FeCAM-Router gate protocol v1.0

**What**: Breach returned a preregistered 14-section protocol document.

**Outcome**: `docs/protocols/fecam-router-gate.md` (v1.0, freeze date 2026-04-26).

Core commitments:
  - **Two-arm paired design**. Arm B (FeCAM-Router) = Arm A (MoB-Full) with β=γ=0
    hard-coded; everything else (backbone, LoRA, tied low-rank Σ, shrinkage λ,
    α calibration, Fisher clamp, optimizer reset, seed, class order, augmentation)
    held strictly identical. This is the KAY/O-invariant strict ablation.
  - **H_alt**: one-sided paired, Δ_practical = **1.0 pp**. Statistically-significant-
    but-trivial wins do not pass the gate.
  - **Primary CI**: BCa bootstrap on paired difference, 10,000 resamples, 95%
    one-sided lower bound.
  - **Matched compute**: FLOP-based ±5% (primary); wall-clock and steps secondary.
  - **Seeds**: {42..51}, paired across arms. Task configs 20T / 5T / 10T, with
    20T as the gate.
  - **3-seed pilot** gates the full run. Pilot passes iff compute-match ≤ 5%,
    Fisher ratio ≤ 2× per seed (addresses KAY/O's 18× Fisher-variance concern via
    pairing + clamp), all 8 acceptance tests pass, pilot σ_Δ permits MDE_80 ≤ 1.0pp
    at n=10.
  - **Fail-safe**: σ_Δ upper > ~1.75pp → escalate to Nosh, do not silently widen
    n or relax Δ_practical.
  - **Timeline**: 2.5–3.5 weeks on ~8 GPU-days (1×A100/H100).
  - **Jett-facing contract**: CLI `mob/gate/run.py` with frozen config YAML,
    8 acceptance tests including β=0 unit test and Arm-A/Arm-B backbone hash
    equality at t=0, FLOP accounting, ciFAIR loader, determinism harness.

**Interpretation**: Methodology is paper-rigor. KAY/O's 18× Fisher-variance threat
addressed by the paired design + Fisher-match gate in the pilot. Any
gate-outcome-dominating-factor other than β/γ is a priori ruled out by the
strict ablation.

**Outstanding decisions**:
  1. **Freeze signature** required by 2026-04-26 (Dev + Nosh).
  2. **KAY/O pre-pilot red-team** explicitly requested by Breach (§11.3 checklist).
  3. **Chamber** to confirm per-task epoch default for the ~300K-param LoRA adapter
     is valid (Jett → Breach escalation flagged).
  4. Jett implementation of the runner + 8 acceptance tests (5–7 days) after
     freeze signature.

**Files**:
  - created: `docs/protocols/fecam-router-gate.md`
  - created: `_bmad/.session/breach.md`

---

### [2026-04-19 ~01:15] KAY/O pre-pilot red-team — APPROVE WITH AMENDMENTS

**What**: KAY/O pre-data adversarial audit of Breach's gate protocol v1.0.

**Outcome**: `docs/protocols/fecam-router-gate-redteam.md`.
**Verdict**: **APPROVE WITH AMENDMENTS**. Freeze can proceed on 2026-04-26 iff three
binding amendments land first. 9 defects total, 8 new acceptance tests on top of
Breach's 8 (total now 16).

**Top-3 binding defects**:

1. **D1 (CRITICAL) — Outcome-branch test framework is broken.** Protocol §1.2
   declares three outcomes (PASS / TIE / FAIL) but §6.1 primary test is one-sided.
   The FAIL branch requires a two-sided CI the protocol never operationalizes, and
   at n=10 with plausible σ_Δ the FAIL region is near-unreachable. Gate currently
   has effectively two outcomes (PASS / NOT-PASS) while §9.3 treats FAIL as
   program-terminating. **Fix**: symmetric one-sided BCa tests — PASS if lower bound
   ≥ +Δ_practical, FAIL if upper bound ≤ −Δ_practical, TIE otherwise.

2. **D2 (CRITICAL) — Fisher-match gate does NOT address KAY/O's R1 threat.**
   Protocol §4.4 checks `2× cross-arm within-seed`, near-trivially satisfied by
   Breach's own protocol invariants. R1 threat was **within-arm across-seed**
   Fisher variance. Different quantity. A PASS under v1.0 leaves R1 unfalsified.
   **Fix**: add `CV(log F̄^A(s)) ≤ 0.5` as a second inclusion criterion; if
   violated, report with limitation annotation or tighten clamp.

3. **D3 (HIGH) — "Marginally powered" band is 46% power, not 80%.** At σ_Δ = 1.75pp,
   n=10, detecting Δ=1.0 at α=0.05 one-sided gives ~46% power. Protocol §6.5
   proceeds in this band with annotation. A 46%-powered gate is a coin flip.
   **Fix**: cap at σ_Δ ≤ 1.30pp OR pre-commit n=20 as backup.

**6 additional defects**: α-calibration source asymmetry, RNG-state-at-first-bid
equality, DataLoader worker seed pinning, loss-equality test at t=0, wall-clock
trap in gate-pass interpretation, paper metric primacy protection.

**Post-PASS caveat**: Even under perfect execution of the amended protocol, a
gate PASS licenses Phase 2/3 but does NOT mechanistically refute the R1 threat.
The β-only and α-only ablations (synthesis §5 Phase 2) remain required to claim
"auction is irreducible." This is an open empirical obligation separate from
Sage's theoretical Pólya-urn result.

**Interpretation**: D1/D2/D3 are ≤1 day of amendment work for Breach. With them
applied, the protocol is a defensible preregistration. Without them, running the
gate and declaring PASS/FAIL is not an honest preregistration. Recommend Breach
v1.0 → v1.1 amendment cycle before 2026-04-26 freeze.

**Follow-up**:
  - Commission Breach amendment cycle (v1.0 → v1.1) addressing D1/D2/D3 + 6 minor
    defects + 8 new acceptance tests.
  - Freeze date 2026-04-26 remains tentative; slip 2-3 days if Breach pushes back
    on any of D1/D2/D3.
  - β-only / α-only ablation protocol is a separate Phase-2 ED obligation — queue
    for post-freeze commissioning.

**Files**:
  - created: `docs/protocols/fecam-router-gate-redteam.md`
  - appended: `_bmad/.session/kayo.md`

---

### [2026-04-19 ~01:30] FeCAM paper + code gap analysis (Nosh, inline)

**What**: Dev asked for a re-read of the FeCAM paper and comparison between
FeCAM's published repo and our MoB Mahalanobis implementation.

**Outcome**: `docs/lit-review/05-fecam-code-comparison.md`. Mapped FeCAM's four
architectural elements (per-class Σ, γ₁/γ₂ additive shrinkage, correlation
normalization, Tukey β=0.5 + L2 feature normalization) against
`contibualmob/prototype_store.py`. **Our implementation has 1 of 4 elements**
(shared Σ with ridge `1e-4·I`) — missing per-class Σ, FeCAM's shrinkage formula,
correlation normalization, Tukey transform, and L2 feature normalization.

**Headline finding**: our Mahalanobis code is not FeCAM. It is a simplified
shared-Σ ridge-regularized Mahalanobis. The v2/v3 MNIST results labeled as
"FeCAM-class prototype routing" (progress_report.md §Current Results) are NOT
comparable to FeCAM. More critically, Breach's Phase-1 gate §2.1 mandates Arm B
be "FeCAM exactly per FeCAM paper." Running the gate with current code as Arm B
would produce an unpublishable straw-man comparison either way (PASS = "beat
weakened FeCAM"; TIE/FAIL = "cannot distinguish auction epiphenomenal from
Arm B beatable for the wrong reason").

**Interpretation**: This is a **pre-freeze blocker** that is orthogonal to
KAY/O's amendment cycle. KAY/O's defects were statistical / methodology;
this defect is implementation. Both must close before pilot.

**Recommended remediation**: Path 2 — Arm B (and A) calls LAMDA-PILOT's own
FeCAM implementation (community-canonical, passes any reviewer challenge).
Breach's protocol already commits to LAMDA-PILOT as the harness, so this is
a natural binding.

**Downstream implications**:
  - Update `project.md` §6 to flag the v3 numbers as weakened-FeCAM, not FeCAM.
  - Flag Sage: `docs/theory/polya-urn-conscience-proof.md` §3 cites Ledoit-Wolf
    as the FeCAM recipe; FeCAM actually uses additive γ₁·V₁·I + γ₂·V₂·(1−I).
    Prop 5.1 DSIC proof survives (fixed public (γ₁, γ₂) generalizes fixed public λ)
    but the citation needs correction in next theory revision.
  - Post-Breach-v1.1: commission Jett to wire the gate runner against
    LAMDA-PILOT's FeCAM method, not against current `prototype_store.py`.

**Files**:
  - created: `docs/lit-review/05-fecam-code-comparison.md`

---

### [2026-04-19 ~01:40] Breach v1.1 amendment pass complete

**What**: Breach returned protocol v1.1 addressing all KAY/O-flagged defects.

**Outcome**: `docs/protocols/fecam-router-gate.md` (v1.0 superseded in place; §15
changelog appended). **Freeze date 2026-04-26 holds.** No escalations.

**Amendments landed as-specified** (KAY/O's prescription followed exactly):
  - **D1**: §1.2/§6/§9 rewritten to symmetric one-sided BCa with mutually exclusive
    PASS/FAIL/TIE regions. Composite α=0.10 declared.
  - **D2**: §4.4 now has dual Fisher-match gate. Within-arm `CV(log F̄) ≤ 0.5`
    inclusion criterion added. Branch (a) preferred: clamp-ladder `{0.1, 0.3, 1.0, 3.0}`.
    Branch (b) fallback: proceed with limitation annotation. Pre-registered.
  - **D4**: α shared pre-routing pre-pass written into frozen_config.yaml.
  - **D5**: RNG-state equality at first bid (§8.4 test 13).
  - **D6**: DataLoader worker seed pinning in §2.1 invariants.
  - **D7**: Bitwise-loss equality at t=0 (§8.4 test 11).
  - **D8**: Wall-clock-ratio disclosure on PASS (§1.8).
  - Paper-metric-primacy clause added (§1.7).
  - All 8 new acceptance tests integrated as §8.4 tests 9–16, cross-referenced
    to KAY/O's list a–h.

**Amendment with modified option-selection** (not an escalation — KAY/O gave
Breach a choice):
  - **D3**: KAY/O offered either "cap σ_upper ≤ 1.30pp" OR "pre-commit n=20 backup."
    Breach selected **n=20 backup**, justified in §6.5: (a) robustness against
    noisy pilot-σ estimates at n=3; (b) better paper outcome than an early
    termination. Seeds {42..61}, +17 seeds/arm past pilot, +~7 GPU-days. Fits
    inside §12 timeline.

**Escalations**: None.

**Freeze-readiness**: 2026-04-26 holds. Jett's §8.4 sprint still 5–7 days even
with 16 tests instead of 8 (tests 9–16 share scaffolding with 1–8).

**Flagged for optional KAY/O round-2** (Breach's explicit callout):
  1. D3 option choice — n=20 backup vs σ_upper cap (§6.5 justification).
  2. D2 branch (b) limitation-annotation text (§4.4 item 4b).
  Both low priority; both explicitly justified. Re-review is optional, not blocking.

**Outstanding pre-freeze blocker (separate from KAY/O track)**:
  - FeCAM-implementation gap documented in `docs/lit-review/05-fecam-code-comparison.md`
    (2026-04-19 01:30 entry above). Not addressed by Breach's v1.1 — this was an
    implementation defect, not a statistical one. Must close before pilot.

**Follow-up**:
  - Dev: signature on v1.1 freeze for 2026-04-26 (or request KAY/O round-2 first).
  - Open question for Dev: how to handle FeCAM-implementation gap — amend v1.1
    to bind Arm B to LAMDA-PILOT's FeCAM, or handle in Jett's implementation
    contract post-freeze.
  - After freeze: commission Jett to implement the gate runner + 16 acceptance tests.
  - Parallel: commission Chamber for empirical bid-surface Lipschitz L (Sage G6).

**Files**:
  - modified: `docs/protocols/fecam-router-gate.md` (v1.0 → v1.1)
  - appended: `_bmad/.session/breach.md`

---

### [2026-04-19 ~02:00] Breach v1.2 FeCAM-canonical binding amendment complete

**What**: Dev chose path (b) from 01:40 decision. Breach commissioned for a tight
v1.1 → v1.2 amendment binding the Mahalanobis core of Arm A + Arm B to the
canonical FeCAM implementation so "MoB beats FeCAM" is a claim with a referent.
Return: v1.2 landed, freeze still holds, three new tests (17–19), and Breach
surfaced a **load-bearing deviation between LAMDA-PILOT and the FeCAM paper**
that forced a composite binding rather than a pure LAMDA-PILOT binding.

**Outcome**: `docs/protocols/fecam-router-gate.md` v1.1 → v1.2 (timestamped
2026-04-19). Purely additive + minor §2.1 row re-scoping. No v1.1 statistically
load-bearing text modified. Jett sprint absorbs +1 day for three new tests +
one synthetic fixture.

**Amendments landed**:
  - **§1.3**: arm definitions cite §2.0 / §4.6 binding; β=γ=0 confirmed as sole
    code-path difference.
  - **§2.0 NEW** (inserted before §2.1): five-element FeCAM recipe pinned —
    per-class μ, per-class Σ, two-parameter additive shrinkage
    `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)`, correlation normalization, L2 normalization.
    **Tukey OFF for ViT-B/16** per paper §7 explicit guidance (ViT features
    have negative values). Explicit out-of-scope: `contibualmob/prototype_store.py`
    stays untouched; gate runs through a **new** `mob/gate/fecam_core.py`.
  - **§2.1**: row re-scoping — Covariance structure, Mahalanobis formulation,
    Shrinkage hyperparameters (replaces "Shrinkage parameter λ"), + new rows
    for L2 normalization, Tukey OFF, and per-class prototype granularity.
  - **§4.6 NEW** (FeCAM-binding implementation invariant): five-clause Jett
    contract, repo-pin table, invalidity-on-violation failure-mode declaration.
    CI-enforced via tests 17–19.
  - **§4.7 NEW** (shrinkage hyperparameter pinning): γ₁=γ₂=1 default (paper §7
    CIFAR-100 MSCIL ResNet-18 values — paper does not publish ViT-B/16 γ values,
    this is tightest match). Pre-freeze empirical override window to γ₁=γ₂=10
    (paper Split-ImageNet-R ViT values) if Jett's V₁/V₂ backbone-only forward
    pass result diverges ≥10× from ResNet-18 reference.
  - **§8.4**: 16 → 19 tests.
    - Test 17: FeCAM-port fidelity ≤1e-5 vs upstream fixture (d=768, N=128, C=10).
    - Test 18: hyperparameter pinning — frozen_config.yaml matches paper §7
      canonical `{γ₁=1, γ₂=1, tukey=false, tukey_beta=0.5, per_class_cov=true,
      l2_normalize=true, inverse_method=pinv}` on both arms.
    - Test 19: L2 + Tukey feature-preprocessing ordering (Tukey correctly
      disabled on mixed-sign ViT; ordering correct if re-enabled for future variant).
  - **§15.8–§15.14 NEW**: v1.2 changelog with B1 (Mahalanobis binding), B2
    (§4.6 Jett contract), B3 (tests 17–19), out-of-scope list, escalations,
    freeze-readiness.

**Composite binding (this is the new finding)**:
  - Harness, trainer, ViT-B/16 backbone, ViT-Tukey-OFF decision →
    **`sun-hailong/LAMDA-PILOT`** commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`.
  - Paper-canonical Mahalanobis recipe (shrinkage, correlation norm, L2 norm) →
    **`dipamgoswami/FeCAM`** commit `e33f39d112ff2d2a2df2e68c490af579a50edd31`,
    `models/base.py`, `exps/FeCAM_cifar100.json` canonical config
    `{alpha1=1, alpha2=1, beta=0.5, per_class=true, full_cov=true, shrink=true, norm_cov=true}`.
  - **Why composite (load-bearing)**: LAMDA-PILOT at the pinned SHA applies
    single-parameter `cov + 100·I`, NOT paper eq. 8's two-parameter additive
    `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)`. This is the escalation Breach's directive
    anticipated. Resolved at Breach's pre-freeze implementation-binding authority
    (§13). Documented in §4.6 and §15.13(1).

**Interpretation**: This vindicates the FeCAM-gap finding from 01:30 — we would
have shipped a non-FeCAM Arm B even if we'd bound against LAMDA-PILOT alone.
The composite binding (paper-canonical recipe + LAMDA-PILOT harness) is the
technically correct target. Tukey-OFF-for-ViT is a **paper-mandated exception**,
not an omission: this means the 01:30 `05-fecam-code-comparison.md` "Tukey
β=0.5 missing" row is misleading for our ViT-B/16 setting and needs correction.

**Escalations**: None that block freeze. Two flags surfaced within Breach's
§13 authority (not asked to adjudicate):
  1. LAMDA-PILOT/paper deviation (resolved via composite binding above).
  2. Paper does not publish CIFAR-100 + ViT-B/16 γ values (resolved via §4.7
     ResNet-18 default + empirical override window).

**Freeze-readiness**: **YES — 2026-04-26 holds.** +1 day for Jett's §8.4
sprint absorbed within the original 5–7 day window.

**Follow-up**:
  - Dev: sign v1.2 freeze by 2026-04-26 (default go if no reject).
  - Optional pre-freeze: Jett backbone-only V₁/V₂ empirical check on seed 42
    for §4.7 γ override decision. Cheap (~1 GPU-hr). Pins γ before freeze
    rather than deferring.
  - Correct `docs/lit-review/05-fecam-code-comparison.md`: reframe "Tukey β=0.5
    missing" row as "Tukey is paper-disabled for ViT-B/16 per §7 — irrelevant
    to our setting."
  - Correct `docs/theory/polya-urn-conscience-proof.md`: Sage's §3
    Ledoit-Wolf-as-FeCAM citation should be replaced with additive
    γ₁·V₁·I + γ₂·V₂·(1−I).
  - Post-freeze: commission Jett for the gate runner + 19 acceptance tests +
    `mob/gate/fecam_core.py` (byte-equivalent port of `dipamgoswami/FeCAM`
    upstream `shrink_cov`/`normalize_cov`/`_mahalanobis`).

**Files**:
  - modified: `docs/protocols/fecam-router-gate.md` (v1.1 → v1.2)
  - appended: `_bmad/.session/breach.md`

---

### [2026-04-19 ~02:30] Inline doc corrections post-v1.2 (Nosh, inline)

**What**: Before dispatching Jett + Chamber, corrected two doc drifts from the
v1.2 FeCAM-canonical binding:
  1. `docs/lit-review/05-fecam-code-comparison.md`: reframed §2.4 Tukey as
     backbone-conditional (paper §7 disables for ViT-B/16, mandatory for ResNet);
     §4 gap table Tukey row downgraded to **N/A at S2 (our setting)**; added §5.5
     resolution-status section documenting v1.2 composite-binding outcome and §5.6
     flagging Sage's §6 scale-bounds rework.
  2. `docs/theory/polya-urn-conscience-proof.md`: added prominent TODO banner at
     top documenting the non-load-bearing Ledoit-Wolf → FeCAM-additive substitutions
     applied inline, and the load-bearing §6.2/§6.3 rework Sage still owes under
     FeCAM's additive shrinkage formula. Substitutions applied: §0 item 4,
     §0 item summary, §1 bid-function definition (Σ_i(λ) → Σ_i,s with additive
     formula), §1 constants list, (A6) shrinkage regularization, (A7) Lipschitz
     metric, Proposition 5.1 proof + Remark 5.2, dealbreaker clause, §11
     Astra-confirmation summary, §8 Ledoit-Wolf citation entry (retitled as
     Goswami/FeCAM with correction note).

**Why**: doc-drift between Breach v1.2 protocol binding (additive
γ₁·V₁·I + γ₂·V₂·(1−I) with Tukey OFF for ViT) and the theory writeup
(Ledoit-Wolf convex-combination Σ(λ) + Tukey-always) creates inconsistency that
future paper co-authors would inherit. Fixed inline now.

**What was NOT changed**: Sage's §6.1/§6.2/§6.3 scale-specialization bounds
(eq. 6.1 `γ_min(λ) = O(α/(K√λ))` and eq. 6.2 `γ_min ≈ O(α · 32 / √λ)`) derive
from the *condition number* of the shrunk covariance, which has a different
functional form under additive vs convex-combination shrinkage. Substitution
is non-trivial — needs Sage to re-derive with κ(Σ_s). TODO banner at top of
doc explicitly blocks this on Chamber's L measurement + Jett's V₁/V₂ measurement
so rework closes both numerical gaps in one pass.

**Files**:
  - modified: `docs/lit-review/05-fecam-code-comparison.md`
  - modified: `docs/theory/polya-urn-conscience-proof.md`

---

### [2026-04-19 ~03:00] Chamber two-part commission return — Q1 epochs + Q2 Lipschitz protocol

**What**: Chamber returned pre-Phase-2 architecture commission covering (Q1)
per-task epoch spec for ViT-B/16 + LoRA on CIFAR-100 CL, and (Q2) empirical
Lipschitz constant L estimation protocol at each scale.

**Q1 outcome — canonical 20 epochs/task**:
  - **20 epochs/task** — LAMDA-PILOT canonical across all adapter/LoRA methods at
    CIFAR-100/ViT-B/16 (`fecam.json: tuned_epoch=20`, `aper_aperpter.json: 20`,
    `ranpac.json: 20`, `mos.json: 20`, `coda_prompt.json: 20`, `slca.json: 20`,
    `ease.json: 20`). L2P/DualPrompt's 5 epochs rejected as outliers (prompt-only
    regime, not applicable to our ~300K-param LoRA+FFN adapter).
  - **AdamW + cosine**, `init_lr=5e-4 (adapter) / 1e-3 (LoRA QKV)`, `min_lr=1e-5`,
    `bs=128`, `wd=1e-4`, **linear 100-step warmup on task 1 only**.
  - **Optimizer reset at every task boundary** for every expert (not just winner) —
    confirmed against LAMDA-PILOT `dualprompt.json`/`mos.json: reinit_optimizer=true`
    and our `memory/MEMORY.md`.
  - **Backbone fully frozen, no LN exception**. Final-block LN unfreeze explicitly
    rejected — breaks DSIC (LN scale becomes private bid-time-hidden parameter)
    and negligible headroom when LoRA QKV captures direction-aware reweighting.
  - **Overfit risk at 20 epochs**: 300K/5k = 60:1 ratio, inside LoRA-CL safe zone
    per InfLoRA/EASE/RanPAC. Overfit-onset ≥30 epochs; 20 epochs is 10-epoch margin.

**Q1 pre-freeze impact**: Protocol v1.2 §7.3 pins **10 epochs** based on Jett's
wall-clock estimate, not a canonical reference. Chamber's Q1 establishes **20 is
the LAMDA-PILOT canonical** for FeCAM's own config — meaning Arm B at 10 epochs
would be training-undertime relative to the published FeCAM. This is a **pre-freeze
protocol correction needed** (Breach v1.2.1 or inline edit by Dev signing).
**Path**: fast Breach amendment or direct §7.3 edit pre-freeze 2026-04-26.

**Q2 outcome — Lipschitz L measurement protocol**:
  - **Operational space**: L2-normalized feature space under the Σ_s-induced
    metric (NOT raw input space, which gives unbounded L on pretrained ViT
    pixel-space gradients). `diam(X) = 2` at S1/S2 (unit-sphere antipodes);
    `diam(X) ≈ 4` at S3 (RMSNorm fixes L2 but not additive shift).
  - **Three-stage estimation**: Stage A random pairs (N=10k), Stage B nearest-
    neighbor-in-Σ pairs (hunts winning-partition boundary), Stage C gradient-
    ascent adversarial (optional). Report `L = max(L_A, L_B, L_C)`.
  - **Prior estimates** (pre-measurement):
    - S1 (MNIST, d=128, K=4): **L ≈ 1.1–2.0 normalized**; γ_min ≈ 1.5 →
      Sage's §6.1 prescription γ=2.5 is correct margin (1.6× above bound).
    - S2 (CIFAR-100+ViT-B/16, d=768, K=8): **L ≈ 6–15 normalized**;
      γ_min ≈ 2.5 → γ is a **first-class bid term, not a small regularizer**.
      Current codebase uses γ=0.1 normalized — two orders of magnitude below.
    - S3 (OLMoE, d=2048, K=64, per-layer): **L ≈ 15–40 per-layer normalized**;
      γ_min ≈ 3 per layer.
  - **Cost**: S1 ~0.05 GPU-hr; **S2 ~0.2 GPU-hr bundle-able with Jett's §7 pilot**
    (same backbone forward pass); S3 ~8 GPU-hr total, deferred to Phase 4.
  - **Closes Sage Gap G6**.

**Q2 escalation triggers flagged**:
  1. If Stage-B reveals **L_S2 > 30 normalized**, γ_min > α in the normalized
     bid scale → forces (α, β, γ) posted-price rebalance before freeze.
  2. If per-layer L_ℓ variance at S3 **exceeds 3×** across measured layers,
     **Sage Conjecture 6.1 (shared γ-across-layers) is refuted** — Scale 3 needs
     per-layer γ_ℓ, mechanism loses "one γ, one auction" posted-price story.

**Interpretation**: Chamber's Q1 found a real pre-freeze defect in v1.2's
§7.3 (10 → 20 epochs). Q2 gives us a measurement recipe, not a number, but the
prior estimates are defensible enough to bound γ choices before the measurement
completes. The L_S2 ≈ 6-15 prior, combined with γ_min ≈ 2.5, means the codebase's
current γ=0.1 normalized scaling is demonstrably two OOM below Theorem 2's lower
bound — **Sage's §6.1 MNIST prescription γ=2.5 is the right target for Phase 2
ablation experiments.** This is a concrete numerical prior we did not have before.

**Follow-up**:
  - **Pre-freeze**: correct protocol §7.3 epochs 10 → 20 (Breach v1.2.1 or inline).
  - **Pilot**: bundle Q2 Stages A+B into Jett's §7 pilot runs. No extra compute.
  - **Post-freeze**: if §4.7 override triggers to γ=10, re-check Q2's prior
    L_S2 estimate (γ₁/γ₂ enter L via shrinkage condition number).
  - **Phase 4**: S3 per-layer L measurement must precede OLMoE integration.

**Files**:
  - appended: `_bmad/.session/chamber.md` (2026-04-19 14:30 entry, 198 words)

---

### [2026-04-19 ~03:30] Breach v1.2.1 training-spec amendment complete

**What**: Breach landed v1.2 → v1.2.1 amendment addressing Chamber's Q1-surfaced
§7.3 epochs defect and pinning Chamber's full 7-item training spec into the
protocol. Verified LAMDA-PILOT `exps/fecam.json` at commit `7a6e904c` directly
from GitHub raw — `tuned_epoch: 20` confirmed as the binding target.

**Outcome**: `docs/protocols/fecam-router-gate.md` v1.2 → v1.2.1 (2026-04-19).
No timeline slip on freeze side. Compute budget grows but stays inside the
Phase-1 window.

**Primary fix + audit deliverables**:
  - **C1 (primary)**: §2.1 Per-task epochs **10 → 20**, cited to LAMDA-PILOT
    `fecam.json: tuned_epoch=20` plus 6 corroborating `cifar224` adapter/LoRA
    configs (aper, ranpac, mos, coda_prompt, slca, ease). New §2.3 training-spec
    block added with T1 (epochs) as the first anchor.
  - **C2 (full 7-item audit)** — all pinned in §2.3:
    - T2 AdamW (β₁=0.9, β₂=0.999, ε=1e-8, wd=1e-4)
    - T3 Cosine decay to min_lr=1e-5; peak 5e-4 (adapter) / 1e-3 (LoRA-QKV)
    - T4 Batch size 128
    - T5 Weight decay 1e-4
    - T6 Linear warmup 100 steps, **task-1 only** (subsequent tasks skip)
    - T7 Optimizer reset at task boundary, **every expert** (not just winner —
      tightened from v1.2's ambiguous wording; this was a second pre-freeze
      defect closed in the same pass)
    - Backbone freeze explicit with `ln_unfreeze_exception: false`
  - **Test 20 added**: `test_training_spec_pinned_v1_2_1` — structural check on
    all 7 T-items + runtime assertion that LN is frozen and all 4 experts get
    optimizer reset at task end. Total acceptance tests: 19 → 20.
  - **§12 re-cost**: ~8 → **~13–14 GPU-days** total gate compute under 20-epoch
    pin. Still inside Phase-1 window.
  - **frozen_config.yaml fragment** appended at end of §15.18 as Jett reference.

**Intentional divergences from `fecam.json`** (Breach's explicit call — within
§13 authority, NOT §4.6 binding violations):
  - AdamW (fecam uses SGD) — LoRA-QKV convention per InfLoRA/SLCA/CODA-Prompt.
  - bs=128 (fecam uses 48 or similar) — Chamber Q1 §3 justification.
  - wd=1e-4 — standard LoRA-CL default.
  These three are training-stack choices, not Mahalanobis-recipe choices;
  do not compromise the Arm B = canonical FeCAM Mahalanobis binding.

**Second defect closed in pass** (not on Chamber's list, found during §2.1
audit): v1.2 §2.1 Optimizer-reset row read "winning expert reset on shift" —
ambiguous whether that meant only-winner or every-expert. v1.2.1 tightens to
**every expert** per LAMDA-PILOT convention + our project memory. Matches
task-aware MoB's intended behavior (reset at task END after Fisher update).

**Escalations**: NONE.

**Freeze-readiness**: **YES — 2026-04-26 unchanged.** Compute budget growth
(5-6 GPU-days) absorbed by Phase-1 window.

**Follow-up**:
  - Nosh freeze signature on v1.2.1 for 2026-04-26.
  - Post-freeze Jett contract must include test 20 implementation + the
    frozen_config training block extensions (§2.3 T1-T7 → YAML).
  - §12 re-cost requires Dev to confirm the ~6 additional GPU-days slot is
    available in our Phase-2 schedule.

**Files**:
  - modified: `docs/protocols/fecam-router-gate.md` (v1.2 → v1.2.1)
  - appended: `_bmad/.session/breach.md`

---

### [2026-04-19 ~03:15] V₁/V₂ backbone measurement — EXECUTED on GPU (Nosh direct-run)

**What**: Dev instructed Nosh to run the previously-sandbox-blocked V₁/V₂ script
directly. Nosh installed `timm`, ran on RTX 4070 SUPER (CUDA), total ~35s
(CIFAR-100 download 5s + ViT extraction 23.5s + RN18 extraction 8.9s).

**Command**:
```
pip install timm
python experiments/fecam_gate/v1v2_backbone_check.py --data_root ./data \
  --out results/fecam_gate/v1v2_empirical_2026-04-19.json --seed 42 --device cuda
```

**Result**: `max_ratio = 4.29 < 10` threshold → **recommendation γ₁=γ₂=1**.
**§4.7 default holds. No protocol edit needed.**

**Numerical detail** (L2-normalized features, per-class Σ_c, n_c=500/class,
10 CIFAR-100 task-0 classes under LAMDA-PILOT RandomState(1993) ordering):

| Quantity | ViT-B/16 (timm in21k, d=768) | ResNet-18 (torchvision IN1K, d=512) | Ratio |
|---|---|---|---|
| V₁ (mean of diag Σ) | 7.10e-4 ± 6.37e-5 | 5.95e-4 ± 4.93e-5 | **1.19×** |
| V₂ (mean of off-diag Σ) | **−8.57e-7** ± 7.19e-8 | **+3.68e-6** ± 2.57e-6 | 4.29× (abs) |

**Key finding — V₂ sign flip**: ViT's off-diagonal covariance is **consistently
negative** across all 10 task-0 classes (V₂ min = -9.79e-7, max = -7.28e-7);
ResNet-18's is consistently positive (1.15e-6 to 8.28e-6). Interpretation: after
L2-normalization onto the unit sphere in high-d, the sum-of-squares constraint
forces near-anticorrelation between feature dimensions; ViT-B/16 at d=768 hits
this regime strongly, while RN18 at d=512 retains residual positive off-diagonal
covariance (likely ImageNet co-activation patterns persisting through
CIFAR-100-upscaled features).

**Implication for additive shrinkage at ViT**: with V₂ < 0 and γ₂=1, the
shrinkage term `γ₂·V₂·(1−I)` **subtracts** from off-diagonals (rather than
adding, as at RN18). FeCAM eq. 8 handles this cleanly — no formula modification
needed — but the effect on Σ_s conditioning differs between backbones. This is
a real distributional finding Sage must account for in §6.2 rework. The
condition-number bound
`κ(Σ_s) ≤ (λ_max + γ₁V₁ + γ₂V₂) / (λ_min + min(γ₁V₁, γ₂V₂))`
has `min(γ₁V₁, γ₂V₂) = γ₂V₂` when V₂ < 0 — potentially negative denominator if
|γ₂V₂| > λ_min + γ₁V₁. At the measured values (γ₁V₁ = 7.10e-4, γ₂|V₂| = 8.57e-7),
`γ₁V₁ ≫ |γ₂V₂|` by ~830×, so positive-definiteness is preserved by the diagonal
boost alone — no degeneracy. Good news: **the formula is numerically safe at the
ViT/γ=1 operating point.**

**Backbone fingerprints** (for audit-trail reproducibility):
  - ViT-B/16: `0a8db43827e31854` (timm `vit_base_patch16_224.augreg_in21k`
    auto-remap from deprecated `vit_base_patch16_224_in21k` name).
  - ResNet-18: `061af67928a02a38` (torchvision `ResNet18_Weights.IMAGENET1K_V1`).

**Rank-deficient Σ_c (expected)**: n_c=500 < d=768 for ViT and ≈ d=512 for RN18;
per-class `lam_min ≈ 0` confirms rank deficiency. This is precisely what the
FeCAM shrinkage recipe exists to address: the additive `γ₁·V₁·I` term lifts
the null space of Σ̂ by γ₁V₁ ≈ 7.10e-4 (ViT) / 5.95e-4 (RN18).

**Caveats** (from script's auto-emitted caveat list):
  - Rank deficiency per-class (documented; shrinkage handles it).
  - No `FALLBACK_NOT_IN21K` flag triggered — timm in21k weights loaded.
  - No |V₂| < 1e-8 escalation triggered.

**Freeze impact**: §4.7 pre-freeze override window CLOSED with data. γ₁=γ₂=1
pinned. Breach v1.2.1 still in flight for §7.3 epochs fix (orthogonal concern).

**Follow-up**:
  - Freeze v1.2.1 on 2026-04-26 includes γ₁=γ₂=1 as empirically-confirmed default,
    not extrapolated from paper ResNet-18 value.
  - Chamber's §6 rework should account for the V₂ sign flip at ViT when
    re-deriving κ(Σ_s) bounds — the measured ratio `γ₁V₁ / |γ₂V₂| ≈ 830` at
    ViT/γ=1 is a concrete numerical anchor for §6.2.
  - Add the JSON to freeze signing packet alongside v1.2.1.

**Files**:
  - created: `results/fecam_gate/v1v2_empirical_2026-04-19.json` (full per-class
    Σ_c eigenvalue diagnostics + summary + divergence-test + caveats).

---

### [2026-04-19 ~03:00] Jett V₁/V₂ backbone empirical check — BLOCKED on sandbox (superseded by 03:15 direct-run)

**What**: Jett commissioned for Phase-1 §4.7 pre-freeze empirical V₁/V₂ divergence
check (ViT-B/16 in21k vs ResNet-18 ImageNet on CIFAR-100 task 0 features, L2-
normalized). Intended outcome: flip γ₁=γ₂ default between 1 (paper CIFAR-100
ResNet-18 value) and 10 (paper Split-ImageNet-R ViT value) based on measured
V-ratio ≥10×.

**Outcome**: **BLOCKED** — script authored (`experiments/fecam_gate/v1v2_backbone_check.py`),
execution denied by sandbox. `python <script>.py` denied on three attempts after
initial `python --version` passed. Jett correctly refused to fabricate numbers
for a protocol-binding decision.

**Script specification** (authored, verified correct, awaits execution):
  - Preferred backbone: `timm:vit_base_patch16_224_in21k` (LAMDA-PILOT canonical);
    fallback: `torchvision:vit_b_16:IMAGENET1K_V1` (IN1K, with `FALLBACK_NOT_IN21K`
    flag in output JSON).
  - Class ordering: LAMDA-PILOT's `np.random.RandomState(1993).shuffle(np.arange(100))`
    matching `exps/cifar100_b0_inc10.json`.
  - Measurement: per-class Σ_c on 500 samples/class, d=768 (ViT) or d=512 (ResNet18),
    L2-normalized features, per-class `V₁ = mean(diag(Σ_c))` and
    `V₂ = (sum(Σ_c) − trace(Σ_c)) / (d·(d−1))` (verified equivalent to paper's
    `(off_diag*mask).sum()/mask.sum()` formulation against `dipamgoswami/FeCAM@e33f39d`).
  - Handles: per-class Σ rank-deficient (n_c=500 < d=768 expected), |V₂| < 1e-8
    escalation branch, numerical precision via float64 covariance.
  - Runtime: ~15 CPU-min or <5 GPU-min.

**Unblocking options for Dev**:
  1. **Grant `python` execution** in Bash tool permission list, re-spawn Jett.
  2. **Human-run** the script:
     ```
     pip install timm
     python experiments/fecam_gate/v1v2_backbone_check.py \
       --data_root ./data \
       --out results/fecam_gate/v1v2_empirical_2026-04-19.json \
       --seed 42
     ```
     Drop the JSON into freeze review.
  3. **Hold γ₁=γ₂=1 default** without measurement. Paper CIFAR-100 canonical value.
     §4.7 override window remains open for post-freeze retroactive change if Dev's
     §15.10 ViT-B/16 extrapolation concern materializes.

**Provisional decision (if measurement cannot run)**: **keep §4.7 default γ₁=γ₂=1**
(paper CIFAR-100 MSCIL ResNet-18 value). Gate proceeds with the pre-registered
override window open. Not ideal — protocol §4.7 was designed to pin this with
data, not defer — but paper-canonical γ=1 is defensible.

**Interpretation**: the sandbox block is a **tooling issue, not a research one**.
The measurement is 15 CPU-min, script is correct. Need Dev to choose path
(1)/(2)/(3).

**Files**:
  - created: `experiments/fecam_gate/v1v2_backbone_check.py` — seed-pinned,
    float64-cov, JSON-output, protocol-aligned V₁/V₂ backbone empirical probe.
  - created: `experiments/fecam_gate/_env_probe.py` — env-check stub.
  - appended: `_bmad/.session/jett.md` — session log + escalation note.
  - NOT created: `results/fecam_gate/v1v2_empirical_2026-04-19.json` (produced
    by script on run).

---

### [2026-04-19 00:30] Onboarding artifacts created

**What**: Generated `project.md` and this `timeline.md` at repo root per Phase 0
of `docs/research-party/synthesis.md`. Honored global directive naming
(`project.md` + `timeline.md`) over the synthesis's `project-context.md` +
`iteration-log.yaml` convention — global `CLAUDE.md` explicitly overrides.

**Why**: The research-party synthesis is the plan-of-record as of 2026-04-18.
Phase 0 of that plan requires a durable project narrative artifact before
specialists start landing experiment results. The worktree
`.claude/worktrees/gracious-mayer-8aceba/CLAUDE.md` already contained a
comprehensive 2026-04-18 project-state snapshot; distilled its blueprint
content into `project.md` and seeded this ledger with pre-2026-04-19 history
reconstructed from commits, session logs, and progress docs.

**Outcome**:
  - `project.md` (blueprint) — 13 sections covering hypothesis, three-scale
    architecture, current phase, sharp edges, dealbreakers, execution plan.
  - `timeline.md` (this file) — ledger with historical reconstruction + live
    appendix for future entries.
  - `.gitignore` updated: adds `project.md`, `timeline.md`, `_bmad/.session/`,
    `node_modules/`, `package-lock.json`, `package.json`.

**Interpretation**: Phase 0 partially complete. Remaining blocker is arxiv
2512.10969 authorship verification — this is a Dev action, not a Nosh action.

**Follow-up**:
  - Dev: verify arxiv 2512.10969 authorship on arxiv.org
  - Next Nosh commissions (user's choice): (A) Breach → FeCAM-Router ablation
    protocol, (B) Sage → Pólya-urn + conscience proof writeup.

**Files**:
  - created: `project.md` — blueprint artifact per global CLAUDE.md §1.A
  - created: `timeline.md` — research ledger per global CLAUDE.md §1.B
  - modified: `.gitignore` — exclude both state files + `_bmad/.session/`

---

## Historical reconstruction (pre-2026-04-19)

> Reconstructed from git log, `_bmad/.session/*.md`, `progress_report.md`,
> `droid_progress.md`, `HANDOVER.md` snapshot, and the worktree `CLAUDE.md`.
> Entries are approximate on time-of-day; dates are authoritative.

---

## 2026-04-18 — Research party (full day)

### [2026-04-18 14:30] Sova: established-literature map

**What**: Comprehensive four-axis literature map — CL canon, prototype/distance-based
CL, MoE routing canon, CIFAR-100 CL benchmarks.

**Why**: Before research party, establish what MoB must cite, beat, or differentiate
from. Target was a SOTA table split by from-scratch-CNN vs ViT-B/16 eras.

**Outcome**: `docs/lit-review/01-established-cl-moe-cifar100.md`. Key findings:
BASE Layers (ICML 2021) is closest auction precedent; Hash Layers is non-learned-router
precedent; FeCAM (NeurIPS 2023) gives the covariance recipe; RanPAC sets the ~92%
CIFAR-100 ViT-B/16 target. Identified 7 unclaimed spaces MoB probes.

**Interpretation**: MoB's positioning constraints are tight. Cannot claim "first
non-learned router" or "no aux loss" — both preempted by 2021 and 2024 respectively.
The clean positioning space is forgetting-immunity under continual routing.

**Follow-up**: Commission scoped 2024-2025 continual-MoE survey (Fade scope) —
MoLE, LoRA-MoE-for-CL, expert-expansion methods. Also targeted low-rank-covariance
survey for d=4096 question.

**Files**: created `docs/lit-review/01-established-cl-moe-cifar100.md`.

---

### [2026-04-18 14:30] Fade: frontier scouting 2024-2026

**What**: Breadth-first scouting report — prompt-pool CL evolution, PTM+prototype
CL, LoRA-MoE-CL wave, MoE routing frontier (DeepSeek-V3 aux-loss-free, MoE++
zero-computation, ReMoE, Lory, GRIN SparseMixer-v2), upcycling, multi-agent
LLM market mechanisms.

**Outcome**: `docs/lit-review/02-frontier-cl-moe-2024-2026.md`. 9 paradigm shifts
identified. **Critical finding**: arxiv `2512.10969` titled "MoB: Mixture of Bidders"
with abstract matching this project verbatim — flagged as most urgent open question
(Dev's own submission or concurrent work?).

**Interpretation**: Whether this is Dev's own arxiv upload or concurrent work is
the single blocker for the entire research program.

**Follow-up**: CONFIRM 2512.10969 authorship. Until resolved, block further work.

**Files**: created `docs/lit-review/02-frontier-cl-moe-2024-2026.md`.

---

### [2026-04-18 19:45] Astra: auction-theory cross-domain synthesis

**What**: Named MoB as first-score sealed-bid reverse procurement auction with
linear-in-attributes quasi-linear scoring rule (Che 1993). Audited DSIC/efficiency/
IR/BB properties. Derived optimizer-reset + Fisher-clamp fixes as Milgrom-Weber
linkage-principle responses to affiliated-signal winner's curse. Mapped Switch/DeepSeek
load balancing as rediscovery of DeSieno 1988 conscience. Mapped forget_cost to
capacity-market stranded-asset pathology.

**Outcome**: `docs/lit-review/03-auction-theory-cross-domain.md`. 10 transferable
design moves + 7 publishable-gap seeds. Identified DeSieno↔Switch equivalence as
highest-impact/lowest-risk paper seed.

**Interpretation**: The mechanism is not ad-hoc — it has a 30-year auction-theoretic
lineage. This is the scaffolding that lets us legitimately claim "principled."

**Follow-up**: Gate any learned-bidder MoB variants on a Myerson-monotonicity proof
before experiments.

**Files**: created `docs/lit-review/03-auction-theory-cross-domain.md`.

---

### [2026-04-18 19:55] Round-1 position papers (Sage, Chamber, Killjoy, Astra, Fade, KAY/O)

**What**: Six specialists wrote independent position papers BEFORE reading each
other, on the best MoB design across three scales.

**Outcomes** (all in `docs/research-party/round1/`):
  - `sage-position.md` — 8 theoretical claims, 3 scale analyses, Lyapunov+two-timescale-SA
    convergence theorem attempt with A1-A7 assumptions, explicit proof gaps. Core claims:
    no global convergence proof exists; Pólya-urn requires DeSieno conscience; Scale-3
    pre-training is NOT grounded (EWC needs prior-task anchor); DSIC non-issue unless
    α/β are learned.
  - `chamber-position.md` — S1 full CNN + W_route 64-d projection; S2 frozen ViT-B/16 +
    LoRA(r=8) + per-expert FFN bottleneck; S3 DeepSeek-style 64+2 + tied low-rank U.
  - `killjoy-position.md` — S1 auction is 5× tax dominated by EWC per-expert backward;
    S2 fits iff Fisher lives on LoRA only (frozen ViT); S3 full per-expert covariance
    is 240 GB (infeasible) and K backward passes = 128× training cost (infeasible).
    Prescribed projected-gradient EWC with one shared backward + per-expert Fisher
    dot products.
  - `astra-position.md` — committed to single mechanism across all three scales via
    Che 1993. Forget_cost is the capacity-market premium that distinguishes MoB from
    FeCAM (energy+capacity vs energy-only analogy).
  - `fade-position.md` — drop Split-MNIST as headline; adopt LAMDA-PILOT with CIFAR-100
    + ImageNet-R; S3 target = OLMoE upcycled via Drop-Upcycling.
  - `kayo-position.md` — central threat: the auction is epiphenomenal. Prototype-argmin
    (FeCAM) does the work; β·forget only matters in a regime dominated by Fisher-magnitude
    seed variance (18× per project memory). Proposed one project-killer experiment:
    FeCAM-Router vs MoB v2 on CIFAR-100 20T × 10 seeds.

**Interpretation**: The research program's survival depends on KAY/O's gate. Every
specialist converged on this independently.

**Follow-up**: Round 2 — resolve the three cruxes (DSIC under shrinkage, posted-price
update cost, bid vs DeepSeek-V3 bias above 7B).

**Files**: created six position-paper files under `docs/research-party/round1/`.

---

### [2026-04-18 ~20:20–20:35] Round-2 crux resolutions (Chamber, Killjoy, Astra)

**What**: Round 2 addressed three round-1 cruxes.

**Outcomes** (all in `docs/research-party/round2/`):
  - `chamber-r2.md` — committed OLMoE-1B/7B as Scale-3 base (no upcycling needed
    at 1B/7B — it's native MoE). Accepted Killjoy's projected-gradient EWC (r_f=32,
    256 bytes/expert Fisher state). Added BidTrace instrumentation (zero-param logging
    hook) as first-class interpretability artifact. Retracted S2/S3 per-layer-vs-single-auction
    inconsistency in favor of per-layer. New secondary dealbreaker: DeepSeek-V3 bias
    magnitude swamping mahal+forget within 1B Dolma tokens.
  - `killjoy-r2.md` — re-costed auction at 5-8B across OLMoE-1B/7B, Mistral-7B upcycled,
    Llama-3-8B upcycled, Phi-3.5-mini upcycled. **OLMoE ranked #1**: fits single
    A100/H100 80GB full FT, 2-5% throughput tax, 8MB bid-log per forward at fp16.
    Priced interpretability: full training-run bid log = 25 TB infeasible; 1% sampling
    → 250 GB tractable. New dealbreaker: interpretability bid tensor must be detached
    from backward graph (else +128 MB activation memory OOMs at B=8 without grad ckpt).
  - `astra-r2.md` — resolved all three R1 cruxes: (1) fix λ as public mechanism
    parameter to restore Che-1993 DSIC under shrinkage; (2) posted-price menu at Scale 3
    is free only if menu updates are SHARD-LOCAL with lazy reduce-at-checkpoint; (3)
    DeepSeek-V3 aux-loss-free bias is operationally a posted-price update but theoretically
    un-framed — MoB is the first to name it as such. Integrated interpretability as
    co-equal claim via the Hurwicz revelation-principle argument: under DSIC the bid
    IS the private type, so bid decomposition is structurally interpretable, not post-hoc.

**Interpretation**: All three round-1 cruxes are resolved. Remaining blocker is the
empirical one — the FeCAM-Router gate.

**Follow-up**: Nosh writes synthesis.

**Files**: created three round-2 files under `docs/research-party/round2/`.

---

### [2026-04-18 ~21:00] Cypher: CIFAR-100 benchmark audit

**What**: Mapped six canonical CIFAR-100 CL protocols (Split-10T/20T/5T, B0, B50,
GCIL). Documented dataset biases: 10% ciFAIR test/train near-duplication,
CIFAR-100N instance-dependent noise, 32×32→224×224 upsampling artifact, 20-superclass
semantic overlap. Assembled provisional SOTA leaderboard for both pretrained-ViT
(L2P/DualPrompt/CODA-Prompt/SLCA/RanPAC/FeCAM/SimpleCIL) and from-scratch-CNN eras.
Produced minimum-credibility checklist + 7 PI-only open questions.

**Outcome**: `docs/lit-review/04-cifar100-benchmark-audit.md`. Flagged FeCAM as
architecturally identical to MoB v2 Mahalanobis routing — reviewers will demand
head-to-head.

**Follow-up**: PI decisions pending: task-count mapping (5T vs 10T), backbone
(frozen ViT vs from-scratch ResNet vs both), rehearsal stance, class ordering.

**Files**: created `docs/lit-review/04-cifar100-benchmark-audit.md`.

---

### [2026-04-18 22:00?] Nosh: research-party synthesis

**What**: Cross-specialist synthesis. Converted the 6 round-1 positions + 3 round-2
crux resolutions into a single plan-of-record: consensus commitments, scale-specific
design decisions, 3 cruxes with resolution paths, 6 dealbreakers, 5-phase execution
order.

**Outcome**: `docs/research-party/synthesis.md` (208 lines). The plan-of-record
as of 2026-04-18.

**Interpretation**: The research program survives iff MoB beats FeCAM-Router on
Split-CIFAR-100 20T × 10 seeds with CIs excluding zero. Everything else conditional
on passing that gate.

**Follow-up**: Phase 0 — Dev verifies arxiv 2512.10969; Nosh commissions onboarding.

**Files**: created `docs/research-party/synthesis.md`.

---

## 2026-04-03 to 2026-04-06 — v3 prototype-routing-collapse work

### [2026-04-03 to 2026-04-06] Droid session: fix training-time prototype routing collapse

**What**: Targeted the 30-35% collapse of prototype routing vs 78% label routing
on Split-MNIST. Implemented 7 mechanisms (conscience, seeding, temperature,
Fisher threshold, task-warmup, blend, online Mahalanobis) — all backward-compatible,
CLI-gated.

**Root-cause analysis (3-layer)**:
  1. **Feedback loop**: `min()` over centroids + 100.0 default cliff for idle
     experts → one expert captures all wins at warmup→prototype switch.
  2. **EWC Fisher poisoning**: Stray wins on idle experts (50-250) with λ=1000
     + Fisher clamp 0.1 = permanent freeze at partially-trained state.
  3. **Euclidean concentration**: In 128-D, gap between "good match" and "bad match"
     distances compresses — routing loses sharp signal.

**Key result**: `v3_blend_0to05` = **65.4%** accuracy with `blend_end=0.5`
(retain 50% label signal throughout). +30pp over prototype baseline, -13pp
vs label ceiling.

**Outcome**: 27 unit tests passing (`tests/test_components.py`). 46 experiment
result files in `results/experiments_v3/`. See `droid_progress.md` for full
results table.

**Interpretation**:
  - Hybrid blend is the breakthrough. Pure prototype training collapses across
    every seed + warmup value.
  - Online Mahalanobis proves the distance metric CAN work for training (99%+
    per-task train acc) but EWC tuning breaks eval (29% avg).
  - Conscience bias even capped at 0.1 forces generalists that conflict with
    EWC task-specific consolidation — open question on reconciling with Sage's
    Pólya-urn convergence requirement.

**Follow-up** (from `droid_progress.md §9`):
  1. Validate `blend_end=0.5` across seeds 123, 456, 789.
  2. Sweep `blend_end ∈ {0.3, 0.4, 0.5, 0.6, 0.7}` for seed 42.
  3. Combine online Mahalanobis WITH 0-to-0.5 blend.
  4. λ_ewc sweep under online Mahalanobis (λ=1000 may be too aggressive with sharper routing).
  5. Mahalanobis-aware blend: label → Mahalanobis (not Euclidean) as prototype component.

**NOTE (2026-04-19 reframing)**: The research-party synthesis deprioritizes
continued MNIST collapse-fighting in favor of the CIFAR-100 killer gate. Items
1-5 are now lower priority than Phase 1 FeCAM-Router protocol design. Return to
these only if (a) conscience mechanism validation requires a working training-time
prototype routing baseline, or (b) the CIFAR-100 gate punts back to MNIST-scale work.

**Files modified** (from `droid_progress.md §7`):
  - `contibualmob/pool.py` — all 7 mechanisms wired
  - `contibualmob/prototype_store.py` — online Mahalanobis in `update()`
  - `tests/run_mob_only.py` — 15+ new CLI flags, `ExpertPoolLocal` mirroring
  - `tests/run_continual_mob.py` — matching CLI flags + Fisher gating
  - `tests/test_components.py` — 19 new tests
  - `results/experiments_v3/` — 46 result files

---

## 2026-03-19 to 2026-04-03 — v2 prototype routing scaffolding

**What**: Built `PrototypeStore`, per-class centroids, Mahalanobis distance, eval-time
prototype routing, `forward_features()` hooks on models, `bid_diagnostics.BidLogger`.
Added prototype-routing CLI to both runners. Ran v2 experiment suite (`results/experiments_v2/`).

**Key v2 results** (Split-MNIST, 4 experts × 5 tasks × 2 digits):
  - MoB + prototype routing (λ=5.0): **86.7%** avg acc (best).
  - MoB + pseudo-label routing (λ=40): 79.35%.
  - Continual MoB (λ=1.0): 80.72%.
  - Gated MoE + EWC: 35.31%.
  - Monolithic EWC: 19.90%.

**Overloaded expert problem** identified: with 4 experts and 5 tasks, one expert
(E1) handles two tasks ({2,3} and {8,9}). High-λ protects {2,3} but blocks {8,9};
low-λ learns {8,9} but forgets {2,3}. This is NOT a routing failure — prototype
routing correctly sends 8,9 to E1. It's a **single-expert capacity limit under EWC**.

**Interpretation**: The overload is the research question. Do not "fix" by going
to 5 experts. (Project memory rule.)

**Files modified / created** (summary):
  - created: `contibualmob/prototype_store.py`
  - modified: `contibualmob/pool.py` — eval-time prototype routing
  - modified: `contibualmob/models.py` — `forward_features()` (see sharp edge §9.1
    of project.md — this hook is incomplete in current tree)
  - created: `progress_report.md` (March 19 snapshot)
  - created: `EXPERIMENTS.md` (v2 experiment guide)

---

## 2026-02-05 to 2026-03-19 — Phase 1: pseudo-label MoB + baselines + Optuna ablations

**What**: Built the original MoB scaffold. Task-aware (`mob/`) and continual (`contibualmob/`)
runners. Four baselines (Naive, Random, MonolithicEWC, GatedMoE). Optuna TPE
hyperparameter search across MoB and all baselines. Generated ablation plots and
the pre-print paper draft.

**Key Phase-1 results** (pseudo-label routing, Optuna-tuned):
  - MoB-TaskAware: 79.03% avg acc (README headline).
  - MoB-Online (continual): 90.22% avg acc (README headline).
  - Best Optuna hyperparameters:
    - MoB: λ=277.54, α=0.3549, β=0.4151
    - Continual: λ=971.27, α=0.5278, β=0.6333, shift_threshold=2.58

**Files created** (summary):
  - `mob/` full package (8 files)
  - `contibualmob/` full package (9 files)
  - `tests/` full runner + baseline suite
  - `results/optuna_search_*.json` (6 files)
  - `results/ablation_plots/` (46 PNG files)
  - `ablation_continual.png`, `ablation_mob.png`, `ablation_results.txt`
  - pre-print paper `MoB.pdf` at `~/Downloads/` (not in repo)

**Key fixes from project memory**:
  - **Fisher clamp min=0.1** in `_normalize_fisher()` — fixes 18× Fisher variance
    across initializations. Overloaded expert retention went 0% → 87% on first task.
  - **Optimizer reset** per task (task-aware) or on shift detection (continual).
    Both needed alongside Fisher clamp.

---

## Cognitive state / open questions carried forward

These are questions that are neither resolved nor explicitly scheduled, but must
not be lost:

1. **arxiv 2512.10969 authorship** — hard Phase-0 blocker. Dev action.
2. **Class ordering for CIFAR-100** — random vs superclass-aware vs adversarial.
   Cypher's open PI question. Defer to Breach during Phase 3 protocol design.
3. **Which continual-fine-tuning benchmark at Scale 3** — sequential domain
   adaptation (math/code/science/biomedical)? Continual instruction tuning?
   Fade to recommend closer to Phase 4.
4. **Does MoB need top-k routing at Scale 3?** Chamber leans top-1; Astra suggests
   combinatorial extension; Killjoy flags NP-hard clearing. Defer until Phase 3.
5. **Reconcile conscience bias with EWC task-specific consolidation**. v3 experiments
   show `max_bias=0.1` still forces unwanted generalists. Sage's Pólya-urn fix
   requires the conscience term — so the tension is real and load-bearing.
6. **`forward_features()` hook audit**: `contibualmob/models.py` silently lacks it;
   runner scripts fall back to logits-only with no warning. Needs explicit fix
   before any new prototype-routing experiment is trusted.
