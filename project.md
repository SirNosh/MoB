# project.md — MoB (Mixture of Bidders) Blueprint

> **Private state file. Governed by global directive in `C:\Users\devya\.claude\CLAUDE.md`:
> this file is in `.gitignore` and MUST NEVER be pushed to GitHub or any public repo.**

> **Purpose**: the macroscopic, stable overview of the project. Updated only on
> fundamental pivots in scope, architecture, or methodology.
> For the chronological ledger of experiments and decisions, see `timeline.md`.

---

## 1. Core Research Hypothesis

**Replace the learned gating network in Mixture-of-Experts with a stateless auction
over expert bids, and demonstrate it transfers intact from CNN-scale to LLM-scale
MoE transformers without structural rewrite.**

Each expert computes a bid on every batch:

```
bid_i   = α · execution_cost_i + β · forgetting_cost_i
winner  = argmin_i(bid_i)
```

- **execution_cost**: "how well do I already handle this input?" — low = good fit.
- **forgetting_cost**: "how much would training on this damage what I know?" — EWC-style, low = safe.
- **winner** trains on the batch; losers observe and update prototypes.
- No learned router parameters → immune to gater-forgetting in continual learning.
- DSIC-truthful under Che-1993 linear-in-attributes scoring (formal argument in §5).

**Positioning (post-research-party, 2026-04-18)**: the claim is **"forgetting-immunity
under continual routing,"** NOT "no auxiliary loss" (DeepSeek-V3 2024 preempted that)
and NOT "first non-learned router" (Hash Layers 2021 preempted that). The load-bearing
differentiator is the `forget_cost` term — the auction-theoretic capacity-market
premium that FeCAM lacks. See `docs/research-party/synthesis.md` §7.

---

## 2. Three-Scale Architecture Target

The single mechanism must survive three scales without structural change:

| Scale | Purpose | Backbone | Experts | Routing granularity | Feature dim | Covariance |
|---|---|---|---|---|---|---|
| **S1 — Split-MNIST / CIFAR-10** | Mechanism development only. Not a paper claim. | Independent SimpleCNN per expert | 4 (intentional — 4-for-5 overload is load-bearing) | Per-batch | 128 | Full (FeCAM on 128) |
| **S2 — Split-CIFAR-100** | Credibility gate. Must beat FeCAM-Router with CIs excluding zero. | Frozen ViT-B/16, shared | **K=4** experts, each = LoRA(r=8) on QKV + per-expert FFN bottleneck (~300K params/expert). **S1/S2 share K=4** per `feedback_4experts.md` — 4-for-5/4-for-20 overload is load-bearing across both scales | Per-sample top-1 at CLS | 768 (CLS) | Tied low-rank U (r=32) + per-expert diag |
| **S3 — LLM MoE FFN** | Same-mechanism-at-all-scales demonstration. | OLMoE-1B/7B (native 16L × 64 experts, d=2048) | 64 routed + 2 shared (DeepSeek-V3 style) | Per-token, per-MoE-layer | 2048 | Tied low-rank U per layer (r=32) + per-expert diag |

**Cross-scale invariant** (merged Chamber + Killjoy + Astra):

> The bid must reduce to a single matvec against a per-expert vector of bounded size,
> depending only on the bidder's own private type plus public mechanism parameters.
> It must never depend on other bidders' bids or on a learned router.

Implementation: a parameter-free `AuctionRouter` with fixed type signature whose
buffers (`μ`, `diag`, `U`, `fisher`) update by EMA.

---

## 3. Current Development Phase (April 2026)

The repo is in **Phase 1 — Killer Gate, HPC deployment track**. Protocol v1.2.1
(FeCAM-composite-binding + training-spec-pinned) is frozen-ready for 2026-04-26
signature. Jett delivered the complete `HPC/` gate-runner (2,629 LOC, 2026-04-19)
for the ACIDSDB cluster (V100 32GB, qGPU48, cuda/12.1). Gate target: MoB vs
FeCAM-Router Mahalanobis, Split-CIFAR-100 LAMDA-PILOT harness, **20T×5C**
(protocol §2.1 headline config), **n=10 primary seeds {42..51} with
D3-conditional n=20 backup {52..61}** prestaged in the same 40-job slurm array
(§4.3 / §6.5 D3).

**Current status (2026-04-19)**: Awaiting Jett's `HPC/` directory delivery
→ Dev upload to ACIDSDB → cluster-side pilot → full slurm array (40 jobs) →
results download → KAY/O post-experiment red-team → BCa bootstrap → pass/tie/fail.

Pre-Phase-1 compute-efficiency work now deprioritized. Phase 3b's prototype-collapse
fight at Split-MNIST is **parked** — research-party decision 2026-04-18 rules
it subordinate to Phase 1 gate outcome. Context for future reference:

- **Phase 3b historical**: Training-time prototype routing collapsed to 10–35%
  accuracy vs ~78% with label-based routing on Split-MNIST. Best v3 result was
  `v3_blend_0to05` = 65.4% (hybrid blend, 50% label signal retained). MNIST work
  continues only for **conscience mechanism validation** (Sage's Pólya-urn fix)
  where its signal is still usable.
- **Phase 1 active**: MoB-vs-FeCAM-Router on Split-CIFAR-100 20T × 20 seeds is
  the survival test. Frozen protocol: `docs/protocols/fecam-router-gate.md` v1.2.1.
  HPC deployment is the engineering path to full n=20 within Phase-1 window.

---

## 4. Pipeline Stages (Current)

```
DataLoader (Split-MNIST, 5 tasks × 2 digits, batch_size=32)
    │
    ▼
ExtendedExpertPool (contibualmob/pool.py)
    ├── ShiftDetector (EMA cost, threshold_multiplier) — task-free stream
    ├── PrototypeStore per expert (MIN_SAMPLES_FOR_MAHALANOBIS=256)
    ├── ConscienceBias (DeSieno, rate=0.005, max_bias=0.1)
    ├── TemperatureAnneal (Boltzmann during training, argmin at eval)
    └── RoutingBlend (λ·label + (1−λ)·prototype, λ linear over training)
    │
    ▼
Per-expert bid = α · exec_cost(x, y̅) + β · EWC(x, F_i, θ*_i)
    ├── α, β posted prices calibrated against empirical medians (Astra)
    ├── Fisher clamp min=0.1 (load-bearing — see §6.2)
    └── Fisher threshold: skip update if expert_wins < 100 (ea2e233)
    │
    ▼
PerBatchAuction (single-winner argmin; second-price payment for logs)
    │
    ▼
Winner trains; losers observe and update prototypes; optimizer-reset on task end
```

**Tokenization / prompt format**: N/A at S1–S2. At S3 inherits OLMoE's tokenizer.

---

## 5. Evaluation Metrics

### At S1 (Split-MNIST, current)

| Metric | Status | Reporting |
|---|---|---|
| Avg accuracy across tasks | Primary | mean ± std over seeds |
| Forgetting (BWT) | Primary | per-task accuracy drop from peak |
| Routing entropy | Secondary | Shannon entropy across experts per batch |
| Gini coefficient of expert utilization | Secondary | target < 0.3 |
| Per-digit routing heatmap | Diagnostic | `bid_diagnostics.BidLogger` |
| Winner-vs-loser distance separation | Diagnostic | currently 24–26% |

### At S2 (CIFAR-100, planned)

LAMDA-PILOT harness, both 5T × 20C (design-faithful) and 10T × 10C (legibility).
Mandatory baselines: FeCAM (headline), L2P, DualPrompt, CODA-Prompt, SLCA, HiDe-Prompt,
RanPAC, EASE + one LoRA-MoE-CL (D-MoLE, SMoLoRA, or OPLoRA).

**Gate criterion**: MoB vs FeCAM-Router 95% CI must exclude zero on Avg Acc. Ties
trigger pivot to continual-fine-tuning-only framing or termination.

### At S3 (LLM, planned)

Continual fine-tuning benchmark (TBD — Fade recommends closer to Phase 4). NOT
pre-training. Candidates: sequential domain adaptation (math/code/science/biomed),
continual instruction tuning. Metrics: per-domain perplexity, routing distribution
entropy, load-balance Gini, forgetting on prior domains, per-token routing cost.

---

## 6. Dataset & Benchmark Strategy

**S1**: Split-MNIST (5 tasks × 2 digits, 4 experts — 4-for-5 overload is intentional,
see `memory/feedback_4experts.md`). Secondary: Split-CIFAR-10.

**S2**: Split-CIFAR-100 via LAMDA-PILOT (20T-canonical, 5T-ours, 10T-community).
Class ordering: defer to Breach during Phase 3 protocol design (random vs superclass
vs adversarial — Cypher's open PI question).

**S3**: OLMoE upcycled via Drop-Upcycling (Fade). Continual-FT corpora TBD at Phase 4.

**Explicitly out of scope**: Split-MNIST is NOT a headline benchmark. The pre-print
paper's Phase-1 numbers (79.03% / 90.22% MoB-TaskAware / MoB-Online) will not appear
in the next paper draft.

---

## 7. Auction-Theoretic Foundations

Via Astra's cross-domain synthesis (`docs/lit-review/03-auction-theory-cross-domain.md`):

- **Mechanism class**: first-score sealed-bid reverse procurement auction with
  linear-in-attributes quasi-linear scoring rule (Che 1993).
- **DSIC**: holds under Che 1993 when the shrinkage parameter λ is a **public posted
  parameter** (not data-dependent). Fixing λ as public was the round-2 crux resolution.
- **Convergence**: no global proof exists. Within-task local convergence holds under
  Borkar two-timescale SA + strict-margin + slow-timescale prototypes (Sage §6 proof
  attempt, 2-4 weeks from publishable).
- **Collapse pathology**: naive argmin auction has a Pólya-urn preferential-attachment
  fixed point. DeSieno conscience term `γ · (f_i / f̄)` is **mathematically necessary**,
  not stylistic. This is Sage's theoretical contribution.
- **Cross-domain ancestors**: BASE Layers (ICML 2021) = closest auction precedent;
  Hash Layers (NeurIPS 2021) = non-learned router precedent; FeCAM (NeurIPS 2023) =
  covariance recipe; DeSieno 1988 = conscience mechanism (rediscovered by Switch/DeepSeek).
- **Explicit non-ancestors**: Holland bucket brigade is NOT a structural ancestor
  (Astra ruled this out).

---

## 8. Codebase Topology (2026-04-19)

```
MoB Final/
├── mob/                    — Task-aware MoB (Phase 1 legacy, stable, pseudo-label only)
├── contibualmob/           — Continual/online MoB (typo intentional). ACTIVE at S1.
│   ├── prototype_store.py  —   PrototypeStore (Mahalanobis w/ online recompute)
│   ├── pool.py             —   ExtendedExpertPool (all v3 mechanisms wired here)
│   ├── bidding.py          —   EWCForgettingEstimator, Fisher clamp min=0.1
│   └── ...
├── HPC/                    — ★ Jett-owned gate-runner for ACIDSDB (in flight 2026-04-19).
│   ├── README.md, HPCguide.md, requirements.txt
│   ├── slurm/run_gate.sh
│   └── gate_runner/        — Self-contained Phase-1 gate code
│       ├── run_arm.py, fecam_core.py, auction.py, experts.py
│       ├── data.py, ewc.py, config.yaml, utils.py
│       └── tests/test_acceptance.py  — 20 acceptance tests from protocol §15
├── experiments/fecam_gate/ — Local measurement scripts (v1v2_backbone_check.py etc)
├── tests/
│   ├── run_mob_only.py     — PRIMARY task-aware runner (1516 lines)
│   ├── run_continual_mob.py — PRIMARY online runner (709 lines)
│   ├── test_components.py  — 27 unit tests, all passing
│   └── ...
├── results/
│   ├── experiments_v3/     — Phase 3b MNIST v3 (24 JSON × 12 experiments, parked).
│   ├── fecam_gate/         — V₁/V₂ empirical measurements.
│   └── gate/               — ★ HPC Phase-1 gate results land here post-download.
├── docs/
│   ├── protocols/          — fecam-router-gate.md v1.2.1 (frozen 2026-04-26)
│   ├── theory/             — polya-urn-conscience-proof.md (Sage §6 rework pending)
│   ├── lit-review/         — 4 literature/frontier/cross-domain surveys + 05-fecam-code-comparison
│   └── research-party/     — synthesis.md + round1/round2/ position papers
├── _bmad/.session/         — Per-specialist session logs (private)
└── project.md, timeline.md — THIS FILE + chronological ledger (both gitignored)
```

Two codebases, one project: GitHub remote `SirNosh/MoB` is well behind local. It
has only Phase 1 (pseudo-label) code. Ignore the remote for now; the local worktree
is authoritative. The pre-print `MoB.pdf` at `~/Downloads/MoB.pdf` reflects the
older pseudo-label Phase 1; it does NOT reflect the FeCAM-Router gate framing.

---

## 9. Sharp Edges (Must Read Before Modifying Core Code)

1. **`contibualmob/models.py` does NOT define `forward_features()`.** The runner
   scripts do `hasattr(...)` and silently fall back to logits-only, which silently
   disables prototype routing. Verify before assuming it runs. (§6.1 of worktree
   CLAUDE.md snapshot, carried forward here.)
2. **Fisher clamp `min=0.1`** in `contibualmob/bidding.py:_normalize_fisher()` is
   load-bearing. Without it, overloaded expert drops to 0% retention on first task
   (18× Fisher variance across initializations). Do not remove.
3. **`contibualmob/__init__.py` exports legacy name `PerBatchVCGAuction`** but
   `auction.py` defines `PerBatchAuction`. Runners import from submodules directly
   so this doesn't currently fire, but `from contibualmob import PerBatchVCGAuction`
   will ImportError. Do not "fix" by renaming without checking downstream.
4. **Optimizer reset is task-type specific.** Task-aware: reset ALL winning experts
   at task END, after Fisher update. Continual: reset the winner on shift detection.
   Always enable `--reset_optimizer` for v3 experiments.
5. **Bid scale invariant**: all experts in the same auction produce bids on the same
   scale. Idle experts get `distance_score = 100.0` sentinel (so they lose), not a
   different cost formula. Current normalizations: `distance/10`, `log1p(forget)/10`,
   `CE/2.5`. Mismatches here produced the v1 bug.
6. **Evaluation uses pseudo-labels, never ground truth.** `pool.evaluate_all()` and
   `evaluate_all_per_sample()` compute bids from each expert's own `argmax(logits)`.
   Preserve this invariant if you write new eval paths.

---

## 10. Dealbreaker Watchlist

From the research-party synthesis (`docs/research-party/synthesis.md` §4). Any one
of these triggers a program-level pivot or termination:

| # | Dealbreaker | Source | Detection |
|---|---|---|---|
| 1 | FeCAM-Router ties MoB on CIFAR-100 20T with overlapping 95% CIs | KAY/O | Phase 1 gate experiment |
| 2 | EMA prototypes stale within 10k steps at Scale 3 | Chamber | Synthetic 8-expert sim before OLMoE compute |
| 3 | No finite γ stabilizes Scale-3 pre-training | Sage | Pre-training out of scope → resolved |
| 4 | Concurrent paper combining DeepSeek-V3 bias + prototype + EWC | Fade | Weekly arxiv watchlist (DeepSeek/AI2/LAMDA) |
| 5 | Linear-in-attributes scoring not DSIC under data-dep. shrinkage | Astra | Fix λ as public → resolved |
| 6 | **arxiv 2512.10969 is not Dev's submission** | all | Manual arxiv.org check — **OPEN blocker** |

---

## 11. Execution Plan (Research-Party Phase Order)

From `docs/research-party/synthesis.md` §5. Gate-structured. Updated 2026-04-19
for HPC-deployment pivot.

- **Phase 0 — Prerequisites**: ✅ arxiv 2512.10969 authorship confirmed (2026-04-19) + onboarding done.
- **Phase 1 — Killer Gate (HPC deployment track, 2–3 weeks)**: FeCAM-Router vs MoB on Split-CIFAR-100 **20T×5C** (§2.1 headline config), **n=10 primary seeds + D3-conditional n=20 backup** (§6.5). Protocol v1.2.1 frozen 2026-04-26. Jett delivered `HPC/` gate-runner 2026-04-19 (2,629 LOC). Dev uploads to ACIDSDB → pilot (array 0–5, 3 seeds × 2 arms) → n=10 primary (array 0–19) → D3 conditional (array 20–39) → KAY/O red-team on returned `results/gate/seed_*_arm_*.json` → BCa bootstrap CI on paired Δ → pass/tie/fail.
  - **Gate criterion**: 95% one-sided lower CI on `AIA(MoB) − AIA(FeCAM-Router)` must exceed Δ_practical = 1.0pp.
- **Phase 2 — Conscience Mechanism (parallel with Phase 1 late)**: Sage §6.2/§6.3 quantitative γ_min rework under FeCAM additive shrinkage (blocked on Chamber's L measurement + V₁/V₂ measurements — both now in hand after 2026-04-19). Jett implements `γ_min = 2αL·diam(X)/K` once Sage returns bound.
- **Phase 3 — CIFAR-100 Full Suite (4–6 weeks)**: assuming Phase 1 passes. Expands to full mandatory baseline set (L2P, DualPrompt, CODA-Prompt, SLCA, HiDe-Prompt, RanPAC, EASE + one LoRA-MoE-CL). Also runs Phase-2 β-only/α-only ablations per KAY/O's post-PASS caveat.
- **Phase 4 — LLM MoE Port (months 3–6)**: OLMoE upcycling via projected-gradient EWC + tied low-rank Σ.
- **Phase 5 — Paper (continuous)**: theory section, related-work discipline, forbidden-claims enforcement.

**Current blocker chain (2026-04-19)**:
`Jett HPC sprint → Dev cluster upload → cluster pilot (array 0–5) → full array (6–39) → results download → KAY/O red-team → statistical test → Phase 1 verdict`.
Compute is on the cluster, not local; Nosh's near-term work is coordination (freeze sign, specialist dispatch on return events) not execution.

---

## 11b. Compute Infrastructure — ACIDSDB HPC (added 2026-04-19)

**Cluster**: ACIDSDB (Arctic). User `dvyas4`. Home `/home/users/dvyas4/`.

**GPU nodes**: `acidsgcn001-007`, V100 32GB. Partitions: qGPU24 (24hr),
qGPU48 (48hr), qGPU120 (120hr). Phase-1 gate uses qGPU48.

**Environment**: `module load cuda/12.1 python/3.10.14`. Python venv pattern
documented in `HPC/HPCguide.md` (Jett-authored, in flight).

**Slurm deployment pattern** (Phase 1 gate):
```
#SBATCH --partition=qGPU48
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-39
```
Array index N → `(seed = 42 + N//2, arm = A if N%2==0 else B)`.

**Results flow**: cluster writes `./results/gate/seed_${SEED}_arm_${ARM}.json`
→ Dev downloads to `C:\MoB Final\results\gate\` → Nosh fires KAY/O for
post-experiment review → BCa bootstrap → verdict.

**Local/HPC split**:
- **Local (RTX 4070 SUPER, dev box)**: measurement scripts, spot checks,
  V₁/V₂ probe, Jett-generated smoke tests, interactive debugging.
- **HPC (V100 32GB, cluster)**: full n=20 gate runs, Phase 3 baseline sweeps,
  any job ≥ 6 GPU-hr that is embarrassingly parallel across seeds.

**Deliverable location**: `HPC/` directory at repo root (Jett-owned; in
flight as of 2026-04-19). **Not** pushed to remote — local upload only.

---

## 12. Positioning Discipline (Forbidden / Permitted Claims)

**NEVER claim** (from Fade's frontier audit):
- "First non-learned router" — Hash Layers (NeurIPS 2021) predates.
- "No auxiliary loss required" — DeepSeek-V3 (2024) achieved this with bias term.
- "Solves catastrophic forgetting" — no method does.
- "Biologically plausible" — no method is.
- "Principled" without auction-theoretic scaffolding (Astra's Che 1993 + DSIC proof).

**DO claim**:
- Forgetting-immune routing for continual fine-tuning.
- Posted-price mechanism framing with DSIC guarantee under fixed public λ.
- Pólya-urn-stabilized via DeSieno conscience (theoretical contribution).
- Same mechanism from 128-dim CNN to 4096-dim transformer (engineering claim).

---

## 13. Project-Specific Memory Rules

From `memory/MEMORY.md` + global CLAUDE.md:

- **4 experts for 5 tasks is intentional.** Never add a 5-expert config "to be fair."
  The overloaded expert IS the continual learning problem MoB exists to solve.
- **Fisher clamp min=0.1** is load-bearing (§9.2 above).
- **Optimizer reset + Fisher clamp together** are required; neither alone is sufficient.
- **lambda_ewc=1000** works well with the Fisher clamping fix.
- **VRAM economy**: on OOM analyze gradient checkpointing, KV cache, optimizer states — NOT blind batch-size reduction.
- **Plan first, execute second** for any non-trivial architecture or distributed-training change.
- **Verify on a micro-batch** before claiming a training loop is complete.
