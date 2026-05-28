---
title: MoB Research Party — Synthesis & Best Way Forward
author: Nosh (Lab Director)
date: 2026-04-18
scope: routing design across MNIST / CIFAR-100 / LLM MoE scales
inputs: astra-position.md, chamber-position.md, killjoy-position.md, sage-position.md, fade-position.md, kayo-position.md
---

# Executive Decision

**The research program survives if and only if one empirical result holds: MoB beats FeCAM-Router on Split-CIFAR-100 (20 tasks, 10 seeds, matched compute) by a delta whose 95% CI excludes zero.** This is KAY/O's killer experiment and it is the first gate. Everything else in this document is conditional on passing it.

**Blocker above the gate**: arxiv 2512.10969 authorship must be verified. If Dev's own submission, proceed. If not, the positioning collapses and the program is scooped.

---

# 1. Consensus Across All Six Specialists

These commitments are not contested by any position paper:

1. **Split-MNIST is not a headline benchmark.** CIFAR-100 via LAMDA-PILOT harness is the credibility bar. (Fade, Cypher, KAY/O.)
2. **Experts become LoRA adapters over a frozen ViT-B/16 at CIFAR-100 scale.** Full independent networks don't scale and don't match the 2024+ PTM-CIL convention. (Chamber, Fade, Killjoy.)
3. **EWC is restricted to LoRA + classifier head (~300K params), not the full backbone.** Full-backbone Fisher is 2.7GB at K=8 on ViT-B. (Killjoy's hard limit.)
4. **Tied low-rank Mahalanobis (shared U across experts, r≈32) is mandatory at Scale 3.** Full per-expert 4096×4096 covariance = 240GB at 128 experts × 60 layers. (Killjoy, Chamber.)
5. **Projected-gradient EWC with one shared backward + per-expert Fisher dot products.** Per-expert backward = 128× training tax; dead on arrival. (Killjoy non-negotiable.)
6. **DeSieno conscience mechanism is necessary, not stylistic.** Sage's Pólya-urn proof shows the naive argmin auction has a preferential-attachment fixed point that collapses to 1-2 experts. A conscience term is mathematically required for convergence. This is the theoretical fix to the training-time prototype-routing collapse.
7. **Reposition the project from "no auxiliary loss" to "forgetting-immunity under continual routing."** DeepSeek-V3's bias-term aux-loss-free router has already claimed the "no aux loss" ground in 2024. MoB's remaining differentiator is the forget_cost term — the auction-theoretic "capacity-market premium" Astra identified. (Fade, KAY/O, Astra converge.)
8. **Fine-grained expert granularity at Scale 3**: DeepSeek-style 64 routed + 2 shared. 4-expert is toy-scale relative to the 2025 frontier. (Chamber, Fade.)
9. **Continual fine-tuning, not pre-training, at LLM scale.** EWC requires prior-task anchors; at pre-training from random init there are none. (Sage and KAY/O agree.)
10. **The cross-scale invariant**: the bid must reduce to a single matvec against a per-expert vector of bounded size, depending only on the bidder's own private type plus public mechanism parameters. This survives from 128-dim CNN to 4096-dim transformer. (Merged Chamber + Killjoy + Astra invariants.)

---

# 2. Scale-Specific Design Commitments

## 2.1 Split-MNIST / CIFAR-10 (debug scale)

**Purpose**: mechanism development only. Not a paper contribution.

- Keep full independent CNN experts (4 experts, do not change).
- Add Chamber's `W_route`: 128 → 64 L2-normalized projection head trained contrastively. Decouples routing space from classification logits. Attack on training-time prototype collapse.
- Add DeSieno conscience term to bid: `b_i = α·d_M + β·c_forget + γ·(f_i / f̄)` where `f_i` is expert i's recent win rate. This is Sage's mathematical fix. Start `γ` at 0.1× scale of α.
- Calibrate α, β as posted prices against empirical medians of `d_M` and `c_forget` (Astra's prescription).
- Standard baselines in same harness: FeCAM, L2P (for completeness), iCaRL, DER++.

## 2.2 CIFAR-100 (credibility scale)

**Purpose**: this is where MoB must beat FeCAM to continue existing.

- Frozen ViT-B/16 backbone. All experts share the backbone.
- Per-expert adapter: LoRA(r=8) on QKV of every block + per-expert bottleneck FFN. ~300K trainable params per expert.
- EWC Fisher computed only over the adapter + classifier head. Full-backbone Fisher is not ours to protect.
- Routing feature: CLS token after the frozen backbone, projected through a shared `W_route` (tied across experts).
- Tied low-rank covariance: shared `U ∈ R^(d × r)` with r=32, per-expert diagonal scaling. Applies FeCAM's shrinkage + correlation-norm + Tukey transform exactly.
- Single auction layer at the CLS output (not per-block) — CLS is where the task signal actually lives.
- Conscience term `γ` active throughout training; DeSieno-style EMA on expert win-frequency.
- Task partition: **5 tasks × 20 classes** (design-faithful — preserves overloaded-expert research question) AND run 10×10 for community-legibility comparison. Cypher's PI question resolves to "do both; report both."
- Mandatory baselines for legibility: FeCAM, L2P, DualPrompt, CODA-Prompt, SLCA, HiDe-Prompt, RanPAC, EASE. Plus at least one LoRA-MoE-CL entry (D-MoLE, SMoLoRA, or OPLoRA).

**Gate decision at this scale**: FeCAM-Router vs MoB with overlapping 95% CIs means the auction is epiphenomenal and the paper either pivots to forgetting-immunity-only framing or terminates.

## 2.3 LLM MoE FFN layer (scaling target)

**Purpose**: demonstrate that the mechanism survives a real transformer without structural rewrite.

- Base system: **OLMoE upcycled via Drop-Upcycling** (Fade's recommendation). Realistic academic compute envelope.
- Expert granularity: 64 routed + 2 shared (DeepSeek-V3 style). Auction runs over the 64 routed experts; shared experts always fire.
- Auction layer: replaces Mixtral-style `gate = softmax(W_g · x)` with `winner = argmin_i(α·d_M,i + β·c_forget,i + γ·conscience_i)` per token per MoE layer.
- Mahalanobis routing: per-layer shared `U ∈ R^(4096 × 32)` projection, per-expert diagonal covariance in the projected space. ~131K covariance params per layer versus ~1B for full per-expert covariance.
- EWC forget_cost: projected-gradient approximation. One shared backward per step; per-expert Fisher dot products in the projected space. ~4K FLOPs/token mechanism overhead versus ~2 TFLOPs FFN. Negligible.
- Training regime: **continual fine-tuning only** (domain adaptation, instruction tuning, RLHF-adjacent). Not pre-training.
- Near-winner prototype update: EMA on shared features for losing experts whose bid is within a window of the winner's. Addresses Chamber's stale-prototype dealbreaker.
- Posted-price mechanism framing (Astra's Scale-3 structural shift): the public α, β, γ are the posted prices; per-token settlement is DSIC under private-value assumption. Preserves truthfulness without cross-token coupling.

---

# 3. The Three Cruxes That Need Resolution Before Implementation

These are the open disagreements among specialists that block concrete design work:

### Crux 1 (empirical — first gate)

**Question**: Is the auction mechanism epiphenomenal?
- KAY/O's central threat: prototype-argmin (FeCAM) does the work; β·forget only matters in a regime dominated by seed variance.
- Astra's counter: forget_cost is the capacity-market premium FeCAM lacks.
- Sage's testable prediction: if `α·var_x[d_M] >> β·‖F_i‖` empirically in our codebase away from task boundaries, KAY/O is right.

**Resolution path**: Run the FeCAM-Router ablation on Split-MNIST first (1 day), then Split-CIFAR-100 20T (~1 week). If the gap is <1% with overlapping CIs on CIFAR-100 with 10 seeds, KAY/O wins and we execute the kill-or-pivot gate.

### Crux 2 (theoretical — blocks Scale 3)

**Question**: Is linear-in-attributes Che-1993 scoring DSIC under a shrinkage estimator with data-dependent λ?
- Astra's dealbreaker: if not DSIC, the same-mechanism-at-all-scales claim breaks.
- Sage's resolution: fix λ as a public parameter (not data-dependent) and DSIC holds.

**Resolution path**: Commit to fixed public λ per layer (calibrated once, held fixed during routing). Sage writes the proof sketch as part of the theory section.

### Crux 3 (empirical — blocks paper claim)

**Question**: Does bid-aware routing outperform DeepSeek-V3 bias-term balancing above 7B parameters?
- Killjoy's crux: if not, the mechanism is "in search of a problem."
- Astra's counter: DeepSeek-V3 bias is a special case of the auction with β=0 (no forget cost). MoB contains it.
- Fade's dealbreaker: a lab combining DeepSeek-V3 bias + prototype + EWC in the next 6 months scoops us.

**Resolution path**: Position the claim as "forgetting-immunity" not "load-balance." Design the Scale-3 evaluation explicitly as a continual-fine-tuning benchmark (e.g., sequential domain adaptation on 5-10 domains) rather than a pre-training benchmark. This is the only regime where MoB's forget_cost differentiator is load-bearing.

---

# 4. Dealbreaker Watchlist

From the specialists' own dealbreaker clauses:

| # | Dealbreaker | Source | Detection |
|---|---|---|---|
| 1 | FeCAM-Router ties MoB on CIFAR-100 20T with overlapping 95% CIs | KAY/O | First gate experiment |
| 2 | EMA prototypes stale within 10k steps at Scale 3 | Chamber | Synthetic simulation before full-scale port |
| 3 | No finite γ stabilizes Scale-3 pre-training | Sage | Pre-training explicitly out of scope (resolved) |
| 4 | Concurrent paper combining DeepSeek-V3 bias + prototype + EWC | Fade | Weekly arxiv watchlist on DeepSeek/AI2/LAMDA |
| 5 | Linear-in-attributes scoring not DSIC under data-dependent shrinkage | Astra | Fix λ as public parameter (resolved) |
| 6 | 2512.10969 is not Dev's | all | Manual arxiv.org check (today) |

---

# 5. Proposed Execution Order

Assuming 2512.10969 is Dev's own submission:

## Phase 0 — Prerequisites (this week)

- [ ] Dev verifies 2512.10969 authorship on arxiv.org
- [ ] Nosh commissions `OB` (onboarding) — generates `project-context.md`, seeds `iteration-log.yaml`

## Phase 1 — The Killer Gate (2-3 weeks)

- [ ] **Breach**: design the FeCAM-Router ablation protocol rigorously (matched compute, seed-variance budget, CI methodology)
- [ ] **Jett**: port MoB to LAMDA-PILOT harness
- [ ] **Jett**: implement FeCAM-Router in the same harness (forget_cost term zeroed, everything else identical)
- [ ] **Jett**: run head-to-head on Split-CIFAR-100 20T, 10 seeds, both frozen ViT-B/16 and from-scratch ResNet-18
- [ ] **Nosh**: adjudicate the gate

**Outcome branch**:
- If MoB wins by measurable delta → proceed to Phase 2.
- If FeCAM ties → pivot MoB's claim to continual-fine-tuning-only, rerun Phase 1 on a sequential-domain benchmark (not CIL), and re-gate.
- If FeCAM beats MoB → terminate the routing-mechanism thesis and extract the prototype-store engineering as standalone contribution.

## Phase 2 — Conscience Mechanism (1-2 weeks, parallel with Phase 1 late stages)

- [ ] **Sage**: formal write-up of the Pólya-urn collapse theorem and DeSieno-conscience fix
- [ ] **Jett**: implement conscience term in MoB code; retest Split-MNIST training-time prototype routing
- [ ] **Omen**: standard code review of the conscience implementation

## Phase 3 — CIFAR-100 Full Suite (4-6 weeks)

- [ ] **Chamber**: formalize the frozen ViT-B/16 + per-expert LoRA architecture (module spec, tensor shapes, parameter budget)
- [ ] **Killjoy**: compute audit of the Scale-2 design (FLOPs, memory, throughput vs baselines)
- [ ] **Jett**: implement the Scale-2 system
- [ ] **Breach**: design the full CIFAR-100 benchmark suite (5T×20C AND 10T×10C, all mandatory baselines)
- [ ] **Jett**: execute benchmark suite
- [ ] **KAY/O**: adversarial review of results before any paper claim
- [ ] **Omen**: standard code review

## Phase 4 — LLM MoE Port (months 3-6)

- [ ] **Fade**: confirm OLMoE is still the best upcycling target (frontier moves fast; re-check in 2 months)
- [ ] **Chamber**: Scale-3 module spec with pseudocode
- [ ] **Killjoy**: projected-gradient EWC implementation plan
- [ ] **Sage**: theoretical argument for posted-price DSIC at Scale 3 (written before implementation, not after)
- [ ] **Jett**: port MoB router into OLMoE. Start with 8-expert subset for sanity; scale to 64+2.
- [ ] **McGonagall / Killjoy**: distributed training setup
- [ ] **Jett**: continual fine-tuning benchmark (TBD — Fade to recommend)

## Phase 5 — Paper (continuous)

- [ ] **Sage**: theory section (Pólya urn, DSIC, convergence conjecture with explicit assumptions)
- [ ] **Astra**: related-work contextualization (BASE Layers, Hash Layers, FeCAM, DeSieno, Che 1993)
- [ ] **Fade**: forbidden-claims enforcement during writing

---

# 6. What This Party Deliberately Did Not Decide

Three design questions deferred to a later round:

1. **Does MoB need top-k routing at Scale 3?** Single-item auction is structurally weaker than Mixtral top-2. Chamber leans top-1; Astra suggests a combinatorial extension; Killjoy flags NP-hard clearing. Defer until Phase 3 results are in.
2. **Class ordering on CIFAR-100.** Random ordering vs superclass-aware vs adversarial. Cypher's open PI question. Low-cost; decide when Breach writes the Phase 3 protocol.
3. **Which continual-fine-tuning benchmark at Scale 3.** Sequential domain adaptation (math/code/science/biomedical)? Continual instruction tuning? Fade to recommend closer to Phase 4.

---

# 7. Positioning Discipline (from Fade's "Forbidden Claims")

The paper must NOT claim:
- "First non-learned router" (Hash Layers predates, 2021)
- "No auxiliary loss required" (DeepSeek-V3 2024 achieved this with bias term)
- "Solves catastrophic forgetting" (no method does)
- "Biologically plausible" (no method is)
- "Principled" without the auction-theoretic scaffolding (Astra's Che 1993 + DSIC proof is the scaffolding; don't claim principled without it)

The paper CAN and SHOULD claim:
- Forgetting-immune routing for continual fine-tuning
- Posted-price mechanism framing with DSIC guarantee under fixed public λ
- Pólya-urn-stabilized via DeSieno conscience (the theoretical contribution)
- Same mechanism from 128-dim CNN to 4096-dim transformer (the engineering claim)

---

*End of synthesis. Round 2 is available on demand if any crux needs adjudication before Phase 1 begins.*
