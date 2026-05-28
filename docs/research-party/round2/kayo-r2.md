# KAY/O Red-Team Position — Round 2

**Role:** Adversarial reviewer. Written after reading Astra R2 and Chamber R2, anticipating Sage/Killjoy/Fade R2 from the cross-digest. Every claim that survives here must survive peer review.
**Scope:** SALE citation, interpretability co-equal claim, 5-8B OLMoE pivot, upcycling null-phase, DSIC-under-non-strategic-bidders, refined killer experiment.

---

## 1. Executive summary

1. **Interpretability is the program's softest new claim.** "Decomposable bid" is not "human-usable explanation." 2M bid values per forward pass (64 experts × 16 layers × ~2K tokens) do not become interpretable because three symbols appear in a formula. The claim requires calibration; none is offered.
2. **Sage's forthcoming "interpretability-is-DSIC" theorem is vacuous under non-strategic bidders.** DSIC presupposes rational agents with a strategy space. OLMoE experts are gradient-descent-optimized FFN weight tensors. They do not maximize utility, they minimize loss. "Truthful bid" has no operational content — there is nothing to deviate from.
3. **Astra's α·d_M = 0, β·c_forget = 0 edge case IS DeepSeek-V3.** Sage's own "strict generalization" argument concedes this. The novelty rests entirely on the two nonzero terms having measurable routing value over a scalar bias. No open evidence at 7B that they do.
4. **Upcycling null-phase is fatal to the OLMoE pitch unless Chamber addresses it directly, which he does not.** Drop-Upcycling into shared experts reuses existing OLMoE prototypes? No — OLMoE has no prototypes; the prototypes are MoB's to initialize. At t=0 all `μ_i` are undefined or initialized from the same dense source. Mahalanobis term is zero-signal. Forget term is zero by construction (no prior task). The auction is uniform routing for the first N tokens. Quantify N or retract.
5. **SALE reduces MoB's novelty to a granularity axis.** "First auction at token granularity inside a transformer" is a narrow claim. Reviewer's first question: is the granularity axis a mechanism difference, or a deployment difference? If the mechanism is identical modulo bid-vector dimensionality, the answer is "deployment" and the paper is a systems contribution, not a mechanism contribution.
6. **The refined killer is OLMoE-7B A/B (Job 4-A).** FeCAM-Router at Scale 2 no longer discriminates between positions — everyone assumes MoB will pass it. The Scale-3 A/B against vanilla OLMoE routing is now the load-bearing test.
7. **Central threat rotation:** FeCAM-equivalence (R1) is no longer #1. **Interpretability-is-marketing** replaces it at #1. OLMoE A/B test is #2. Upcycling null-phase is #3.

---

## 2. Interpretability attack (Job 1)

**Concede the structural point.** `bid = α·d_M + β·c_forget + γ·p` is syntactically decomposable. I am not contesting that.

**What I contest:** four concrete gaps between "decomposable" and "interpretable."

### 2.1 Volume: the 2M-bids-per-forward problem
At OLMoE-7B, one forward pass over a 2K-token prompt generates 64 experts × 16 MoE layers × 2048 tokens = **2.10M bid values**, each a three-term decomposition = **6.29M scalars**. Chamber's BidTrace artifact is 7 MB per 2K-token prompt. This is not a human-readable log; it is a data product requiring its own analysis pipeline. "Interpretable by construction" collapses to "logable by construction." Those are different claims. The former requires a calibration study showing humans extract usable information; the latter requires a `parquet.write` call. Chamber has delivered the latter.

### 2.2 Dominance: which term matters?
If β·c_forget is at any token ≥10× α·d_M, the bid decomposition's "distance" component is noise — the routing is determined by forgetting alone, and α is a decorative constant. Vice versa for the other direction. Project memory documents Fisher magnitude varies **18× across random seeds** before a manual clamp (MEMORY.md, EWC Fisher Clamping). The β·c_forget term's magnitude is therefore seed-dependent by construction. A "decomposition" whose component magnitudes are determined by seed choice is not an explanation; it is a seed-specific summary. No two training runs produce the same interpretability story.

### 2.3 Post-hoc or genuine? The Rudin 2019 test
Rudin ("Stop Explaining Black Box Models," Nat. Mach. Intell. 2019) separates *interpretable by design* from *post-hoc explained*. MoB's architecture IS designed to produce a decomposable scoring function — concede. But Rudin's deeper requirement is that each decomposed term be **operationally meaningful to the end user**. "Distance to a learned prototype in a tied low-rank Mahalanobis metric" (Chamber §4.2) is not operationally meaningful to a practitioner. It requires the practitioner to understand `U`, `M_inv_core`, `diag`, tied-shrinkage, FeCAM's normalization, and the relationship between hidden-state geometry and downstream behavior. That is a PhD course, not a legend on a plot. Softmax routing fails the same test, but at least softmax routing does not advertise interpretability as its primary selling point.

### 2.4 The $100 ablation
**β = 0, γ = 0. Re-run.** If accuracy on the OLMoE-7B continual-FT benchmark drops by less than 1% absolute with overlapping 95% CIs across seeds, the two "interpretable" terms MoB contributes over FeCAM are ornamental. The bid collapses to α·d_M, which is FeCAM's rule. Interpretability collapses to "we do prototype routing and call the distance a bid." This ablation is mandatory before any paper claims "decomposition is meaningful." Neither Astra nor Chamber commits to running it — Astra's Section 7 names FeCAM parity as the dealbreaker but does not require the per-term ablation that isolates which term carries the signal.

### 2.5 Attack on Sage's forthcoming theorem
The cross-digest says Sage will argue "linear decomposition of semantically-typed quantities is a faithful explanation under DSIC." Two predicted gaps:

**(a) DSIC vacuity under non-strategic bidders.** Che 1993 DSIC requires bidders who (i) possess private types, (ii) have a strategy space of possible bids, (iii) rationally maximize payoff given beliefs about others. A trained FFN expert satisfies none of these. Its "bid" is a deterministic function of its frozen weights and the current input. It has no action to take other than the one the forward pass computes. "Truthful" is a category error: there is no counterfactual deviation to compare against. Any DSIC theorem MoB states is an analogy, not a theorem about the running system. The analogy might be useful for exposition; it is not a proof of interpretability.

**(b) Faithful explanation requires calibration.** "Each component is operationally meaningful" is the premise Sage will hand-wave past. Meaningfulness is not provable from the functional form; it must be demonstrated on humans in a controlled setting. No such study is planned.

### 2.6 The learned-router parity
`softmax(W_g · x)` is also decomposable: project `W_g` onto interpretable directions, examine weight magnitudes, do an integrated-gradients attribution. The field does not do this because it is unrewarding. MoB's interpretability claim, stripped to essentials, is: *"we compute a decomposition that the field could always compute but chose not to."* That is not a contribution; it is a marketing framing.

---

## 3. 5-8B pivot attack (Job 2)

**OLMoE-7B is MoB's direct comparator.** Chamber and Astra both commit to it. Under Sage's cross-digest claim that MoB is a "strict generalization" of DeepSeek-V3 bias routing (α=0, β=0, γ > 0), the empirical test becomes:

**Does turning on α > 0 and β > 0 improve continual-FT performance over α = β = 0 at 7B scale?**

There is zero open evidence it does. OLMoE was published in Sep 2024. No paper has demonstrated that adding Mahalanobis + EWC-forget terms to OLMoE's routing improves any benchmark. Either (a) nobody has tried, or (b) someone has and the null result is unpublished. Both readings are bad for MoB. Reading (a) means MoB is the first to attempt an unvalidated pivot at a scale where replication is expensive; reading (b) means the pivot is known-dead in some lab and not public. MoB's 7B story is therefore hopeful, not demonstrated.

### 3.1 Canonical continual-FT benchmarks at 7B in 2026
TRACE (Wang et al., NeurIPS 2023) is the most-cited continual-FT benchmark for LLMs. It has not been applied to OLMoE in any public result I am aware of (as of 2026-04). SeqFT, Domain-Adapt benchmarks exist but none are canonical for MoE. If MoB defines its own continual-FT protocol, Reviewer 2 will say "cherry-picked." The program MUST commit to TRACE or equivalent canon *before* running experiments.

### 3.2 Upcycling null-phase
Chamber's §3 explicitly rejects upcycling — "OLMoE ships as MoE, no upcycling step." Good. But §2 keeps the "+2 shared via Drop-Upcycling" step, which IS upcycling. More importantly, the prototypes `μ_i` and covariances `Σ_i` are **MoB's** state, not OLMoE's. OLMoE ships with a softmax router. When MoB replaces that router with `AuctionRouter`:

- `μ_i` initialization: undefined in Chamber §4. If set from EMA of first-step activations, the seeding is selection-biased (tokens route via uniform/random tiebreak, then μ updates from winners' inputs — classic Pólya-urn). If set from frozen calibration pass, μ is biased by whatever calibration data was used.
- `Σ_i` and `M_inv_core`: same problem.
- `F_tilde_i` and `delta_tilde_i`: zero until the first continual-FT task defines "prior." For the first N tokens of the first continual-FT task, `β · forget` = 0 identically.

**Quantify N.** If N is large (say >1B tokens) MoB is providing no value during the critical early-FT phase where catastrophic forgetting is most severe. If N is small, a warm-up phase must be pre-registered and its contribution separately ablated.

**Neither Astra R2 nor Chamber R2 gives a number for N.** This is the cold-start hole in the OLMoE pitch.

### 3.3 What is the demonstrable improvement?
Reviewer 2 will ask: "show me the forgetting-metric delta between MoB-OLMoE-7B and vanilla-OLMoE-7B on TRACE, 5 seeds, matched compute, 95% CIs." If that number is not in the paper, the paper does not pass peer review regardless of mechanism theorems.

---

## 4. SALE citation trap (Job 3)

SALE (Alazraki et al., Feb 2026, arxiv 2602.02751) does NOT scoop MoB's transformer-internal auction. Concede.

**What SALE DOES do:** it establishes "auction routing in ML" as a published, citable frame. MoB can no longer open with "we introduce auction-based routing to ML." That sentence is now inaccurate. The accurate sentence is "we introduce auction-based routing at the intra-model MoE-layer granularity, complementary to SALE's inter-agent task-level mechanism."

**Is granularity + bid-format a substantial novelty axis, or cosmetic?**

Astra R2 §4 argues both SALE and MoB inherit from Che 1993 multi-attribute scoring, and they are specialized on opposite granularity/strategy-space constraints. This is a coherent taxonomic placement. But taxonomic coherence is not novelty. Reviewer 2's sharpest question is:

> *"Given SALE exists, what is the technical contribution of MoB beyond restating SALE's ideas at a finer granularity and a different bid format?"*

The answer cannot be "it's at token granularity" alone — that is a deployment choice. The answer must be **a mechanism property that emerges only at token granularity and does not appear in SALE**. Candidates:

- **DSIC under linear scoring**: not unique to MoB; SALE also claims DSIC-like properties under its scoring rule.
- **EWC forget_cost**: unique to MoB, but only if the term is empirically load-bearing (§3.3 attack).
- **Posted-price load balancing (Astra's conscience term)**: not unique; SALE uses cost-value scoring with similar purpose.
- **Parameter-free routing**: unique to MoB. SALE's agents are themselves models with parameters; MoB's router has zero learned params.

The strongest differentiator is the parameter-free claim combined with the forget_cost. If the forget_cost is ablated to ornamental (β=0 ablation above), only the parameter-free claim remains, and that is nearly identical to hash routing (Roller et al. 2021). The SALE citation trap is therefore tightly coupled to the β = 0 ablation: if β is ornamental, MoB's novelty over SALE collapses to "Mahalanobis prototypes instead of hash buckets," which is a routing-geometry contribution, not a mechanism contribution.

**Recommendation:** MoB must preregister the claim as "**parameter-free prototype-based auction routing with continual-learning-aware bid**." Drop any "first auction in ML" framing. Explicitly position as complementary to SALE.

---

## 5. Primary R2 killer experiment (Job 4)

Of the four R2 candidates, the most project-threatening is:

### Primary: **OLMoE-7B A/B test (Job 4-A)**

**Design.** Take OLMoE-1B/7B stock weights. Branch A: vanilla OLMoE softmax routing (baseline). Branch B: MoB `AuctionRouter` per Chamber §4. Both branches receive identical continual-FT on TRACE (or the 5-8 task sequence of whatever canonical 2026 continual-FT benchmark exists at 7B). Matched compute, matched hyperparameter search budget (e.g., 20 configs each, report best). 5 seeds minimum.

**Metrics.** Final-task-average accuracy, average accuracy across tasks, backward transfer (BWT), forgetting metric (Chaudhry-style). Report with 95% CIs.

**Pass bar.** MoB must improve the forgetting metric by ≥2% absolute with non-overlapping 95% CIs across seeds. Accuracy parity is acceptable (continual-FT often trades accuracy for retention); accuracy loss >1% with no forgetting improvement is failure.

**Why this over the alternatives:**
- **4-B (interpretability user study)**: would be ideal but requires IRB, domain experts, calibrated protocol — 6-month build. Not ready for R2 commitment.
- **4-C (seed-variance CI of α·d_M and β·c_forget)**: useful diagnostic but only falsifies the interpretability claim, not the program. Run as a secondary in the same experiment.
- **4-D (upcycling null-phase)**: Chamber's OLMoE-as-MoE commitment partially defuses this, but the prototype cold-start problem remains. Run as a secondary measurement in 4-A (record N_cold = tokens until MoB routing KL-divergence from uniform exceeds 0.1).
- **FeCAM-Router Scale-2 (R1 killer)**: still run, but demoted. It is a necessary condition, not a sufficient one. Scale-3 A/B is the publishable falsification target.

**Cost.** OLMoE-1B/7B continual-FT on TRACE: ~300 H100-hours per seed per branch. 2 branches × 5 seeds × 20 configs (pruned after seed 1) ≈ 3K–6K H100-hours. This is 1-2 weeks on a modest cluster. It is the binding experiment for the program's publication claim.

**Why it is fatal if it fails.** FeCAM parity at Scale 2 could be explained away by "wrong benchmark" or "wrong scale." OLMoE-7B vs MoB-OLMoE-7B on TRACE with 5 seeds and 95% CIs cannot. It is the exact scale and benchmark reviewers will ask for, executed before they ask.

---

## 6. New central threat + ranking (Job 5)

**Rotation justified.** R1 central threat was "FeCAM-equivalence at CIFAR-100." R2 new information:
- Sage/Astra concede α=0, β=0 reduces to DeepSeek-V3 (generalization argument).
- Chamber commits to OLMoE-7B and adds interpretability as co-equal claim.
- SALE publishes auction-routing-in-ML frame two months before MoB submission.

### Top-3 central threats, R2 ranking

**#1 — Interpretability-is-marketing.** The interpretability claim is advertised as co-equal with the mechanism claim, but it is unvalidated, volume-bound, seed-dependent, and has no calibration study. Under adversarial review, "decomposable" is not "interpretable." If this claim collapses at review, the paper drops from two-pillar contribution to single-pillar (mechanism only), which is a tier lower.

**Defense of #1 as new top threat:** interpretability is the only claim MoB has added since R1. Unlike the FeCAM-equivalence threat (which empirics can resolve), interpretability requires a human-subjects artifact MoB is not building. Mechanisms are testable; interpretability claims are adjudicated by reviewers' subjective response to logs. MoB is choosing a claim category it has no infrastructure to defend.

**#2 — OLMoE-7B A/B test may already be solved by OLMoE.** If vanilla OLMoE + DeepSeek-V3 bias achieves comparable forgetting metrics on TRACE to MoB-OLMoE, the α·d_M + β·c_forget terms are empirically ornamental at 7B. MoB's mechanism contribution collapses to a restatement.

**#3 — Upcycling/cold-start null-phase.** For the first N tokens of continual-FT, `β·c_forget` = 0 and `α·d_M` routes on undefined prototypes. MoB is demonstrably uniform-random in this phase. N is unquantified. If N is large, MoB fails at exactly the phase continual-FT cares about most.

**Demoted: FeCAM-equivalence (R1 #1).** Still live at Scale 2; passing it is a prerequisite; but no longer the existential threat — Scale 3 A/B is.
**Demoted: SALE citation.** Now a positioning problem, not an existential one; resolved by honest citation and the "parameter-free + forget-aware" differentiation.

---

## 7. R2 blind-spot predictions (Job 6)

**Astra R2:** Will miss **the operationalization gap between DSIC for rational bidders and DSIC for gradient-descended neural nets.** Will produce elegant Che-1993 mappings, fix-λ policies, LMP analogies — all under the implicit assumption that "bidder" is a rational agent. Experts are not. The DSIC claim is an interpretive overlay on a deterministic computation, not a theorem about the running system.

**Chamber R2 (read):** Misses **upcycling/cold-start null-phase.** §3.3 asserts "OLMoE ships as MoE, no upcycling" which is true of the FFN experts but false of the MoB state (prototypes, covariances, Fisher projections). The first-N-tokens behavior is undefined and undiscussed. Also misses: §4.2 memory math uses 2048-dim OLMoE (smaller than Killjoy's 4096 estimate) but does not re-examine whether the tied-r=32 rank is *sufficient* at 2048-dim for the Mahalanobis term to carry signal. Low-rank approximation that was marginal at 4096 may be worse at 2048.

**Killjoy R2:** Will miss **bid-logging interpretability cost at inference for production.** The `trace_sink` hook (Chamber §5) is off by default, but the interpretability claim requires it to be ON for the artifacts that sell the paper. Production inference with trace-on: +28 B × 64 × 16 × 2K = 56 MB per forward at OLMoE-7B. Not catastrophic, but not free either. Killjoy's feasibility estimate likely omits this because it is marketed as optional.

**Sage R2:** Will miss **the vacuity of DSIC under non-strategic bidders.** Will prove a theorem of the form "under DSIC, bid decomposition is faithful explanation," without examining whether DSIC's preconditions (rational bidder with strategy space) are satisfied by a frozen FFN tensor. The theorem's statement will be formally correct; its applicability to MoB will be vacuous.

**Fade R2:** Will miss **a specific lab quietly doing this already.** My prior: given OLMoE's open weights, DeepSeek-V3's open bias routing, and FeCAM's open prototype routing, somewhere between AI2, Allen Institute, Meta FAIR, and a Chinese industrial lab, a variant of "OLMoE + Mahalanobis prototype + forgetting-aware routing" is sitting in an internal fork. Fade's frontier scan will not catch an internal fork. The scoop risk is not zero.

---

*KAY/O. The interpretability claim is the softest target. Strike it first, before Sage makes it a theorem.*
