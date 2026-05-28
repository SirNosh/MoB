# KAY/O Red-Team Position: MoB's Path Forward

**Role:** Adversarial reviewer. This is not a design. This is the case against MoB written before reading any teammate's defense.
**Scope:** MoB = stateless first-score reverse auction routing for MoE, bid = alpha * Mahalanobis(features, expert centroids) + beta * EWC-forget-cost. 4 experts, 5 tasks, 87% on Split-MNIST. Target: LLM MoE (8-128 experts, 4096-dim, per-token).

---

## 1. Executive Summary

1. **The auction is decorative.** Once prototypes exist, argmin-distance routing is FeCAM. The EWC term is an additive constant per expert per step; it does not change the argmin unless forget-costs straddle the Mahalanobis gap -- an edge case, not a mechanism.
2. **The 87% Split-MNIST headline is not a paper.** It is an architectural stress test on a dataset that saturates at >99% with a single linear layer. The delta over a frozen prototype baseline is not reported, and there is no reason to believe it is non-zero.
3. **Project memory documents a known failure mode:** Fisher magnitude varies 18x across initializations; the fix is a hand-tuned clamp. This is p-hacking waiting to happen at CIFAR and LLM scale.
4. **The prior-art ghost (arxiv 2512.10969 "MoB: Mixture of Bidders") is an unresolved existential risk.** Until Dev confirms authorship, every design step is a bet the paper is his. If it isn't, the project is scooped on the name, the mechanism, and likely the claims.
5. **Aux-loss-free routing (DeepSeek-V3) already solves load balancing with a scalar bias update.** MoB's "no aux loss" pitch is post-dated by a deployed 671B-param system. The delta must now be articulated against DeepSeek-V3, not Switch.
6. **EWC forget_cost is undefined at LLM pre-training.** There are no prior "tasks." The beta term collapses to zero or to an arbitrary drift regularizer, gutting half the bid formula.
7. **Per-token auctions at 4096-dim with 128 experts cost a full Mahalanobis evaluation per token per layer per step.** Killjoy will estimate the FLOPs; my claim is the memory bandwidth for 128 inverse covariance matrices at d=4096 is the binding constraint, not compute.
8. **The project-killer experiment is cheap:** FeCAM vs MoB v2, same backbone, same protocol, matched compute, 10 seeds, CIFAR-100 20T. If MoB's accuracy advantage is <1% absolute with overlapping CIs, the auction adds zero signal and the program terminates.

---

## 2. The Central Threat: FeCAM Equivalence

Cypher flagged it; I will sharpen it to the point where it cuts.

**Claim (to be falsified by MoB, not by me):** The auction in MoB is epiphenomenal. The work is done by the prototype geometry. If you delete the auction and keep only per-expert Mahalanobis argmin with frozen prototypes, you reproduce MoB v2's numbers within noise.

**Formal argument.** The MoB winner is:
  `i* = argmin_i [alpha * d_M(x, mu_i, Sigma_i) + beta * c_i]`
where `c_i` is the EWC forget-cost of expert i.

Consider two regimes.

**Regime A (EWC term dominated by Mahalanobis term).** `beta * c_i` is small relative to `alpha * d_M`. Then `i* = argmin_i d_M(x, mu_i, Sigma_i)`. This is FeCAM's prediction rule applied to experts rather than classes. No auction. No bids. Just nearest-prototype.

**Regime B (EWC term dominant).** `beta * c_i` dominates. Then the winner is the expert with the smallest forget-cost, independent of the input features. This is not routing; it is priority scheduling, and it is pathological (the least-trained expert always wins, starving every other expert and defeating continual learning).

**The useful regime is a narrow strip** where `beta * c_i - beta * c_j` is comparable to `alpha * (d_M(x, i) - d_M(x, j))` for the top-two experts. Outside that strip the auction is a no-op. Inside that strip, MoB is betting the fine structure of EWC Fisher matrices is informative enough to override Mahalanobis geometry at the margin. Given that project memory reports Fisher magnitudes vary 18x across seeds before a manual clamp -- the signal is mostly noise.

**What this implies.**
- MoB v1 (raw features, no prototypes) was probably routing on noise.
- MoB v2 (Mahalanobis prototypes) is doing FeCAM and attributing the result to the auction.
- The auction mechanism has not demonstrated irreducible contribution.

**What would refute this?** A controlled ablation on CIFAR-100 20T: (i) FeCAM-per-expert argmin, (ii) MoB v2 full bid, (iii) MoB v2 with beta=0, (iv) MoB v2 with beta-only (alpha=0). If (i) approx (ii) approx (iii) and (iv) is chaos, the auction is ornamental. If (ii) > (iii) by >2% with non-overlapping 95% CIs across >=10 seeds, the auction is doing work. This ablation has not been run. Nothing else in the design discussion matters until it is.

---

## 3. Scale-Specific Vulnerabilities

### 3.1 MNIST / CIFAR-10

**The 87% number is under-powered.** Split-MNIST baselines: EWC reaches ~80%, iCaRL ~95%, even a single MLP with replay hits ~90%. 87% with 4 experts and 5 tasks is not a frontier number; it is a debug result. The headline claim is about the architecture (auction with overloaded experts), not the accuracy.

**Statistical power.** With 4 experts, 5 tasks, and ~5 seeds, the per-expert-task cell is n=5. Fisher magnitude varies 18x across initializations (project memory, documented). A delta of 2-3 accuracy points requires, at minimum, n=20 seeds under a paired design (Welch t-test assumptions violated by the Fisher-variance heteroscedasticity). Otherwise every reported improvement is within the seed envelope.

**Falsifying ablation.** Run MoB v2 with a random-bid baseline: replace the alpha * d_M + beta * c term with `b_i ~ Uniform(0,1)` on the same prototypes. If random routing with prototype-anchored experts reaches 80%+, the auction adds <7 points, and we need to ask whether that delta survives seed variance.

### 3.2 CIFAR-100

**ViT-B/16 ImageNet-21k leakage is not a rumor, it is a documented benchmark artifact.** ImageNet-21k contains WordNet synsets that overlap CIFAR-100 classes (apple, chair, train, various animals). The "continual learning" on CIFAR-100 with frozen ViT is evaluating *how well the model rotates a fixed classifier head* over classes the backbone already represents. This is why SimpleCIL -- literally averaging features with no training -- reaches ~83% on CIFAR-100 B0. Any "continual learning" claim built on frozen ViT-B/16 + CIFAR-100 is a ceiling-bounded toy. RanPAC, FeCAM, SLCA all live in the same 1-3% band because they are all doing roughly the same thing.

**Consequence for MoB:** if MoB reaches 88% and FeCAM reaches 87%, that 1% delta will not survive Bonferroni across the 6 MoB hyperparameters (alpha, beta, Fisher clamp, prototype update rate, expert count, optimizer reset policy).

**Overloaded-expert claim is contrived at CIFAR-100 scale.** With 4 experts and 20 tasks, 16 of them are "overloaded" (handling >=2 tasks each). This is no longer a property; it is the regime. The paper would need to show the mechanism degrades gracefully under increasing overload, and that requires sweeping E/T ratios. If the graceful-degradation curve is flat or noisy, the contribution is unpublishable.

### 3.3 LLM MoE

**Pre-training infeasibility.** No academic lab has the compute to pre-train an 8B-param MoE from scratch. Upcycling from a dense checkpoint is the only path, which means every "MoB at LLM scale" claim is a claim about *upcycling*, not pre-training. The comparison is: upcycled-Mixtral-style vs upcycled-MoB. That is a narrow claim, and the baseline (Mixtral's top-2 with load-balance aux loss) is a moving target -- DeepSeek-V3's bias-term scheme and MoE++'s zero-computation experts both compete on the same axis.

**Top-k problem.** Mixtral, DeepSeek-V2/V3, GLaM all use top-2 or top-8. Single-winner auctions return top-1. If MoB is extended to top-k, it becomes a "k-winner auction" -- which is either (a) the first k argmin of bids (then load balancing must be added back, destroying the "no aux loss" pitch) or (b) a combinatorial allocation (computationally infeasible per-token). There is no free top-k auction formulation that preserves MoB's selling points.

**Prototype provenance at 4096-dim.** Where do `mu_i` come from during pre-training? If from EMA of winning-expert activations, they are selection-biased: experts that win early define their own prototype, reinforcing winning, producing the Matthew effect. This is not a bug in the implementation; it is a fixed point of auction dynamics with endogenous prototypes. The prototype-routing collapse observed in earlier MoB runs is the theoretically predicted behavior.

**Covariance at d=4096.** Full Sigma_i is 4096 x 4096 = 16.7M params per expert. 128 experts = 2.1B params just for routing covariances. Inverse is O(d^3) per update. Low-rank / diagonal approximations (what FeCAM does) resurrect the covariance-vs-rank tradeoff; the rigorous comparison is "MoB with low-rank Sigma vs FeCAM-router with low-rank Sigma," and both are approximations of the same operation.

**Delta over DeepSeek-V3.** DeepSeek-V3 trains a 671B MoE with a scalar bias term per expert updated via a simple load-balance rule, no auxiliary loss, near-perfect balance, state-of-the-art quality. MoB must explain what it delivers beyond a scalar bias update per expert. Current formulation: replaces scalar bias with a learned/EMA Mahalanobis prototype. This is a 4096x parameter increase per expert for an effect that the scalar achieves. The burden of proof is on MoB to show the 4096x cost is recouped by measurable routing quality.

---

## 4. Prior-Art / Novelty Audit

**arxiv 2512.10969 "MoB: Mixture of Bidders"** (Fade's find). Until Dev confirms this is his own submission, it is a scooping risk of the first order. The ID prefix "2512" is anomalous (arxiv IDs are YYMM.number; 2512 means December 2025). Three possibilities, in descending probability:

1. **Dev's own paper.** Project proceeds, but all design discussions must align to the submitted version or risk a self-scooping conflict.
2. **Concurrent work.** Common name, common mechanism -- the field has converged. Novelty arguments evaporate; the program must pivot to "first to deploy at scale" or "first rigorous ablation."
3. **Search-index artifact / hallucinated ID.** Fade flagged 26xx-prefix IDs as likely mis-captures; 2512 could be similar. Must be verified on arxiv.org directly.

**This is blocking.** No design proposal has integrity until 2512.10969 is resolved.

**Other prior-art vectors.**
- **Per-token routing as an auction, formalized 2024-2025:** None identified at the transformer-layer granularity. BASE Layers (2103.16716) is the closest -- it is a linear assignment problem (Hungarian algorithm), not a first-score auction. Argument: MoB is differentiated from BASE. Risk: reviewers conflate them.
- **Mahalanobis-prototype routing at LLM scale:** Self-Routing MoE (Fade's scan) uses learned router logits, not Mahalanobis. Navigating Semantic Drift uses drift detection, not distance-bid routing. No direct prior at LLM scale -- but also no evidence anyone has tried it, which could mean (a) it's novel or (b) it has been tried and does not work. Null result is not published.
- **DeSieno 1988 conscience mechanism** (Astra's claim, to be challenged in Section 6): introduces a per-neuron bias that penalizes over-winners. This is isomorphic to DeepSeek-V3's bias term. If MoB's forget-cost is argued as a DeSieno descendant, so is DeepSeek-V3's bias, and the comparative advantage collapses.

**What "scooped" looks like:** a 2025 NeurIPS or ICLR paper titled "auction-based routing for MoE" with (a) per-token Mahalanobis bids, (b) CIFAR or LLM results, (c) a novelty claim identical to MoB's. The 2512.10969 find is one keystroke away from that description.

---

## 5. The Project-Killer Experiment

**Design.** FeCAM-Router vs MoB v2, CIFAR-100 20T, frozen ViT-B/16 backbone.

- **FeCAM-Router (null):** for each expert i, maintain class prototypes `mu_c^i` and shared covariance `Sigma^i` computed per FeCAM. Route each input x to expert `i* = argmin_i min_c d_M(x, mu_c^i, Sigma^i)`. No auction, no EWC, no learned bidding. Train experts independently on their routed subset.
- **MoB v2 (alternative):** as specified, alpha * d_M + beta * EWC-forget-cost.

Matched compute budget (same wall-clock, same GPU, same backbone freezing). 10 seeds. Report mean +/- 95% CI on final accuracy (after task 20), average accuracy, forgetting, and BWT.

**Expected under null (MoB dies):** mean(MoB v2) - mean(FeCAM-Router) is within +/- 1% absolute with overlapping 95% CIs. FeCAM-Router matches or beats MoB v2 on at least one of {final accuracy, forgetting}. Interpretation: the auction is ornamental; prototype geometry is doing the work; the project has no mechanism claim. **Terminate and re-scope.**

**Expected under alternative (MoB survives):** mean(MoB v2) exceeds FeCAM-Router by >=2% absolute with non-overlapping CIs; the gap widens as tasks accumulate (>=T15), showing auction-specific continual-learning benefit; the beta=0 ablation degrades MoB to FeCAM levels (proving the EWC term is load-bearing). Interpretation: the auction is irreducible; proceed to CIFAR-100 + LLM.

**Why this test is fatal.** It cannot be hand-waved. It uses the exact backbone the CIFAR-100 literature uses, the exact protocol (20T), the exact metrics. There is no room for protocol-shopping, and FeCAM is a peer-reviewed baseline (NeurIPS 2023) not a straw man.

**Cost.** Estimated 2-3 days on a single A100. Blocks nothing; enables everything. This should run before any LLM-scale design is written.

---

## 6. What Each Specialist Will Miss

**Astra (mechanism refinement).** Will map MoB onto auction theory (first-score, DSIC, linkage principle, DeSieno). Will propose mechanism refinements (VCG payments, Myerson monotonicity, reserve prices). **Will miss:** none of this matters if FeCAM matches MoB without an auction. Auction theory describes the mechanism; it does not justify it. The foundational question is whether the mechanism is needed at all, and auction theory is silent on that.

**Chamber (architecture pitch).** Will propose LLM-scale instantiations -- per-layer MoB heads, top-k auction variants, learned bidders. **Will miss:** (a) the covariance cost at d=4096 (full Sigma is infeasible, low-rank reopens the FeCAM parity question), (b) the selection-biased prototype problem under endogenous-prototype auctions, (c) that upcycling is the only feasible pre-training path and the comparison-set-of-one is Mixtral.

**Killjoy (feasibility estimate).** Will estimate FLOPs/memory per token and report feasibility with throughput numbers. **Will miss:** the memory-bandwidth constraint on inverse-covariance updates; the EMA-vs-second-moment tradeoff at 4096-dim; the per-GPU-per-step latency impact of 128 per-expert Mahalanobis evaluations in the forward path. Throughput in FLOPs is optimistic; walltime on A100s will be 2-3x worse.

**Sage (convergence argument).** Will frame MoB as a dynamical system and argue for fixed-point stability. **Will miss:** the auction is non-differentiable (argmin), so convergence arguments require a softmin relaxation, which moves MoB toward standard MoE routing. The "stateless" selling point dies under any relaxation strong enough to prove convergence. Sage will also likely not address prototype-routing collapse as a fixed point (it is one, for any selection-biased EMA prototype update).

**Fade (frontier framing).** Will pitch MoB as aligned with the router-free 2024-2025 trend. **Will miss:** "router-free" in DeepSeek-V3 means "no learned router params, just a scalar bias." MoB's Mahalanobis prototypes are 16.7M params per expert at d=4096 -- MoB is not router-free; it is *router-reframed*. The alignment argument misreads the trend.

---

## 7. Verdict

**Status:** claims-invalidated (pending FeCAM head-to-head). The 87% Split-MNIST result does not isolate auction contribution. No ablation against FeCAM-Router has been run. The prior-art 2512.10969 is unresolved. The LLM-scale pitch is under-specified against DeepSeek-V3. At least two foundational assumptions (Fisher stability, EWC applicability at pre-training) are demonstrably fragile.

**What unlocks the program:**
1. Confirm ownership of arxiv 2512.10969 (1 day).
2. Run FeCAM-Router vs MoB v2 on CIFAR-100 20T, 10 seeds, paired design (3 days).
3. If MoB survives: run the beta=0 and alpha=0 ablations to localize the auction contribution (2 days).
4. Only then write an LLM-scale design.

**What should not happen:** any LLM-scale design or theory work before steps 1-3 complete. Every hour spent on 128-expert Mahalanobis routing before the FeCAM parity question is answered is hours wagered on a mechanism not yet shown to exist.

---

*KAY/O. Claims must hold under confounders and stronger baselines.*
