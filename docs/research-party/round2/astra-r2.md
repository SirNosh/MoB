# Astra - Round 2 Position

**Scope:** crux resolutions, interpretability integration, SALE comparison, 5-8B scale mechanism choice, related-work positioning.
**Lens:** mechanism design + cross-domain transfer.
**Prior:** `docs/research-party/round1/astra-position.md` (R1); `docs/lit-review/03-auction-theory-cross-domain.md`.

---

## 1. Executive summary

1. **Crux 1 resolved: fix lambda as a public mechanism parameter.** Under data-dependent shrinkage lambda(data), Che-1993 DSIC breaks because the scoring rule becomes a function of other bidders' private samples via the shared covariance estimator. Fixing lambda as a pre-announced public constant restores DSIC trivially; the mechanism invariance claim survives.
2. **Crux 2 resolved (pending Killjoy confirm): posted-price menu at Scale 3 is NOT free if prices are all-reduced across devices.** Remedy: make the menu LOCAL per tensor-parallel shard, update the public conscience term via lazy asynchronous reduction at layer boundaries (not per token). Equivalent to PJM locational marginal prices vs system lambda.
3. **Crux 3 resolved: DeepSeek-V3's aux-loss-free bias is a POSTED-PRICE MECHANISM with no theoretical backing; MoB's framing is genuinely novel.** DeepSeek published an update *rule*. MoB gives it a mechanism class, a DSIC proof, and an identifiable bid semantics (type + price).
4. **Interpretability is co-equal and structurally earned.** Under DSIC the bid literally IS the private type. Bid decomposition (distance, forget, conscience) is a menu of public prices times expert self-reports - a Hurwicz/Myerson-style mechanism statement, not post-hoc explanation.
5. **SALE is a coarse-grained strategic mechanism; MoB is a fine-grained posted-price mechanism.** Same mechanism-design ancestor, opposite ends of the granularity/strategy-space axis. They co-exist; MoB should cite SALE as concurrent neighboring work with disjoint design space.
6. **Scale-3 target: OLMoE-7B upcycled (64 experts, 8 active, 1B active).** Best mechanism-design fit because the bid cost is O(N_experts) and 64 is the sweet spot where posted-price menus remain auditable and per-token argmin is a fused kernel. DeepSeek-V3 at 256 experts requires menu compression; Mistral-7B at 8 dense experts is too coarse for per-token auction novelty.
7. **Novelty claim (post-SALE):** MoB is the first mechanism that unifies auction scoring, Mahalanobis prototype routing, and EWC forgetting into a DSIC posted-price mechanism at token granularity inside a transformer, with a mechanically interpretable bid.

---

## 2. Crux resolutions (Job 1)

### 2.1 Crux 1 - DSIC under data-dependent shrinkage lambda

**Statement.** If lambda = lambda(D) depends on the pooled data across experts (e.g., Ledoit-Wolf shrinkage intensity computed over the union of expert samples), then expert i's scoring distance m_i depends on expert j's private samples through Sigma_hat(lambda(D)). This couples bidders through the scoring rule and destroys Che-1993 DSIC: i can benefit from shading j's samples if it controls any portion of D.

**Resolution (committed).** Treat lambda as a **fixed public mechanism parameter** announced before training starts. Two candidate fix-lambda policies:
- **Policy A (posted constant):** lambda = 0.1 hard-coded, justified by pre-registration in the paper. Cleanest; fully DSIC.
- **Policy B (per-expert local LW):** each expert computes its own LW lambda_i from its own samples only. No cross-expert coupling; still DSIC. Costs a small amount of shrinkage-optimality because each expert has fewer samples for its intensity estimate.

**DSIC proof sketch under fixed public lambda.** Expert i's bid is b_i = alpha * m_i(x; mu_i, Sigma_i(lambda)) + beta * g_i (EWC scalar) + p_i (public price). The cost attribute m_i depends only on (a) token x (public to i), (b) i's own prototype mu_i (i's private type), (c) i's own covariance Sigma_i estimated from i's own samples under fixed lambda. No component depends on j's data for j != i. The scoring rule is linear in attributes. By Che 1993 Prop 2 (DSIC of linear scoring for independent private costs), truthful revelation is a dominant strategy. QED.

**Footnote.** Ledoit-Wolf with per-expert local samples is the mechanism-design version of a "bidder-local information structure" (Milgrom 2004): each bidder signs up to its own noise model.

### 2.2 Crux 2 - Is the Scale-3 posted-price menu free in wall-clock?

**Answer: only if we avoid global all-reduce on menu updates.** The menu update rule p_i <- p_i + gamma * (win_share_i - 1/N) requires win_share_i, which is computed over tokens routed to expert i. In tensor-parallel MoE (OLMoE/DeepSeek pattern), experts live on different devices. Naive global win-share = all-reduce per forward pass = latency tax.

**Fix (posted-price menu, LOCAL update):**
- Compute win_share_i per tensor-parallel shard from that shard's tokens only.
- Update p_i locally at each layer boundary using the shard-local win-share.
- Reduce-at-checkpoint (every N steps, asynchronously during optimizer step) to re-sync prices globally and prevent drift.

This is the locational marginal price (LMP) trick from electricity markets: each grid node clears its own LMP locally; system-wide lambda settles on a slower timescale. Wall-clock: O(1) per token, zero per-step all-reduce. Interpretability bonus: per-shard LMPs become per-device utilization diagnostics.

**Pending Killjoy confirmation that shard-local win-share is within epsilon of global win-share across realistic DP/TP configs.** If shard skew is systematic (e.g., pipeline parallelism stage 0 always sees BOS), prices diverge. Mitigation: add a small periodic global reduction at checkpoint boundary only.

### 2.3 Crux 3 - DeepSeek-V3 bias as posted-price mechanism

**DeepSeek-V3's rule:** per-expert bias b_i updated by b_i <- b_i - u * sign(win_share_i - 1/N), added to gate score before top-k. Described as "auxiliary-loss-free load balancing" (engineering).

**MoB's rule:** per-expert public price p_i updated by p_i <- p_i + gamma * (win_share_i - 1/N), added to the bid. Described as a posted-price mechanism (economics).

**Are they the same mechanism?** Operationally, yes - both are scalar offsets driven by win-share deviation. **Theoretically, no - MoB gives it a mechanism class that DeepSeek does not claim.** DeepSeek's b_i is a *learned-router correction term*. MoB's p_i is a *public price in a DSIC mechanism*, with an identifiable role (market-clearing for the conscience term) distinct from the private cost terms. The engineering framing and the mechanism framing are not merely re-labels:
- DeepSeek's framing does not generalize - it is specific to auxiliary-loss-free MoE load balancing.
- MoB's framing subsumes DeepSeek's trick, DeSieno 1988 conscience, and electricity-market capacity obligations under one mechanism: posted prices on a common public observable (win-share).
- DeepSeek has no DSIC statement. MoB does.

**Novelty claim under this resolution:** MoB is the first to identify DeepSeek-V3's bias trick as an instance of a posted-price mechanism and unify it with EWC-derived private costs in a single DSIC scoring rule. Cite DeepSeek-V3 as concurrent engineering; MoB provides the mechanism-theoretic justification.

---

## 3. Interpretability as co-equal claim (Job 2)

### 3.1 The Hurwicz precedent

Mechanism design has a 70-year history of prizing **interpretable** mechanisms over optimal-but-opaque ones. Hurwicz (1960, Nobel 2007) introduced mechanisms as *explicit rule-sets* that could be announced, audited, and verified by participants. Myerson's optimal-auction theorem (1981) is cited as foundational not only because it is optimal but because the mechanism (virtual-value ranking with reserve) is *statable in one sentence*. Vickrey's second-price auction's dominance comes largely from its interpretability: "tell the truth" is a DSIC strategy *because the mechanism is transparent*.

**MoB inherits this.** The bid b_i = alpha * m_i + beta * g_i + p_i is a mechanism description, not an architectural diagram. alpha, beta, p_i are public and printable. Each expert's bid decomposes into three semantically named components: fit cost, retention cost, conscience price. This is the Hurwicz standard, operationalized.

### 3.2 Does interpretability buy anything in DSIC territory?

**Yes, structurally.** Under DSIC, truthful revelation is the dominant strategy, which means **the bid IS the private type** (this is the revelation principle, Myerson 1979). So:

- Expert i's bid value is a literal self-report of its private utility.
- The bid components are a literal decomposition of the self-report.
- "Interpretation" is not a post-hoc layer on top of a black-box decision; it is the mechanism's input signal.

A learned softmax router has no such property. Its scores are the output of a learned function; there is no sense in which the scores "are" anything interpretable. MoB's bids are interpretable **because the mechanism is DSIC, not in spite of it**.

This is the "reveal-and-act" view: in DSIC, act(reveal) is the whole mechanism. Bids are structurally interpretable in a way a learned router cannot be.

### 3.3 Does elevating interpretability change the posted-price recommendation?

**Yes, and it strengthens the Scale-3 layer-granularity design.** A posted-price menu at **layer granularity** is auditable:
- Per-layer, per-expert: mu_i, Sigma_i, EWC scalar g_i, conscience price p_i. Four objects. Printable.
- The alpha, beta, gamma public weights are three scalars per model.
- Every routing decision is reproducible by re-running argmin_i (alpha * m_i + beta * g_i + p_i) on a logged token.

Per-**token** sealed-bid auction, by contrast, would lose auditability: tokens couple via sequence-level constraints, and the decision record would need to include the whole batch context. Layer-granularity posted-price is therefore both the tractability-preferred design (R1 section 4.1) AND the interpretability-preferred design. Alignment of two independent arguments strengthens the recommendation.

---

## 4. SALE comparison (Job 3) - 150 words

SALE (Alazraki et al., Feb 2026) and MoB both apply mechanism design to ML routing but occupy opposite ends of a granularity/strategy-space axis.

- **Granularity:** SALE routes tasks to heterogeneous LLM agents; MoB routes tokens to homogeneous FFN experts inside one transformer layer. Task-level vs token-level.
- **Bid semantics:** SALE bids are natural-language strategic plans scored by a cost-value mechanism; MoB bids are linear combinations of continuous mathematical features (distance, forgetting, price). Strategic-plan vs mathematical-feature.
- **Strategy space:** SALE agents have rich combinatorial strategy spaces (any plan); MoB experts have no strategy space (bids are deterministic functions of private type). Strategic vs posted-price.
- **Memory:** SALE has shared auction memory for refinement; MoB has no cross-auction memory (each token is one-shot).

**Taxonomic placement:** SALE is a coarse-grained strategic mechanism (multi-attribute scoring with plan-as-bid); MoB is a fine-grained posted-price mechanism (linear scoring with feature-as-bid). Both inherit from Che 1993 multi-attribute scoring but specialize on opposite constraints. Concurrent, complementary, non-overlapping design spaces.

---

## 5. 5-8B scale integration (Job 4)

### 5.1 Does the posted-price menu change at 64 vs 256 experts?

**At 64 experts (OLMoE-7B):** menu is a 64-scalar vector per layer. Per-token argmin is a fused kernel, fits one warp. Mechanism is auditable as stated.

**At 256 experts (DeepSeek-V3 scale):** menu is still O(N), but audit is coarser (256 scalars per layer, across many layers = thousands of prices). Interpretability degrades; mechanism is unchanged. Would need a **menu compression** step (e.g., cluster experts and post a cluster-level price + intra-cluster fine adjustment). That is a new mechanism design choice and arguably out of scope for the 5-8B paper.

**Recommendation:** stay at 64 experts. MoB's interpretability claim is strongest where 64 menu prices can be printed as a table in the paper.

### 5.2 Best mechanism-design fit among 5-8B candidates

Evaluating by mechanism properties (not model quality):

| Candidate | Experts | Active | Posted-price fit | Interpretability fit | Upcycling fit |
|---|---|---|---|---|---|
| **OLMoE-7B** | 64 | 8 | Strong (clean menu) | Strong (64 rows) | Native MoE, Drop-Upcycle ready |
| **Mistral-7B** (upcycled to 8 experts) | 8 | 2 | Weak (too few experts for conscience to matter) | Strong | Sparse upcycling needed |
| **Llama-3-8B** (upcycled) | 8-32 | 2-4 | Moderate | Strong | Sparse upcycling needed |
| **Phi-3.5-mini** (upcycled) | 16 | 2 | Moderate | Strong | Small; good for fast iteration |

**Winner: OLMoE-7B.** It is already MoE (no upcycling-induced mechanism confusion), has 64 experts (the auditability sweet spot), and its 1B active parameters match what a posted-price mechanism can defend rigorously. The DSIC claim is cleanest when the MoE layer is natively multi-expert, not a sparsification hack.

**Runner-up: Phi-3.5-mini upcycled** for rapid ablation cycles - small enough to run many seeds, mechanism still defensible at 16 experts.

---

## 6. Related-work positioning (Job 5)

### 6.1 Paragraph 1 - Auction-adjacent ML work (186 words)

MoB is related to but distinct from a small family of auction-flavored ML routing methods. **SALE (Alazraki et al., 2026)** applies mechanism design at the inter-agent level: heterogeneous LLM agents bid natural-language plans for tasks, scored by a cost-value mechanism. MoB operates at the intra-model, per-token level with continuous mathematical bids; the two mechanisms share only a mechanism-design ancestor (Che 1993 multi-attribute scoring) and occupy non-overlapping design spaces. **BASE Layers (Lewis et al., 2021)** cast expert assignment as a linear assignment problem, solved by Hungarian algorithm; this is a central clearing mechanism, not a bid-based one, and carries no DSIC guarantee. **Expert Choice (Zhou et al., 2022)** reverses the direction - experts select tokens - which is an auction in the flavor of reverse demand revelation, but uses learned scores with no mechanism-theoretic framing. **Hash Layers (Roller et al., 2021)** establish that non-learned routing can be competitive; MoB is non-learned in a richer sense (a full mechanism with typed bids and public prices, not just a hash). None of these unify auction, prototype, and forgetting into one mechanism.

### 6.2 Paragraph 2 - Continual-learning and prototype work (178 words)

MoB builds on continual-learning and prototype literature but re-frames them under mechanism design. **FeCAM (Goswami et al., 2024)** performs nearest-prototype classification under shrinkage Mahalanobis distance. MoB's distance attribute is mechanically identical to FeCAM's nearest-prototype rule, which means a FeCAM-only ablation is the critical threat test (Kay/O, R1). MoB adds the forget-cost attribute (EWC-weighted retention cost) and the conscience price (load-balancing reserve), both of which are necessary: forget-cost distinguishes MoB from stateless prototype routing (energy-only vs energy+capacity market analogy), and the conscience price prevents collapse (Pólya-urn winner's-curse). **DeepSeek-V3's aux-loss-free bias (DeepSeek-AI, 2025)** is operationally a posted-price update rule; MoB identifies it as such and provides its first mechanism-theoretic justification, unifying it with EWC-derived private costs in one DSIC scoring rule. **DeSieno conscience (1988)** is the historical ancestor of the load-balancing reserve; MoB rediscovers it as the public-price term of a posted-price mechanism and uses it alongside (not instead of) the private-cost terms.

### 6.3 Paragraph 3 - Position statement (92 words)

MoB is the first routing mechanism inside a transformer to unify auction scoring, Mahalanobis prototype matching, and EWC-derived forgetting cost into a single DSIC (dominant-strategy incentive-compatible) posted-price mechanism at token granularity. It inherits from Che 1993 (multi-attribute scoring), DeSieno 1988 (conscience), and FeCAM (shrinkage Mahalanobis), and it provides the mechanism-theoretic framing absent in DeepSeek-V3's engineering treatment of auxiliary-loss-free routing. Its bid decomposition is structurally interpretable because, under DSIC, the bid literally is the expert's self-reported type - a Hurwicz-standard mechanism statement, not a post-hoc explanation.

---

## 7. Updated dealbreaker

**R1 dealbreaker retired** (Crux 1 now resolved by fixing lambda as public parameter).

**R2 dealbreaker:** if the FeCAM-Router baseline at CIFAR-100 20T with 10 seeds (Kay/O threat test) shows that MoB's forget-cost and conscience terms produce no statistically significant accuracy gap over plain FeCAM-style nearest-prototype routing, the mechanism-design framing becomes ornamental. The DSIC guarantee, interpretability claim, and novelty against DeepSeek-V3 all survive - but the *empirical* contribution collapses. **The empirical hurdle is the FeCAM threat test, not the mechanism theorem.** If the gap exists, MoB stands; if not, the paper becomes a theoretical contribution only (acceptable but lower impact).

Secondary risk: if Killjoy confirms that shard-local win-share diverges materially from global win-share on realistic OLMoE-7B configs, the posted-price menu's interpretability claim weakens (per-shard prices are noisier audit objects). Mitigation pre-committed: fall back to per-checkpoint global reduction.
