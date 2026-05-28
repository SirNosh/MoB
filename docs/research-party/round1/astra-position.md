# Astra — Position Paper, Round 1

**Scope:** best auction design for MoB at three scales (Split-MNIST -> CIFAR-100 -> LLM MoE FFN).
**Lens:** mechanism design + cross-domain transfer.
**Prior:** see `docs/lit-review/03-auction-theory-cross-domain.md` for the Che 1993 / DeSieno 1988 / capacity-market groundwork this paper builds on.

---

## 1. Executive summary

1. **Keep the first-score sealed-bid reverse procurement auction as the canonical MoB mechanism at every scale.** Changing mechanisms between scales breaks the narrative and forces re-derivation of truthfulness, IR, and budget-balance. The scoring rule stays linear-in-attributes (Che 1993). Only the item count, the reserve, and the tie-breaker change.
2. **Treat `alpha` and `beta` as posted prices, not hyperparameters.** Calibrate them once, per scale, against the empirical distributions of Mahalanobis distance and EWC forget cost so that the two bid components are on the same utility scale. No grid search.
3. **Replace the `optimizer_reset`-on-shift fix with a principled entry-cost bidder-rejuvenation primitive.** Optimizer reset works because it restores the bidder's cost function to its true value after a type change; re-frame it as Milgrom-Weber *linkage principle* compliance and make it an explicit rule of the mechanism, not an engineering hack.
4. **Scale 1 collapse is a winner's-curse / common-value contagion, not a bidding-weights problem.** The fix is a conscience-style dynamic reserve (DeSieno 1988), not rebalancing alpha/beta. Do NOT add an auxiliary load-balance loss; that re-introduces learned-router coupling MoB was designed to remove.
5. **Scale 2 should stay single-winner per sample but introduce a `must-run` capacity obligation** so that every expert sees at least a floor fraction of samples per task. This is the electricity-market fix for stranded-asset collapse, transferred one-for-one.
6. **Scale 3 must become a posted-price menu mechanism, not a sealed-bid auction.** Per-token sealed bidding at 128 experts is intractable and also theoretically wrong: tokens within a sequence are not independent bidders' items. A clock/menu auction at layer granularity with per-token posted prices preserves truthfulness and is O(N_experts) per token.
7. **Low-rank shrinkage covariance at d=4096 preserves the Che 1993 scoring rule.** The rule is linear in cost attributes; any PSD metric on the attribute vector is admissible. We do not need to rework the mechanism, only the *implementation* of the distance attribute.
8. **Cross-scale invariant (committed): the bid must remain a function only of the expert's own private type and public parameters — never of other experts' bids or of a learned global router.** Violating this is what would turn MoB back into Switch.

---

## 2. Scale 1 — Split-MNIST / CIFAR-10 (4 experts, current)

### 2.1 Is first-score reverse auction still the right mechanism here?

**Yes, and this is the scale at which we should *prove* its properties.** At 4 experts and ~5 tasks, the auction is small enough to verify DSIC, IR, and ex-post efficiency empirically — the mechanism-design analogue of a unit test. Mechanism design is a ladder: the scoring rule must be the same object at 4 experts as at 128, or the scaling claim is empty.

### 2.2 `alpha` and `beta` as posted prices

The bid is `b_i = alpha * mahalanobis_i + beta * forget_cost_i`. Today alpha and beta are hyperparameters. They should be *posted prices* announced by the mechanism before bidding — i.e., the principal's marginal rates of substitution between fit cost (Mahalanobis) and retention cost (EWC).

Concrete procedure:
1. Run a pilot on task 1. Collect the empirical distributions of `mahalanobis_i` across experts-and-samples, and of `forget_cost_i` across experts (post-task-1 Fisher).
2. Set `alpha / beta = median(forget_cost) / median(mahalanobis)`. This puts the two terms on equal *expected* utility scale — the principal's indifference curve.
3. Freeze them for the rest of the run.

This is the posted-price analogue of Myerson's virtual-value shading, adapted for the reverse-procurement case. Do NOT do Myerson virtual-cost shading directly here: the bidders are not strategic (they report true features), so virtual-cost inversion has no role. Posted prices are the right primitive for non-strategic bidders with common public parameters.

### 2.3 Optimizer reset as a principled primitive

The optimizer-reset-on-shift fix works because when a bidder's type changes (its expert is re-assigned to a new task), the Adam momentum encodes the *old* type. The bidder then bids against its old cost function while its true cost has changed. That is a mechanism-design violation, not a Pytorch bug.

Rename and re-scope it: **`bidder rejuvenation`**: when the auction detects a type change (task-shift detector fires, or the expert wins a sample whose Mahalanobis distance exceeds its running centroid radius by k sigma), the mechanism resets the bidder's internal state so its next bid reflects its new true type. This generalizes cleanly to:
- Scale 2: rejuvenation fires at task boundary AND on sustained Mahalanobis drift.
- Scale 3: rejuvenation fires only at checkpoint cadence (per-token rejuvenation would be absurd).

This is the Milgrom-Weber linkage principle in operational form: require bidders to publish enough state that their current bid is consistent with their current type.

### 2.4 Training-time prototype routing collapse through an auction lens

Section 3.7 reports that training-time routing collapses to 1-2 experts. This is **not** a bidding-weights problem; it is a **common-value winner's-curse cascade**.

Mechanism-design diagnosis: when expert A wins early samples, its centroids update toward those samples, lowering its future Mahalanobis distance, which lowers its future bid — a positive feedback loop. Experts B, C, D have centroids drifting from initialization noise, not from any signal. The auction is mechanically correct, but the *cost function itself* is endogenous to the winner's history. Classic winner's-curse structure: the bidder whose private signal is most correlated with its won samples appears to be the cheapest bidder on all future samples.

**Fix: dynamic reserve price (DeSieno conscience).** Each expert gets a reserve adjusted by `gamma * (empirical_win_share_i - 1/N)`. An expert that has won too much gets a positive reserve added to its bid (it must beat a higher bar). This is exactly Switch/DeepSeek load balancing, but as a bidder-side reserve rather than a global auxiliary loss. Critically: the reserve is **public and announced**, it does not couple experts' bids to each other, and it preserves DSIC because each bidder still truthfully reports its own type.

Do not try to fix collapse by rebalancing alpha/beta. The collapse is in the *data-generating process* of the bids, not in the weighting. Rebalancing chases a symptom.

---

## 3. Scale 2 — CIFAR-100 (4-8 experts, 5-20 tasks, 100 classes)

### 3.1 Single-item or combinatorial?

**Stay single-winner per sample.** Moving to top-k combinatorial here would import the VCG complexity (core-selecting combinatorial auctions are NP-hard in general; see spectrum auction literature) without a payoff: at 8 experts and 100 classes, the marginal value of a second winner is small if the primary winner's centroids are well-fit.

Deferred multi-winner to Scale 3 only, where per-token routing genuinely requires top-2.

### 3.2 Reserve / capacity obligation / must-run

**Add a must-run obligation**: each expert must win at least `floor_fraction * batch_size` samples per epoch, enforced by a sliding-window dynamic reserve (the DeSieno primitive from 2.4, but with a hard floor). This is the electricity-market capacity-obligation transfer from the cross-domain review: in ISO-NE and PJM, merchant generators are paid a capacity payment to stay online even when they are not the cheapest bidder. Same logic here: underused experts get a negative reserve (bid discount) when their win-share falls below the floor.

Mechanism: `b_i_effective = b_i - max(0, floor_fraction - win_share_i) * nu`, with `nu` = median bid magnitude. This is IR (the principal pays at most `nu` per underused expert) and preserves DSIC (each expert still truthfully reports its own cost; the discount is public and parameterized on observables).

### 3.3 Frozen ViT-B/16 vs from-scratch ResNet

This matters because the router sees very different feature distributions.

- **Frozen ViT-B/16:** features are pre-trained and already clustered. Centroids converge fast; feature-to-expert is nearly deterministic. The forget_cost term becomes almost decorative because ViT features don't drift. **Drop `beta` toward zero; rely on Mahalanobis alone.** This is why RanPAC works — the prototype classifier is already near-optimal on frozen features.
- **From-scratch ResNet:** features co-evolve with training; centroids chase a moving target. `beta` matters, and bidder-rejuvenation (2.3) must fire more aggressively.

The mechanism is invariant across these two cases; only the posted prices change.

### 3.4 Response to the FeCAM-equivalence threat (Cypher)

FeCAM is: nearest prototype with shrinkage Mahalanobis distance. MoB v2 prototype routing is: argmin over experts of Mahalanobis to their centroids. **Cypher is correct that FeCAM and MoB-v2 are functionally near-identical** *for a single-task classifier*.

What the auction adds over FeCAM is the **forget_cost term**. FeCAM has no retention pressure — it just picks the nearest prototype. MoB picks the nearest prototype *among experts that would not be damaged by accepting the sample*. On a single-task classifier this term is zero and the two collapse. On continual learning with Fisher-weighted retention cost, the two diverge: MoB will route a hard sample to a slightly-worse-fit expert if the best-fit expert has critical knowledge at risk.

This is **exactly** the cross-domain analogy from the lit review: FeCAM is "dispatch to cheapest generator"; MoB is "dispatch to cheapest generator subject to stranded-asset avoidance." In electricity markets, that distinction is the difference between an energy-only market (FeCAM) and an energy + capacity market (MoB). The capacity-market literature is 30 years deep on why you need the second term.

**Operationally,** the MoB-over-FeCAM paper is: demonstrate an accuracy gap on a task sequence where `beta * forget_cost` is non-trivial. If that gap exists, MoB is not redundant.

---

## 4. Scale 3 — LLM MoE FFN (8-128 experts, per-token, d=4096)

### 4.1 Tractability: which mechanism survives?

Per-token sealed bidding at 128 experts is O(T * N_experts * d) — roughly the same cost as softmax routing. The actual problem is **theoretical**: tokens within a sequence are not independent items. Sealed-bid assumes independent allocation; if tokens `t` and `t+1` both prefer expert 3, the auction has no coordination mechanism, and per-sequence coordination would destroy parallelism. This is the auction-theory analogue of the per-token vs per-sequence load-balancing trade-off in DeepSeek-V3.

**Recommendation: posted-price menu mechanism at layer granularity, per-token settlement.**

Concrete design:
1. At each forward pass, each expert publishes a **menu** of posted prices: `p_i = f(expert_i_state)`. The menu is a scalar per expert per layer, updated once per forward pass.
2. Each token computes its Mahalanobis-distance vector `m_i(token)` and picks `argmin_i (m_i + p_i)` for top-1, and the second-smallest for top-2. Pure local decision per token; no coupling.
3. The menu price `p_i` is updated between forward passes via the DeSieno rule on observed win-share (the reserve lives here, not in the bids).

This is:
- O(N_experts * d) per token, no dependence on sequence length or other tokens.
- Parallelizable perfectly.
- Preserves DSIC at the expert level: each expert truthfully publishes its private cost.
- Implements load balance without an auxiliary loss.
- Is the DeepSeek-V3 aux-loss-free bias-term trick, but re-derived as a posted-price mechanism with theoretical backing.

### 4.2 Truthful multi-item approximation of top-2 routing

Top-2 per token is a 2-unit uniform-price auction per token. Uniform-price is DSIC when items are identical (Vickrey). Tokens are not identical items to the experts, but **the tokens are not the bidders** — the experts are. So the correct frame: each expert bids a posted-price menu, each token is an auctioneer who runs a one-shot second-price clock for 2 units. This is trivially DSIC because the expert's bid (its menu price) is token-independent.

This is the right decomposition. It avoids combinatorial complications that would appear if we incorrectly modeled tokens as bidders.

### 4.3 Low-rank covariance and the linear scoring rule

At d=4096, a full-rank covariance per expert is 4096x4096 floats per expert — 67MB per expert in fp32. Shrinkage + low-rank (FeCAM's recipe) is forced.

**Does this break the Che 1993 scoring rule?** No. The rule requires linearity *in the cost attributes*. The attribute is `distance`. Any PSD metric produces a valid distance. Shrinkage `Sigma_hat = (1-lambda) * Sigma_lowrank + lambda * tr(Sigma_lowrank)/d * I` is PSD by construction. The mechanism is unchanged; only the statistical object under it is regularized — the auction analogue of ambiguity aversion (Gilboa-Schmeidler).

### 4.4 DeSieno-conscience vs auction bid: use both

The `forget_cost` term in the bid is **type-private**: it is each expert's own marginal EWC retention cost. It is *not* redundant with DeSieno conscience, which is **market-public**: the sliding-window reserve that punishes over-winning.

- **forget_cost:** "I am expensive to retrain; my bid is higher."
- **DeSieno reserve:** "You have been winning too much; your bid gets a penalty."

These address orthogonal failure modes. forget_cost without DeSieno reserve collapses (Scale 1 evidence). DeSieno reserve without forget_cost re-creates Switch (loses continual-learning protection). **Both are required at LLM scale,** and they compose without interference: forget_cost goes in the bid, reserve goes in the posted price menu.

---

## 5. Cross-scale invariant

**The bid must be a function only of (a) the bidder's own private type and (b) public mechanism parameters — never of other bidders' bids, never of a learned router's parameters, and never of a loss gradient that couples bidders to each other.**

That single constraint is what distinguishes MoB from Switch, from learned sparse routing, and from every aux-loss-balanced MoE. It must hold at Scale 1 (where we prove it), Scale 2 (where we stress it with capacity obligations), and Scale 3 (where it becomes the parallelism argument). Violating it — even once, even with the best intentions, even in a "small auxiliary head" — collapses MoB into a conventionally learned router with extra steps.

---

## 6. Where I defer to others

1. **Sage (theory/math):** does the shrinkage+low-rank Mahalanobis attribute actually satisfy the Che 1993 linear-scoring-rule identifiability conditions, or does the rank-deficient case introduce a scoring-rule non-uniqueness I am glossing over? I need a formal statement, not an analogy.
2. **Killjoy (systems):** is the posted-price menu update at Scale 3 genuinely free in terms of latency (one scalar per expert per forward pass), or does the synchronization across devices re-introduce an all-reduce that defeats the parallelism claim?
3. **Chamber (architecture) + Fade (frontier):** DeepSeek-V3's aux-loss-free bias-term update rule is operationally very close to my posted-price menu (4.1). Is there prior art I am missing that already names this as a posted-price mechanism, or is the auction-theoretic framing genuinely novel against their engineering framing?

## 7. Dealbreaker

**Single finding that would force retraction:** if Sage or KAY shows that the linear-in-attributes scoring rule (Che 1993) is **not** DSIC when the attribute distance is computed under a shrinkage estimator with data-dependent shrinkage intensity `lambda(data)`, then the "keep the same mechanism across scales" claim collapses. At Scale 3 we would either need to fix `lambda` as a public parameter (likely fine) or transition to a different mechanism class at Scale 3 (breaking the ladder). The whole paper rests on scoring-rule invariance; if that invariance does not hold under the covariance estimator we are forced to use at d=4096, the three-scale story needs rebuilding.
