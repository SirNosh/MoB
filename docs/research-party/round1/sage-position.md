# Sage — Position Paper, Round 1

**Role:** Mathematical foundations and theory.
**Scope:** Whether MoB's auction-based routing converges, generalizes, and remains well-defined at Split-MNIST, CIFAR-100, and LLM-MoE-FFN scales.
**Stance:** Guarded green-light for Scale 1; conditional green-light for Scale 2 under FeCAM-style covariance regularization; provisional red-light for Scale 3 pre-training with existing EWC bid term; green-light for Scale 3 continual-FT only.

---

## 1. Executive summary

1. **No global convergence proof exists** for the joint system (experts + auction + prototypes + Fisher). A local two-timescale argument (Borkar, 1997) gives convergence *within a task* under standard SA assumptions plus a winner-takes-all margin condition. Across tasks, the best available guarantee is "bounded drift" — not convergence.
2. **Stateless auction ≠ stateless dynamics.** The auction couples experts through the selection rule `argmin_i b_i(x; θ, μ, F)`. Under a non-degeneracy (strict-margin) assumption the argmin is piecewise-constant a.s., so gradient flow per expert is well-defined; without it, the dynamics are a differential inclusion (Filippov sense).
3. **Prototype convergence is conditional on coverage.** Per-expert centroids converge to conditional means *on the winning partition* — not on the true class posterior. This is a selection-bias fixed point, not the Bayes-optimal prototype. Empirically benign at MNIST; theoretically the driver of the "rich-get-richer" collapse.
4. **EWC forget-cost is a valid *local quadratic surrogate* for forgetting**, not an upper bound, under the Laplace approximation (Kirkpatrick 2017; Huszár 2018 correction). Small `forget_cost` implies small KL to old-task posterior to second order — nothing globally. Sova's collapse / drift failure modes are real and propagate through the bid.
5. **Rich-get-richer is a genuine auction pathology.** The Mahalanobis term is monotonically decreasing in samples already routed to expert i (Σ̂ tightens on its own data); the Fisher term grows only on task boundaries. Between boundaries, the bid has a negative-feedback-to-winner loop that is a Pólya-urn / preferential-attachment process. A conscience / load term (DeSieno 1988; Switch aux loss) is mathematically necessary, not stylistic.
6. **Scale-2 calibration requires shrinkage.** With K centroids per expert in dimension d, the sample covariance is rank-deficient once K < d. Mahalanobis is undefined without regularization. FeCAM's tied-shrinkage Σ is the minimal fix; Ledoit-Wolf gives an optimal shrinkage coefficient in closed form.
7. **Scale-3 per-token routing is a concentration question.** Under sub-Gaussian features and Lipschitz prototypes, the auction winner matches the oracle with high probability *only if the bid-margin exceeds the concentration radius*. Margin scales as O(1/√K_experts) for random features; the Lipschitz constant of the bid grows linearly in centroid count. These scalings fight each other.
8. **Cross-scale invariant:** monotonicity of the bid in expert-specialization-distance. If this breaks, truthfulness, regret bounds, and convergence all break simultaneously.

---

## 2. Scale 1 — Split-MNIST / CIFAR-10

### 2.1 Training-loop convergence

Let expert i have parameters θ_i ∈ Θ_i, prototype set μ_i, Fisher diagonal F_i. At step t with batch B_t drawn iid from current-task distribution D_τ, the winner is

  i*(t) = argmin_i b_i(B_t) = argmin_i ( α · d_M(φ(B_t); μ_i) + β · C_F(θ_i; F_i) ).

**Claim (within-task local convergence).** Fix task τ. Assume:
(A1) Features φ are bounded and the per-expert loss L_i(θ_i) is L-smooth with bounded stochastic gradients (standard SGD assumptions).
(A2) **Strict-margin:** there exists ρ > 0 such that for D_τ-a.e. x, the gap b_{(2)}(x) − b_{(1)}(x) ≥ ρ.
(A3) Step sizes η_t satisfy Robbins-Monro (Σ η_t = ∞, Σ η_t² < ∞).
(A4) Prototypes/Fisher update on a slower timescale (η_μ, η_F = o(η_θ)).

Then θ_{i*} follows the standard stochastic approximation ODE and converges a.s. to a stationary point of the restricted expected loss E_{x|i*=i}[ℓ(θ_i; x)]. Proof sketch: (A2) implies the winner indicator is constant on a neighborhood, so per-expert SGD decouples; (A4) implies μ, F are locally frozen on the θ timescale; apply Borkar two-timescale SA.

**What this does *not* give you:** (i) global convergence — Θ_i is non-convex (NN); stationary ≠ optimal. (ii) cross-task convergence — when D_τ switches, the bid surface moves, A2's ρ can collapse, and we re-enter a transient. (iii) guarantee that the fixed point is the *right* assignment — selection bias (§2.2) means the fixed point is only locally efficient.

**Can it diverge?** Yes, in two regimes:
- If A2 fails on a positive-measure set (ties / near-ties), the winner index oscillates and θ_i updates become a switched system. Without dwell-time or hysteresis, Filippov solutions can exhibit sliding modes — no convergence.
- If α, β are learned (Astra's non-truthful case), the bid becomes a function of θ_i itself and the system is no longer a projection: the SA ODE may not have a Lyapunov function.

### 2.2 Prototype / centroid convergence

Expert i's centroid update on winning batch B:
  μ_i ← (1 − η_μ) μ_i + η_μ · mean_φ(B).

In the continuous-time limit μ̇_i = E[φ(x) − μ_i | i*=i] · p(i*=i). Fixed point: μ_i = E[φ(x) | i* = i].

**This is selection-biased.** The conditional distribution p(x | i*=i) is NOT the task-conditional p(x | τ) unless experts are perfectly specialized (one expert per task). Under rich-get-richer (§2.4) the winning partition is determined largely by initialization — the fixed point is initialization-dependent.

**When does this match Bayes-optimal prototypes?** Only if the winning partition coincides with true latent clusters, which requires either (a) orthogonal task supports in feature space (plausible Split-MNIST, borderline CIFAR) or (b) a balancing force ensuring partition covers the distribution.

### 2.3 EWC forget-cost as gradient-interference proxy

The standard EWC penalty Ω(θ) = ½ Σ_k F_k (θ_k − θ*_k)². Under the Laplace approximation of the old-task posterior, Ω is the second-order Taylor expansion of KL(p_old || p_θ). So:

**Valid claim:** Ω ≤ ε ⇒ KL to old-task posterior ≤ ε + O(‖Δθ‖³), locally.

**Invalid claims sometimes made:** (i) Ω is not a global upper bound — the cubic term can dominate under large drift; (ii) Ω bounds *forgetting* (loss increase on old data) only through the KL-to-loss chain, which requires bounded log-likelihood ratios; (iii) empirical Fisher ≠ true Fisher (Martens 2020; Kunstner et al. 2019) — the bound degrades under miscalibrated models.

For the bid's purposes, **low forget_cost implies small local parameter drift**, under (a) Laplace-approximation validity, (b) full-rank Fisher (the 0.1 clamp from the codebase is exactly a regularizer to enforce this), (c) no task-boundary stale-ness (Fisher was updated at the *relevant* old task).

### 2.4 Rich-get-richer / Matthew effect

**Proposition (Pólya-urn dynamics of the bid).** Consider a batch drawn iid from a fixed distribution, α > 0, β = 0 (between task boundaries). Let n_i(t) = cumulative batches won by expert i up to step t. Then:
1. Σ̂_i (expert i's sample covariance over μ_i) has effective sample size proportional to n_i.
2. d_M(x; μ_i) has variance that *decreases* in n_i (tighter covariance ⇒ lower distances on in-distribution x).
3. p(i wins next batch) is increasing in n_i by monotonicity of the argmin in bid.

This is the defining signature of a preferential-attachment / Pólya process. Classical result (Arthur 1989; Pemantle 2007): such processes converge to an absorbing vertex of the simplex with probability 1 under mild conditions — i.e., one or two experts absorb all mass. This is your empirical training-time prototype-routing collapse, *on the nose*.

**Structural remedy (theory-approved):** break the positive feedback. Options:
- Conscience term −γ · n_i / t (DeSieno 1988) → provably stabilizes to uniform partition under convex settings.
- Capacity cap (Switch/GShard overflow) → hard constraint, distributes but is non-differentiable.
- Prior-weighted bid (Bayesian auction, Astra's direction) → encode a belief that the task distribution is balanced.
The label-prototype linear-routing blend (recent commits `0046042`) is a soft version of this — it injects a task-balanced signal that breaks the urn dynamics.

---

## 3. Scale 2 — CIFAR-100

### 3.1 Calibration with growing centroid count

At 100 classes / 5–20 tasks, K_i (centroids per expert) ranges up to ~50. Feature dimension d is typically 512 (ResNet) or 768 (ViT). With n_i samples per centroid, the Mahalanobis plug-in estimator has

  E[d̂_M − d_M]² = O(d² / n_i) (Bodnar-Okhrin 2008 bounds on sample Mahalanobis).

As K grows, per-centroid samples thin, variance grows, and the bid's ordering becomes noisy. Calibration fails when the bid noise exceeds the margin ρ (A2).

### 3.2 Shrinkage — FeCAM as theoretical minimum

**FeCAM recipe:** Σ_i ← (1 − λ) Σ̂_i + λ · (tr(Σ̂_i)/d) · I, with tied Σ across classes within expert.

**Theoretical justification:** Ledoit-Wolf 2004 give the closed-form optimal λ* minimizing E‖Σ_ℓw − Σ‖_F². When n < d, λ* → 1 (full regularization); when n >> d, λ* → 0. The trade is between bias (λ ≠ 0 biases toward isotropy) and variance (low λ inflates estimation error in low-rank directions).

**What breaks at d=4096 (LLM-scale):** if per-expert n < 4096, the *unregularized* Σ̂ is rank-deficient and d_M is undefined (pseudo-inverse gives unbounded distances on null-space components). FeCAM is *mandatory*, not optional, at Scale 3.

### 3.3 No-regret interpretation

Treat each expert as an arm, the auction as a context-dependent selection rule, and forget_cost as a stateful arm-cost. The relevant framework is **contextual bandits with non-stationary costs** (Auer 2002; Slivkins 2019). The auction is a *greedy* policy, not a bandit algorithm — no exploration, no regret bound against best-in-hindsight.

**Conjecture.** With probability p_explore > 0 of routing to a random expert (ε-greedy), and with forget_cost clamped, a simple adversarial-bandit reduction (EXP3-style) achieves regret O(√(T · E log E)) vs. best-fixed-expert under bounded losses. **Proof gap:** standard bandit regret assumes stochastic or oblivious adversary; here, the cost of expert i depends on its own history (EWC), so the adversary is adaptive. I don't have a tight result.

---

## 4. Scale 3 — LLM MoE FFN layer

### 4.1 Per-token auction concentration

**Claim (conditional).** Suppose (i) features φ(x) are σ-sub-Gaussian, (ii) prototypes μ_i are L-Lipschitz in expert parameters, (iii) the bid has a margin Δ(x) between true-winner and runner-up. Then

  P(argmin b̂_i ≠ argmin b_i) ≤ 2 E · exp(−Δ² / (8 σ² L²)).

**What this requires for LLM scale (E=128):**
- Δ must exceed O(σ L √(log E)) — log-factor grows slowly, manageable.
- L depends on centroid count K_i. As K grows, the bid surface becomes more bumpy and L increases — roughly L = O(√K · σ_μ) for random prototypes.
- Margin scales as O(Δ_intrinsic / √K_experts) under random-feature symmetry (each expert occupies less of the feature space).

**Net:** for fixed d, Δ / (σ L) ≈ Δ_intrinsic / (σ · K_experts · √K_centroid). This is not vacuous, but it is not automatic — margins must be *engineered* (entropy-regularized training; specialization-aware init; Fade's load-balance-by-bias).

### 4.2 Pre-training vs continual fine-tuning

EWC forget-cost is defined relative to a *reference θ\** and *reference Fisher F\**. Both come from "end of old task." During pre-training from random init, no old task exists — F\* is uninformative (Fisher of a random network concentrates near zero except through architecture-induced priors), and θ\* is the init.

**Consequence:** β · C_F(θ_i; F_i) is dominated by noise during pre-training. The bid degenerates to pure Mahalanobis routing (α term only), which accelerates rich-get-richer (§2.4).

**When forget_cost becomes meaningful:** after sufficient exposure that (i) F has stabilized (per-parameter Fisher estimates have low variance) and (ii) θ has moved into a region where ΔL (loss on earlier data) is actually measurable. Heuristically this is ≥ 1 epoch of a stable distribution. During pre-training's early phase, β should be annealed from 0 — mathematically and empirically.

**Strong recommendation:** at Scale 3 pre-training, MoB in its current form is **not theoretically grounded**. At Scale 3 continual fine-tuning from a pre-trained MoE, MoB reduces to the CIFAR-100 regime and §3 applies.

### 4.3 Specialization theorems and aux-loss-free routing

DeepSeek-V3 aux-loss-free bias updates: b_i ← b_i − γ · (load_i − load_target). This is a primal-dual step on the Lagrangian of a load-constrained routing problem. It converges to a KKT point under convexity assumptions that do not hold for neural routers — empirical stability is documented (DeepSeek-V3 report) but no general theorem exists.

**What 2024–2025 theory actually says** (Fade's scan + my read): expert specialization in MoE is driven primarily by (i) top-k sparsity forcing discrete choices and (ii) aux losses for load balance. Neither mechanism has a proper specialization theorem of the form "expert i becomes Bayes-optimal for cluster i". Equifinality results (Fade's reference) suggest *many* specialization configurations are empirically reachable — i.e., specialization is *underdetermined* without additional constraints. This is a warning sign for MoB: greedy auction without load balance almost certainly collapses to a non-canonical specialization.

### 4.4 DSIC at scale

Myerson's revenue equivalence requires symmetric bidders. Experts are *asymmetric* (different parameters, different Fisher, different histories). Revenue equivalence fails; first-price and second-price can yield different outcomes.

**Practical take:** DSIC is not the primary desideratum at LLM scale. What matters is (a) computational cheapness (first-score is O(E), VCG is O(E²) or worse with externalities) and (b) stable incentives (bidders shouldn't game). If α, β are fixed (not learned per-expert), the mechanism is trivially strategy-proof and DSIC is moot. If they're learned, Astra's Myerson-monotonicity gate is the right theoretical check.

---

## 5. Cross-scale invariant

**The one property that must hold across all three scales:**

> **Bid monotonicity in expert-specialization-distance.** For each expert i and input x, b_i(x) must be a monotonically increasing function of a meaningful distance from x to expert i's specialization region.

Formally: there exists a feature-space distance d(x, S_i) (S_i = expert's support) such that b_i(x) = f_i(d(x, S_i)) with f_i monotonic non-decreasing. The Mahalanobis term satisfies this by construction (when Σ_i is well-conditioned). The EWC term does *not* inherently — forget_cost depends only on θ_i, not on x. So the bid as a whole is *input-x-monotonic only in the α term*.

**Consequence:** if β is too large relative to α, the bid can become non-monotonic in x (the forget_cost shift dominates input-driven changes). This violates the invariant and breaks:
- Truthfulness (non-monotone ⇒ Myerson fails).
- Regret bounds (bandit reductions require monotone-in-context costs).
- Convergence of the selection rule under A2.

**Design rule:** keep α · var_x[d_M] >> β · ‖F_i‖ as an order-of-magnitude check. This is the theoretical version of "alpha dominates beta unless you're at a task boundary."

---

## 6. Rigorous theorem attempt

**Theorem (MoB within-task stability, informal).** Under assumptions (A1)–(A4) of §2.1 plus
(A5) Fisher clamp F_i ⪰ c · I for some c > 0 (enforced in codebase),
(A6) Prototype covariance shrinkage Σ_i = (1−λ)Σ̂ + λ·(tr/d)I with λ > 0,
(A7) Conscience or load-balance term with strength γ > 0 such that the bid's net preferential-attachment coefficient is negative,

the MoB joint system (θ_i, μ_i, F_i, assignment) converges a.s. to a local equilibrium in which (a) each θ_i is stationary for its induced local loss, (b) μ_i equals the conditional winning mean, (c) the assignment partition is stable up to measure-zero boundaries.

**Proof sketch.**
1. (A5) ensures the EWC-induced bid term is bounded and C¹ in θ.
2. (A6) ensures d_M is well-defined and C¹ in (x, μ, Σ̂).
3. (A7) ensures the urn process has a non-absorbing stationary distribution.
4. (A1)–(A4) give standard two-timescale SA convergence per expert.
5. Combine: the joint dynamics admit a Lyapunov function V = Σ_i L_i + KL-to-balanced-partition penalty; V is non-increasing in expectation; LaSalle gives convergence to a level set.

**Honest gaps.** (i) "Local equilibrium" is weak — non-convex loss landscapes admit many. (ii) The Lyapunov construction requires (A7)'s strength γ to exceed a computable but scenario-dependent threshold. (iii) The argument is *within-task*. Cross-task stability requires an additional "bounded drift" assumption on D_τ that I have not written. (iv) This is not a rate result — no O(1/t) bound, only asymptotic convergence.

**Status.** Principled conjecture. The pieces exist in the literature (Borkar SA, Ledoit-Wolf, Pemantle urn theory, LaSalle); assembling them into a publishable theorem is a 2–4 week effort, not a one-shot.

---

## 7. Where I defer to others

1. **Astra (auction theory):** Is Myerson-monotonicity achievable for *learned* α, β per expert? My (A7) conscience term is a principal-side intervention; Astra's angle is bidder-side. If learned bidders violate monotonicity, my convergence theorem's (A6)/(A7) are insufficient — we need a mechanism-side redesign.

2. **Chamber (model architecture):** What is the empirical Lipschitz constant L of the realized bid surface at each scale? My Scale-3 concentration bound (§4.1) is tight only if L can be bounded or estimated; without that, the claim is qualitative.

3. **Killjoy (systems tradeoffs):** At per-token routing with E=128, can conscience / load-balance updates happen at the required timescale without memory-bandwidth blowup? If the fix that saves convergence (A7) is infeasible at scale, theory says "won't work" regardless of mechanism design.

4. **Fade (frontier):** Is there a 2024–2026 result establishing specialization for *aux-loss-free* MoE under any precise assumption? My §4.3 reads the literature as "no formal specialization theorem exists"; if Fade has found one, it changes the Scale-3 green-light/red-light call.

---

## 8. Dealbreaker

**Single result that forces retraction:** A construction showing that, for any α, β > 0 and any Fisher clamp c, there exists a sequence of tasks on which the MoB assignment is a measure-zero set of experts with probability approaching 1 as t → ∞ — *even with* conscience term γ > 0. Equivalently: a formal proof that MoB's collapse is generic rather than regime-specific.

This would mean (A7)'s remedy is insufficient and the mechanism itself is structurally broken. In that case, MoB must be replaced by a non-greedy, exploration-augmented routing rule (e.g., probabilistic bids with softmax temperature, or proper bandit routing). I consider this outcome **plausible but not likely** — the empirical label-prototype blend's success argues against it.

**Negative-result-lite (more likely):** if experiments show that *no* finite conscience term stabilizes Scale-3 pre-training MoB, the red-light on §4.2 hardens to permanent, and MoB is a continual-FT-only mechanism. That's not a retraction — it's a scope reduction.

---

**Word count:** ~2,380. Notation kept minimal where intuition suffices; formal where it must. Proof gaps flagged explicitly per Sage principles.
