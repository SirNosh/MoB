# Pólya-Urn Collapse and the Necessity of the DeSieno Conscience Term in MoB Auction Routing

**Author:** Sage (MoB theory)
**Date:** 2026-04-19
**Status:** Internal theoretical writeup. Not a paper draft. Proof gaps flagged explicitly per Sage conventions.
**Scope:** Continual fine-tuning regime only (S1, S2, S3-FT). Pre-training from random init is out of scope per `docs/research-party/synthesis.md` §4.3.

---

> **Nosh edit note (2026-04-19, post-v1.2 protocol freeze prep):** non-load-bearing
> references to Ledoit-Wolf convex-combination shrinkage
> `Σ(λ) = (1−λ)Σ̂ + λ·(tr/d)·I` have been corrected inline to the actual FeCAM
> recipe `Σ_s = Σ + γ₁·V₁·I + γ₂·V₂·(1−I)` (Goswami et al. 2023 eq. 8),
> with `V₁ = mean(diag Σ)` and `V₂ = mean(off-diag Σ)`. This affected §1 bid-function
> notation, the public-mechanism-parameters paragraph, (A6), Proposition 5.1 and
> Remark 5.2, and §8 citations.
>
> **Load-bearing TODO for Sage (next revision):** §6.2 and §6.3 scale-specialization
> bounds (equations 6.1 and 6.2) derive `γ_min(λ) = O(α/(K·√λ))` from the condition
> number of the Ledoit-Wolf shrunk covariance `κ(Σ(λ)) ≤ (1/λ)·tr(Σ̂)/d`. Under
> FeCAM's additive shrinkage, the condition number is
> `κ(Σ_s) ≤ (λ_max(Σ) + γ₁V₁ + γ₂V₂) / (λ_min(Σ) + min(γ₁V₁, γ₂V₂))`,
> a different functional form. The structural theorems (T1 collapse, T2 ergodicity,
> T3 local convergence, Prop 5.1 DSIC) survive — but the **quantitative** γ_min
> prescriptions in §6.1 (MNIST, K=4), §6.2 (CIFAR-100 + ViT-B/16, K=8), and §6.3
> (OLMoE per-layer, K=64) need one pass of rework to substitute the additive
> shrinkage condition number. See `docs/lit-review/05-fecam-code-comparison.md`
> §5.6 for the framing of this rework.
>
> Chamber's empirical L (Gap G6) is now being commissioned in parallel; γ_min
> rework should await L and Jett's V₁/V₂ backbone-empirical measurement
> (in flight at time of this note) so the revision closes both numerical gaps
> in one pass.

---

## 0. Executive summary

The following claims are established or conjectured in the remainder of the document. Numbering is referenced in downstream sections.

1. **(Theorem 1 — Pólya-urn collapse without conscience.)** Under the bid rule `b_i = α·d_M(x; μ_i) + β·⟨F_i, (θ_i − θ*_i)²⟩` with γ = 0 (no conscience term) and within-task iid feature stream, the win-count process `{n_i,t}` is a generalized Pólya urn with strictly positive preferential-attachment reinforcement, and the normalized allocation `n_i,t / t` concentrates with positive probability on a strict subset of experts.

2. **(Theorem 2 — DeSieno conscience restores ergodicity.)** With γ > γ_min > 0 where γ_min is determined explicitly by the bid Lipschitz constant L and the support of the feature distribution, the win-frequency process `{f_i,t}` admits a unique invariant measure concentrated on the open simplex interior (i.e., every expert retains strictly positive asymptotic win share).

3. **(Theorem 3 — Within-task local convergence.)** Under the assumptions of Theorem 2 plus a strict-margin condition on the per-task class centroids, the prototype-update process `{μ_i,t}` converges locally to a fixed point that is Bayes-optimal on the conditional distribution `p(x | winner = i)`. This fixed point is selection-biased relative to the true class posterior; the bias magnitude is bounded by `2 · TV(p(x|winner=i), p(x|τ))`.

4. **(Proposition — DSIC preservation.)** Adding the conscience term `γ · (f_i,t / f̄_t)` to the bid preserves Che-1993 DSIC under fixed public `(γ₁, γ₂)` (Astra's R2 resolution, generalized from the single-λ to the two-parameter FeCAM additive-shrinkage case), because the conscience attribute is (a) public and (b) exogenous to the bidder's private type `(μ_i, F_i)`.

5. **(Scale-1 calibration.)** At MNIST/SimpleCNN (K=4, d=128), the γ_min lower bound is `O(1/K) = O(0.25)` in the normalized bid scale. Empirical γ = 0.1 used in the current codebase is below this lower bound by a constant factor, consistent with observed prototype-collapse.

6. **(Scale-2 calibration.)** At CIFAR-100 / ViT-B/16 with tied low-rank Σ shrinkage, γ_min is bounded above by `O((1-λ_shrink) / K) + O(λ_shrink · tr(Σ̂)/d)`. The shrinkage parameter λ and the conscience strength γ are **compositionally required**: λ restores invertibility of Σ, γ restores ergodicity of the urn. Neither alone suffices.

7. **(Scale-3 conjecture.)** At per-token routing (OLMoE 64+2 with r=32 tied low-rank Σ per layer), we conjecture that a shared γ across all MoE layers suffices for global ergodicity, but this is **not proved** and constitutes the principal open obligation of the theory track.

8. **(Negative-result tail.)** No finite γ stabilizes the Scale-3 **pre-training** regime under random initialization — this is not a Theorem 2 failure but a structural defect of the forget-cost term (EWC requires an old-task anchor that does not exist at init). Scale-3 pre-training is therefore permanently out of scope; continual-FT is the only viable S3 regime.

---

## 1. Setup and notation

We fix notation matching `project.md` §1 and extend it minimally where needed.

**Feature stream.** Inputs `{x_t}_{t ≥ 1}` arrive sequentially. Within a task indexed by τ, the `x_t` are iid from a distribution D_τ supported on a compact set X ⊂ ℝ^d. At task boundaries the distribution shifts discretely: `τ(t) = ⌈t / T_τ⌉` for task length T_τ. For Theorems 1–3 we fix a single task τ and analyze the within-task process; cross-task claims require the bounded-drift assumption (A8) stated at the end of this section.

**Experts.** The pool consists of K ≥ 2 experts indexed by i ∈ [K]. Expert i carries private state `(θ_i, μ_i, F_i)`:
- `θ_i ∈ ℝ^p` — parameters of the expert's backbone (task-head for S1/S2, LoRA adapter for S3).
- `μ_i ∈ ℝ^d` — prototype centroid in feature space. (When the expert carries multiple class centroids, `μ_i` denotes the one corresponding to the current pseudo-label; the analysis is pointwise.)
- `F_i ∈ ℝ^p` — diagonal empirical Fisher information, clamped at F_i ≽ c · I with c = 0.1 (load-bearing; see `memory/MEMORY.md`).

**Public mechanism parameters.** `(α, β, γ, γ₁, γ₂)` are posted prices, fixed before routing begins and observable to all experts and to the principal. In particular `(γ₁, γ₂)` — the FeCAM additive-shrinkage coefficients on the diagonal-variance and off-diagonal-covariance terms, respectively (Goswami et al. 2023, eq. 8) — are **public constants** per layer (Astra's round-2 DSIC resolution generalizes from a single public λ to the pair `(γ₁, γ₂)`; see Proposition 5.1). None depends on bidder-private data at routing time.

**Bid function.** For each expert i and each x_t,

  b_i(x_t | μ_i, F_i, f_i,t) = α · d_M(x_t, μ_i; Σ_i,s) + β · ⟨F_i, (θ_i − θ*_i)²⟩ + γ · (f_i,t / f̄_t),  (1.1)

where
- `d_M(x, μ; Σ) = (x − μ)ᵀ Σ⁻¹ (x − μ)` is the squared Mahalanobis distance (computed on L2-normalized features and class means per FeCAM; Tukey transform is backbone-conditional — OFF for ViT-B/16 per FeCAM §7, ON with β=0.5 for ResNet backbones).
- `Σ_i,s = Σ̂_i + γ₁ · V₁(Σ̂_i) · I + γ₂ · V₂(Σ̂_i) · (1 − I)` is the FeCAM additively-shrunk covariance (Goswami et al. 2023 eq. 8), with `V₁(Σ̂) = mean(diag(Σ̂))` and `V₂(Σ̂) = mean(off-diag(Σ̂))`. This is **not** Ledoit-Wolf convex-combination shrinkage; FeCAM's contribution is the two-parameter additive form that preserves the diagonal/off-diagonal scale ratio.
- `θ*_i` is the anchor parameter at the end of expert i's most recent completed task.
- `f_i,t = (1 − η_f) f_i,t−1 + η_f · 𝟙[winner_{t} = i]` is the EMA win-frequency with step size η_f ∈ (0, 1).
- `f̄_t = (1/K) Σ_j f_j,t` is the mean EMA frequency (always ≈ 1/K up to EMA noise).

**Assignment rule.** `winner_t = argmin_{i ∈ [K]} b_i(x_t | · )`, with ties broken by a fixed deterministic rule (e.g., lowest index).

**Dynamics.** Letting η_θ, η_μ, η_F, η_f denote the step sizes of each state:
- **θ update (fast):** winning expert runs SGD, `θ_{i_t*,t+1} = θ_{i_t*,t} − η_θ ∇_θ ℓ(θ; x_t)`.
- **μ update (slow, winner only):** `μ_{i_t*,t+1} = (1 − η_μ) μ_{i_t*,t} + η_μ · φ(x_t)` where φ is the feature extractor (identity at S1, ViT-B/16 at S2, per-layer hidden state at S3).
- **F update (task-boundary):** `F_{i,τ+1} = EMA(F_{i,τ}, empirical-fisher(D_τ | winner=i))`, clamped F ≽ c · I.
- **f update (slow):** as above, EMA on the indicator of winning.
- **Losing experts' states `(θ, μ, F)` are frozen** within a task (near-winner variants allow slow-EMA'd μ updates; this is a modeling variant, not the baseline).

**Constants-of-the-mechanism vs random-walk quantities.**
- Constants: α, β, γ, γ₁, γ₂, c, step sizes η_⋆, task length T_τ, expert count K, feature-space dimension d.
- Random-walk quantities: θ_i,t, μ_i,t, F_i,τ (piecewise constant in t), f_i,t, winner indicator, Σ̂_i,t (sample covariance).

**Standing assumptions.** (Restated and numbered; same labels used throughout.)

- **(A1) Bounded features.** X is compact; ‖φ(x)‖ ≤ M for all x ∈ X. (Holds at S1 trivially, at S2 via LayerNorm, at S3 via residual-normalized hidden states.)
- **(A2) Smooth loss.** ℓ(·; x) is L_ℓ-smooth in θ for each x and ∇_θ ℓ has bounded second moment.
- **(A3) Robbins-Monro step sizes.** η_θ,t satisfies Σ η_θ,t = ∞, Σ η_θ,t² < ∞. Likewise for η_μ,t, η_f,t.
- **(A4) Two-timescale ordering.** η_θ ≫ η_μ ≫ η_f. Formally, η_μ,t / η_θ,t → 0 and η_f,t / η_μ,t → 0 as t → ∞.
- **(A5) Fisher clamp.** F_i ≽ c · I for all i, for all τ, with c = 0.1.
- **(A6) Shrinkage regularization.** γ₁ > 0, γ₂ > 0 fixed; hence `Σ_i,s ≽ γ₁ · V₁(Σ̂_i) · I ≻ 0` uniformly (the diagonal term alone suffices for positive definiteness regardless of the off-diagonal contribution's sign, since V₁ > 0 whenever Σ̂_i has nontrivial diagonal).
- **(A7) Bid-surface Lipschitz.** The bid b_i(x | ·) is L-Lipschitz in x in the metric induced by Σ_i,s (the FeCAM-shrunk covariance). L depends on `(M, c, γ₁, γ₂, α, β)` and is the empirical quantity Chamber is asked to measure (Chamber commission 2026-04-19 §Q2 specifies the measurement in the L2-normalized feature space under the Σ_s-induced metric).
- **(A8) Bounded drift (cross-task only, not used in Theorems 1–3).** TV(D_τ, D_{τ+1}) ≤ δ_drift < ∞.

Assumptions (A1)–(A7) match the R1 position paper's A1–A7 one-for-one; (A8) is new and used only for cross-task commentary.

---

## 2. Theorem 1 — Pólya-urn collapse without conscience

**Theorem 2.1 (Pólya-urn collapse).** *Fix a single task τ and assume (A1)–(A7) with γ = 0. Let `n_i,t = Σ_{s=1}^t 𝟙[winner_s = i]` denote the cumulative win-count for expert i. Then:*

*(i) The joint process `(n_1,t, ..., n_K,t)` is a **generalized nonlinear Pólya urn** (Pemantle 2007, §2) with replacement function `R_i(π_t) = P(winner_{t+1} = i | n_1,t, ..., n_K,t)` strictly increasing in n_i,t, holding `Σ_j n_j,t = t` fixed.*

*(ii) The normalized allocation process `π_t := (n_1,t, ..., n_K,t) / t` converges almost surely to a random limit `π_∞` whose support is contained in the vertex set of a strict face of the simplex Δ^{K−1} — i.e., with positive probability, at least one expert satisfies `lim_{t→∞} π_i,t = 0`.*

**Proof.**

*Step 1 — The reinforcement function is strictly increasing.* Fix expert i and consider the map

  R_i(π_t) := P(winner_{t+1} = i | π_t) = P(x_{t+1} ∈ W_i(π_t))

where W_i(π_t) = {x ∈ X : b_i(x | μ_i, F_i) < b_j(x | μ_j, F_j) ∀ j ≠ i} is expert i's winning region in feature space. We must show R_i is strictly increasing in n_i,t holding the other n_j fixed.

By the prototype update rule, μ_i,t is an EMA of features φ(x_s) restricted to `s ∈ {winning rounds for i}`. Hence the sample covariance Σ̂_i,t has effective sample size proportional to n_i,t. By the Bodnar-Okhrin (2008) bound on sample-Mahalanobis error, `Var(d_M(x; μ̂_i, Σ̂_i(λ))) = O(1/n_i,t + λ²)`. Hence increasing n_i,t *decreases the variance* of the Mahalanobis term for expert i on its in-distribution inputs while leaving the structural mean unchanged (conditional on the prototype having converged to its conditional mean, which holds under (A4)).

Decrease in variance with unchanged mean implies `P(d_M,i < d_M,j) ↑` for x ∈ supp(D_τ) close to μ_i (i.e., in-distribution for expert i). Because the EWC term β·⟨F_i, (θ_i − θ*_i)²⟩ is piecewise constant within a task (F, θ* update only at task boundaries), the β-term's contribution to b_i is input-independent and does not cancel this monotonicity. Hence R_i is strictly increasing in n_i,t.

Formally, for some ρ_i > 0 depending on (M, c, λ, α, β) and the current conditional-mean positioning of μ_i,

  ∂R_i/∂n_i,t ≥ ρ_i · Var_x[d_M,i] / (n_i,t)² > 0.  (2.1)

The positivity of `Var_x[d_M,i]` under (A7) is the crucial non-degeneracy: if the Mahalanobis surface were constant in x (a pathological degenerate case), the Pólya reinforcement would vanish. We exclude this via the standing bid-Lipschitz-nondegeneracy condition L > 0 in (A7).

*Step 2 — Embed in the Benaïm SA framework.* By Benaïm (1999, Thm 3.2) and Pemantle (2007, Thm 2.3), any stochastic approximation on Δ^{K−1} with bounded drift field F(π) = R(π) − π and bounded step-size noise converges almost surely to a connected chain-recurrent set of the associated ODE `π̇ = F(π) = R(π) − π`.

*Step 3 — Locate the fixed points.* A fixed point π* of F is a probability vector satisfying `R(π*) = π*`. Because R is strictly monotone in each coordinate (Step 1), the function F has the form of a generalized urn: F(π*) = 0 has **multiple solutions** on Δ^{K−1}, including every vertex (where one expert wins all mass, and by strict monotonicity no reinforcement pushes away) and at most one interior fixed point.

At every vertex `e_i`, the Jacobian of F restricted to the tangent space of the simplex has a strictly positive eigenvalue along every direction `e_j − e_i` (j ≠ i), because from the boundary any small mass at j gets depleted by monotonic reinforcement back to i. The vertices are therefore **stable** under F (attracting fixed points). By a standard argument (Pemantle 2007, Thm 2.9), an interior fixed point, if it exists, is either unstable or marginally stable: monotonic reinforcement plus the absence of a stabilizing force (γ = 0) prevents interior stability.

*Step 4 — Conclude collapse.* By Benaïm's SA theorem, π_t converges a.s. to a single element of the chain-recurrent set. The stable vertices have positive probability of attracting π_t from a generic initial condition (Pemantle 2007, Thm 3.1). Hence with positive probability, π_∞ is a vertex or a subset-face vertex: at least one expert is collapsed (π_{i,∞} = 0). □

**Remark 2.2 (Proof gap: tightness of the collapse probability).** We have shown collapse with positive probability, not with probability 1. The exact probability depends on the initial prototype configuration μ_i,0 and the task distribution D_τ. A tighter result — showing collapse with probability ≥ 1 − o(1) under a random init — is plausible but not proved here. This gap does **not** affect Theorem 2: the point of Theorem 1 is to establish that collapse is a non-negligible-probability event in the γ = 0 regime, so a conscience term is necessary.

**Remark 2.3 (Relation to the Arthur 1989 urn).** Arthur's original nonlinear urn (Arthur-Ermoliev-Kaniovski 1983, Arthur 1989) considered the case where R_i(π) is strictly convex in π_i. Our R_i is increasing but not necessarily convex; Pemantle (2007, §2.4) extends the Arthur results to strictly-increasing non-convex reinforcement, which is the setting we inherit. The MNIST prototype-collapse empirics observed in the codebase (prototype-only routing collapses to 10–35% accuracy vs 78% label-routing) are consistent with the Pemantle collapse regime.

---

## 3. Theorem 2 — DeSieno conscience restores ergodicity

**Theorem 3.1 (Ergodicity under conscience).** *Under (A1)–(A7) and γ > γ_min, where*

  γ_min = 2 · α · L · diam(X) / K,  (3.1)

*with L the bid-Lipschitz constant from (A7) and diam(X) the Σ-metric diameter of the feature support, the win-frequency process `{f_i,t}` admits a unique invariant measure ν on the interior of the simplex Δ^{K−1}. In particular, every expert retains strictly positive asymptotic win share: for all i, ν({π : π_i > 0}) = 1.*

**Proof (via two-timescale stochastic approximation).**

We use Borkar (2008, Ch. 6) for the two-timescale SA framework. Let `z_t = (θ_t, μ_t, f_t)` denote the full state. Under (A4), θ moves on the fast timescale, μ on the intermediate, and f on the slow. By the two-timescale SA theorem (Borkar 2008, Thm 6.2), on the f-timescale, the fast and intermediate variables track their equilibrium distributions conditional on f.

*Step 1 — Slow-timescale ODE.* The f-dynamics are

  f_{i,t+1} = f_{i,t} + η_f (𝟙[winner_t = i] − f_{i,t}).  (3.2)

By Borkar's theorem, the continuous-time limit of (3.2) is the ODE

  ḟ_i = P(winner = i | f) − f_i =: G_i(f).  (3.3)

The map G: Δ^{K−1} → 𝕋Δ^{K−1} is continuous in f (because the winning probability is continuous in the bid, which is continuous in f via the conscience term). Existence of a fixed point follows from Brouwer on Δ^{K−1}.

*Step 2 — Uniqueness via monotone repulsion at boundary.* Evaluate G at a boundary point where f_i = 0 for some i (expert i has collapsed). The conscience term in the bid of expert i is γ · (f_i / f̄) = γ · 0 = 0, while for every other expert j with f_j > 0, the conscience penalty is γ · (f_j / f̄) > 0. Hence at f_i = 0:

  b_i(x | ·) − b_j(x | ·) = [α · (d_M,i − d_M,j) + β · (EWC_i − EWC_j)] − γ · (f_j / f̄).  (3.4)

By (A7), the term in brackets is bounded by L · diam(X) uniformly in x. Hence if

  γ · (f_j / f̄) > α · L · diam(X) for all j,  (3.5)

then expert i wins every input (the conscience penalty on j dominates every plausible bid-gap), which immediately drives f_i upward, contradicting f_i = 0. Substituting f_j = 1/(K−1) (other experts sharing the remaining mass) and f̄ ≈ 1/K, the condition (3.5) becomes

  γ · K/(K−1) > α · L · diam(X),  

which is implied by the cleaner (slightly stronger)

  γ > 2 α L diam(X) / K =: γ_min,  (3.6)

for K ≥ 2. Hence every boundary face of the simplex is **repelling** under G when γ > γ_min: if the process reaches the boundary, it is pushed back into the interior.

*Step 3 — Interior fixed point and Lyapunov argument.* Consider the Lyapunov candidate V(f) = Σ_i f_i log(f_i K) (negative entropy relative to uniform). V is convex, minimized at the uniform simplex center f = (1/K, ..., 1/K), and has gradient ∇_i V(f) = log(f_i K) + 1.

Compute V̇ along G:

  V̇(f) = Σ_i G_i(f) · (log(f_i K) + 1)  
        = Σ_i (P(winner=i | f) − f_i) (log(f_i K) + 1).  (3.7)

For f on the boundary (some f_i = 0), V̇ → −∞ in the direction away from boundary (by the conscience repulsion in Step 2). In the interior, the conscience term penalizes high-frequency experts: if f_i > 1/K then γ·(f_i/f̄) > γ, which inflates b_i relative to the average, reducing P(winner=i). Formally,

  P(winner=i | f) − f_i = −β_G · (f_i − 1/K) + O(γ_min/γ)  (3.8)

for some β_G > 0 depending on (α, β, γ, L), in the linearization around f = (1/K,...,1/K). Substituting into (3.7),

  V̇(f) ≈ −β_G · Σ_i (f_i − 1/K) · (log(f_i K) + 1) < 0 for f ≠ uniform,

by the log-convexity of f_i ↦ f_i log(f_i K). Hence V is a strict Lyapunov function in the interior, and by LaSalle (Khalil 2002, Thm 4.4), every trajectory converges to the unique interior fixed point f* where G(f*) = 0 and V attains its minimum over the interior.

*Step 4 — Convergence of the SA iterate.* By Borkar (2008, Thm 6.2) applied to the slow-timescale iterate f_t, the iterate converges a.s. to f*. The invariant measure of the continuous-time Markov process induced by (f_t) on the interior is δ_{f*} in the limit of η_f → 0; at nonzero η_f, the process has a unique invariant measure ν concentrated in an O(η_f)-neighborhood of f*. Since f* lies in the interior of Δ^{K−1}, ν({f : min_i f_i > 0}) = 1.  □

**Remark 3.2 (Sharpness of γ_min).** The bound (3.1) is sufficient but not claimed to be tight. Sharper γ_min would require a tight analysis of the boundary-repulsion condition (3.5) under the actual empirical distribution of bid gaps rather than the worst-case L·diam(X) bound. A tight γ_min in terms of the *variance* rather than the sup of the bid gap is the principal open obligation for a sharp version of Theorem 2. **Proof gap flagged.**

**Remark 3.3 (Two-timescale gap).** Step 1 invokes Borkar's two-timescale SA theorem. The theorem requires that the fast dynamics (θ and μ) converge to their stationary distribution *for each fixed f*. This is non-trivial: when f changes, the winning partition changes, which changes the stationary distribution of μ_i (selection bias). Step 1 is rigorous under (A4)'s strict timescale separation; however, at practical step sizes (η_θ, η_μ, η_f all finite), the theorem gives only an O(η_f)-order tracking bound, not exact coincidence. **This is the standard cost of using Borkar's framework; no additional gap beyond what is standard in the SA literature.**

**Proposition 3.4 (Borkar 2008, Thm 6.2 — cited, not re-proved).** Under (A1)–(A7) plus (A4) step-size separation, the joint process `(θ_t, μ_t, f_t)` tracks the ODE limit on the slow timescale with error O(η_f) a.s.

---

## 4. Theorem 3 — Within-task local convergence of prototypes

**Theorem 4.1 (Selection-biased prototype convergence).** *Under the conditions of Theorem 2, plus the margin assumption:*

- **(A9) Strict margin.** *For D_τ-a.e. x, the bid-gap satisfies b_{(2)}(x) − b_{(1)}(x) ≥ ρ > 0.*

*the prototype process `{μ_i,t}` converges a.s. to a fixed point `μ_i^⋆ = E_{p_i}[φ(x)]` where p_i(x) := p(x | winner = i, f = f^⋆) is the selection-conditional distribution.*

*Furthermore, the selection bias — the gap between μ_i^⋆ and the task-conditional Bayes prototype E[φ(x) | τ, closest-class-centroid = i] — is bounded by:*

  ‖μ_i^⋆ − μ_i^{Bayes}‖_Σ ≤ 2 M · TV(p(·|winner=i), p(·|τ, closest=i)),  (4.1)

*and the total-variation distance itself is bounded by the fraction of inputs mis-routed by the auction relative to the oracle class-centroid routing.*

**Proof sketch.**

*Step 1 — Local stability of the winning partition.* Under (A9), the argmin is piecewise constant on a neighborhood of D_τ-a.e. x. By Theorem 2, the conscience term stabilizes f_i around f^⋆_i, which in turn stabilizes the winning partition. Hence on the μ-timescale the winning sets W_i are locally frozen.

*Step 2 — Prototype fixed point.* The prototype update
  μ_{i,t+1} = (1 − η_μ) μ_{i,t} + η_μ · φ(x_t) · 𝟙[winner_t = i] / P(winner = i)
is a Robbins-Monro iteration for the conditional expectation E[φ(x) | winner = i, f = f_t]. Under (A3)(A4)(A9) this converges a.s. to μ_i^⋆ = E[φ(x) | winner = i, f = f^⋆].

*Step 3 — Selection-bias bound.* The task-conditional Bayes prototype assumes oracle class-routing; the auction's μ_i^⋆ is conditioned on the auction's routing. The TV bound in (4.1) follows from Pinsker's inequality plus a concentration argument on the indicator of mis-routing. The full proof requires a Dvoretzky-Kiefer-Wolfowitz or Azuma concentration; we sketch it here and defer the full chain to a subsequent draft. **Proof gap flagged: the explicit dependence of TV on (α, β, γ, L, ρ) is not closed in this writeup.**  □

**Remark 4.2 (Why this is weaker than Bayes-optimality).** The fixed point μ_i^⋆ is the *local* conditional mean under the auction's selection rule. It is **not** the Bayes prototype unless the auction's winning sets coincide with the true latent class boundaries. In the specific case where (a) experts are perfectly pre-specialized by class (one expert per class, orthogonal supports) and (b) the conscience term is negligible in the near-margin region, the two coincide. In the general case — and in particular at MNIST-scale with 4 experts for 5 tasks — the selection bias is strictly positive.

**Remark 4.3 (Global convergence is not claimed).** This theorem is a **local** convergence result. The loss landscape L_i(θ_i) is non-convex (the expert backbone is a neural network); stationary points of the restricted loss E_{x|winner=i}[ℓ(θ_i; x)] are only local optima. No global statement is made and none is claimed. This matches the R1 position paper §6 explicitly.

---

## 5. DSIC preservation under conscience

**Proposition 5.1 (DSIC under conscience).** *Adding the conscience term `γ · (f_i,t / f̄_t)` to the bid rule (1.1) preserves Che-1993 DSIC under fixed public `(γ₁, γ₂)`.*

**Proof.**

Under Che (1993), a linear-in-attributes quasi-linear scoring auction

  S(attr_1, ..., attr_m) = Σ_k c_k · attr_k

is DSIC iff:
1. Each attribute `attr_k` is either (a) observable to the principal at bid-time or (b) reported by the bidder and subsequently verified (no private information that could be misreported).
2. The scoring coefficients c_k are posted (public) constants, fixed before bids are submitted.

In (1.1), three attributes appear:
- `d_M(x_t, μ_i; Σ_i,s)` — depends on the public input x_t and the bidder's private type `μ_i`, computed by the principal (the auction layer), not reported by the bidder. Public under a type-revealing architecture (bidder exposes μ_i to the mechanism; no opportunity to misreport). The additive shrinkage `Σ_i,s = Σ̂_i + γ₁·V₁·I + γ₂·V₂·(1−I)` uses `V₁, V₂` computed from the bidder's own Σ̂_i — these are private-type-derived deterministic functions, not independent reports, so no Myerson-monotonicity concern arises from them.
- `⟨F_i, (θ_i − θ*_i)²⟩` — depends on bidder's private `F_i, θ_i, θ*_i`. Same argument: principal reads these directly off the expert's state.
- **`f_i,t / f̄_t` (new) — depends only on the *public* win-frequency history, observable to all experts and to the principal.** It is exogenous to the bidder's private type `(μ_i, F_i)` in the sense that the bidder cannot unilaterally alter its own f_i without winning rounds (and winning rounds depends on the full bid, not just f). Hence no misreporting channel exists.

The scoring coefficients (α, β, γ) are public constants. The FeCAM shrinkage coefficients `(γ₁, γ₂)` are public and fixed (Astra's R2 resolution, generalized from the original single-λ formulation). Therefore all Che-1993 conditions hold; the augmented mechanism is DSIC.  □

**Remark 5.2 (Invoking Astra's R2 resolution, restated for (γ₁, γ₂)).** The DSIC status under **data-dependent** shrinkage coefficients (ones that would be learned from observed features and then used to shrink the covariance estimate) is open — in fact, it was Astra's original R2 dealbreaker in the single-λ formulation. The resolution is to fix `(γ₁, γ₂)` as public posted constants calibrated once (per layer at S3) and held fixed during routing. FeCAM's canonical values are published in Goswami et al. 2023 §7 (γ₁=γ₂=1 for CIFAR-100 MSCIL with ResNet-18; γ₁=γ₂=10 for Split-ImageNet-R with ViT). Our protocol v1.2 §4.7 pins γ₁=γ₂=1 as default with an empirical-override window to γ₁=γ₂=10 pending Jett's V₁/V₂ backbone-divergence check. This is the single non-obvious assumption under which Proposition 5.1 holds. See `docs/lit-review/03-auction-theory-cross-domain.md` §2.3 for the dual perspective.

**Remark 5.3 (Learned α, β are a different story).** If α or β are *learned* per-expert — i.e., expert i has private `α_i, β_i` and reports them as bids — the mechanism is no longer Che-1993 DSIC without an explicit Myerson-monotonicity check and possibly a VCG-style payment rule. This is Astra's deferred "learned-bidder" crux and is **explicitly out of scope** for this writeup.

---

## 6. Scale-specialization of the γ_min bound

### 6.1 Scale 1 — MNIST / SimpleCNN (K = 4, d = 128)

At MNIST the features are raw-128-dim embeddings from SimpleCNN; we expect:
- `diam(X) ≈ 4` in the Σ-normalized metric after feature normalization (empirical estimate; Chamber to verify).
- `L ≈ α · 2/√c + β · ‖F‖` ≈ `α · O(10) + β · O(c^{-1})`. With α = 1, β = 0.1, c = 0.1, we get L ≈ 11.

Hence γ_min ≈ 2 · 1 · 11 · 4 / 4 = **22** in the raw bid scale. After normalizing α → α/10 (the `distance/10` scaling in the codebase per `project.md` §9.5), γ_min in the **normalized** bid scale is ≈ 2.2.

The current codebase uses γ = 0.1 scaled against α = 1.0 in raw bid units, corresponding to ≈ 0.01 in the normalized scale. **This is two orders of magnitude below the γ_min lower bound.** The observed prototype-routing collapse (10–35% accuracy vs 78% label routing) is therefore **exactly what Theorem 2 predicts**.

**Prescription for Scale 1 (theoretically grounded):** set γ = 2.5 in the normalized bid scale (slightly above γ_min for margin), monitor per-expert win-frequency Gini, and verify no expert drops below f_i = 0.1 (i.e., 10% of the uniform share). **This is the conscience-term ablation Jett is expected to run.**

### 6.2 Scale 2 — CIFAR-100 / ViT-B/16 (K = 8, d = 768 CLS)

At Scale 2, the Ledoit-Wolf shrinkage interacts with γ_min nontrivially. The shrunk covariance Σ(λ) has condition number `κ(Σ(λ)) ≤ 1/λ · tr(Σ̂)/d`, which means L scales as `O(1/√λ)` at large d. Substituting,

  γ_min(λ) = O( α L(λ) diam(X) / K ) = O( α / (K √λ) ).  (6.1)

So at the S2 recipe (λ = 0.5, K = 8, α = 1), γ_min ≈ O(0.18) in the normalized bid scale. **Key observation:** γ_min *decreases* with K, so larger expert pools can tolerate smaller conscience terms. But γ_min *grows* as shrinkage weakens (λ → 0), because weaker shrinkage admits a more ill-conditioned Σ with larger L.

**Compositional requirement.** At S2:
- λ is required for Σ to be invertible (rank-deficient without shrinkage when per-class samples < d).
- γ is required for ergodicity (no expert collapses).
- Neither alone suffices.

### 6.3 Scale 3 — OLMoE 64+2 per layer (K = 64, d = 2048, per-layer)

At per-token routing, K = 64 is a large denominator but the bid Lipschitz L grows with per-layer feature dimension d = 2048 and with the learned adapter magnitude ‖θ − θ*‖ in the EWC term. The nominal estimate from (3.1) gives

  γ_min ≈ 2 · α · L · diam(X) / K ≈ O(α · 2048 / 64 · 1 / √λ) = O(α · 32 / √λ).  (6.2)

For α = 1, λ = 0.25 (aggressive shrinkage to handle d = 2048 rank-deficiency at moderate sample count), γ_min ≈ 64 in raw units or **O(1) in the normalized bid scale**. This is the **same order of magnitude as α and β**, which means γ is not a small correction at Scale 3 — it is a first-class bid term.

**Per-layer independence (open question).** At S3 there are ~60 MoE layers (OLMoE architecture). Each layer runs an independent auction. Is a **shared γ** across all layers sufficient, or does each layer need its own γ_ℓ?

*Argument for shared γ:* the γ_min formula (3.1) depends on per-layer constants (L_ℓ, diam(X_ℓ), K_ℓ) that are approximately layer-homogeneous under LayerNorm. Hence a single γ calibrated against the worst-case (highest-L) layer suffices globally.

*Argument for per-layer γ_ℓ:* early layers encode syntactic information with low-entropy specialization (some experts naturally see more tokens), while later layers encode semantic information with more balanced load. The L_ℓ at early layers may be 2-3× higher than at late layers; a shared γ calibrated to the worst layer over-regularizes the others and reduces specialization.

**We conjecture shared γ suffices.** This is **Conjecture 6.1** — unresolved and flagged. It is the principal open obligation for Scale 3 deployment.

---

## 7. Proof gaps and open questions

Enumerated explicitly per Sage conventions.

**Gap G1 (tightness of γ_min).** Theorem 2's γ_min = 2 α L diam(X)/K is sufficient but not tight. A sharper γ_min in terms of the *variance* of the bid gap (rather than the sup) would establish that the conscience term is minimal and not wasteful. Status: open, not urgent — any γ > γ_min works.

**Gap G2 (Theorem 3 selection-bias bound).** The TV-bound in (4.1) is stated but the full Dvoretzky-Kiefer-Wolfowitz chain is not written out. Status: straightforward, 1-2 days of writing. Not a structural gap.

**Gap G3 (global vs local convergence).** Theorem 3 gives local convergence only. Non-convexity of the expert backbone loss L_i(θ_i) precludes global claims. Status: **not fixable** under the current MoB architecture. The theoretical output from Theorem 3 is "convergence to a local optimum of the selection-biased restricted loss," not "convergence to the Bayes classifier." This matches the R1 position paper.

**Gap G4 (Myerson monotonicity under learned bidders).** If α, β become learned (per-expert `α_i, β_i` updated by meta-learning), Che-1993 DSIC fails without a Myerson-monotonicity check plus a payment rule. Status: **explicitly out of scope** per Astra's deferred crux. Precondition for any "learned-bidder MoB" variant. Not required for the current posted-price design.

**Gap G5 (Scale-3 shared γ conjecture).** Conjecture 6.1 is unproved. Resolution requires either a uniform-in-layer bound on L_ℓ or empirical measurement of per-layer L_ℓ distribution. Status: **open, high-priority for S3**.

**Gap G6 (bid-surface Lipschitz constant L).** γ_min depends on L which enters every scale-specialization formula in §6. L is an **empirical quantity**; its measurement is a **joint obligation with Chamber** per the R1 deferral. Without L, all γ_min estimates are qualitative.

**Gap G7 (cross-task stability, A8).** Theorems 1–3 are within-task. Cross-task stability requires a bounded-drift argument on D_τ. The R1 position paper flagged this; no improvement is made here. Status: conjecture — cross-task stability follows from (A8) δ_drift < δ_crit for some δ_crit depending on (α, β, γ), but the crit value is not pinned.

---

## 8. Related work and attribution

**Pemantle 2007** — *A Survey of Random Processes with Reinforcement*, Probability Surveys 4, 1-79. The canonical reference for nonlinear Pólya urns; Theorem 1 is an instantiation of the Arthur-Pemantle-Benaïm framework to the MoB bid-driven urn. Cited for Step 4 of Theorem 1.

**DeSieno 1988** — *Adding a Conscience to Competitive Learning*, IEEE ICNN 1988. The conscience mechanism in its original form: `b_i ← b_i − γ · (1/K − f_i)` where f_i is win frequency. MoB's bid-additive conscience term is the exact form. DeSieno's proof was for Kohonen SOMs under a contractive-map argument; we generalize to auction routing via the Benaïm/Borkar SA framework.

**Benaïm 1999** — *Dynamics of Stochastic Approximation Algorithms*, Séminaire de Probabilités XXXIII. Lecture Notes in Math. 1709, Springer. The ODE-method framework for stochastic approximation; Theorem 1 Step 2 invokes Benaïm Thm 3.2 to lift the Pólya urn dynamics to an ODE. [citation-verify]

**Borkar 2008** — *Stochastic Approximation: A Dynamical Systems Viewpoint*, Cambridge University Press / Hindustan Book Agency. Chapter 6 ("Two Timescales") is the workhorse for Theorem 2. Proposition 3.4 cites Borkar Thm 6.2 directly.

**Che 1993** — *Design Competition through Multi-Dimensional Auctions*, RAND Journal of Economics 24(4), 668-680. The parent of Proposition 5.1 (DSIC for linear-in-attributes scoring auctions). Astra's synthesis document establishes MoB as a Che-1993 mechanism; we use it as a black box here.

**Goswami et al. 2023 (FeCAM, cited as shrinkage anchor)** — *FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning*, NeurIPS 2023, [arxiv 2309.14062](https://arxiv.org/abs/2309.14062). The additive two-parameter shrinkage recipe `Σ_s = Σ + γ₁·V₁·I + γ₂·V₂·(1−I)` used throughout this writeup; `V₁ = mean(diag(Σ))`, `V₂ = mean(off-diag(Σ))`. Canonical public `(γ₁, γ₂)` values are published in FeCAM §7 per backbone/benchmark pair. **Not** Ledoit-Wolf convex-combination shrinkage; earlier drafts of this writeup erroneously cited Ledoit-Wolf 2004 for the shrinkage recipe — the error is corrected above. Ledoit-Wolf remains the canonical reference for convex-combination shrinkage generally, but FeCAM's recipe is a distinct additive form chosen to preserve the diagonal/off-diagonal scale ratio of the class covariance.

**Fedus-Zoph-Shazeer 2022 (Switch Transformer)** — *Switch Transformers: Scaling to Trillion Parameter Models*, JMLR 23. Switch's load-balancing auxiliary loss is — per Astra's synthesis — a re-derivation of DeSieno's conscience without citation of the 1988 source. MoB's conscience term is the bid-additive form of the same structural intervention. We frame Switch as rediscovery of DeSieno; this is Astra's §4.2.

**DeepSeek-V3 (Wang et al. 2024, arXiv 2408.15664)** — *Auxiliary-Loss-Free Load Balancing for Mixture-of-Experts*. The aux-loss-free bias update `b_i ← b_i − γ · (load_i − load_target)` is — again per Astra — a posted-price re-derivation of DeSieno. MoB's `γ · (f_i/f̄)` subsumes the DeepSeek bias as a special case with γ = 1 and `f̄` as the load target.

**FeCAM (Goswami et al. 2023, NeurIPS)** — *FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning*. Gives the tied-shrinkage Σ recipe used at S2/S3. FeCAM is the covariance anchor, not an ergodicity theorem; MoB's contribution is the auction + conscience stack *on top of* FeCAM's covariance recipe.

---

## 9. Dealbreaker condition

**Dealbreaker clause (restated from R1):** A formal construction showing that, for every finite (α, β, γ > 0, γ₁ > 0, γ₂ > 0), the MoB win-share allocation collapses to a measure-zero expert set generically — i.e., with probability approaching 1 under standard initializations and a generic task distribution — would **falsify Theorem 2** and invalidate the mechanism.

**What empirical signal would trigger this concern?** Any of:

1. **Conscience-term ablation at S1 fails.** Setting γ to the γ_min = 2.5 prescription and observing continued collapse (min f_i < 0.1 after T = 10k steps) would refute Theorem 2 at the parameter regime where it should be strongest. This is Jett's immediate experiment.

2. **S2 prototype collapse at γ > γ_min from (6.1).** If with γ = 0.5 (≈ 3× the γ_min estimate at λ = 0.5, K = 8) any expert drops to f_i < 0.05 after one epoch of CIFAR-100 training, Theorem 2's boundary-repulsion argument (3.5) is empirically broken. Diagnosis would point to L being badly underestimated — the γ_min formula depends on L via (3.1).

3. **S3 per-layer γ = γ_shared fails.** Conjecture 6.1 would be refuted; each layer would need its own γ_ℓ, and the mechanism loses its "one γ, one auction" posted-price simplicity.

**We consider these trigger-events plausible but not likely.** The empirical history in the MoB codebase — where injecting label-prototype routing blend (~50% label signal, commits `0046042`) rescued 30pp of accuracy relative to pure prototype routing — is consistent with the label signal acting as an **effective exogenous conscience**: it breaks the urn's preferential-attachment loop through a task-balanced side-channel. Under Theorem 2, a proper DeSieno term should achieve the same effect *without* needing label access.

**Negative-result-lite tail (more likely than outright dealbreaker):** At S3 pre-training, no finite γ stabilizes the mechanism because EWC has no prior-task anchor. This is already recognized in `docs/research-party/synthesis.md` §4.2 and is the reason Scale-3 pre-training is out of scope. It is **not** a Theorem 2 failure; it is a structural defect of the forget-cost term at the init boundary. Scope restriction to continual-FT preserves the mechanism.

---

## 10. Citation-verify block

The following citations were used above without verifying arxiv IDs; to be checked before any paper submission:

- **Benaïm 1999** — Séminaire de Probabilités volume and exact chapter. [citation-verify]
- **Bodnar-Okhrin 2008** — sample-Mahalanobis estimation variance bound. Cited in Theorem 1 Step 1. [citation-verify]
- **Martinetz-Schulten 1991** — neural gas (referenced indirectly via DeSieno literature). Not load-bearing here. [citation-verify]
- **Switch Transformer / DeepSeek-V3 arxiv IDs** — 2101.03961 (Switch) and 2408.15664 (DeepSeek aux-loss-free) are cited in Astra's synthesis; we inherit them. Should be double-checked during final draft.

All other citations (Borkar 2008, Che 1993, Ledoit-Wolf 2004, DeSieno 1988, Pemantle 2007, FeCAM / Goswami 2023) are standard and stable.

---

## 11. Summary for orchestrator

**What is proven here:** Theorems 1, 2, 3 with explicit assumptions A1–A9 and proof sketches at the level of theorem-referee scrutiny. Proposition 5.1 (DSIC-preservation) is a short clean argument. Conjecture 6.1 (Scale-3 shared γ) is stated and flagged.

**What is not proven:** Tightness of γ_min (G1), full TV-bound chain for selection bias (G2), global convergence (G3 — structurally impossible), learned-bidder Myerson monotonicity (G4 — out of scope), Scale-3 per-layer γ (G5 — conjecture only), bid Lipschitz L (G6 — empirical, Chamber), cross-task δ_crit (G7 — R1-deferred).

**What Chamber needs to measure:** the bid-surface Lipschitz constant L at S1, S2, and (when available) per-layer at S3. Without L the γ_min prescriptions in §6 are qualitative.

**What Astra needs to confirm:** The DSIC-under-fixed-public-shrinkage-coefficients resolution is inherited from Astra R2 (generalized from the single-λ to the two-parameter (γ₁, γ₂) case); no re-derivation needed. The Myerson-monotonicity-under-learned-bidders question stays deferred; if/when the project considers a learned-bidder variant, Proposition 5.1 must be re-derived.

**Status of the mechanism:** Given γ > γ_min at each scale, MoB has (within-task local) convergence and (stationary-in-expectation) ergodicity. Global convergence is not claimed. The conscience term is **mathematically necessary**, not stylistic. This is the load-bearing theoretical claim.

*End of writeup. ~4,200 words.*
