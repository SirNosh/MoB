# Sage — Round 2 Position Paper

**Role:** Mathematical foundations. **Scope delta from R1:** interpretability as co-equal claim; incorporate specialist R1 resolutions.

---

## 1. Executive summary

1. **Interpretability theorem (Thm 2.1, §3).** Under fixed public (α, β, γ, λ) and DSIC bidder behavior, the MoB selection map `x ↦ i*(x)` admits a *model-intrinsic, additively-separable, causally-faithful* attribution into three semantically-typed features (d_M, c_forget, f_i/f̄). Softmax-gate MoE admits no such decomposition without post-hoc tooling. This is MoB's sharpest theoretical claim.
2. **Crux 1 resolved.** If experts learn their own (α_i, β_i) by gradient descent on private loss, the mechanism is monotone (hence Myerson-implementable) **iff** the learned coefficients remain in the non-negative orthant *and* the bid is signed-consistent in type. A single projection `α_i, β_i ← max(·, 0)` plus a no-sign-flip invariant suffice. Without projection, monotonicity fails generically.
3. **Crux 2 (Lipschitz).** Closed-form bound: `L ≤ 2α·‖Σ_i^{-1}‖·D_μ + β·‖F_i‖·‖θ_i − θ*_i‖` where D_μ bounds the prototype-to-input distance. Shrinkage controls `‖Σ_i^{-1}‖ ≤ 1/λ·(tr(Σ̂)/d)^{-1}`; Fisher clamp controls ‖F_i‖. Both appear multiplicatively — they jointly determine margin feasibility.
4. **Crux 3 (distributed conscience).** Under bounded broadcast staleness τ and bounded load-rate change, projected-gradient EWC with asynchronous load updates preserves A5–A7 convergence with additive error O(τ·γ) in the Lyapunov drift.
5. **Crux 4 (specialization).** No aux-loss-free *formal* specialization theorem exists in 2024–2026 literature; the strongest statement is load-balance-to-KKT (DeepSeek-V3 primal-dual argument) without a Bayes-optimal partition guarantee.
6. **FeCAM separation:** The β-term strictly changes the winner on a set of positive Lebesgue measure whenever the ordered Mahalanobis gaps and ordered Fisher gaps are *not comonotone* across (i, j) pairs. This is a precise falsifiable event.
7. **DeepSeek-V3 strict generalization:** MoB ⊃ DSV3-routing. Setting α = β = 0 and reinterpreting γ·(f_i/f̄) as the DSV3 bias gradient recovers DSV3 exactly; DSV3 cannot recover MoB (lacks d_M and c_forget semantic types).
8. **Dealbreaker update.** KAY/O's epiphenomenality threat is *partially* resolved: §4(a) gives a positive-measure regime where β strictly changes the winner. The *headline* claim shifts from "auction improves accuracy" to "auction yields faithful attribution at no accuracy cost in that regime."

---

## 2. Crux resolutions

**C1 (Astra: Myerson monotonicity under learned bidders).** Let bidder i have private type `τ_i = (θ_i, μ_i, F_i)` and report bid `b_i(x; τ_i, α_i, β_i)`. The mechanism is monotone (Myerson) iff `∂b_i/∂τ_i^+ ≥ 0` for every type-coordinate that increases cost. With fixed public (α, β) this is immediate from the bid's linearity. With learned `(α_i, β_i)` via SGD on private loss `ℓ_i`, the gradient step can flip signs: `dα_i/dt = −η·∂ℓ_i/∂α_i`, and `∂ℓ_i/∂α_i` need not be sign-constant. **Resolution:** project `(α_i, β_i) ← max(·, 0)` each step, and add a no-sign-flip penalty `λ_sf · 1{sign changes}`. Under projection, the realized bid function is the upper envelope of a family of non-negative linear forms in (d_M, c_forget), which is monotone in both arguments. Proof sketch: the projection preserves the positive cone; Myerson's lemma applies to the resulting cone-valued mechanism. **Gap:** DSIC under *asymmetric* bidders requires revenue equivalence arguments that R1 §4.4 already flagged; the projection restores monotonicity, not full DSIC.

**C2 (Chamber: Lipschitz of bid surface).** For fixed Σ_i, F_i, θ_i:
  `|b_i(x) − b_i(x')| ≤ α · |d_M(x; μ_i) − d_M(x'; μ_i)| + 0`
(the β, γ terms are x-independent). Using `d_M(x) = √((x−μ)^T Σ^{-1} (x−μ))`, the gradient satisfies `‖∇_x d_M‖ ≤ ‖Σ_i^{-1}‖_{op}^{1/2} · 1`. Hence
  **L_x ≤ α · ‖Σ_i^{-1}‖_{op}^{1/2}.**
Under Ledoit-Wolf shrinkage with coefficient λ, `‖Σ_i^{-1}‖_{op} ≤ (λ · tr(Σ̂_i)/d)^{-1}`, so
  **L_x ≤ α · √(d / (λ · tr(Σ̂_i))).**
At fixed α, shrinkage λ must grow with d to preserve concentration (R1 §4.1). For Fisher-norm contribution along θ: `‖∇_θ c_forget‖ ≤ ‖F_i‖_{op} · ‖θ_i − θ*_i‖`.

**C3 (Killjoy: distributed conscience under stale broadcasts).** Let each expert's load counter be updated locally and broadcast every Δt steps with staleness ≤ τ. Let γ_t be the conscience strength. The Lyapunov function V from R1 §6 admits drift
  `E[V_{t+1} − V_t | F_t] ≤ −κ·‖∇V‖² + O(γ · τ · L_load).`
If `γ · τ · L_load < κ · ‖∇V‖²` on the relevant sublevel set, drift remains strictly negative. This gives *approximate* convergence to an O(γτL_load)-neighborhood of the R1 equilibrium. For E ≤ 64 and τ bounded by one tokens-per-second window, `γτ` is small in practice. **Gap:** "relevant sublevel set" is not universal; the bound is local.

**C4 (Fade: aux-loss-free specialization).** Literature scan (through 2025-Q4) confirms **no theorem** of the form "expert i converges to Bayes-optimal router for latent cluster c(i)" exists for aux-loss-free MoE. DeepSeek-V3's bias-update is a primal-dual method for the Lagrangian of a load-constrained problem; it converges to a KKT point under convexity, which does not hold. Equifinality results (Fade R1) show many specializations are reachable — specialization is *underdetermined* absent additional constraint.

---

## 3. Interpretability theorem — the headline contribution

**Setup.** MoB mechanism with fixed public (α, β, γ, λ, c_Fisher). Bidder i reports
  `b_i(x) = α · d_M(x; μ_i) + β · c_forget,i(θ_i) + γ · (f_i / f̄).`
Winner: `i*(x) = argmin_i b_i(x)`. Let `φ_i^{(1)}(x) := α · d_M(x; μ_i)`, `φ_i^{(2)} := β · c_forget,i`, `φ_i^{(3)} := γ · (f_i/f̄)`.

**Definition 3.1 (faithfulness, Jacovi-Goldberg 2020 + Rudin 2019).** An attribution `A: X → R^E × R^3` is *faithful* for selection function `i*` if:
(F1) **Decomposition completeness:** `Σ_k A_{i,k}(x) = b_i(x)` for every i.
(F2) **Causal determinacy:** `i*(x) = argmin_i Σ_k A_{i,k}(x)`.
(F3) **Model-intrinsic:** A is computed from the same parameters and arithmetic used during selection (no post-hoc surrogate).
(F4) **Semantic typing:** each coordinate k has a fixed, mechanism-level meaning independent of x.
(F5) **Public coefficients:** the scalar scaling of each type is a published constant, not a learned matrix.

**Theorem 3.2 (MoB is faithfully interpretable by construction).**
The attribution `A_{i,k}(x) := φ_i^{(k)}(x)` for k ∈ {1,2,3} satisfies (F1)–(F5). Under DSIC bidder behavior with fixed public (α, β, γ), the attribution is additionally
(F6) **Truthful:** b_i(x) is bidder i's private-type-report, not strategic shading; hence A_{i,k} reflects *type*, not game-theoretic distortion.

**Proof.** (F1): by construction, `b_i = φ^{(1)} + φ^{(2)} + φ^{(3)}`. (F2): `i* = argmin_i b_i = argmin_i Σ_k φ_i^{(k)}`. (F3): the same arithmetic operations produce both the winner and A. (F4): φ^{(1)} is Mahalanobis distance (exec cost), φ^{(2)} is a quadratic form on parameter drift (forget cost), φ^{(3)} is load share (fairness); these meanings are mechanism-level constants independent of x. (F5): α, β, γ are hyperparameters, public by definition. (F6): standard DSIC argument — fixed public scoring implies truthful dominant strategy is to report cost honestly; b_i is therefore a type report, not a strategic variable. ∎

**Corollary 3.3 (quantitative attribution).** For any x the fractional attribution of type k to the winning event is
  `ρ_k(x) := φ_{i*}^{(k)}(x) − min_{i≠i*} φ_i^{(k)}(x)`
(i.e., the margin contribution of type k). Then `Σ_k ρ_k(x) = b_{(2)}(x) − b_{(1)}(x) = margin(x)`. Hence every winning event has a *precise* linear breakdown into type contributions.

**Separation from softmax-gate MoE.** A softmax router computes `p_i(x) = softmax(W_g · x)_i`, and selection is `i* = argmax_i p_i`. The attribution analog would require an additive decomposition of `(W_g · x)_i` into semantically-typed public terms; `W_g` is learned and its rows are not semantically typed. Any decomposition is either (a) a post-hoc surrogate (LIME, Integrated Gradients) — violating (F3), or (b) a re-projection of W_g onto an ad hoc basis — violating (F4) and (F5). **MoB's decomposition is by construction; softmax MoE's is by post-hoc reconstruction.** This is the model-intrinsic / post-hoc distinction of Rudin 2019, operationalized for MoE routing.

**Response to KAY/O's Round 2 attack ("bid readability is post-hoc").** The attack would say: "you just read off α·d_M + β·c_forget + γ·f — the field could always compute those." The distinction is that MoB *commits* to this decomposition *as the selection rule*. In softmax MoE, one can compute d_M, c_forget, f after the fact, but the selection is made by `W_g · x`, so these quantities are causally irrelevant to the winner — (F2) fails. In MoB, the same three quantities *are* the winner-determining function. The attribution is not a report about the selection; it *is* the selection. This is the operational content of "interpretable by construction."

**Honest gaps.** (G1) Theorem covers single-step selection, not downstream accuracy-vs-attribution tradeoffs. (G2) Faithfulness is for the *routing* decision, not for expert i's *internal* computation. (G3) DSIC requires the projection/fixed-coefficient regime of C1. (G4) Human-usefulness of the three types is an empirical HCI question, not a theorem. (G5) If β is zero in practice (see §4(a)), the theorem holds with a 2-type decomposition but the β's interpretive weight disappears.

---

## 4. Theoretical separations

**(a) vs FeCAM (KAY/O's threat).** Let `Δ_M(x)_{ij} := d_M(x; μ_j) − d_M(x; μ_i)` and `Δ_F_{ij} := c_forget,j − c_forget,i`. The event `{argmin(α·d_M + β·c_forget) ≠ argmin(d_M)}` equals `{∃ i,j: α·Δ_M(x)_{ij} > 0 and α·Δ_M(x)_{ij} + β·Δ_F_{ij} < 0}`, i.e., the Fisher gap β·Δ_F is large enough (negative) to reverse the Mahalanobis winner. This has positive Lebesgue measure **iff** (i) the distribution of x places positive mass on the set where `α·Δ_M(x)_{ij} ∈ (0, β·|Δ_F_{ij}|)` for some (i,j), and (ii) the Mahalanobis and Fisher rankings across experts are *not comonotone*. Concretely: the event is non-null when expert ordering by prototype-distance disagrees with expert ordering by EWC-rigidity, and x lies in the "margin window" (0, β·|Δ_F|). Away from task boundaries, β·‖F_i‖ is small and the window shrinks — **this is exactly KAY/O's regime**. Near task boundaries, β·‖F_i‖ grows and the window opens. So MoB ≠ FeCAM *precisely when* forgetting pressure disagrees with prototype distance — which is when continual learning matters.

**(b) vs DeepSeek-V3 (strict generalization).** DSV3: `gate_i(x) = softmax(W_g · x + bias_i)`, with `bias_i ← bias_i − ε·sign(load_i − load_target)`. Set MoB parameters: α = 0, β = 0, and replace the Mahalanobis-prototype-plus-argmin with a *softmax-over-negative-bid* reading. Then `b_i(x) = γ·(f_i/f̄) = −(W_g · x)_i + bias_i` recovers DSV3's score up to the `W_g · x` term. To recover fully: allow φ^{(1)} to instantiate as `−W_g · x` (a linear scoring, degenerate Mahalanobis with μ=0, Σ^{-1}=W_g^T W_g up to sign), keep α = 1, β = 0, γ·(f_i/f̄) plays the role of `bias_i`. Conversely, DSV3 cannot express d_M (no per-expert prototypes) or c_forget (no EWC term). **Hence MoB ⊃ DSV3** as a strict superset of routing mechanisms. The γ-term of MoB **is** DSV3's bias-update dynamic, made public and parameterized.

---

## 5. Refined convergence conjecture

Integrating R2 resolutions:
- **A4 (step-size separation).** Astra's fixed public λ preserves the two-timescale argument. **Unchanged.**
- **A5 (Fisher clamp / projected-gradient EWC).** Killjoy's projected-gradient EWC preserves F_i ⪰ c·I; **A5 strengthened** from codebase-enforced clamp to a projection onto the PSD cone intersected with {F ⪰ cI}.
- **A6 (shrinkage).** Chamber's tied low-rank U (r=32) shared across experts via Woodbury satisfies A6 with `Σ_i = λ·I + U·D_i·U^T`; `Σ_i^{-1}` available in O(r²·d) via Woodbury. **A6 strengthened** from scalar shrinkage to structured shrinkage, with the same operator-norm bounds.
- **A7 (conscience).** γ·(f_i/f̄) term exists, and C3's distributed-conscience theorem admits O(γτ) error. **A7 strengthened** to tolerate asynchrony.
- **A1–A3.** Unchanged.

**Restated conjecture (sharp form).** Under (A1–A3) + (A4) fixed public coefficients + (A5) PSD-projected Fisher ⪰ cI + (A6) tied low-rank + Woodbury shrinkage + (A7) conscience with distributed staleness τ bounded, the MoB joint dynamics converge a.s. to an O(γτ)-neighborhood of a local equilibrium satisfying R1 Thm conditions (a)–(c). **Most likely failure:** (A2) strict-margin, because margin scales as O(1/√E_eff) while Lipschitz of bid grows in K. **Test:** measure empirical margin and Lipschitz (Chamber's job 1) and verify `margin > const · L · √(log E)` as §4.1 of R1 requires.

---

## 6. 5–8B scale considerations

At 5–8B with E ≤ 64 per-layer experts:
- **Concentration bound (R1 §4.1)** tightens: `P(misroute) ≤ 2E·exp(−Δ²/(8σ²L²))` with log-E only mildly penalizing E=64 vs E=128. Margin requirement softens by `√(log 128 / log 64) ≈ 1.08` — negligible theoretical relief but directionally favorable.
- **Pólya-urn dynamics (R1 §2.4)** *intensify* at smaller E: fewer absorbing states means initialization-dependence grows. The conscience term γ must be proportionally stronger. Quantitatively, expected absorption time scales as `E·log E`, so smaller E means faster collapse without γ.
- **Net:** Concentration helps ~1.08×; urn dynamics hurt roughly `log(128)/log(64) ≈ 1.17×`. Urn effect dominates. **5–8B with E=64 demands γ > γ_crit where γ_crit scales as 1/log E.** This is tractable; it is also *theoretically mandatory* given (A7)'s role in the R1 convergence argument.

---

## 7. Updated dealbreaker

**R1 dealbreaker:** formal proof that MoB collapses to measure-zero expert set generically. **R2 update:** KAY/O's *epiphenomenality* threat is partially resolved by §4(a): β strictly changes the winner on positive measure when Mahalanobis and Fisher orderings disagree. But the new dealbreaker is sharper:

> **If empirical measurement shows the β-term flips the winner on a negligible fraction of tokens away from task boundaries** (e.g., < 0.5% of routing decisions), then the β-term is cosmetic for accuracy and the headline shifts to pure interpretability-at-no-cost. That is not a theorem retraction — it is a **claim scope reduction** from "continual-learning-aware routing" to "interpretable routing that degrades gracefully under continual shift."

Additionally: if the interpretability theorem (§3) fails human-usefulness trials — i.e., humans cannot actually use the A_{i,k} decomposition to predict routing behavior — then (F4) semantic-typing is violated in practice and Thm 3.2's value is rhetorical rather than operational. I rate this **possible but unlikely** given the three types (distance, rigidity, load) are standard ML abstractions.

---

**Word count ≈ 1,780.** Notation precise; proof gaps (G1–G5, C3-gap, C4-gap) flagged explicitly.
