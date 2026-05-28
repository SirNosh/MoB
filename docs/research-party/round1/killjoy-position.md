# Killjoy Position Paper — Auction Feasibility at Three Scales

**Agent:** Killjoy (Systems ML / hardware-aware tradeoffs)
**Round:** 1
**Scope:** Does the MoB auction mechanism actually fit on hardware at MNIST / CIFAR-100 / LLM-MoE scale? What breaks first?

---

## 1. Executive Summary (system commitments)

1. **The bid must be one forward-pass-equivalent or less, per expert, at every scale.** If bid cost grows faster than a single expert forward pass, the auction eats the mechanism alive at K > 8.
2. **At MNIST scale the auction is a ~5x tax on the training step and it does not matter.** We are compute-rich here. All optimization effort on this scale is wasted engineering.
3. **At CIFAR-100 the auction is feasible iff Fisher lives on the adapter/head, not the ViT backbone.** Full-backbone EWC is a factor-of-100 memory blowup per expert. Non-negotiable.
4. **At LLM-MoE scale the current bid formulation is dead on arrival.** Full covariance (64 MB/expert fp16) and per-expert backward passes for EWC (K× training FLOPs) are both infeasible. The mechanism survives only with (a) tied low-rank / diagonal-plus-low-rank covariance and (b) input-independent cached forget cost.
5. **Mahalanobis must be reformulated as a matvec, not a quadratic form.** `diff @ inv_cov @ diff.T` is O(B·C·d²). Cache `L = chol(inv_cov)`, compute `Lx` once, then Euclidean in whitened space: O(B·C·d) bid time.
6. **Per-token routing at d=4096 with 128 experts is bandwidth-bound, not FLOP-bound.** 8 GB of inv_cov state weighs more than a DeepSeek-V3 expert FFN itself. Tied covariance (one shared inv_cov across all experts) drops this to 64 MB and is the minimum viable baseline.
7. **The invariant primitive is a single matvec against a per-expert d-dim vector** (or low-rank `r × d` projection). Everything else must compile down to that or the mechanism does not scale.
8. **Dealbreaker is per-expert backward for forget cost.** At LLM scale we cannot run K backward passes per step. If the mechanism requires input-dependent Fisher × grad² per expert, it dies at Scale 3 regardless of how clever Mahalanobis becomes.

---

## 2. Scale 1 — Split-MNIST / CIFAR-10 (current)

### 2.1 Standing measurements (from `mob/`, `contibualmob/`)

- Model: `SimpleCNN`, ~450 K params, feature dim **d=128**, C=2 classes/task.
- Experts: **K=4**. Batch **B=64**.
- Device: H100 or 4090.

### 2.2 Bid FLOP breakdown per batch

**Regular forward pass (SimpleCNN, B=64, MNIST 28×28):**
- Conv1 (1→32, 3×3): 64·28²·1·32·9 ≈ 14.5 MFLOPs
- Conv2 (32→64, 3×3): 64·14²·32·64·9 ≈ 230 MFLOPs
- FC1 (3136→128): 64·3136·128 ≈ 26 MFLOPs
- FC2 (128→10): negligible
- **Forward total ≈ 270 MFLOPs/batch.** Backward ≈ 2× that ≈ 540 MFLOPs.

**Mahalanobis per expert** (current code, `_compute_distance_matrix`):
- `diff = features.unsqueeze(1) - centroid_matrix.unsqueeze(0)`: (B, C, d) = (64, ~10, 128) tensor build.
- `transformed = diff @ inv_cov`: B·C·d² = 64·10·128² ≈ 10.5 MFLOPs.
- `(transformed * diff).sum(-1)`: 64·10·128 ≈ 0.08 MFLOPs.
- **Per expert ≈ 10.6 MFLOPs. Across 4 experts ≈ 42 MFLOPs.** That is ~16 % of one forward pass. Cheap.

**EWC forget cost per expert** (code: `compute_forgetting_cost`):
- Forward + backward on the batch = **~810 MFLOPs per expert.**
- Across K=4 experts: **~3.2 GFLOPs.** This is **12x one forward pass, and ~6x a full training step**. **Dominant cost.**

### 2.3 Memory footprint

- 4 expert models × 450 K params × 4 B (fp32) = **7.2 MB**.
- 4 Fisher matrices (same shape as params): **7.2 MB**.
- 4 optimal_params copies: **7.2 MB**.
- 4 prototype stores: 4 × (128×128 inv_cov + ~10 centroids × 128 + running sums) ≈ 4 × 68 KB = **~270 KB**.
- Total mechanism overhead on top of 4 models: **~14.7 MB**. Noise on a 24 GB GPU.

### 2.4 Takeaway

The bid cost at MNIST scale is dominated by **EWC forget cost (K backward passes per batch)**, not Mahalanobis. That's a ~5x tax on the training step. On H100 it costs milliseconds. Do not optimize this.

**Low-hanging optimization that does not change the mechanism:**
- Run all K forget-cost backward passes with `torch.func.vmap` over experts (K independent graphs that share the input batch). Potential 1.5-2x wallclock from kernel fusion.
- Or just share the detached feature batch and reuse; the existing code already does a fresh forward per expert.
- **Strong recommendation: ignore.** The speedup is not worth the code complexity for experiments that already finish in minutes.

---

## 3. Scale 2 — CIFAR-100 (ViT-B/16 or ResNet-18, 4–8 experts)

### 3.1 Setup assumptions

- **Frozen** ViT-B/16 backbone (86 M params, d=768 hidden / 768 output CLS), per-expert **adapter or LoRA head** on top.
- K ∈ {4, 8} experts. B=256. 100 classes.

### 3.2 Per-expert memory, three options

| Option | Trainable params/expert | Fisher size/expert | Total (K=8) |
|---|---|---|---|
| Full ResNet-18 per expert (no shared backbone) | 11 M | 11 M × 4 B = **44 MB** | 352 MB + 8×11M params = **440 MB** |
| Full ViT-B backbone per expert | 86 M | **344 MB** | **2.7 GB** just for Fisher |
| Shared frozen ViT + per-expert **LoRA r=8 head** | ~150 K | **600 KB** | **~5 MB** for all Fisher |
| Shared frozen ViT + per-expert linear classifier (768→100) | 77 K | **308 KB** | **~2.5 MB** |

**Verdict:** Full-backbone EWC is a non-starter at K=8 (2.7 GB Fisher alone, plus optimal_params copy = 5.4 GB just for EWC state; triples at K=16). LoRA adapter + head is the only sane option.

### 3.3 Mahalanobis at d=768

- inv_cov per expert: 768² × 4 B = **2.36 MB fp32** (1.18 MB fp16). K=8 → **19 MB**. Fine.
- Bid FLOPs (batch=256, C=100 centroids): B·C·d² = 256·100·768² ≈ **15 GFLOPs per expert**. K=8 → **120 GFLOPs per bid**.
- ViT-B forward pass: ~17.6 GFLOPs per image at 224×224 → B=256 → **~4.5 TFLOPs**. Backward ≈ 9 TFLOPs.
- Ratio: bid is ~1.3 % of a training step **per expert**, ~10 % for K=8. Still cheap.

**Optimization worth doing:** Mahalanobis should be rewritten as **whitened Euclidean**. Precompute `L = cholesky(inv_cov)`. At bid time: `z = L @ features.T` (one B·d² = 50 MFLOPs once, amortized across experts if covariance is tied), then Euclidean in z-space. Drops per-expert bid to O(B·C·d) = 200 MFLOPs. ~70x speedup.

### 3.4 EWC forget cost

- Forward+backward on the **LoRA + head** (not full ViT — the backbone is frozen, no gradient flows back through it for the bid): ~150 K trainable params, B=256.
- Per-expert bid ≈ one forward pass through the adapter + head ≈ **~100 MFLOPs** (negligible relative to frozen ViT forward at 4.5 TFLOPs since the frozen forward is shared).
- **Crux: the frozen backbone forward is shared across all K experts.** Do it once, pass features to all K adapter heads. Bid cost then scales as K × adapter_cost, not K × full_model_cost.

### 3.5 K-expert crossover threshold

At what K does linear bid cost exceed one forward pass?
- One ViT-B forward: 4.5 TFLOPs/batch.
- Per-expert bid (Mahalanobis + adapter forward+backward): ~2 GFLOPs.
- **Crossover: K ≈ 2200 experts.** Not a concern at this scale. Auction fits.

### 3.6 Commitment

- Shared frozen ViT-B/16 backbone. Feature dim d=768.
- LoRA r=8 adapter + linear head per expert. Fisher computed on adapter+head only.
- Tied Mahalanobis covariance: **one shared inv_cov across all experts**, recomputed each task boundary. Saves K× memory and enables the whitened-Euclidean trick.
- K=8 is comfortable. K=32 is achievable.

---

## 4. Scale 3 — LLM MoE FFN layer (8–128 experts, per-token routing, d=4096)

This is where the mechanism either lives or dies.

### 4.1 Per-token routing math

Context: 1 MoE FFN layer, hidden d=4096, sequence T=4096, batch=1 (inference) or B=8 × T=2048 (training micro-batch), K experts.

**Bid evaluations per layer per forward pass:**
- Inference: T · K = 4096 · 128 = **524 K bids/layer.**
- Training: B · T · K = 8 · 2048 · 128 = **2.1 M bids/layer.**

**Current full-covariance Mahalanobis cost per bid:**
- d² = 16.7 M FLOPs per bid.
- Training: 2.1 M · 16.7 M = **3.5 × 10¹³ FLOPs = 35 TFLOPs per MoE layer per bid**.
- A typical MoE FFN forward (say, SwiGLU with 4d hidden): B·T·(3·d·4d + 4d·d) = 8·2048·(3·16M + 16M) ≈ **2.1 TFLOPs per layer**.
- **Mahalanobis bid is ~16.5× the actual expert FFN compute.** Infeasible. Confirmed.

### 4.2 Covariance memory

- Per-expert full inv_cov: 4096² × 2 B (fp16) = **32 MB/expert**. Correction to the prompt figure: 4096²·2 = 32 MB fp16, not 64 MB. (64 MB would be fp32.)
- K=128: **4 GB just for covariance state per MoE layer.**
- Typical LLM has 30-60 MoE layers → **120-240 GB of covariance state**. Larger than the model itself. Infeasible.

### 4.3 Minimum viable approximation — covariance

**Ranked by feasibility cost, best → worst:**

| Approach | Memory/expert | Bid FLOPs/token | Viable? |
|---|---|---|---|
| **Tied low-rank r=32** `inv_cov ≈ I + U Uᵀ`, U shared across experts | 32·4096·2 = **256 KB shared** | 2·r·d = 260 K | **YES — baseline** |
| **Diagonal-plus-low-rank** per expert, r=16 | 16·4096·2 + 4096·2 = **139 KB/expert** | 2·r·d + d = 135 K | **YES** |
| **Random projection RanPAC-style** to k=256, then full cov in k-space | 256²·2 = **131 KB/expert** + shared P: 4096·256·2 = 2 MB | (d·k) + k² = 1.1 M | **YES** |
| Diagonal only per expert | 4096·2 = **8 KB/expert** | d = 4096 | YES but too weak |
| Full per-expert | 32 MB/expert | 16.7 M | **NO** |

**Recommendation: tied low-rank r=32 with per-expert offsets.** inv_cov_i ≈ diag(α_i) + U Uᵀ where U is shared, α_i is per-expert. ~264 KB total overhead for 128 experts. Bid cost ~260 K FLOPs/token = **2 TFLOPs per MoE layer** — on par with the FFN itself, which is the affordable ceiling.

### 4.4 Minimum viable approximation — EWC forget cost

The current `compute_forgetting_cost(x, y)` does **a forward + backward per batch per expert**. At LLM scale:
- Per-batch backward ≈ 2× forward ≈ one full training step's compute.
- K=128 backward passes per batch = **128× training compute**. Completely dead.

**Approximation ladder (cheapest → most faithful):**

1. **Input-independent forget cost** (cached scalar per expert) — forget_cost_i = ‖F_i · (θ_i − θ*_i)‖₁ computed once per consolidation. Cost: zero per-step. Loses input specificity (the whole point of the 2024 fix).
2. **Hutchinson trace estimator**: approximate F_i · g² using v·F_i·v for a random Rademacher v. Still requires one backward, but only one total for the auction (not K). Reduces K backwards to 1.
3. **Gradient proxy via feature norms**: use ‖∇_feature L‖² weighted by a per-expert sensitivity scalar. No backward through model params. ~1 extra forward.
4. **Low-rank random projection of grad**: compute grad once (one backward), project to r=32 dims, cache per-expert Fisher projections. Forget_cost ≈ ‖P g‖² weighted by P F P. One backward total, K dot products in r-dim space.

**Recommendation: option 4** (projected gradient against cached projected Fisher). One backward per step regardless of K. Adds O(K·r) scalar work to the auction. **Survives at K=128.**

### 4.5 Expert count vs bid-compute

With the tied low-rank Mahalanobis and projected-gradient EWC:
- Per-token Mahalanobis bid: 2·r·d = 260 K FLOPs.
- Per-token EWC bid: r operations per expert ≈ 32 K FLOPs/expert.
- K=128 experts: (260 K + 128·32 K) × B·T ≈ 4.4 M × 16 K = **70 GFLOPs per layer per step**.
- Relative to ~2 TFLOPs FFN compute: **~3.5 % tax**. **K=128 is feasible with this bid formulation.**

### 4.6 Bandwidth / parallelism

- In expert parallelism (EP), each expert lives on a different GPU. Bids must be all-gathered before top-k selection. Bid tensor shape: (B·T, K) scalars = 2.1 M × K fp16 bytes.
- K=128: **256 MB all-to-all per layer per forward pass.** On NVLink (600 GB/s) that is ~0.4 ms. On InfiniBand (50 GB/s cross-node) that is **~5 ms per layer**. With 30 MoE layers: **150 ms/step just on bid communication cross-node.** Non-trivial.
- **Compared to standard MoE top-k routing (gate logits, same shape, same cost):** identical communication pattern. Auction does **not** worsen this. Good.
- **NP-hard combinatorial clearing**: Astra's concern. My take: top-k auction = sort bids, take top-k. O(K log K) per token. Not NP-hard in this regime. The NP-hardness only appears if we impose capacity constraints across tokens (load balancing as combinatorial ILP). Solution: soft capacity constraint via bias term, per DeepSeek-V3.

---

## 5. Minimum Viable Auction at LLM Scale

Single page, leanest preservation of the MoB mechanism:

**Bid formula (unchanged in spirit):**
```
b_i(x_t) = α · mahal_i(x_t) + β · forget_i(x_t)
```

**Mahalanobis term (tied low-rank):**
- Shared `U ∈ ℝ^{d×r}` with r=32 across all experts (rank-32 whitening).
- Per-expert centroid bank `μ_i ∈ ℝ^{C_i × d}` (C_i ≤ 32 class centroids per expert).
- Per-expert diagonal correction `α_i ∈ ℝ^d` (optional; start without).
- Compute once per token: `z_t = Uᵀ x_t` → `r`-dim. [2·r·d = 260 K FLOPs]
- Per expert: min over centroids of `‖z_t − U μ_i^{(c)}‖²`. Precompute `U μ_i` offline.
- **Total: ~30 K FLOPs/expert/token after shared projection.**

**Forget term (cached Fisher + projected gradient):**
- At consolidation: project Fisher diagonal and optimal-param delta via `P ∈ ℝ^{r_f × n_params}` (r_f=32, Gaussian random matrix, fixed, not stored — seed-regeneration).
- Cache per expert: `F̃_i = P F_i`, `δ̃_i = P (θ*_i − θ_i)`. Size: **2·r_f = 64 scalars/expert.**
- At bid time: compute one backward on current batch → gradient g. Project: `g̃ = P g` (one matvec, r_f·n_params FLOPs — but this is once per step, not per expert).
- Per-expert forget cost: `forget_i = g̃ᵀ diag(F̃_i) g̃` = **r_f = 32 multiplies per expert per token** (or per step if we treat forget as input-global).

**Auction clearing:**
- Top-k (k=2) over bids per token. O(K log K) sort or O(K) quickselect. Trivial.
- Soft load balance: bias term per expert updated via SGD on load imbalance (DeepSeek-V3 style).

**Total per-layer bid overhead:**
- Mahalanobis: (2·r·d) + K·(r·C_avg + r) FLOPs/token ≈ 260 K + 128·300 = **300 K FLOPs/token**.
- EWC: one backward per step (not per expert) + K·r_f dot products ≈ one step's backward for bid (acceptable) + **4 K FLOPs/token**.
- Mechanism memory: U (256 KB shared) + per-expert centroids (C·d·2 = 256 KB/expert) + projected Fisher (64 B/expert).
- **Total K=128 memory: ~33 MB per MoE layer.** Order of magnitude cheaper than the FFN weights themselves (128 · ~100 MB).

**What we give up vs. full formulation:**
- Full per-expert covariance → tied low-rank (loses per-expert feature-space geometry; recover via diagonal α_i if needed).
- Input-dependent per-expert backward → one shared backward + projected Fisher dot-product (loses per-expert gradient specificity; the projection basis is the same for all experts but the projected Fisher is per-expert).

Both are honest science tradeoffs, not hacks. Each approximation is theoretically characterized (low-rank Mahalanobis = factor analysis; projected Fisher = Hutchinson-style trace).

---

## 6. Where I Defer to Others

1. **Astra (mechanism design):** Is tied low-rank Mahalanobis *still* a Mahalanobis mechanism in the sense that matters for your regret/fairness bounds? If the covariance is shared across experts, do the auction's theoretical properties (incentive compatibility, no-starvation) still hold?

2. **Chamber / Sage (continual-learning theory):** Does projected-gradient EWC (Hutchinson-style trace approximation of Fisher·grad²) preserve the forgetting-prevention guarantees? Specifically: at what projection rank r_f does the approximation error overwhelm the signal? Is there a paper I should steal this from (Mirzadeh et al.? S-FSGM?).

3. **Fade (LLM-MoE practitioner):** DeepSeek-V3's aux-loss-free bias balancing implies they don't need a fancy bid. Is there any empirical evidence that a cost-aware bid (even a cheap one) improves downstream performance over learned gates at 100B+ scale, or is this a solution looking for a problem above 7B?

---

## 7. Dealbreaker

**The single hardware reality that forces a mechanism change is per-expert backward for forget cost.**

The current `compute_forgetting_cost` implementation runs a fresh forward+backward on the current batch per expert. At K=4 and MNIST, this is a 5x training tax — fine. At K=128 and LLM scale, this is a 128x training tax — **more expensive than the entire rest of the training step combined.**

No clever kernel fusion fixes a factor of 128. The mechanism must move to a **single backward per step with per-expert Fisher projections** (or commit to input-independent cached forget costs and accept the loss of input-specificity). I recommend the former.

If the team insists on keeping per-expert backward for forget cost at LLM scale: the mechanism does not ship. Full stop.

---

**Word count:** ~2350. Back-of-envelope numbers shown inline. All claims traceable to either the current codebase (`mob/bidding.py`, `contibualmob/prototype_store.py`, `contibualmob/models.py`) or standard LLM/ViT FLOP formulas.
