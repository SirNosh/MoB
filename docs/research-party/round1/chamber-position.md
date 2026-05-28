# Chamber — Round 1 Position Paper
**Subject:** The MoB routing layer at three scales — concrete module definitions
**Author:** Chamber (architecture)
**Status:** Opinionated first pass. Defers to Astra/Sage/Fade/Killjoy on flagged cruxes.

---

## 1. Executive summary — eight architectural commitments

1. **The invariant module is `AuctionRouter`**, a parameter-free callable with signature `(features: [B, D]) -> (winner_idx: [B], bid_matrix: [B, E])`. Everything downstream specialises only the feature extractor and the expert body. This module survives from 4-CNN MNIST to 128-FFN LLM.
2. **Prototype extraction is always a single vector per sample in a dedicated "routing space"**, never multi-scale, never block-wise. Multi-scale routing is an unforced complexity tax; FeCAM shows a single penultimate vector suffices.
3. **At Scale 1 (MNIST/CIFAR-10) we keep full independent CNN experts** but add a 128-dim **linear projection head `W_route: [128 -> 64]` with L2 normalisation** before prototype accumulation. This is the architectural fix for training-time prototype collapse — decouples the routing space from the classification space.
4. **At Scale 2 (CIFAR-100) we switch to frozen ViT-B/16 + per-expert LoRA(rank=8) in attention QKV + per-expert 2-layer MLP adapter in FFN.** EWC applies only to the adapter parameters. Prototype space is the **frozen [CLS] token**, shared across experts, normalised with FeCAM's Tukey+shrinkage recipe.
5. **At Scale 3 (LLM FFN) the auction fires per-token per-MoE-layer**, replacing the gate MLP in a Mixtral/DeepSeek block. Prototypes live per-expert per-layer in the hidden space (e.g. 4096-dim). We do **not** store full covariance — we use **per-expert diagonal variance + one tied low-rank factor `U ∈ [D, r=32]` shared across experts per layer** (FeCAM's correlation matrix is layer-universal in their ablations).
6. **Fine-grained (DeepSeek-style, 64 routed + 2 shared) wins over Mixtral-style coarse at LLM scale.** Auction routing benefits from granularity because the bid signal has more separation when experts are specialised. This is not a free parameter — it is required for MoB to avoid winner-take-all collapse.
7. **Losing experts update their prototypes via cheap EMA on the shared features of the winning-routed batch, gated by a bid-proximity mask** (prototype refresh only on samples where the loser's bid was within 2x of the winner's). This is how we keep 127 non-winning experts' statistics from staling without giving them gradient updates.
8. **Backward pass is always straight-through-on-winner only**: gradients flow through the selected expert body, zero through the auction. The auction is non-differentiable by design. Top-k is implemented as k independent winners with their outputs summed (not averaged) — gradients accumulate naturally.

---

## 2. Three scales, concrete

### Scale 1 — Split-MNIST / CIFAR-10 (current, 4 experts)

**Verdict:** Keep full independent experts. *Do not* move to shared backbone yet. Reason: the current regime studies whether the auction can maintain specialisation under the overload constraint (4 experts, 5 tasks). A shared backbone contaminates the experiment.

**Changes I commit to:**

- Add `W_route: Linear(128, 64)` + L2 norm per expert. Train *only* via a contrastive pull-to-own-centroid loss during expert training (not via task CE). This is the fix for prototype collapse: the routing space is no longer yoked to logits, so an expert that overfits to two tasks does not drag its centroids into a degenerate overlap.
- **Prototype extraction point:** penultimate of CNN body, then through `W_route`, then L2-norm. Shape chain: `[B, 3, 32, 32] -> CNN -> [B, 128] -> W_route -> [B, 64] -> L2norm -> [B, 64]`.
- **Per-sample routing for eval, per-batch routing for train.** Reason: per-sample at train time breaks the EWC Fisher accumulation (Fisher assumes a single task-consistent batch per expert). Per-sample at eval time is free and gives better accuracy on mixed test streams.
- **Backward pass:** winner-only. Losers do not update CNN weights. Losers *do* update `W_route` via a stop-gradient on the CNN features but a small (weight 0.01) contrastive push-away term. This is cheap and keeps the routing space globally consistent.

**Forward pass pseudocode (Scale 1):**

```
# Inputs:   x: [B=128, 3, 32, 32]
# Experts:  E=4, each has CNN body + W_route: [128->64] + PrototypeStore
# Proto store per expert: mu_c: [C_seen, 64], Sigma_c: [C_seen, 64, 64] (FeCAM-shrunk)

for expert_i in experts:
    feat_i, _     = expert_i.cnn.forward_features(x)      # [B, 128]
    z_i           = L2norm(expert_i.W_route(feat_i))      # [B, 64]
    # Mahalanobis to nearest own-centroid
    d_i           = min_c ( (z_i - mu_c)^T Sigma_c^{-1} (z_i - mu_c) )  # [B]
    bid_i[b]      = alpha * d_i[b] + beta * fisher_cost_i # [B]

bids          = stack([bid_1, ..., bid_E], dim=1)         # [B, E]
winner_idx    = argmin(bids, dim=1)                       # [B] (eval) or
                mode(argmin(bids, dim=1))                 # [] (train, per-batch)

# Train: one expert gets the whole batch
y_hat = experts[winner_idx].forward(x)                    # [B, 10]
loss  = CE(y_hat, y) + lambda_ewc * fisher_penalty(experts[winner_idx])
       + 0.01 * contrastive_route_loss(z_winner, mu_winner_own, mu_losers)

# Eval: per-sample dispatch, stitch outputs
y_hat = zeros(B, 10)
for i in range(E):
    mask_i = (winner_idx == i)
    if mask_i.any():
        y_hat[mask_i] = experts[i].forward(x[mask_i])
```

---

### Scale 2 — CIFAR-100 (4-8 experts, 5-20 tasks)

**Verdict: Frozen ViT-B/16 + per-expert (LoRA on attention + bottleneck adapter in FFN).** Reject full ResNet-18 from scratch. Reasoning matrix:

| Option | Params/expert | VRAM @ 8 experts | EWC cost | Prototype quality | Pick |
|---|---|---|---|---|---|
| Full ResNet-18/expert | 11M | ~2.4 GB | Fisher over 11M | Strong specialisation, slow | no |
| Shared ViT-B/16 + LoRA(r=8) | ~0.3M | ~0.9 GB | Fisher over 0.3M | Uses ViT features, rich | **yes** |
| Shared ViT-B/16 + full adapter | ~2M | ~1.4 GB | Fisher over 2M | Richer but 6x heavier | no |
| Shared CNN + heads only | ~0.05M | ~0.3 GB | trivial | Too weak, heads collapse | no |

**Commitments:**

- **EWC applies only to adapter/LoRA parameters.** Backbone is frozen so there is nothing to protect there. Fisher is computed over the ~300k LoRA+adapter params per expert — cheap, fits in fp32 without approximation.
- **Prototype space = frozen ViT [CLS] output (768-dim), passed through a tiny per-expert `W_route: [768 -> 128]` that IS trainable and IS EWC-protected.** This is the right answer to the "shared features lose specialisation signal" worry: the routing projection specialises per expert even though the backbone does not.
- **Module layout: single auction layer at the top, not per-block.** Per-block auction multiplies routing complexity by 12 and gains nothing — ViT [CLS] already aggregates.
- **FeCAM recipe applied verbatim** on the 128-dim routed features: diagonal shrinkage (`gamma=0.1`), correlation normalisation, Tukey transform (`power=0.5`) before Mahalanobis.

**Forward pass pseudocode (Scale 2):**

```
# Inputs:  x: [B=64, 3, 224, 224]
# Frozen:  ViT-B/16 trunk -> cls: [B, 768]
# Per expert i: LoRA(QKV, r=8), FFN_adapter: [768 -> 64 -> 768], W_route: [768 -> 128]
# PrototypeStore_i: mu_c: [C_seen, 128], Sigma_shrunk_c: FeCAM diag+corr

with torch.no_grad():
    cls_shared = frozen_vit.encode_cls(x)                 # [B, 768]

# Auction uses per-expert route projection of shared CLS
for i in experts:
    z_i        = L2norm(experts[i].W_route(cls_shared))   # [B, 128]
    z_i_tukey  = sign(z_i) * |z_i|^0.5                    # FeCAM Tukey
    d_i        = fecam_mahal(z_i_tukey, mu_i_c, Sigma_i_c)# [B]
    bid_i      = alpha * d_i + beta * ewc_cost_adapter_i  # [B]

winner_idx = argmin(stack(bids), dim=1)                   # [B] eval / batch-mode train

# Winning expert runs the *adapted* forward through frozen ViT
for i in winning_experts:
    cls_i   = vit_with_adapter(x, adapter=experts[i])     # [B_i, 768]
    logits_i= experts[i].head(cls_i)                      # [B_i, 100]
```

Critical: the backbone forward runs **once** (frozen, shared), the adapter path runs only on the winner's sample subset. This is where the compute savings live — dense backbone, sparse adapter+head.

---

### Scale 3 — LLM MoE FFN layer (8-128 experts, per-token)

**Verdict: DeepSeek-style fine-grained, 64 routed + 2 shared, auction per token per MoE layer. Diagonal+shared-low-rank covariance.**

This is the scale where MoB must earn its keep. The concrete module:

```
class AuctionFFN(nn.Module):
    def __init__(self, d_model=4096, d_ff=1792,  # DeepSeek ratio ~0.44
                 n_routed=64, n_shared=2, top_k=2,
                 rank_shared_cov=32):
        self.shared = nn.ModuleList([FFN(d_model, d_ff) for _ in range(n_shared)])
        self.routed = nn.ModuleList([FFN(d_model, d_ff) for _ in range(n_routed)])
        # Prototype params (non-learned, updated by EMA)
        self.register_buffer('mu',      zeros(n_routed, d_model))    # per-expert centroid
        self.register_buffer('diag',    ones (n_routed, d_model))    # per-expert diag var
        self.register_buffer('U',       randn(d_model, rank_shared_cov) * 0.01) # tied low-rank
        self.register_buffer('fisher',  ones (n_routed))              # aggregate EWC cost
        self.top_k = top_k
```

**Parameter budget check (per layer, d_model=4096):**

- Expert FFN bodies: 64 × (2 × 4096 × 1792) ≈ 940M (this is the model, inevitable).
- `mu`: 64 × 4096 = 262k. Negligible.
- `diag`: 64 × 4096 = 262k. Negligible.
- `U` (tied low-rank): 4096 × 32 = 131k **per layer**. Across 32 layers: 4.2M total. Cheap.
- `fisher` scalars: 64. Ignore.

Rejected alternative: full 4096×4096 covariance per expert = 16M × 64 = **1B params per layer just for covariance**. Infeasible. Chamber commits to **diagonal + layer-tied low-rank shared `U`**.

**Integration:** Replace the gate linear `W_g: [D -> n_routed]` in a Mixtral block with the auction. The rest of the block (norm, attention, residual) is unchanged.

**Forward pass pseudocode (Scale 3):**

```
# Inputs:  h: [B, T, D] hidden states, D=4096
# Per token we must produce: routed expert selection + shared experts always on.

h_flat = h.reshape(B*T, D)                                # [N, D], N = B*T

# Shared experts always fire on all tokens
y_shared = sum(shared_i(h_flat) for shared_i in self.shared) / n_shared  # [N, D]

# Bid: Mahalanobis with diag + tied-low-rank correction
#   Sigma_i^{-1} ≈ diag(1/diag_i) - diag(1/diag_i) U (I + U^T diag(1/diag_i) U)^{-1} U^T diag(1/diag_i)
# Woodbury identity; r=32 so the (r x r) inverse is trivial.
delta   = h_flat.unsqueeze(1) - self.mu.unsqueeze(0)      # [N, E=64, D]
diag_term = (delta**2 / self.diag.unsqueeze(0)).sum(-1)   # [N, E]
proj    = einsum('ned,dr->ner', delta/self.diag, self.U)  # [N, E, r]
# (I + U^T D^-1 U) is [E, r, r]; per-expert small matrix, batch-invert
M_inv   = batched_woodbury_core(self.U, self.diag)        # [E, r, r]
lowrank_term = einsum('ner,erq,neq->ne', proj, M_inv, proj) # [N, E]
mahal   = diag_term - lowrank_term                        # [N, E], Woodbury correction

bid     = alpha * mahal + beta * self.fisher.unsqueeze(0) # [N, E]

# Top-k per token
topk_vals, topk_idx = bid.topk(self.top_k, dim=-1, largest=False)  # [N, k]

# Dispatch: gather token indices per expert
y_routed = zeros(N, D)
for e in range(n_routed):
    mask = (topk_idx == e).any(dim=-1)                    # [N]
    if mask.any():
        y_routed[mask] += self.routed[e](h_flat[mask])    # additive combine

y = y_shared + y_routed                                   # [N, D]
y = y.reshape(B, T, D)

# --- Prototype / Fisher update (no gradient) ---
with torch.no_grad():
    for e in range(n_routed):
        won_mask  = (topk_idx == e).any(dim=-1)
        if won_mask.any():
            ema_update(self.mu[e],   h_flat[won_mask].mean(0), tau=0.99)
            ema_update(self.diag[e], h_flat[won_mask].var(0),  tau=0.99)
        # Losers-near-winner: refresh on samples where bid was within 2x of winner's
        near_mask = (bid[:, e] < 2 * topk_vals[:, 0]) & (~won_mask)
        if near_mask.any():
            ema_update(self.mu[e],   h_flat[near_mask].mean(0), tau=0.999)  # slower
```

**Backward pass:** gradients flow through `self.routed[e](h_flat[mask])` only for winning-per-token selections. Losing experts see no gradient — their bodies drift only via the shared `U` and the EMA proto updates described above. Fisher is updated offline per task boundary (same as today) summed over gradient norms of winning tokens.

**Initialization of prototypes:** run one forward pass of pretraining data through a frozen reference model, cluster hidden states per layer into `n_routed` k-means clusters → those are the initial `mu`. This avoids the "all prototypes at zero" collapse.

---

## 3. Cross-scale architectural invariant

The single module that survives from 4 CNNs to 128 FFNs:

```python
class AuctionRouter(nn.Module):
    """
    Parameter-free routing. All learned parameters live in the experts'
    projection heads / adapters, NOT here. This module is pure arithmetic.

    Buffers (updated by EMA, not gradient):
        mu:     [E, D_route]                 per-expert centroid
        diag:   [E, D_route]                 per-expert diagonal variance
        U:      [D_route, r]  (optional)     tied low-rank covariance factor
        fisher: [E]                          per-expert EWC aggregate cost

    Type signature:
        forward(z: Tensor[N, D_route],
                alpha: float,
                beta: float,
                top_k: int = 1
               ) -> Tuple[LongTensor[N, top_k],    # winner indices
                          FloatTensor[N, E]]      # full bid matrix

    Semantics:
        bid[n, e] = alpha * mahal(z[n], mu[e], Sigma_e) + beta * fisher[e]
        winner[n] = argmin_e bid[n, e]  (top-k for k>1)
    """
```

**What does NOT live in the router:** feature extraction, expert bodies, heads, any learned weights. The router is stateless-across-steps except for the EMA buffers — and even those updates are explicit and external, not part of the backward graph.

**What differs across scales:** only the definition of `z` (CNN penultimate → projected → L2-normed at S1; frozen ViT CLS → per-expert W_route at S2; transformer hidden state per token at S3) and whether `Sigma` is full, diagonal-only, or diagonal+tied-low-rank.

This invariance is the load-bearing claim: if it holds, the MoB paper's contribution is a *module*, not three unrelated systems.

---

## 4. Where I defer to others — three cruxes

**Crux 1 (for Astra / Sage): Is EWC Fisher even the right forgetting-cost at LLM scale?**
At 128 experts × billions of params, a per-expert Fisher is enormous and offline-expensive. Astra's auction-theory framing suggests the "cost to serve" should be whatever is locally cheap and monotone in forgetting risk. Candidates: (a) running loss on a held-out rehearsal buffer, (b) gradient norm of the last K steps on the winning tokens, (c) a shadow-weight drift penalty (`||theta_t - theta_{t-1000}||`). I committed to EWC in the pseudocode but I have low confidence this survives Scale 3. **Please adjudicate.**

**Crux 2 (for Fade / Sage): Fine-grained (64+2) vs coarse (8) at Scale 3 — does MoB require fine-grained to avoid single-expert bid monopoly?**
My intuition is yes: with only 8 experts, one expert will minimise its Fisher fastest and win every token until the prototype EMA catches up. Fine-grained spreads bid-space entropy. But this is a claim about auction dynamics I cannot prove analytically — needs Astra's monopoly analysis and Fade's empirical MoE-scaling priors.

**Crux 3 (for Killjoy): Am I correct that per-block auction (one per transformer layer) is wasteful vs single auction at top?**
I rejected per-block routing at Scale 2 on complexity grounds, but per-block *is* standard in Mixtral/DeepSeek at Scale 3 (every MoE layer routes independently). There is an inconsistency in my paper: I argue single-auction at S2 and per-layer-auction at S3. Killjoy, please surface whether this is principled (CLS is a fixed aggregate at S2, hidden states evolve at S3) or whether I am rationalising.

---

## 5. Dealbreaker

**One finding that forces retraction of this entire paper:**

> *Empirical demonstration that EMA-updated prototypes for losing-but-near-winner experts do NOT track the true per-expert feature distribution at Scale 3 — specifically, that losing experts' `mu` and `diag` stale within 10k training steps to the point that they cannot re-win tokens they should win.*

If Cypher or Killjoy shows this via a small-scale simulation (e.g. 8-expert synthetic routing over shifting mixtures), the entire cross-scale invariant breaks. Every non-winning expert's prototype becomes unreliable, bids become noise, and the auction degenerates to a rich-get-richer monopoly. The only fix would be full-batch proto updates (expensive) or adding a learned routing gate as a fallback — at which point MoB is no longer parameter-free and the thesis collapses into a Mixtral variant.

Secondary (non-fatal but painful): if Astra proves the reverse procurement auction has a degenerate Nash equilibrium where `beta` must scale with `n_experts` faster than O(sqrt(E)), then the bid formula needs a normalisation term I have not written, and the Fisher clamp values from Scale 1 will not transfer.

---

*Chamber, end of Round 1.*
