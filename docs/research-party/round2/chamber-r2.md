# Chamber — Round 2 Position
**Subject:** Pick the 5-8B base, re-spec Scale 3, add interpretability as first-class output
**Author:** Chamber (architecture)
**Status:** Commits to OLMoE-1B/7B as Scale 3 substrate. Accepts Killjoy's projected-gradient EWC. Accepts Sage's conscience-term necessity. Rejects my own per-layer-vs-single-auction inconsistency — resolves in favor of per-layer.

---

## 1. Executive summary

1. **Base model is OLMoE-1B/7B (arxiv 2409.02060).** 6.9B total / 1.3B active, 16 layers, d_model=2048, d_ff=1024, 64 routed experts + 0 shared, top-8. Already MoE. Router swap is a one-line surgical edit. Rejects Mistral-7B / Llama-3-8B upcycling: the upcycling recipe adds a confound we do not need to fight, and OLMoE's native 64-way fine granularity is exactly what the auction mechanism needs.
2. **Scale 3 expert count: 64 routed + 2 shared per layer (keep OLMoE's 64, add 2 shared via Drop-Upcycling of existing experts).** Top-k=8 matches OLMoE defaults. Fine granularity is not optional for MoB — confirmed by Sage §2.4 Pólya-urn result and my own R1 Crux 2.
3. **Accept Killjoy's projected-gradient EWC.** One shared backward per step + per-expert 32-dim Fisher dot product. Drops EWC cost from 64x training tax to +2% overhead. My R1 Crux 1 resolves: stay with EWC as forgetting proxy (Sage validated local-Laplace surrogate), swap **implementation** from per-expert backward to projected gradient. Forget_cost semantics unchanged.
4. **Accept tied low-rank r=32 shared across experts per layer + per-expert diagonal.** My R1 spec was right on this; Killjoy's memory math (32 MB/expert full cov × 64 experts × 16 layers = 32 GB DEAD at d=2048) reconfirms. Total mechanism buffer: **44 MB across all 16 MoE layers**.
5. **Per-layer auction, not single-auction-at-top.** Crux 3 resolves: the S2 single-auction argument was CLS-specific (CLS is already an aggregate). Scale 3 hidden states are per-token and transform layer-to-layer — routing must too. I retract the S2/S3 inconsistency framing; they are consistent once you condition on feature type (aggregate vs per-token).
6. **Interpretability is now a first-class output surface.** Add `BidTrace` instrumentation (zero parameters, pure logging) that emits `[token_id, layer, top_k_experts, bid_decomposition]` per forward pass. This is the human-inspectable artifact: reviewers can project prototype centroids to nearest training tokens and see that expert-17 at layer-9 owns "mathematical identifiers," etc.
7. **`AuctionRouter` module type signature survives from R1, with one addition: a `trace_sink` callable for the `BidTrace` hook.** Parameter-free, buffer-only, drop-in for OLMoE's `OlmoeSparseMoeBlock.gate`.

---

## 2. Crux resolutions

### Crux 1 — EWC at LLM scale: accept Sage, adopt Killjoy's implementation
Sage R1 §2.3 establishes that low forget_cost implies low KL to old-task posterior *to second order, locally*, under Laplace + full-rank Fisher. This is the weakest-acceptable claim; it is not a global upper bound. I accept it for the bid.

What I reject: the per-expert backward implementation. Killjoy R1 §4.4 showed 64x training tax at Scale 3 — dead on arrival. The implementation switches to **projected-gradient EWC**:

- Project per-expert Fisher diagonal and parameter-delta via fixed Gaussian `P ∈ R^{r_f × n_params_per_expert}`, `r_f=32`, seed-regenerated (no storage).
- Cache per expert: `F_i_tilde = P · diag(F_i)` (32 floats), `delta_i_tilde = P · (theta_i_star - theta_i)` (32 floats). **256 bytes per expert.**
- At bid time: one shared backward through the currently active token batch → raw gradient `g`. Project `g_tilde = P · g` (one matvec, once per step regardless of E).
- forget_cost_i = `g_tilde^T · diag(F_i_tilde) · g_tilde` = 32 multiplies per expert per step.

This preserves input-dependent forget-cost semantics (the bid reacts to *current* gradient geometry) while paying one backward, not E. Sage's local-Laplace validity argument transfers because projection is linear — KL-to-second-order becomes KL-to-second-order-restricted-to-projection-subspace, still a valid local surrogate.

**Commitment:** forget_cost formula in the paper is unchanged. The implementation swap is a footnote.

### Crux 2 — Fine-grained vs coarse at Scale 3: fine-grained, resolved by base model
OLMoE ships with 64 experts. The 5-8B cap does not force coarse — the cap forces OLMoE. Mixtral-8x7B is 47B total and above cap. Forced-choice Mistral/Llama upcycling with 8 coarse experts would have been the 5-8B-consistent coarse option, but we are not taking that path. **64 routed + 2 shared stands.**

The +2 shared experts are not in OLMoE natively. Adding them via Drop-Upcycling: copy weights from two of the top-load experts (measured over a calibration corpus), tag them as `shared=True`, exempt them from the auction, always fire. This is a well-defined Drop-Upcycling variant and preserves OLMoE's native FFN dimensions.

### Crux 3 — Single vs per-block auction: per-block at Scale 3
I retract the S2-vs-S3 inconsistency claim. The correct invariant is:

> *Route at the granularity of the feature aggregation.*

At S2 the ViT CLS is already the aggregate — one auction decision per sample. At S3 the hidden state is per-token and evolves layer-to-layer — one auction decision per token per MoE layer. These are not inconsistent; they are the same rule evaluated against different feature structures.

Per-layer at S3 costs 16x the memory (one `mu`/`diag`/`U`/`fisher_proj` bank per MoE layer) but is unavoidable: OLMoE's MoE layers route independently and the hidden state representation drifts between them. Killjoy's 32 MB/expert full-cov number is per-layer; my tied-low-rank math (§4 below) is also per-layer and fits.

---

## 3. Base model choice — OLMoE-1B/7B

### Comparison matrix

| Option | Total / active | Layers × d_model | Experts native | Upcycling needed | MoB integration cost |
|---|---|---|---|---|---|
| **OLMoE-1B/7B** | 6.9B / 1.3B | 16 × 2048 | 64, top-8, no shared | None — swap router | 1 file, `OlmoeSparseMoeBlock` |
| Mistral-7B dense | 7.3B / 7.3B | 32 × 4096 | 0 | Drop-Upcycling to 8 × 7B→8E | 2-4 weeks, FLOP-heavy recipe |
| Llama-3-8B dense | 8.0B / 8.0B | 32 × 4096 | 0 | Drop-Upcycling (bigger) | Same, bigger |
| Phi-3.5-mini 3.8B | 3.8B / 3.8B | 32 × 3072 | 0 | Upcycling — under cap | Same risk profile as Mistral |
| Mixtral-8x7B | 47B / 13B | 32 × 4096 | 8, top-2 | None | **Over 8B cap** — excluded |

### Architectural rationale for OLMoE

1. **Already MoE.** Upcycling dense checkpoints into MoE introduces a training recipe confound (does MoB help, or is Drop-Upcycling doing the work?). OLMoE lets us isolate the router contribution.
2. **Fine granularity native.** 64 experts matches the granularity MoB needs for auction separability (R1 §2.6, Sage §2.4). Upcycled 8-way Mixtral-style alternatives would have forced the Crux-2 regression.
3. **Fully open weights, data, training code.** AI2 ships the router code in `olmoe_modeling.py` — the swap surface is trivially identifiable. Mistral and Llama-3 do not expose pretraining stacks.
4. **d_model=2048, not 4096.** Half the memory per expert, half the covariance state, half the bid FLOPs. This is the hidden win from OLMoE — Killjoy's S3 budget was calculated at d=4096 and comes in under budget at d=2048 by 4x in every memory term.
5. **Continued-FT regime is legible.** Continue-pretrain on a Dolma slice for 10-20B tokens; compare to stock OLMoE on the same slice. Publishable delta without a from-scratch run.

**Rejected:** Mistral / Llama-3 dense. Upcycling would be a second novel contribution we would then have to defend separately. Phi-3.5-mini is small enough to fit budget but shares the upcycling confound.

**Commitment: OLMoE-1B/7B as-is. No upcycling step. Router is the only swap.**

---

## 4. Scale 3 module spec (OLMoE-specialised)

### 4.1 Constants

```
n_layers         = 16     # OLMoE
d_model          = 2048   # OLMoE hidden
d_ff             = 1024   # OLMoE expert FFN hidden
n_routed         = 64     # keep native
n_shared         = 2      # added via Drop-Upcycling
top_k            = 8      # OLMoE default
rank_cov         = 32     # tied low-rank covariance
rank_fisher      = 32     # projected-gradient EWC
```

### 4.2 Per-layer memory budget

| Buffer | Shape | dtype | Bytes |
|---|---|---|---|
| `mu` (per-expert centroid) | [64, 2048] | fp16 | 262 KB |
| `diag` (per-expert diag var) | [64, 2048] | fp16 | 262 KB |
| `U` (tied low-rank factor) | [2048, 32] | fp16 | 131 KB |
| `M_inv_core` cached Woodbury core | [64, 32, 32] | fp16 | 131 KB |
| `F_tilde` (projected Fisher) | [64, 32] | fp32 | 8 KB |
| `delta_tilde` (projected delta) | [64, 32] | fp32 | 8 KB |
| `bias` (DeepSeek-V3 load balance, Sage/Fade) | [64] | fp32 | 256 B |
| **Per-layer total** | — | — | **~803 KB** |
| **All 16 layers** | — | — | **~12.8 MB** |

Killjoy's 44 MB across-all-layers estimate included per-expert centroid banks; ours is tighter at 2048-dim. Either way, this is noise next to 7B model weights (~14 GB fp16).

### 4.3 LoRA vs full FFN expert at this scale

Reject LoRA for Scale 3. OLMoE's experts are native full FFNs — we inherit them. Adding LoRA on top would be an unforced extra adaptation layer between the hidden state and the expert output, and would break the "replace only the gate" commitment that makes this a clean experiment. **Full FFN experts. No LoRA. No adapter.**

LoRA stays at Scale 2 (ViT adapters) because there experts did not pre-exist. At Scale 3 they do.

### 4.4 Forward pass pseudocode (single MoE layer, OLMoE-integrated)

```python
# h: [B, T, D=2048] hidden states from attention block
# Replaces OlmoeSparseMoeBlock.forward

def forward(self, h, trace_sink=None):
    B, T, D = h.shape
    h_flat = h.reshape(B*T, D)                             # [N, D], N=B*T
    N = B * T

    # --- 1. Shared experts (always on) ---
    y_shared = sum(self.shared[i](h_flat) for i in range(self.n_shared)) \
               / self.n_shared                             # [N, D]

    # --- 2. Bid: Mahalanobis (diagonal + tied low-rank via Woodbury) ---
    # delta[n, e, d] = h_flat[n, d] - mu[e, d]
    delta = h_flat.unsqueeze(1) - self.mu.unsqueeze(0)     # [N, 64, D]
    inv_diag = 1.0 / self.diag                             # [64, D]
    diag_term = (delta ** 2 * inv_diag.unsqueeze(0)).sum(-1)  # [N, 64]
    # Low-rank correction: proj = (delta / diag) @ U      [N, 64, r]
    proj = torch.einsum('ned,dr->ner',
                        delta * inv_diag.unsqueeze(0),
                        self.U)
    # Woodbury core M_inv = (I_r + U^T diag(1/diag_e) U)^{-1} is cached per expert
    lowrank_term = torch.einsum('ner,erq,neq->ne',
                                proj, self.M_inv_core, proj)
    mahal = diag_term - lowrank_term                       # [N, 64]

    # --- 3. Forget cost: one shared backward (offline to forward) ---
    # g_tilde cached from last training step; at inference = 0.
    # forget_i = g_tilde^T diag(F_tilde_i) g_tilde
    forget = torch.einsum('r,er,r->e',
                          self.g_tilde, self.F_tilde, self.g_tilde)  # [64]
    # DeepSeek-V3 bias term (load balance, Fade-imported)
    bias = self.bias                                        # [64]

    # --- 4. Bid matrix ---
    bid = (self.alpha * mahal
           + self.beta  * forget.unsqueeze(0)
           + bias.unsqueeze(0))                             # [N, 64]

    # --- 5. Top-k selection ---
    topk_vals, topk_idx = bid.topk(self.top_k,
                                    dim=-1, largest=False)  # [N, 8]

    # --- 6. BidTrace hook (interpretability, zero params) ---
    if trace_sink is not None:
        trace_sink.emit(
            layer=self.layer_idx,
            token_positions=torch.arange(N),
            topk_experts=topk_idx,
            topk_bids=topk_vals,
            bid_components=dict(mahal=mahal.gather(-1, topk_idx),
                                forget=forget[topk_idx],
                                bias=bias[topk_idx]))

    # --- 7. Dispatch to routed experts ---
    y_routed = torch.zeros_like(h_flat)
    weights = softmax(-topk_vals, dim=-1)                   # convert bid→weight
    for e in range(self.n_routed):
        tok_mask = (topk_idx == e)                          # [N, 8]
        selected = tok_mask.any(-1)
        if selected.any():
            w = (weights * tok_mask.float()).sum(-1)[selected].unsqueeze(-1)
            y_routed[selected] += w * self.routed[e](h_flat[selected])

    # --- 8. Combine + reshape ---
    y = y_shared + y_routed
    return y.reshape(B, T, D)
```

Backward: gradients flow through `self.routed[e](h_flat[selected])` for winning per-token selections only, and through `self.shared[i](h_flat)` always. Zero gradient through `mahal`, `forget`, `bias`, `bid`, `topk_idx`, or `trace_sink`. The auction is non-differentiable by construction.

Prototype / diag / Fisher updates: EMA under `torch.no_grad()` after each step, as in R1 §2 Scale 3, applied per layer. DeepSeek-V3 bias updates on load imbalance, scalar per expert per step.

---

## 5. Interpretability instrumentation

### 5.1 Design principle
**Instrumentation, not a new module.** The interpretability surface is a logging hook, not a trainable component. This preserves the R1 "parameter-free router" invariant and adds zero backward compute.

### 5.2 `BidTrace` contract

```python
class BidTrace:
    """Append-only ring buffer of routing decisions. Zero params, no grad."""
    def emit(self, layer: int, token_positions: LongTensor,
             topk_experts: LongTensor, topk_bids: FloatTensor,
             bid_components: Dict[str, FloatTensor]) -> None: ...

    def dump(self, path: str) -> None:
        """Flush to parquet. Schema:
           (step, layer, token_pos, expert_id, rank_in_topk,
            bid_total, bid_mahal, bid_forget, bid_bias)
        """
```

At training, trace is sampled at 1% of steps (calibration) and fully on at eval. At inference, configurable — off by default, on with `--trace-routing` flag. Memory cost per recorded token-layer-expert triple: 28 bytes. A 2k-token prompt × 16 layers × 8 top-k = 256k rows = ~7 MB. Acceptable.

### 5.3 Three artifacts the architecture produces

**(a) Per-token routing trace.** The raw BidTrace dump. Reviewer query: "why did expert 17 win on this token?" → grep by `(token_pos, layer, expert_id=17)`, read `bid_mahal`, `bid_forget`, `bid_bias`. Answer: human-readable by construction.

**(b) Per-expert specialization map.** Offline post-processor over the trace: for each expert, cluster the tokens it wins into top-50 by mean `mahal` (closest-to-prototype tokens). Project those token IDs back to source strings via the tokenizer. Output: `"Expert 17, layer 9: wins on tokens {=, ==, !=, :=, ->, +=} — operator specialization."` This is the **headline interpretability artifact**. FeCAM cannot produce this by construction (it has no expert specialization).

**(c) Prototype-to-input nearest neighbor.** For each `mu[layer, expert]`, find the top-5 training tokens with smallest Mahalanobis distance. Tokens are decodable by the OLMoE tokenizer. This projects prototype centroids directly to interpretable inputs — the MoB-specific answer to "what does this prototype mean?" which prompt-CL and LoRA-MoE-CL cannot answer.

### 5.4 Routing consistency as an interpretability metric
Reviewer will ask: "does the same token-in-same-context always route to the same expert?" We answer with a metric: over a held-out corpus, compute routing entropy per `(token_id, context_hash)` tuple. Low entropy (≈ 0) = deterministic routing = "this token type belongs to this expert." High entropy = "this token is routed by local context." Both are publishable; either is falsifiable.

---

## 6. SALE architectural contrast (Job 4)

SALE (arxiv 2602.02751) runs a reverse auction between heterogeneous LLM *agents* at the task level: given a request, bidding agents quote expected cost-to-serve, cheapest-qualified wins, one model handles the full request. The auction is *outside* any model. MoB puts the auction *inside* a single model's MoE layer, fires it *per token per layer*, and uses *internal hidden states* as the bid input. This produces three things SALE cannot: (1) fine-grained sub-task specialization visible in per-token bid logs, enabling per-token interpretability (Job 3); (2) continual fine-tuning of a single model where forgetting-cost is measured on *parameter* drift, not agent reliability; (3) token-specific routing decisions that interleave within a single forward pass, not per-request escalation. SALE is concurrent neighboring work at a different granularity — cite, don't reframe.

---

## 7. Updated cross-scale invariant

The R1 `AuctionRouter` type signature holds, with one added hook:

```python
class AuctionRouter(nn.Module):
    """Parameter-free router. Buffers update by EMA, not gradient."""
    # Buffers: mu[E, D], diag[E, D], U[D, r] (optional), F_tilde[E, r_f],
    #          delta_tilde[E, r_f], bias[E]
    def forward(
        self,
        z: Tensor,                    # [N, D]  features
        alpha: float,
        beta: float,
        top_k: int = 1,
        trace_sink: Optional[BidTrace] = None,   # NEW in R2
    ) -> Tuple[LongTensor,            # [N, top_k] winner indices
               FloatTensor,           # [N, E]     full bid matrix
               Dict[str, Tensor]]:    # NEW: bid_components for trace
        ...
```

**Survives the 5-8B cap:** module definition is independent of d_model, expert count, layer count. OLMoE's d=2048 and Scale 2's d=768 both instantiate the same signature.

**Survives interpretability:** the trace_sink is optional and stateless; absence does not change behavior. Production inference can run without trace; research runs with trace.

**What changed from R1:** only the trace hook. The core contract — Mahalanobis + forget + bias → argmin_k — is identical. This is the load-bearing cross-scale invariant and I stand behind it.

---

## 8. Updated dealbreaker

R1 dealbreaker was about EMA prototype drift at Scale 3. It stands, but I add a second, sharper one specific to OLMoE:

> *Empirical demonstration that the DeepSeek-V3 bias term, once added to the bid to satisfy Sage's conscience-term necessity, dominates both Mahalanobis and forget-cost in magnitude within the first 1B continued-pretraining tokens on Dolma, reducing MoB to "DeepSeek-V3 bias routing with decorative prototypes."*

If the bias update rate is tuned to actually balance load, and its scale is comparable to or larger than `alpha · d_M`, then MoB collapses to KAY/O R1's Regime B (priority scheduling) and the auction adds nothing over DeepSeek-V3. The architectural fix, if this triggers, is to cap `|bias_i|` at a fixed fraction of the per-step `max_n,e mahal[n,e]` — but that is a late-stage patch and I would rather accept the retraction risk than pre-bake the clamp.

Original R1 dealbreaker (prototype drift on losing-near-winner experts) still stands and should be tested first because it is cheaper to falsify.

---

*Chamber, end of Round 2. ~1740 words.*
