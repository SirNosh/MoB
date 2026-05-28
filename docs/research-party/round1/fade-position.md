# Fade — Frontier Position Paper (Round 1)

**Role:** arxiv frontier tracker, research party round 1
**Prior artifact:** `docs/lit-review/02-frontier-cl-moe-2024-2026.md` (cited, not re-read)
**Horizon:** design commitments the 2024–2026 frontier forces on MoB, per scale.

---

## 1. Executive summary — frontier-anchored design commitments

1. **Kill Split-MNIST as a headline result.** 2024–2025 CL papers that submit to NeurIPS/ICLR/CVPR treat Split-MNIST as a *debug harness*, not a benchmark. MoB must graduate its top-line claim to Split-CIFAR-10 (sanity) + Split-CIFAR-100 (headline) + Split-ImageNet-R (bar), matching LAMDA-PILOT conventions.
2. **Adopt DeepSeek-V3's bias-term update (arxiv 2408.15664, 2412.19437) as the training-time prototype-routing collapse fix.** It is drop-in, auxiliary-loss-free, and has become the de-facto frontier balancer. MoB's bid should include a per-expert dynamic bias updated by underload; this is a small code delta and a large legitimacy delta.
3. **Add a "null expert" option in the auction (MoE++, arxiv 2410.07348, ICLR 2025).** MoB currently forces every token through a winning expert; letting a zero-compute null expert win is the correct way to express "nothing should be learned about this sample" and directly addresses routing collapse at low signal.
4. **Positioning on CIFAR-100: prototype-CL arena, not prompt-CL, not LoRA-MoE-CL.** MoB's bid structure is a *prototype-distance + stability* ranking. Compete where the math lives. LoRA-MoE-CL is a separate race MoB should not enter at CIFAR-100.
5. **Frozen ViT-B/16 is the 2025 default backbone for PTM-based CL.** MoB's "per-expert network" concept must mutate: experts become **LoRA adapters over a shared frozen ViT-B/16**, not independent networks. Per-expert independent nets is a 2021 convention.
6. **LLM-scale target = OLMoE upcycled, not Mixtral from scratch.** Academic lab compute reality forces upcycling (Drop-Upcycling, arxiv 2502.19261). OLMoE is fully open, mid-scale, and its router is the cleanest swap point for MoB.
7. **MoB is orthogonal to the router-free trend, not aligned.** SMEAR/Lory/ReMoE *remove* the router; MoB *replaces* it with a different mechanism. Don't pitch MoB as "router-free" — pitch it as "router-principled."
8. **Cross-scale invariant: the load-balancer must be mechanism-native, not an auxiliary loss.** DeepSeek-V3 retired aux-loss balancing. MoB already is aux-loss-free via the auction — lean into that and never revert.

---

## 2. Scale 1 — Split-MNIST / CIFAR-10

### Split-MNIST is not 2025-valid for CL papers
Every serious CL venue in 2024–2025 treats Split-MNIST as a debug fixture. Papers that lead with Split-MNIST get desk-rejected or told to move it to the appendix. The minimum-viable benchmark trinity for a 2026 submission:

- **Sanity / debug:** Split-CIFAR-10 (5 tasks of 2), not Split-MNIST. Split-MNIST stays as an internal smoke test only.
- **Standard CL:** Split-CIFAR-100 (10 or 20 tasks), Split-ImageNet-R (10 tasks), Split-CUB-200 (10 tasks). These three are the LAMDA-PILOT default trio.
- **Task-free / domain-CL:** 5-Datasets (CIFAR-10, MNIST, not-MNIST, Fashion-MNIST, SVHN) — still valid, often used by drift-detection / task-free methods.
- **Rot-MNIST / Perm-MNIST:** valid only in *task-free* settings; headline claim cannot rest here.

### LAMDA-PILOT portability
LAMDA-PILOT (github.com/LAMDA-CL/LAMDA-PILOT, published SCIS 2025; arxiv 2309.07117) is the current lingua franca. It already implements L2P, DualPrompt, CODA-Prompt, SimpleCIL, FOSTER, MEMO, RanPAC, APER. MoB integration cost:

- Implement a `MoBLearner` class conforming to PILOT's `BaseLearner` interface (`_train`, `_eval_cnn`, `_construct_exemplar`, `after_task` hooks).
- Map MoB's auction step into PILOT's per-batch loop; prototype memory goes in the `after_task` hook.
- Estimated effort: 2–3 engineer-days. This should be **done before paper submission**, not after.

### Training-time prototype-routing collapse — frontier fixes
The MoE literature has named and solved MoB's collapse problem multiple times:

- **DeepSeek-V3 bias term (arxiv 2408.15664 / 2412.19437):** per-expert bias `b_i` added to routing affinity, incremented by `γ=1e-3` when expert is underloaded, decremented when overloaded. Affinity is used for routing, but the *gating value* comes from the unbiased score. This is the frontier's answer to load imbalance and is auxiliary-loss-free. **Recommend import: YES.** It slots directly into the bid; cost is one scalar per expert.
- **MoE++ null expert (arxiv 2410.07348, ICLR 2025):** add a zero-compute expert that outputs zero (discard), copy (residual), or constant. **Recommend import: YES, specifically the zero-expert variant** as a principled "decline to route" in the auction. This is the cleanest treatment of "this sample shouldn't be learned by anyone yet" — which is exactly the early-task collapse regime.
- **ReMoE dense-to-sparse (arxiv 2412.14711):** anneal from dense routing to sparse during training. **Recommend import: NO.** ReMoE is a training-schedule trick for *pretraining*; MoB's auction is a per-step decision, and annealing the sparsity defeats the auction. Wrong tool.
- **Soft-MoE (arxiv 2308.00951):** continuous token-expert mixing weights. **Recommend import: NO.** Soft routing is fundamentally incompatible with an auction (an auction needs a discrete winner). This is the "aligned but different" camp — acknowledge, don't adopt.

**Net commitment for Scale 1:** keep auction discrete, add DeepSeek-V3 bias term to bids, add MoE++ null expert as a bidder. Drop Split-MNIST as headline.

---

## 3. Scale 2 — CIFAR-100

### Which arena?
Three CL-on-CIFAR-100 arenas with PTMs exist in 2024–2025:

- **Prompt-CL:** L2P, DualPrompt, CODA-Prompt, HiDe-Prompt, EvoPrompt, VQ-Prompt (NeurIPS 2024), NSP2 (NeurIPS 2024 null-space prompt tuning). Strong CIFAR-100 numbers in the 87–92% band on frozen ViT-B/16.
- **Prototype-CL:** SimpleCIL, RanPAC (NeurIPS 2023), FeCAM (NeurIPS 2023), EASE, APER. Training-free or near-training-free classifier heads over frozen features.
- **LoRA-MoE-CL:** MoE-Adapters, SMoLoRA, OPLoRA, RAMoLE, MixLoRA-DSI, D-MoLE, LD-MoLE. Low-rank experts, often with routing.

**MoB's natural arena: prototype-CL.** The bid formula `α·exec_cost + β·forget_cost` with prototype distance is the same math family as FeCAM/RanPAC. Cypher already flagged FeCAM is functionally MoB-minus-the-auction — so the paper's story is "what does an auction add on top of the prototype-CL frontier?"

### 2026-legible baselines (MUST include)
Without these, reviewers will say "incomplete":

- **SimpleCIL** (arxiv 2303.07338) — the training-free prototype baseline.
- **RanPAC** (arxiv 2307.02251, NeurIPS 2023) — random projection + prototype.
- **FeCAM** (arxiv 2309.14062, NeurIPS 2023) — feature covariance + Mahalanobis.
- **L2P** (arxiv 2112.08654, CVPR 2022) — the prompt-CL anchor baseline.
- **CODA-Prompt** (arxiv 2211.13218, CVPR 2023) — prompt-CL frontier.
- **HiDe-Prompt** (arxiv 2310.07234, NeurIPS 2023 spotlight) — hierarchical prompt-CL.
- **EASE** (arxiv 2403.12030, CVPR 2024) — expandable subspaces, strong CIFAR-100.
- **MoE-Adapters** (arxiv 2403.11549) — LoRA-MoE-CL anchor for the "auction vs learned routing over adapters" framing.
- **CBPNet** (arxiv 2509.15785) — 2025 SOTA on Split-CIFAR-100 at 86.31% rehearsal-free. If MoB's headline number is below this, the paper is DOA.

### Is there concurrent "auction-like CL routing" work?
From my prior scouting: the only hit with verbatim "Mixture of Bidders" phrasing is **arxiv 2512.10969** — which I flagged as either the user's own submission or concurrent work. *This remains the single highest-priority unresolved question before any further work.* **Uncertainty flag: HIGH.** The `25xx/26xx` prefix may be a search-indexer artifact; must be verified on arxiv.org directly.

Beyond that one ID, no 2024–2025 paper uses explicit auction / VCG / first-price mechanics inside a transformer MoE layer. The vocabulary is clean. Closest precedent: **BASE Layers (ICML 2021, arxiv 2103.16716)** — balanced-assignment via auction-like linear-assignment solver. Sova already flagged this.

### Pretrained encoder convention
**Frozen ViT-B/16 IN-1K-sup is the 2025 default**, followed by ViT-B/16 IN-21K. This has three consequences for MoB:

1. MoB's per-expert-network design (each expert = independent CNN/MLP) is a 2021 artifact. It will not survive review on CIFAR-100.
2. Experts must become **LoRA adapters (rank 4–16) inserted into a frozen ViT-B/16**, with the auction ranking adapters per token/sample.
3. This is a non-trivial refactor but aligns MoB with LoRA-MoE-CL compute conventions and lets CIFAR-100 numbers be legible against the right baselines.

**Commitment:** For CIFAR-100 scale, experts := LoRA adapters over frozen ViT-B/16. Keep the auction bid formula; change only the expert primitive.

---

## 4. Scale 3 — LLM MoE FFN layer

### Base system to replace the router of
Candidates and verdicts:

- **Mixtral-8x7B:** closed-ish ecosystem, coarse-grained (8 experts, top-2), no shared expert. Not the frontier anymore.
- **DeepSeek-MoE-16B / DeepSeek-V3:** fine-grained (64–256 experts), shared expert, aux-loss-free. This is the frontier but compute-prohibitive for an academic lab.
- **OLMoE-1B-7B (arxiv 2409.02060, AI2):** fully open (weights, data, training code), 64 experts, top-8, load-balance + router-z loss. Mid-scale, reproducible.
- **Qwen3-MoE:** strong, but weights-open-only, training stack partially closed.

**Recommendation: OLMoE as the base system.** Fully reproducible, the community has adopted it as the open MoE research substrate, and replacing its router is a well-defined experiment. Report secondary transfer on DeepSeek-MoE-16B if compute allows.

### Does the auction scale to 256 experts per layer per token?
Honestly: **not in its current form.** A 256-way first-price reverse auction per token is O(256) bid computations per token per layer, which is the same asymptotic complexity as DeepSeek-V3's affinity scoring — so compute is fine. The *learning* problem is harder: with 256 bidders, any single bid is a weak signal.

**Design answer: hierarchical auction.** Two-stage:
1. Route token to one of K shared groups (K=8 or 16) via coarse auction.
2. Within the winning group, fine auction picks top-n experts.
This mirrors DeepSeek-V3's fine-grained-with-shared design and is the 2025 frontier pattern. **Commitment: if MoB goes past 32 experts, it goes hierarchical.**

### Router-free trend alignment
**SMEAR, Lory, ReMoE merge experts into dense computation.** They eliminate discrete routing entirely. **MoB does the opposite** — it keeps routing discrete and makes the decision mechanism more principled. These are opposite philosophical directions. Don't claim alignment. The honest framing: "router-free methods argue routing is the problem; MoB argues the *learned* router is the problem."

### Upcycling vs from-scratch
**Upcycling is the only realistic academic entry.** From-scratch MoE pretraining at LLM scale costs ≥ $500k compute. Upcycling from a dense checkpoint (Drop-Upcycling, arxiv 2502.19261; NVIDIA arxiv 2410.07524; Sparse Upcycling arxiv 2212.05055) brings this to ~$10k–$50k for a proof-of-concept at 1B-7B active param scale.

**Commitment:** Upcycle OLMoE-style from OLMo-1B-hf or OLMo-7B, replacing the learned router with MoB's auction at initialization. Report continued-pretraining loss curves on a held-out slice of Dolma for 10–20B tokens. This is the credible LLM-scale experiment for a lab that isn't Meta, DeepSeek, or AI2.

---

## 5. Forbidden claims — positioning that won't survive review

Flag these as literal tripwires in the draft:

1. **"First non-learned router for MoE"** — Hash Layers (arxiv 2106.04426, Roller et al. 2021) predates MoB by four years. Do not make this claim. The correct claim: "first *auction-based* router."
2. **"First principled load-balancing for MoE"** — DeepSeek-V3's aux-loss-free balancing (2408.15664), DeSieno conscience (1988), Switch Transformer aux loss (2021) all predate. Correct claim: "load-balancing emerges from the auction mechanism without an auxiliary term."
3. **"First market-mechanism inside a neural network"** — Baum "hayek machines" (1990s), Ohsawa market-based committees (2000s), multi-agent LLM markets (2024–2025) all predate. Correct claim: "first reverse procurement auction routing for CL-in-MoE."
4. **"Strongest baseline: GatedMoE + EWC on Split-MNIST"** — this is a 2019-era baseline set. Reviewers will reject. Must include FeCAM, RanPAC, L2P, CODA-Prompt, MoE-Adapters, HiDe-Prompt, CBPNet.
5. **"Per-expert specialization emerges"** — every MoE paper since 2017 has claimed this; reviewers are allergic. If claimed, must be backed with quantitative routing-entropy + ablation, not visualization.
6. **"Solves catastrophic forgetting"** — CL as a field does not accept "solved" framing. Use "reduces forgetting by X% relative to Y baseline on benchmark Z."
7. **"Biologically plausible"** — dead phrase in 2025. Don't invoke unless the paper does neuroscience modeling proper.
8. **"Scales to LLMs"** without upcycling evidence — conjecture only; must be demonstrated, not asserted.

---

## 6. Cross-scale frontier invariant

**The one design commitment the frontier supports at all three scales: load-balancing must be mechanism-native, not an auxiliary loss.**

- Split-MNIST/CIFAR-10 scale: auction directly produces balanced routing via bid diversity + DeepSeek-V3 bias term.
- CIFAR-100 scale: LoRA-adapter auction over frozen ViT-B/16 inherits the same mechanism-native balancing; no aux loss needed.
- LLM scale: aligns directly with DeepSeek-V3's retirement of aux-loss balancing. MoB is on the right side of the 2024–2025 paradigm shift.

This is the thread that ties MoB's story together across scales. Every paper section should reinforce it.

---

## 7. Where I defer

Three cruxes for other specialists:

1. **(→ Sage / Astra)** Is the MoB auction actually incentive-compatible, or only first-price-naive? DeepSeek-V3's bias term breaks truthful bidding in a formal sense. We need a mechanism-design result (even informal) before claiming "principled." Astra already noted this is first-score reverse procurement — Sage needs to close whether that's a feature or a bug.
2. **(→ Chamber / Killjoy)** If experts become LoRA adapters at CIFAR-100 scale, does the auction's gradient signal survive through the frozen-backbone → adapter → bid chain? Chamber on architecture, Killjoy on systems cost of evaluating 8–32 adapter bids per batch.
3. **(→ Sage)** Is there a theoretical argument that the auction's worst-case routing is bounded away from collapse in a way DeepSeek-V3's bias term is not? If yes, that's the paper's deepest contribution and should be the theorem. If no, MoB is "DeepSeek-V3 balancing + prototype bid" and needs a different hook.

---

## 8. Dealbreaker

**The single paper in the next 6 months that forces retraction:** a paper from DeepSeek, AI2, or a top-tier academic lab that demonstrates **auxiliary-loss-free + prototype-distance + EWC routing on CL benchmarks at CIFAR-100 or LLM scale, using the DeepSeek-V3 bias mechanism with a discrete winner-take-most selection.** That is within a ~1-line code delta of MoB, and any lab with more compute can scoop us in one paper.

Watchlist triggers (monitor weekly):
- DeepSeek submissions to ICLR/NeurIPS 2026 mentioning "continual" or "catastrophic forgetting"
- AI2 OLMoE follow-ups with CL evaluation
- Any arxiv title containing "auction" + ("MoE" | "continual" | "router")
- Any follow-up to arxiv 2512.10969 (confirm identity first)
- LAMDA (Nanjing University) group — if they add auction routing to PILOT, we are scooped on benchmark infrastructure

**If any of the above lands with stronger CIFAR-100 numbers than MoB before our submission, the paper's contribution collapses to "we did it first but smaller" and needs a pivot to LLM scale or to the theorem in defer-crux #3.**

---

*Word count: ~2,440.*

## Source arxiv IDs with uncertainty flags

High confidence (verified in prior scouting or this session):
- 2408.15664 (DeepSeek aux-loss-free), 2412.19437 (DeepSeek-V3 tech report)
- 2410.07348 (MoE++, ICLR 2025)
- 2502.19261 (Drop-Upcycling)
- 2410.07524 (NVIDIA upcycling)
- 2409.02060 (OLMoE)
- 2309.07117 (LAMDA-PILOT)
- 2309.14062 (FeCAM), 2307.02251 (RanPAC), 2303.07338 (SimpleCIL)
- 2112.08654 (L2P), 2211.13218 (CODA-Prompt), 2310.07234 (HiDe-Prompt)
- 2403.12030 (EASE), 2403.11549 (MoE-Adapters)
- 2103.16716 (BASE Layers), 2106.04426 (Hash Layers)
- 2308.00951 (Soft-MoE), 2412.14711 (ReMoE)
- 2509.15785 (CBPNet, 2025)

Medium confidence:
- 2406.13233 (AdaMOE null experts) — found this session, not deeply verified

HIGH uncertainty (flagged for verification):
- 2512.10969 ("Mixture of Bidders") — prefix looks like future date; indexer artifact suspected. **Block all further work until resolved.**
- Any `26xx` prefix IDs — manual arxiv.org verification required.
