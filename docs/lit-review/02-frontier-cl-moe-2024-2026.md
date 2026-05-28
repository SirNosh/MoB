# Frontier Scouting: Continual Learning & MoE Routing, 2024–2026

**Scout:** Fade
**Date:** 2026-04-18
**Scope:** Arxiv + conference proceedings, Jan 2024 through Apr 2026
**For project:** MoB (auction-based MoE routing for continual learning)
**Stance:** Surface-level mapping, NOT recommendations. Interpret later.

---

## 1. Executive summary

1. **MoB's own arxiv listing (2512.10969) is live.** A paper titled "MoB: Mixture of Bidders — A Truthful Auction Mechanism for Continual Learning in Mixture of Experts" is on arxiv (Dec 2025 ID, discoverable April 2026). Language in the abstract — VCG auctions, execution+forgetting bid, stateless routing — matches this project exactly. **Confidence: high this is either the user's own submission or a near-concurrent work; the team should confirm provenance immediately before any follow-up work.** If not the user's, it's direct priority prior art.
2. **The auxiliary-loss-free routing direction is the biggest MoE router shift of 2024–2025.** DeepSeek-V3 (Dec 2024) dropped the load-balance auxiliary loss entirely and instead nudges routing via per-expert additive bias terms updated *outside* backprop — the field is moving away from differentiable-through-the-router loss plumbing. MoB's "no auxiliary loss" framing is now mainstream, not contrarian.
3. **Prompt-pool CL has plateaued; the frontier moved to prototype + covariance methods on frozen PTMs.** FeCAM, RanPAC, EASE, and 2025 follow-ups (Sculpting CLS / TOSCA, Knowledge Memorization & Rumination CVPR'25) dominate the replay-free PTM-CIL leaderboard. Mahalanobis-distance routing is now a well-studied primitive here.
4. **Mixture-of-LoRA for CL is the single most active sub-field.** LoRAMoE, MoLA, MoLE, O-LoRA (2023) spawned a 2024–2025 wave: D-MoLE, LD-MoLE, LoRAC-IPC, OPLoRA, OLieRA, MH-MoE, SMoLoRA, RAMoLE, MixLoRA-DSI, Brainstacks. Most of these learn a router over LoRA experts, so MoB's auction substitute directly competes with them.
5. **Router-free / soft-routing is having a moment.** SMEAR (2023), Lory (2024, Princeton+Meta), ReMoE (ICLR 2025, Tsinghua) are all arguing learned top-k routing is the wrong abstraction. ReMoE in particular replaces top-k with ReLU gating, reporting "universally outperforms top-k." This is a genuine paradigm candidate.
6. **Zero-computation experts (MoE++, ICLR 2025 oral).** Introduces null / copy / constant experts so easy tokens skip the MoE layer. This reframes "bid" as "is any expert worth engaging at all?" — MoB could inherit this as a bid threshold.
7. **Fine-grained + shared experts won.** DeepSeekMoE's architectural recipe (many tiny experts, a few always-on shared experts) is now the default in DeepSeek-V2/V3, Qwen3-MoE, OLMoE, Arctic (128 experts), Jamba, and Mixtral successors. Expert granularity is rising; MoB's 4-expert scale is a toy but the granularity increase strengthens the case for structure-based routing.
8. **Task-free CL with drift detection is mature enough to cite.** Online-LoRA (NeurIPS'24) already uses training-dynamics-based drift detection to trigger adaptation without task labels. MoB's "shift detection → winner reset" is well-aligned with this trajectory.
9. **No other 2024–2026 paper uses auction/VCG language for neural routing.** Market-mechanism vocabulary appears in multi-agent LLM coordination (Microsoft's Magentic Marketplace, MIT's Ripple Effect Protocol) but not inside transformer layers. MoB is novel on the vocabulary axis; whether the mechanism is novel depends on whether 2512.10969 is the user's own paper.
10. **Post-foundation-model CL consensus is forming: continual pre-training + continual fine-tuning + continual compositionality.** (ACM CSUR 2025 survey.) Classical Split-MNIST/CIFAR CL is now treated as a scientific testbed, not a target — the target is LLM-scale CPT/CFT.

Calibration flags: arxiv IDs below are copied verbatim from search-result metadata; I did not independently verify every ID. Dates on papers with IDs beginning `26xx` are ambiguous and flagged. I did not open every PDF.

---

## 2. Continual learning frontier, 2024–2026

### 2.1 Prompt-pool CL: past the peak

The L2P → DualPrompt → S-Prompt → CODA-Prompt → HiDe-Prompt arc essentially closed in 2023. 2024–2025 follow-ups are incremental:

- **LW2G — Learning Whether to Grow (2024, arxiv 2409.18860)** — decides *per-task* whether to expand the prompt pool; attacks the uncontrolled growth problem in CODA/HiDe.
- **PEARL — Input-Agnostic Prompt Enhancement with Negative Feedback (2024, arxiv 2412.10900)** — only updates a subset of the pool per step.
- **REP — Resource-Efficient Prompting (2024, arxiv 2406.04772)** — compute budget for rehearsal-free prompt-based CL.
- **Prompt Customization (PC, 2024, arxiv 2404.18060)** — prompt-assemblage modulation over CODA.
- **C-ADA — Continual Adapter (ECCV 2024, arxiv 2407.10281)** — adapter-path successor, explicitly claims "beyond prompt learning" and beats CODA by ~2.7%.

**Read:** prompt-pool methods are losing ground to adapter/LoRA-per-task methods. The routing problem (which prompt to use?) still uses a learned key-query match — MoB could target this.

### 2.2 Pretrained-encoder + prototype CL: the current leaderboard

- **FeCAM (NeurIPS 2023)** — already uses Mahalanobis distance over class covariances on frozen ViT. MoB's Mahalanobis bid is conceptually close to FeCAM's classifier.
- **RanPAC (NeurIPS 2023)** — frozen random projection + prototypes; "RanPAC-imp" fixes instability in the official code.
- **EASE — Expandable Subspace Ensemble (CVPR 2024, arxiv 2403.12030)** — one lightweight adapter per task → ensemble of subspaces.
- **SLCA / SLCA++** — slow-learner + classifier alignment; strong on ImageNet-R.
- **Calibrating Higher-Order Statistics for FSCIL with ViT (CVPRW 2024, arxiv 2404.06622)** — explicit covariance shrinkage + correlation normalization for a full-rank Mahalanobis matrix on pretrained features. **Directly relevant to MoB's 4096-dim scaling plan.**
- **Sculpting CLS Features / TOSCA (Feb 2025, arxiv 2502.14762)** — LuCA module on the CLS token, ~8× fewer params than EASE-style methods.
- **Knowledge Memorization and Rumination (CVPR 2025)** — PTM-CIL with memorization+consolidation scheduling.
- **Navigating Semantic Drift (Feb 2025, arxiv 2502.07560)** — Mahalanobis-aligned covariance matrices between current and old representations for task-agnostic CIL. Most relevant precedent for MoB's prototype drift problem.
- **LAMDA-PILOT toolbox (ICCV 2025 track)** — 15 PTM-CIL algorithms in one benchmark harness; the field uses this.
- **Continual Learning with PTMs: A Survey (IJCAI 2024)** — Zhou et al., Nanjing.

**Read:** Prototype + covariance + frozen backbone is the 2024–2025 PTM-CIL consensus. FeCAM-style Mahalanobis is mainstream. MoB isn't novel in *using* Mahalanobis — it's novel in using it as a *bid* that competes with an EWC term inside an MoE router. This is a narrow but real gap.

### 2.3 Mixture-of-LoRA CL: crowded, moving fast

Base generation (2023–early 2024): LoRAMoE, MoLE, MoLA, O-LoRA, PESC (EMNLP 2024, arxiv 2401.02731).

2024–2026 successors:

- **D-MoLE — Dynamic Mixture of Curriculum LoRA Experts (ICML 2025, arxiv 2506.11672)** — evolves architecture with controlled parameter budget; dynamic per-layer expert allocator; continual multimodal instruction tuning.
- **LD-MoLE — Learnable Dynamic Routing (arxiv 2509.25684, Sep 2025)** — end-to-end learnable dynamic routing allocating experts to tokens across layers.
- **MoLA — Layer-wise Expert Allocation (NAACL Findings 2025)** — middle layers benefit from more experts than lower layers.
- **LoRAC-IPC (2025, arxiv 2504.13407)** — orthogonal LoRA composition with critical-parameter constraints; +6.35% over O-LoRA on Split-CIFAR-100, −3.24% forgetting.
- **OLieRA (2025, arxiv 2509.06100)** — orthogonal LoRA in Lie groups; preserves intrinsic parameter geometry.
- **OPLoRA (2025, arxiv 2510.13003)** — double-sided orthogonal projections preserving dominant singular directions.
- **MH-MoE — multi-head routing (2025, arxiv 2602.12587 [ID looks mis-captured; flagged])** — routes per attention head to reduce composition collisions; cuts backward transfer degradation on Qwen3-0.6B from 11.2% (LoRAMoE) to 4.5%.
- **SMoLoRA (ICCV 2025)** — separable MoLoRA with dual routers for visual understanding vs instruction-following.
- **Brainstacks (2026 arxiv 2604.01152 [ID flagged])** — frozen MoE-LoRA stacks, hard null-space projection, cross-domain composition.
- **Mixture of LoRA Experts for Continual IE (EMNLP Findings 2025)** — MoLE-CIE.
- **RAMoLE, MixLoRA-DSI** — expert pool *grows* when OOD detected via router energy statistics; "sublinear parameter growth." Most MoB-adjacent in spirit.

**Read:** The LoRA-MoE CL wave is where MoB's closest competition lives. Almost all of these keep a *learned* router and add regularization or structure to prevent router forgetting. MoB's differentiator is removing the router parameters entirely.

### 2.4 Task-free / online / drift-based CL

- **Online-LoRA (WACV/arxiv 2411.05663)** — task-free online CL via LoRA + loss-dynamics-based drift detector on ViT. Closest method-spirit to MoB's "shift detection → winner reset."
- **Holistic CL under Concept Drift with Adaptive Memory Realignment (arxiv 2507.02310)** — concept-drift handling in replay CL.
- **Random Representations Outperform Online CL Representations (arxiv 2402.08823)** — striking result: frozen random features beat online-learned representations on several online-CL benchmarks. Suggests the routing signal, not the features, is the bottleneck.
- **Online CL: A Systematic Literature Review (arxiv 2501.04897, Jan 2025)** — 81 methods + 83 datasets surveyed.
- **MCD-DD — Maximum Concept Discrepancy Drift Detector** — contrastive-embedding drift without labels.
- **LDC — Learnable Drift Compensation** — fits on any moving backbone.
- **State-Space Plug-and-Play for Online CL (arxiv 2412.18177)** — SSM-based enhancement.
- Classical **ADWIN / Page-Hinkley** detectors are being hybridized with loss-based drift signals in 2024 papers (e.g. H-CLAS for IoT).

**Read:** "Drift detector triggers structural change" is a well-established pattern now. MoB's optimizer reset on shift is consistent with this, not novel. Novelty is the *auction bid* as the detection substrate.

### 2.5 Auction / market / bidding language in CL specifically

Nothing else uses auction vocabulary inside a CL method besides `arxiv 2512.10969` (MoB itself). I searched "auction routing," "bidding routing," "market routing," "economic routing" in the CL and MoE contexts and found no other 2024–2026 precedent.

---

## 3. MoE routing frontier, 2024–2026

### 3.1 The model zoo

| Model | Lab | Date | Architecture | Routing |
|---|---|---|---|---|
| **Mixtral 8x7B** | Mistral | Dec 2023 | 8 experts | Static top-2 softmax |
| **Mixtral 8x22B** | Mistral | Apr 2024 | 8 experts × 22B | Top-2 |
| **DBRX** | Databricks | Mar 2024 | 16 experts, top-4 | Fine-grained, fused MoE kernels |
| **DeepSeekMoE 16B** | DeepSeek | Jan 2024 (arxiv 2401.06066) | Fine-grained + shared | Top-k with shared experts |
| **DeepSeek-V2** | DeepSeek | May 2024 (arxiv 2405.04434) | 160 routed + 2 shared | Same recipe, scaled |
| **DeepSeek-V3** | DeepSeek | Dec 2024 (arxiv 2412.19437) | 256 routed + 1 shared, top-8 | **Auxiliary-loss-free** (bias-term balancing) |
| **Jamba** | AI21 | Mar 2024 (arxiv 2403.19887) | Transformer+Mamba+MoE hybrid, 16 experts | Top-2, every-other-layer MoE |
| **Arctic 480B** | Snowflake | Apr 2024 | Dense-MoE hybrid, 128 experts | Top-2, residual MoE alongside dense |
| **GRIN-MoE 16×3.8B** | Microsoft | Sep 2024 (arxiv 2409.12136) | Phi-3.5 MoE | **SparseMixer-v2** — gradient-informed router |
| **OLMoE 1B-7B** | Allen AI | Sep 2024 (arxiv 2409.02060) | 64 experts, top-8 | Learned + load-balance + router z-loss; strong specialization reported |
| **Qwen3-MoE 235B-A22B** | Alibaba | May 2025 (arxiv 2505.09388) | Fine-grained | Global-batch-level aux loss (relaxed) |
| **Qwen3-Next hybrid MoE** | Alibaba | late 2025 | Hybrid MoE | Open weights |
| **Llama 4 (Scout/Maverick)** | Meta | 2025 | Large MoE | Details partially proprietary |

### 3.2 Routing innovations by paper

- **DeepSeekMoE (arxiv 2401.06066)** — the fine-grained + shared-expert recipe that the rest of the field adopted.
- **Auxiliary-Loss-Free Load Balancing (arxiv 2408.15664, Aug 2024)** — the DeepSeek-V3 recipe written up as a standalone contribution. **Replaces the entire load-balance loss with a bias-update rule outside backprop.** This is the single most important MoE-routing paper since Switch Transformer.
- **GRIN / SparseMixer-v2 (arxiv 2409.12136)** — better gradient estimator through the discrete top-k, no capacity factor, no token dropping.
- **MoE++ (ICLR 2025 oral, arxiv 2410.07348)** — zero-computation experts (null/copy/constant) — tokens can skip the MoE layer. 1.1–2.1× throughput.
- **ReMoE (ICLR 2025, arxiv 2412.14711)** — ReLU routing replaces top-k softmax. Fully differentiable. Claims to universally outperform top-k.
- **Lory (arxiv 2405.03133)** — Princeton+Meta: fully differentiable MoE for autoregressive pretraining via causal segment routing + similarity-based batching.
- **Load Balancing MoE with Similarity Preserving Routers (arxiv 2506.14038)** — router orthogonality as a balancing regularizer.
- **From Score Distributions to Balance (arxiv 2510.03293)** — plug-and-play balancing.
- **Omni-Router (arxiv 2507.05724)** — shares routing decisions across layers.
- **Router Upcycling (arxiv 2509.00679)** — initialize routers' queries from attention heads, experts' keys similarly; attention-like expert matching.
- **Self-Routing (arxiv 2604.00421 [ID flagged as anomalous])** — parameter-free routing from hidden states; competitive with learned router, more balanced utilization.
- **Hash routing** (Roller et al. 2021 reference point) — still cited in 2024–2025 as the non-learned baseline. Recent analyses confirm hash routing loses only ~1-2 PPL to learned routing, which is surprisingly close.
- **Mixture of Routers (arxiv 2503.23362)** — routing the router; latent prototype routing (LPR) and collaboration-constrained routing (C2R) report near-perfect load balancing.
- **Optimizing MoE Routers: Design, Implementation, Analysis (arxiv 2506.16419)** — systematic comparison paper; useful reference.
- **Equifinality in MoE (arxiv 2604.14419 [ID flagged])** — argues routing topology does not determine LM quality. Strong signal that router design is less load-bearing than the community thinks.

### 3.3 Upcycling

- **UpIT (arxiv 2410.01610)** — upcycling instruction tuning from dense → MoE via parameter merging; 1% of training data to pre-optimize routing vectors.
- **CLIP-UP (arxiv 2502.00965)** — sparse upcycling recipe for CLIP MoE.
- **Adaptive Upcycling (EMNLP Findings 2025)** — efficient MoE pretraining via upcycling.
- **NVIDIA dense→MoE (Oct 2024)** — virtual group initialization to match dense function at startup.
- **Read-ME (NeurIPS 2024)** — refactorize LLMs with decoupled router.

### 3.4 Router collapse, stability, load balance

- Router collapse remains the canonical MoE failure mode (early+late layers funnel to 1-2 experts; middle layers balance).
- DeepSeek-V3 bias-term balancing, Loss-Free Balancing, and similarity-preserving routers are the three cleanest 2024–2025 cures.
- Shared experts (always on) are now standard as a collapse hedge.
- Expert-orthogonality regularizers (at the router level, not expert level) are cheaper to train than per-expert orthogonality.

**Read:** MoB's current "collapses to 1-2 experts during training-time prototype routing" is exactly the classic MoE router-collapse pathology. DeepSeek-V3 bias-term balancing and MoE++ zero-experts are both directly importable as analogs (bias-the-bid; null-bid).

### 3.5 Non-learned routing precedents

- **Hash routing** (2021, resurfaced in 2024–2025 surveys) — deterministic, surprisingly competitive.
- **Self-Routing** — parameter-free from hidden states, competitive with learned.
- **Expert Choice Routing (ICLR 2023)** — reversed causality; formally, each expert ranks tokens and picks top-k. This is a reversed-auction reading already, though not framed that way.
- **MoE Routing Testbed (arxiv 2604.07030 [ID flagged])** — small-scale study of specialization behavior.

No 2024–2026 work re-derives expert-choice as an auction; the auction framing in 2512.10969 (MoB) is fresh vocabulary even if expert-choice is spiritually similar.

---

## 4. Auction / mechanism / market precedents in 2024–2026 ML

### 4.1 Inside the model

Empty — nothing else except MoB (2512.10969). That's a meaningful negative finding.

### 4.2 Multi-agent LLM systems (outside the model)

- **Magentic Marketplace (Microsoft Research 2025)** — open-source marketplace simulator for multi-agent LLM commerce.
- **Ripple Effect Protocol (Chopra et al., MIT)** — agent-population coordination, shares sensitivities not just decisions.
- **Strategic Collusion of LLM Agents (arxiv 2410.00031)** — Cournot market behavior of LLM agents.
- **From Competition to Coordination: Market-Making for Safe Multi-Agent LLM (arxiv 2511.17621)** — market-making as alignment-safe coordination framework.
- **AgenticPay (arxiv 2602.06008 [ID flagged])** — buyer-seller negotiation benchmark.
- **LLM-Based Routing in MoE for Trading (arxiv 2501.09636)** — uses MoE language in the trading domain but not inside a transformer.
- **Online Learning for Dynamic VCG (arxiv 2506.19038)** — dynamic VCG mechanisms in sequential auctions under unknown environments; closest prior work to VCG-for-ML in 2024–2025, but applied to classic auctions, not neural routing.

**Read:** Market language exists in multi-agent LLM orchestration but doesn't reach inside transformer layers. MoB's positioning — "put the market inside the MoE layer" — is a clean gap.

---

## 5. Paradigm shift watch

Changes in the last 12 months that would reshape MoB's design:

**SHIFT A — Auxiliary losses are losing.**
DeepSeek-V3's bias-term balancing (arxiv 2408.15664, Dec 2024) is a paradigm move. The field is accepting that router behavior can be steered *outside backprop*. MoB already has no auxiliary loss — this is now mainstream, not contrarian. **Implication:** the paper's "no aux loss" framing has less rhetorical weight in 2026 than it did in 2024. Lead with catastrophic-forgetting-immunity instead.

**SHIFT B — Router-free is a real research direction.**
SMEAR → Lory (ICLR 2025 Princeton/Meta) → ReMoE (ICLR 2025 Tsinghua). Three labs independently arguing top-k softmax routing is the wrong abstraction. Plus hash routing only ~1-2 PPL behind learned. **Implication:** MoB's "no learned router" is aligned with a real frontier trend, not a heresy. Cite this cluster.

**SHIFT C — Fine-grained + shared expert is the default.**
DeepSeekMoE's recipe (many tiny experts, 1-2 shared) adopted by V3, Qwen3, OLMoE, Arctic, Jamba. **Implication:** MoB at 4 experts is toy-scale. For LLM deployment the unit of analysis should be ~64 routed + 1-2 shared experts. Shared experts also serve as a collapse hedge — relevant to MoB's prototype-routing collapse problem.

**SHIFT D — Zero-computation / null experts.**
MoE++ (ICLR 2025 oral) legitimizes tokens skipping the MoE layer entirely. **Implication:** MoB could add a "null bid" — an implicit expert that bids the identity function. Fixes hard cases where all experts' bids are high (genuine OOD) without distorting the auction.

**SHIFT E — Prototype-routing collapse is a known MoE pathology, not a MoB-specific bug.**
Router collapse at early/late layers is the canonical failure mode documented across Cerebras, DeepSeek, ReMoE, MoE++. The solutions in order of popularity: (1) shared/always-on experts, (2) bias-term balancing outside backprop, (3) ReLU-style continuous gating, (4) similarity-preserving router regularization. **Implication:** MoB has documented prior art for *all* these fixes when it needs them.

**SHIFT F — LoRA-MoE-CL is the competitive arena.**
D-MoLE, LD-MoLE, MH-MoE, SMoLoRA, RAMoLE, Brainstacks, OPLoRA, OLieRA are all MoB's actual competition. Almost all retain a learned router. **Implication:** MoB's benchmark plan should include at least one of these as a baseline, not just vanilla EWC.

**SHIFT G — Post-LLM CL canon is forming.**
ACM CSUR 2025 survey crystallizes: CPT (continual pre-training) + CFT (continual fine-tuning) + continual compositionality. Split-MNIST/CIFAR are now scientific testbeds, not targets. **Implication:** MoB's Split-MNIST results are table-stakes existence proofs; the LLM-MoE-layer replacement claim needs at least a small CFT experiment to hit the 2026 bar.

**SHIFT H — Task-free with drift detection is mature.**
Online-LoRA (NeurIPS 2024) + MCD-DD + LDC established the pattern. **Implication:** MoB's shift-triggered optimizer reset has clear 2024 precedent. Cite, don't claim novelty on the mechanism — only on the bid as the signal source.

**SHIFT I — Benchmarks have tightened.**
LAMDA-PILOT (ICCV 2025) and the Online CL survey (arxiv 2501.04897, 81 methods) define the bar. Reviewers will expect comparisons inside PILOT. **Implication:** port the MoB method into the PILOT harness before submission.

**Watch but do not trust yet:**
- Equifinality in MoE (arxiv 2604.14419 [ID flagged]) — claims routing topology doesn't determine LM quality. If this replicates, MoB's *quality* claims need to shift onto the forgetting axis specifically, not generalization.

---

## 6. Reference list

**Notes on arxiv IDs:** IDs beginning with `24xx` = 2024, `25xx` = 2025, `26xx` should be 2026 but several `26xx` and `2602-2604-2606` IDs surfaced in search snippets — these may be mis-captured by search indexers and should be verified manually before citing in a paper. Flagged with `[verify]`.

### 6.1 The probable prior-art shock

- `2512.10969` — **MoB: Mixture of Bidders — A Truthful Auction Mechanism for Continual Learning in Mixture of Experts** (Dec 2025). Abstract matches this project verbatim. **Immediately determine whether this is the user's own submission or a concurrent work.**

### 6.2 Continual learning — PTM-CIL & prototypes

- `2307.02251` RanPAC (NeurIPS 2023) — reference point for 2024 successors.
- `2312.xxxxx` FeCAM (NeurIPS 2023) — reference point; exact ID not re-verified in this pass.
- `2403.12030` EASE — Expandable Subspace Ensemble (CVPR 2024).
- `2404.06622` Calibrating Higher-Order Statistics for FSCIL (CVPRW 2024).
- `2502.14762` Sculpting CLS Features / TOSCA (Feb 2025).
- `2502.07560` Navigating Semantic Drift in Task-Agnostic CIL (Feb 2025).
- `2402.08823` Random Representations Outperform Online CL Representations (2024).
- `2309.07117` PILOT toolbox (ICCV 2025).

### 6.3 Continual learning — prompt-based & adapter

- `2409.18860` LW2G — Learning Whether to Grow (2024).
- `2412.10900` PEARL (2024).
- `2406.04772` REP (2024).
- `2404.18060` Prompt Customization (2024).
- `2407.10281` C-ADA — Continual Adapter (ECCV 2024).

### 6.4 Continual learning — LoRA-MoE

- `2310.14152` O-LoRA (Oct 2023) — base.
- `2401.02731` PESC (EMNLP 2024).
- `2506.11672` D-MoLE (ICML 2025).
- `2509.25684` LD-MoLE (Sep 2025).
- `2504.13407` LoRAC-IPC (2025).
- `2509.06100` OLieRA — O-LoRA in Lie Groups (2025).
- `2510.13003` OPLoRA (2025).
- `2602.12587` MH-MoE Multi-Head Attention as Source of Catastrophic Forgetting [ID verify].
- `2604.01152` Brainstacks [ID verify].
- `2411.13949` (2024–2025) — additional LoRA-MoE-CL work found in survey index.
- MoLA (NAACL Findings 2025, aclanthology link).
- SMoLoRA (ICCV 2025, openaccess link).
- MoLE-CIE (EMNLP Findings 2025).

### 6.5 Continual learning — task-free / online

- `2411.05663` Online-LoRA (2024).
- `2501.04897` Online CL Systematic Literature Review (Jan 2025).
- `2507.02310` Holistic CL under Concept Drift with Adaptive Memory Realignment (2025).
- `2412.18177` State-Space CL + Class-Conditional Mixture of Discretization (2024).
- `2505.12512` Scalable Strategies for CL with Replay (2025).
- `2402.01364` / `2404.16789` LLM CL surveys (2024).
- ACM CSUR 2025 — Continual Learning of Large Language Models: A Comprehensive Survey.
- `2506.03320` The Future of Continual Learning in the Era of Foundation Models (Jun 2025).

### 6.6 MoE — headline model releases

- `2401.06066` DeepSeekMoE — Ultimate Expert Specialization (Jan 2024).
- `2405.04434` DeepSeek-V2 (May 2024).
- `2412.19437` DeepSeek-V3 Technical Report (Dec 2024).
- `2403.19887` Jamba: Hybrid Transformer-Mamba LM (Mar 2024).
- `2409.02060` OLMoE 1B-7B (Sep 2024).
- `2409.12136` GRIN-MoE (Sep 2024).
- `2505.09388` Qwen3 Technical Report (May 2025).
- Snowflake Arctic (blog + weights, Apr 2024).
- Mixtral 8x22B (Mistral release, Apr 2024).

### 6.7 MoE — routing innovations

- `2408.15664` Auxiliary-Loss-Free Load Balancing (Aug 2024) — DeepSeek-V3 recipe.
- `2410.07348` MoE++ Zero-Computation Experts (ICLR 2025 oral).
- `2412.14711` ReMoE — ReLU Routing (ICLR 2025).
- `2405.03133` Lory — Fully Differentiable MoE (2024, Princeton+Meta).
- `2506.14038` Load Balancing MoE with Similarity-Preserving Routers (2025).
- `2510.03293` From Score Distributions to Balance (2025).
- `2507.05724` Omni-Router (2025).
- `2509.00679` Router Upcycling (2025).
- `2503.23362` Mixture of Routers (2025).
- `2505.00315` Mixture of Sparse Attention — expert-choice in attention (2025).
- `2505.00792` Improving Routing with Graph of Tokens (2025).
- `2506.16419` Optimizing MoE Routers (Jun 2025).
- `2604.00421` Self-Routing — Parameter-Free Expert Routing from Hidden States [ID verify].
- `2604.14419` Equifinality in MoE — Routing Topology Doesn't Determine Quality [ID verify].
- `2604.07030` MoE Routing Testbed [ID verify].
- `2202.09368` Expert Choice Routing (reference; Google 2022).

### 6.8 MoE — upcycling

- `2410.01610` UpIT — Upcycling Instruction Tuning (2024).
- `2502.00965` CLIP-UP (2025).
- Adaptive Upcycling (EMNLP Findings 2025, aclanthology).
- Read-ME (NeurIPS 2024).

### 6.9 Multi-agent / market / auction

- `2506.19038` Online Learning for Dynamic VCG (2025).
- `2511.17621` From Competition to Coordination: Market-Making for Multi-Agent LLM (Nov 2025).
- `2410.00031` Strategic Collusion of LLM Agents (Oct 2024).
- `2501.09636` LLM-Based Routing in MoE for Trading (Jan 2025).
- Magentic Marketplace (Microsoft Research, Oct 2025).
- Ripple Effect Protocol (MIT, Chopra et al.).

### 6.10 Surveys & toolboxes

- `2507.11181` MoE in LLMs survey (Song 2025).
- `2503.07137` Comprehensive Survey of MoE (Mar 2025).
- `2302.03648` Class-Incremental Learning: A Survey (TPAMI 2024 version).
- ACM CSUR 2025 — Continual Learning of LLMs (Wang-ML-Lab).
- IJCAI 2024 — Continual Learning with PTMs Survey.
- LAMDA-PILOT (GitHub, toolbox).

---

## 7. Uncertain but interesting (worth tracking)

- **MH-MoE (`2602.12587` [verify])** — if the "attention-head-as-MoE-bottleneck" claim holds, it implies the *router input* is information-lossy by construction, which would support MoB's claim that a learned router over a collapsed input is the wrong object.
- **Equifinality in MoE (`2604.14419` [verify])** — strongest possible signal that router design is second-order. If true, MoB's selling point shifts hard onto the *forgetting* axis, not the *routing-quality* axis.
- **Self-Routing (`2604.00421` [verify])** — parameter-free routing from hidden states; this is the *closest mechanism in spirit* to MoB besides `2512.10969` itself. If verifiable, must be a baseline.
- **RAMoLE / MixLoRA-DSI** — expert pool grows when OOD detected by router energy. Closest analog to MoB's task-boundary expansion philosophy without the auction framing.
- **Random Representations Outperform Online CL Representations (`2402.08823`)** — strong signal that online feature learning is fragile; frozen features + smart routing is a reasonable frontier bet, which is effectively MoB's setup.
- **Router Upcycling (`2509.00679`)** — uses attention-head queries as router queries. A "features-as-bid" view consistent with MoB's prototype bid.
- **Omni-Router (`2507.05724`)** — shares routing decisions across layers. For LLM-scale MoB this would let one auction result propagate up the stack, cutting decision cost.
- **Navigating Semantic Drift (`2502.07560`)** — explicit Mahalanobis alignment of covariances across tasks; most immediate reference for MoB's prototype-drift problem under continual training.
- **GRIN / SparseMixer-v2 (`2409.12136`)** — if MoB ever needs differentiable bids (it doesn't now), this is the gradient estimator.

---

## 8. What I did NOT cover (known gaps for next pass)

- I did not read PDFs. All summaries are from abstract / first-page snippets surfaced by search.
- I did not verify arxiv IDs with `26xx` and `2602-2606` prefixes — search indexers sometimes return malformed IDs; these should be cross-checked on arxiv.org directly.
- I did not map citation velocity / Twitter / workshop traction.
- I did not check specific MoE-routing theory papers (e.g. sample complexity of MoE, identifiability). Frontier-scout scope, not theory scope.
- I did not evaluate the actual empirical strength of any method — that's for the next analyst.
- **I did not confirm whether `2512.10969` is the user's own paper.** This is the single most urgent open question from this scouting pass.

---

*Fade signing off. Interpret, don't act yet.*
