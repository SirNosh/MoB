# Established Literature Map for MoB: Continual Learning, MoE Routing, and CIFAR-100 CL Benchmarks

**Prepared by:** Sova (research tracking specialist)
**Date:** 2026-04-18
**Scope:** Peer-reviewed canon (pre-2024) plus foundational 2024 papers. This is a *provenance* document — every claim is backed by a specific paper, venue, and year. Claims that could not be independently verified carry an explicit `[uncertain]` marker.
**Out of scope:** Routing redesigns for MoB, proposed fixes for the training-time collapse problem, anything under `docs/literature/` (stale).

---

## 1. Executive Summary

1. **BASE Layers (Lewis et al., ICML 2021) is the closest published precedent to MoB.** It formulates MoE routing as a balanced linear-assignment problem solved with the **Bertsekas (1992) auction algorithm** — the only MoE paper that literally runs an auction. It still uses learned token and expert embeddings to compute assignment scores, so it is not "stateless," but it is the most direct evidence that auction mechanisms train stably at LLM scale.

2. **Hash Layers (Roller et al., NeurIPS 2021) is the strongest evidence that non-learned routing can match learned routing** at the sparse-MoE LM scale. A fixed token-hash routed to FFN experts reportedly matches or beats Switch Transformer in validation perplexity with no load-balancing loss. This is the existence proof MoB's "no learned router" thesis needs.

3. **The published MoE canon uses learned token-choice (Shazeer 2017, GShard, Switch, GLaM, Mixtral, DeepSeek-MoE) or learned expert-choice (Zhou et al., NeurIPS 2022)** with one principled exception — Soft-MoE (Puigcerver et al., ICLR 2024) — which goes fully dense-soft. Nothing in the canon uses **distance-to-prototype** as a routing signal.

4. **The closest prototype-based routing precedent is FeCAM (Goswami et al., NeurIPS 2023)** — it uses *per-class* Mahalanobis distance with covariance shrinkage, Tukey transform, and correlation normalization, but classifies into a single classifier, not a pool of sub-models. FeCAM's covariance estimation recipe is directly transferable to MoB's execution-cost term.

5. **RanPAC (McDonnell et al., NeurIPS 2023) reports the highest published class-incremental accuracy on CIFAR-100 with ViT-B/16 that I verified: ~92.2%** (vs. CODA-Prompt ~86.3%). RanPAC uses a frozen random projection plus class-prototype accumulation — no gradient updates to the backbone. This sets the realistic "ceiling" MoB must approach on CIFAR-100 before LLM claims are credible.

6. **Kirkpatrick et al. (2017, PNAS) EWC was empirically challenged by Huszár (PNAS 2018) over "multiple penalties."** Online-EWC (Schwarz et al., ICML 2018) uses a *running-sum* Fisher that drops per-task storage to O(1). MoB's current architecture keeps per-expert Fisher — that's the same storage profile as per-task EWC, which is *against* the direction Online-EWC took. Worth re-examining.

7. **Van de Ven & Tolias (Nature Machine Intelligence 2022) formalized the three CL scenarios** — Task-IL, Domain-IL, Class-IL. Their central empirical claim: regularization methods (EWC, SI, MAS) *fail* on Class-IL; only replay or architectural isolation succeeds. MoB is an architectural-isolation method **combined with** regularization; this two-dimensional positioning is novel relative to the taxonomy.

8. **CIFAR-100 CL has two distinct regimes** with very different SOTA numbers: (a) train-from-scratch CNN, 10-task Class-IL, DER++/iCaRL-era peaks reported in the 60–70% range; (b) ViT-B/16 pretrained era, 10-task Class-IL, RanPAC ≈92% average accuracy. MoB's CIFAR-10/100 experiments need to declare which regime they're targeting — Split-MNIST results do not transfer.

9. **Expert collapse is a named, studied failure mode in the MoE canon** (Shazeer 2017 load-balance loss; Switch Transformer z-loss; Fedus/Zoph stability papers). The specific form MoB is hitting — training-time collapse to 1–2 experts at 30–35% accuracy while eval-time routing works — most resembles **mode collapse under a non-differentiable router**, which is exactly the pathology Shazeer 2017's differentiable top-k gate was designed to avoid.

10. **No established paper I located combines {EWC-Fisher forgetting cost} + {Mahalanobis distance execution cost} + {argmin auction routing}.** MoB occupies genuinely unclaimed space. The gap is real; the question is whether the space is unclaimed because it's novel or because prior attempts failed silently.

---

## 2. Axis 1 — Continual Learning Canon

### 2.1 Regularization family

**Elastic Weight Consolidation (EWC) — Kirkpatrick et al., PNAS 2017** (arXiv:1612.00796)
- Quadratic penalty: `L(θ) = L_new(θ) + (λ/2) Σᵢ Fᵢ (θᵢ − θ*ᵢ)²` where F is the diagonal of the empirical Fisher information.
- Motivated as a Laplace approximation to sequential Bayesian posterior updates.
- **Failure modes:** (a) Diagonal Fisher assumption breaks when task gradients have strong off-diagonal structure; (b) Huszár (PNAS 2018) "Note on Quadratic Penalties" showed that stacking per-task Fisher sums is not the posterior-consistent recombination that Kirkpatrick originally claimed — Online-EWC's single running-sum Fisher is the mathematically cleaner form; (c) per-task Fisher storage is O(T × |θ|), linear in task count.
- **MoB relevance:** MoB uses EWC-style Fisher as the *forgetting-cost* bid per expert. MoB's per-expert Fisher is conceptually *per-task* Fisher (one expert per lineage of tasks), so the Huszár critique applies proportionally. The `Fisher clamp min=0.1` fix documented in MEMORY.md is addressing a known EWC pathology — Fisher values are heavily skewed toward zero on most parameters, and division/regularization with un-clamped Fisher is numerically fragile.

**Online EWC / Progress & Compress — Schwarz et al., ICML 2018** (arXiv:1805.06370)
- Replaces per-task Fisher with a running sum: `F_running = γ · F_prev + F_current`, γ<1 emphasizes recent tasks.
- Storage drops from O(T|θ|) to O(|θ|).
- **MoB relevance:** The *right* canonical reference for per-expert Fisher accumulation. MoB's implementation should be described as "Online-EWC per expert" rather than "EWC per expert" to match the established literature.

**Synaptic Intelligence (SI) — Zenke, Poole, Ganguli, ICML 2017** (arXiv:1703.04200)
- Computes parameter importance as the path-integral contribution to loss reduction during training: `ω_i = Σ_t (∂L/∂θᵢ)(Δθᵢ)`.
- Runs *online*, no separate Fisher-computation pass.
- **Failure modes:** Sensitive to learning rate schedules; path-integral approximation is biased when Δθ is large.
- **MoB relevance:** Viable alternative to Fisher for the forgetting-cost term if the Fisher computation becomes a bottleneck at LLM scale. SI computes importance *during* training for free.

**Memory Aware Synapses (MAS) — Aljundi et al., ECCV 2018** (arXiv:1711.09601)
- Unsupervised importance: `Ωᵢ = E_x ||∂||f(x;θ)||² / ∂θᵢ||`. Uses gradient of the *output magnitude*, not the loss, so it needs no labels.
- **MoB relevance:** Interesting for LLM-scale MoB where labeled Fisher computation on pretraining data is expensive; MAS gives an unsupervised importance estimator.

**RWalk (Riemannian Walk) — Chaudhry et al., ECCV 2018** `[uncertain: I recall this from the CL survey canon; venue is ECCV 2018, combines EWC Fisher with SI path-integral; have not re-read the paper to verify the exact combination formula]`
- Combines Fisher-style and path-integral-style importance.

**EBLL (Encoder-Based Lifelong Learning) — Rannen et al., ICCV 2017** `[uncertain: I know the paper exists and is cited in CL surveys as a functional-regularization method using autoencoders per task; have not re-verified specifics]`
- **MoB relevance:** Marginal — autoencoder-per-task is orthogonal to MoB's design.

### 2.2 Replay family

**iCaRL — Rebuffi, Kolesnikov, Sperl, Lampert, CVPR 2017** (arXiv:1611.07725)
- Combines three things: (a) **Nearest-Class-Mean (NCM) classifier** in feature space for test-time prediction; (b) knowledge distillation loss on replay buffer; (c) herding-based exemplar selection.
- NCM means: at inference, assign `y = argmin_c ||φ(x) − μ_c||` where μ_c is the mean of stored exemplars of class c.
- **Failure modes:** Feature drift between tasks invalidates old prototypes; requires exemplar storage (privacy-unfriendly).
- **MoB relevance:** iCaRL's NCM classifier is the *Euclidean* ancestor of MoB's per-expert Mahalanobis prototype. The "which prototype" question (handled by NCM as argmin over class means) is structurally identical to MoB's "which expert" question — except MoB uses per-expert *covariance* and routes to a sub-model, not a class label.

**GEM — Lopez-Paz & Ranzato, NeurIPS 2017** (arXiv:1706.08840)
- Stores memory per task; at each update, projects the gradient onto a cone that does not increase loss on any stored memory: constraint `⟨g_new, g_mem_t⟩ ≥ 0` for all t.
- Quadratic program at every step.
- **Failure modes:** QP cost scales with number of tasks; can get stuck when constraints conflict.

**A-GEM — Chaudhry et al., ICLR 2019** (arXiv:1812.00420)
- Replaces per-task constraints with a single *average* constraint. Much cheaper: one gradient projection per step.
- **MoB relevance:** GEM/A-GEM's constraint `gradient must not increase loss on past tasks` is *exactly* what MoB's forgetting cost encodes — MoB just does it through Fisher rather than through gradient projection. Worth positioning MoB as a "gradient-projection-free" alternative to A-GEM in prose.

**Experience Replay (ER)** — canonical baseline; multiple variations. Robins (1995) is the original cite for rehearsal in neural networks. Chaudhry et al. (arXiv:1902.10486, "Tiny Episodic Memories") is the modern ER baseline.

**DER / DER++ — Buzzega et al., NeurIPS 2020** (arXiv:2004.07211)
- DER: store `(x, logits)` in replay buffer; loss = CE(current) + α · MSE(current_logits, stored_logits). Uses reservoir sampling, no task boundaries needed.
- DER++: adds ground-truth CE on replayed examples for robustness to sudden shifts.
- **Failure modes:** Stored logits can become stale as the network changes dramatically; MSE regularization is weaker than KL on distribution.
- **MoB relevance:** DER++ is the modern "Class-IL without task labels" baseline. It is the standard rehearsal baseline the community compares to. MoB should report against DER++ numbers.

**MER — Riemer et al., ICLR 2019** (arXiv:1810.11910)
- Combines Reptile meta-learning with experience replay. Trains such that replayed-example gradients align with current gradients.
- **MoB relevance:** Orthogonal to MoB's mechanism; cited for completeness.

### 2.3 Parameter isolation family

**Progressive Networks (PNN) — Rusu et al., 2016** (arXiv:1606.04671)
- Add a new column of parameters per task; laterally connect to all prior columns; freeze old columns.
- **Failure modes:** Parameter count grows linearly with tasks; no expert *sharing* across tasks.
- **MoB relevance:** MoB shares PNN's "freeze-and-add" discipline *per expert* but forces experts to multiplex tasks when expert count < task count — the 4-expert/5-task overload problem is precisely the case PNN avoids by design.

**PackNet — Mallya & Lazebnik, CVPR 2018** (arXiv:1711.05769)
- Iterative pruning creates task-specific binary masks on a fixed-size network. After task t: prune least-important weights, retrain remaining weights to recover accuracy, freeze weights used by task t.
- **Failure modes:** Fixed capacity limit; needs pruning heuristic; task identity required at inference.
- **MoB relevance:** MoB's expert-isolation is much softer than PackNet's mask — MoB shares backbone features, isolates only the expert head.

**PathNet — Fernando et al., 2017** (arXiv:1701.08734)
- Evolves routing paths through a multi-module network using tournament selection.
- **MoB relevance:** **PathNet uses evolutionary search, not gradient, to pick routes.** This is the closest established parallel to MoB's non-learned routing, although the mechanism is different (evolution vs. auction).

**HAT (Hard Attention to the Task) — Serra et al., ICML 2018** (arXiv:1801.01423)
- Learns per-task gating masks via a stochastic sigmoid that anneals to hard {0,1} over training.
- Claims "cuts forgetting rates by 45–80%" per verified summary.
- **Failure modes:** Requires task identity at inference; mask learning is sensitive to annealing schedule.
- **MoB relevance:** HAT's masks *are* learned task routers. MoB's pitch is that the auction replaces HAT's mask-learning with a stateless mechanism.

**Piggyback — Mallya et al., ECCV 2018** `[arxiv ID: 1801.06519 — uncertain on exact venue paper count, but confirmed the paper exists]`
- Learns per-task binary masks on a frozen pretrained backbone via a surrogate gradient on the mask threshold.
- **MoB relevance:** Prototype for "freeze backbone, learn routing-like structure per task."

**Supermasks in Superposition — Wortsman et al., NeurIPS 2020** (arXiv:2006.14769) `[uncertain: verifying venue]`
- Finds task-specific subnetworks within a randomly-initialized fixed network — no weight training at all.
- **MoB relevance:** Radical precedent for "routing without training weights." The supermask-per-task idea is selection over fixed computation, analogous to MoB's expert selection over fixed expert weights (after an expert is trained).

### 2.4 Functional regularization

**Learning without Forgetting (LwF) — Li & Hoiem, ECCV 2016 / TPAMI 2017** (arXiv:1606.09282)
- Uses the current model's output on new-task inputs (before training) as a distillation target — no old-task data needed.
- **Failure modes:** Weak protection against large distribution shifts because the distillation signal is computed from the *new* task's inputs, not old data.
- **MoB relevance:** Modest. MoB could use LwF-style distillation *per expert* to smooth prototype drift, but this is mechanism-level speculation.

**LFL (Less-Forgetful Learning) — Jung et al., 2016** `[uncertain: I recall the paper as a feature-space L2 regularizer between old and new networks — have not re-verified details]`

### 2.5 Van de Ven & Tolias scenario taxonomy

**Three Scenarios for Continual Learning — van de Ven & Tolias, arXiv 2019** (arXiv:1904.07734); expanded as **Three Types of Incremental Learning — van de Ven, Tuytelaars, Tolias, Nature Machine Intelligence 2022** (doi:10.1038/s42256-022-00568-3).

| Scenario | Task ID at test | Shared output head? | Example |
|---|---|---|---|
| **Task-IL** | Given | Per-task head | Split-MNIST, told which split |
| **Domain-IL** | Not given | Shared head, same classes | Permuted MNIST |
| **Class-IL** | Not given | Shared head, growing classes | Split-MNIST, not told which split; must distinguish all 10 digits |

**Central empirical claim (verified via the 2022 NMI paper abstract):** regularization methods (EWC, SI, MAS) **fail** on Class-IL; replay is essentially required.

**MoB relevance — critical for positioning:**
- MoB's Split-MNIST experiments with 4 experts, 5 tasks operate in **Task-IL** if you route using task labels (which MoB does not — it routes by bid). If MoB routes by Mahalanobis distance without ever seeing task ID, it is operating in **Class-IL**. This needs to be declared explicitly in MoB's papers.
- MoB is simultaneously a regularization method (EWC on expert) and an architectural method (bid-selected expert). The taxonomy predicts regularization alone fails on Class-IL; MoB's architectural dimension is what should make it viable on Class-IL. The Split-MNIST results *are* the test of this prediction.

---

## 3. Axis 2 — Prototype and Distance-based Continual Learning

### 3.1 The Mahalanobis-OOD lineage

**A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks — Lee et al., NeurIPS 2018** (arXiv:1807.03888)
- Fits a class-conditional Gaussian `p(φ(x)|y=c) = N(μ_c, Σ)` with a **shared (tied) covariance Σ** across classes — this is Linear Discriminant Analysis in feature space.
- OOD score: `max_c −(φ(x) − μ_c)ᵀ Σ⁻¹ (φ(x) − μ_c)`.
- **Tied-covariance is critical** — the shared Σ makes the decision boundary linear and statistically stable with few samples per class.
- **MoB relevance:** The canonical reference for "Mahalanobis distance to class prototypes" in deep features. MoB's current *per-expert* covariance is a step beyond Lee 2018 toward *untied* covariance, which is higher-variance and needs more data per expert. The tradeoff between tied and untied covariance is well-studied in statistics but **not** re-examined for CL routing.

### 3.2 Prototype methods in the pretrained-encoder CL era

**FeCAM (Feature Covariance-Aware Metric) — Goswami et al., NeurIPS 2023** (arXiv:2309.14062)
- Bayes classifier using **anisotropic Mahalanobis distance** — per-class covariance, not tied.
- Three key tricks for making per-class covariance work with few samples:
  1. **Covariance shrinkage:** `Σ̂_c = (1−α) Σ_c + α Σ_shared`.
  2. **Correlation normalization:** rescale Σ_c so diagonal = 1 before inversion.
  3. **Tukey's ladder-of-powers transformation** on features: `φ_new = φ^λ` with λ < 1.
- Reports **70.9% average accuracy on CIFAR-100 5-task Class-IL with a pretrained backbone** (vs. FeTrIL 67.6%).
- Exemplar-free — no replay buffer.
- **MoB relevance: DIRECT.** FeCAM's recipe for per-class covariance with few samples is *exactly* what MoB's per-expert covariance needs. The three tricks (shrinkage, correlation normalization, Tukey) transfer one-to-one. **MoB should adopt the FeCAM covariance estimation recipe as baseline.**

**SLCA (Slow Learner with Classifier Alignment) — Zhang et al., ICCV 2023** (arXiv:2303.05118)
- Observation: the representation layer overfits fast during continual fine-tuning of pretrained models; the classifier head needs the most fixing.
- Fix: dramatically slower LR on the backbone (~1e−4) vs. classifier (~1e−2), plus a post-hoc classifier alignment step that fits a Gaussian per class and resamples to retrain the head.
- Reports **91.53% Last-Acc on Split CIFAR-100 with ImageNet-21K pretrained ViT.**
- **MoB relevance:** SLCA's decoupled LR schedule (slow body, fast head) is a general recipe that maps onto MoB's "freeze backbone + train expert heads" decomposition.

**SimpleCIL — Zhou et al. (surveyed in IJCAI 2024 survey arXiv:2401.16386)**
- Baseline: take a *frozen* pretrained ViT, compute class prototypes in feature space, classify by cosine similarity. No training at all on the CL sequence.
- The IJCAI 2024 survey reports SimpleCIL matches or beats L2P and DualPrompt across several benchmarks.
- **MoB relevance:** SimpleCIL is the "null hypothesis" — with a good enough pretrained encoder, CL reduces to prototype matching. MoB must beat SimpleCIL to justify its complexity.

**ADAM (Adapt And Merge) — Zhou et al., 2023** `[uncertain: I know the method exists and is cited in the IJCAI 2024 survey as a "first-session adaptation" + prototype method; have not verified the exact publication venue]`
- Fine-tune ViT on the first task only, then freeze and use prototypes for the rest.

**RanPAC (Random Projections + Pre-trained Models) — McDonnell et al., NeurIPS 2023** (arXiv:2307.02251)
- Injects a **frozen random projection + nonlinearity** between the pretrained features and the output head.
- Accumulates class prototypes in the projected space and decorrelates them (Gram-matrix-based).
- Reports **92.2% on CIFAR-100** (vs. CODA-Prompt 86.3%). 20–62% relative error reduction across seven CIL benchmarks.
- No rehearsal memory.
- **MoB relevance:** Two specific takeaways. First, the random projection expands dimensionality before prototype matching — directly relevant to MoB scaling to 4096-dim LLM features (projection can *reduce* to a tractable d for covariance). Second, the "decorrelate the prototypes" step is the FeCAM covariance intuition in a different form.

**EASE (Expandable Subspace Ensemble) — Zhou, Sun, Ye, Zhan, CVPR 2024** (arXiv:2403.12030)
- Per-task lightweight adapter creates a task-specific subspace.
- Semantic-guided prototype complement: synthesizes features for old classes without storing old data.
- Ensemble decision across all subspaces at inference.
- **MoB relevance:** EASE's "one adapter per task" is architecturally analogous to "one expert per task" in MoB. The prototype-synthesis trick is specifically designed to combat prototype drift — a problem MoB's online prototype updates are also addressing. Worth reading EASE carefully.

### 3.3 The prototype-drift problem (canonical)

**Prototype drift** = as the backbone continues to learn, old class prototypes (computed in old feature space) no longer match current feature distributions.

Two canonical mitigations:
1. **Freeze the backbone** (SimpleCIL, ADAM, FeCAM, RanPAC) — trivially solves drift by refusing to update.
2. **Replay-based drift correction** (iCaRL + its successors) — store exemplars, recompute prototypes.

A third emerging mitigation is **semantic feature synthesis** (EASE) — generate proxy features for old classes without storing them.

**MoB's online prototype update (per MEMORY.md) is a fourth approach** — continuously rebase prototypes from the live training stream. This is **not** in the established canon as a standalone mitigation. Closest precedent: exponential moving average prototypes in FedAvg-style federated continual learning. `[uncertain: I cannot cite a specific peer-reviewed paper that does exactly this for single-model CL.]`

### 3.4 Summary — which-prototype-to-use question

| Method | Covariance choice | Prototype source | Drift mitigation |
|---|---|---|---|
| NCM/iCaRL 2017 | Isotropic (Euclidean) | Exemplar mean | Replay |
| Lee 2018 OOD | Tied Σ across classes | Class mean in features | N/A (static) |
| FeCAM 2023 | Per-class Σ, shrunk + normalized | Class mean in frozen features | Freeze backbone |
| SLCA 2023 | Per-class Σ | Class mean, Gaussian resample for head | Slow backbone LR |
| RanPAC 2023 | Per-class, in random-projection space | Class mean, decorrelated | Freeze backbone |
| EASE 2024 | Subspace-specific | Synthesized for old classes | Per-task adapter |
| **MoB** | **Per-expert Σ** (currently) | **Expert-mean in pen layer** | **Online updates** (not canonical) |

The MoB cell's "per-expert" dimension is novel because experts are not the same unit as classes or tasks. The coverage question "which expert owns which input" has no established prototype-method answer.

---

## 4. Axis 3 — MoE Routing Canon

### 4.1 Learned-router lineage

**Sparsely-Gated Mixture-of-Experts — Shazeer et al., ICLR 2017** (arXiv:1701.06538)
- Per-token gate `G(x) = Softmax(TopK(x · W_g + Noise))`.
- **Load-balance loss:** `L_balance = w · CV(importance)²` where importance_e = Σ_x G_e(x). This forces roughly equal expert usage.
- **Failure modes documented in the paper itself:** expert imbalance, router collapse (one expert always wins). The load-balance loss is specifically engineered against these.
- **MoB relevance:** Shazeer 2017 is the reference that says loud and clear: **without a load-balance term, MoE routers collapse.** MoB's training-time collapse to 1–2 experts is exactly this known pathology. The MoB thesis is that a *non-learned* router cannot collapse in the same way — but an auction with a *learned Fisher term* can collapse if one expert's Fisher stays low forever.

**GShard — Lepikhin et al., ICLR 2021** (arXiv:2006.16668)
- Top-2 token-choice routing with a **capacity factor** (each expert has a fixed token-capacity; excess tokens are dropped/overflowed).
- Load-balance via Shazeer-style auxiliary loss.
- 600B-parameter multilingual MT model.
- **Failure modes:** Token dropping under load imbalance; communication costs for all-to-all dispatch.

**Switch Transformer — Fedus, Zoph, Shazeer, JMLR 2022** (arXiv:2101.03961)
- Top-1 routing (simpler than GShard's top-2).
- Introduces **router z-loss** for training stability.
- Scales to 1.6T parameters.
- **Failure modes documented:** router instability, requires bf16/selective fp32 to train.
- **MoB relevance:** Switch established that **top-1** per-token routing is viable if you carefully stabilize the router. MoB's winner-take-all is top-1 by design — so MoB inherits Switch's failure modes (instability, collapse) even though its router is different.

**GLaM — Du et al., ICML 2022** (arXiv:2112.06905)
- 1.2T parameters, 64 experts per layer, 32 MoE layers, top-2 routing.
- Demonstrated that MoE LLMs match dense LLM quality at ~1/3 the training energy.
- **MoB relevance:** GLaM is the scale reference. For MoB to claim LLM-scale relevance, its per-token routing compute cost must be competitive with GLaM's learned top-2 gate.

### 4.2 Non-learned and alternative routers — the key references for MoB

**Hash Layers — Roller et al., NeurIPS 2021** (arXiv:2106.04426)
- **No learned router.** Each input token is mapped to an expert by a deterministic hash function on the token ID.
- "Balanced and random hashes focused on the most local features work best."
- Matches or beats Switch Transformer on perplexity (23.16 vs. 23.65 at 1×64 experts; 22.89 vs. 23.52 at 1×128). `[verified via search summary]`
- No load-balance loss needed.
- **MoB relevance: CRITICAL PRECEDENT.** Hash Layers is the published existence proof that non-learned routing works at LLM scale. MoB is philosophically closer to Hash Layers than to Switch. Specifically:
  - Hash Layers: route by static hash of input ID
  - MoB: route by dynamic auction on features
  - Both: no learned router parameters; both forgo load-balance losses
- The Roller 2021 paper is the natural entry point for positioning MoB against the MoE community.

**BASE Layers — Lewis et al., ICML 2021** (arXiv:2103.16716)
- Formulates routing as a **linear assignment problem** solved by **Bertsekas (1992) auction algorithm**.
- Perfect load balance by construction (assignment is constrained equal per expert).
- No auxiliary load-balance loss, no new hyperparameters.
- Matches Switch on quality; better than Switch on validation perplexity per verified summary.
- **MoB relevance: THE MOST DIRECT PRECEDENT.** BASE literally runs an auction. The differences:
  - BASE uses learned token and expert embeddings; auction scores are inner products.
  - MoB uses Mahalanobis distance to prototypes + Fisher overlap; scores are statistics, not learned parameters.
  - BASE imposes perfect per-expert load balance (combinatorial constraint).
  - MoB imposes no explicit load constraint.
- A proper MoB paper must cite BASE as the auction-routing precedent and explicitly contrast the "learned vs. statistical score" axis.

**Expert Choice Routing — Zhou et al., NeurIPS 2022** (arXiv:2202.09368)
- Inverts the token-choice perspective: each expert picks its top-k tokens from the batch.
- **Perfect load balance by construction** (each expert gets exactly k tokens).
- Reports 2× training speedup over Switch/GShard.
- **Failure modes:** requires batched dispatch — breaks in streaming/autoregressive decode settings; some tokens may be picked by no expert.
- **MoB relevance:** Expert Choice is the closest learned-router competitor to BASE Layers. Load balance is achieved by *selection mechanism design*, not by loss. MoB could in principle run in an "expert-choice" mode — each expert bids on the top-k batch items it wants — but this is speculation.

**Soft-MoE — Puigcerver et al., ICLR 2024** (arXiv:2308.00951)
- Fully differentiable: **weighted combinations of all tokens** are passed to each expert; outputs are re-weighted back.
- No discrete routing decision, no token dropping, no load-balance loss.
- Reports 40× parameter scaling of ViT with 2% inference-time overhead.
- **Failure modes:** Not sparse at inference in the hard-routing sense; memory grows with #experts × #tokens.
- **MoB relevance:** Soft-MoE is the *opposite* design philosophy — keep routing learned and differentiable by going soft. MoB and Soft-MoE are the two axes of "non-standard MoE routing" in 2023–2024.

**Mixtral of Experts — Jiang et al., 2024** (arXiv:2401.04088)
- Open-weights 8-expert Mistral-7B-based SMoE, top-2 routing per token.
- Industrial validation of standard top-2 learned routing at open-source scale.
- **MoB relevance:** Mixtral is the production-deployment reference. If MoB claims to replace learned routers, Mixtral's routing cost and quality are the baseline to beat at 7B-parameter scale.

**DeepSeek-MoE — Dai et al., ACL 2024** (arXiv:2401.06066)
- Two new techniques: (a) **fine-grained expert segmentation** (split each expert's FFN into m smaller ones, activate m·K of them); (b) **shared-expert isolation** (a subset of experts always active, capture common knowledge).
- DeepSeekMoE-16B matches LLaMA-7B at ~40% compute.
- **MoB relevance:** Fine-grained experts is orthogonal to routing mechanism and composable with MoB. Shared-expert isolation is an *anti-collapse* mechanism: even if routed experts collapse, the shared expert always fires. This is a published technique MoB could adopt to mitigate its training-time collapse symptom.

### 4.3 MoE failure-mode taxonomy (synthesized from the canon)

| Failure | Canonical reference | Mechanism | MoB exposure |
|---|---|---|---|
| **Expert collapse** (one expert wins all tokens) | Shazeer 2017 | Positive feedback in softmax router | **Yes — current open problem** |
| **Load imbalance** (some experts rarely used) | Shazeer 2017; GShard | Unequal scoring → capacity dropping | Yes |
| **Router instability** (logit blowup) | Switch Transformer z-loss | Unbounded routing logits | Less direct (auction has no learned logits) |
| **Expert specialization redundancy** | DeepSeek-MoE | Multiple experts learn near-duplicate functions | Possible |
| **Token dropping** under capacity constraints | GShard | Capacity factor < 1 | Not applicable in current MoB (no capacity constraint) |

**None of these are unique to learned routers.** MoB's training-time collapse belongs to the "expert collapse" row. What Hash Layers showed (NeurIPS 2021) is that *deterministic non-learned* routers don't collapse because the input→expert map is fixed. MoB is neither — the Mahalanobis score is data-dependent but not learned — so it sits in a new cell of this table and needs a specific mitigation.

### 4.4 Summary table — is there anything non-learned or auction-like?

| Method | Router type | Load balance | Auction-like? |
|---|---|---|---|
| Shazeer 2017 | Learned softmax+TopK | Auxiliary loss | No |
| GShard 2020 | Learned softmax, Top-2 | Aux loss + capacity | No |
| Switch 2021 | Learned softmax, Top-1 | Aux loss + z-loss | No |
| GLaM 2022 | Learned softmax, Top-2 | Aux loss | No |
| BASE 2021 | Learned scores, **linear-assignment auction** | By construction | **Yes — Bertsekas auction** |
| Hash Layers 2021 | **Static hash, no parameters** | By hash balance | No (but non-learned) |
| Expert Choice 2022 | Learned expert-side top-k | By construction | Partial (experts "pick") |
| Soft-MoE 2024 | Learned soft weights | By construction (dense) | No |
| Mixtral 2024 | Learned softmax, Top-2 | Aux loss | No |
| DeepSeek-MoE 2024 | Learned softmax + shared expert | Aux loss | No |
| **MoB** | **Statistical bid (Mahalanobis + Fisher), argmin** | **None explicit** | **Yes (argmin auction)** |

Only BASE Layers uses a literal auction algorithm. Only Hash Layers has no learned router. MoB is the *intersection* — statistical scores + argmin selection — and that cell is empty in the published literature.

---

## 5. Axis 4 — CIFAR-100 CL Benchmarks

### 5.1 Standard protocols

**Split-CIFAR-100 task structures:**
- **5 tasks × 20 classes** — common in the prompt-based / pretrained-backbone era.
- **10 tasks × 10 classes** — most-cited split; classical DER++/iCaRL benchmarks.
- **20 tasks × 5 classes** — stress-test split for longer sequences.
- **50-base + N×incremental** (50 initial classes, then increments of 5 or 10) — "B0/B50 Inc5/Inc10" protocols popularized by PODNet and subsequent works.

**Protocols:**
- **Class-IL**: standard default; no task ID at test.
- **Task-IL**: task ID given at test (generally considered too easy for CIFAR-100).
- **GCIL (Generalized Class-IL)**: allows class *reappearance* across tasks, imbalanced counts. Mi et al. 2020 `[uncertain: checking exact venue]`.
- **Domain-IL**: rarely used for CIFAR-100 because the 100 classes are semantically fixed.

### 5.2 Canonical metrics

- **Average Accuracy (A_T)**: mean accuracy across all tasks at end of sequence. Primary metric.
- **Last-Step Accuracy / Final Accuracy**: accuracy at end of training (equivalent to A_T in most papers).
- **Average Forgetting (F_T)**: mean over tasks of `max_{t' < T} acc(t, t') − acc(t, T)`. Chaudhry et al. ECCV 2018 (RWalk paper) formalized this.
- **Backward Transfer (BWT)**: Lopez-Paz & Ranzato NeurIPS 2017 — how learning new tasks affects old. Negative = forgetting; positive = constructive transfer.
- **Forward Transfer (FWT)**: Lopez-Paz & Ranzato — zero-shot performance on future tasks.
- **Intransigence**: Chaudhry et al. 2018 — how hard it is to learn new tasks given past constraints.

### 5.3 Representative SOTA on CIFAR-100 — pretrained ViT-B/16 era

Numbers below are from verified search summaries or widely-cited paper abstracts. Where I could not independently confirm the exact number, I mark `[paper-reported, unverified by me]`.

| Method | Split | Backbone | Avg Acc (%) | Last Acc (%) | Exemplars? | Paper / Venue |
|---|---|---|---|---|---|---|
| L2P | 10×10 | ViT-B/16 IN-21K | ~83 | ~83 | No | Wang et al., CVPR 2022 [paper-reported, unverified by me] |
| DualPrompt | 10×10 | ViT-B/16 IN-21K | ~86 | — | No | Wang et al., ECCV 2022 [paper-reported, unverified by me] |
| CODA-Prompt | 10×10 | ViT-B/16 IN-21K | 86.3 | — | No | Smith et al., CVPR 2023 (verified via RanPAC's reported comparison) |
| SLCA | Split-100 (various) | ViT-B/16 IN-21K | — | **91.53** | No | Zhang et al., ICCV 2023 (verified) |
| FeCAM | 5×20 | ViT-B/16 IN-21K | **70.9** | — | No | Goswami et al., NeurIPS 2023 (verified) |
| RanPAC | Split-100 | ViT-B/16 IN-21K | **92.2** | — | No | McDonnell et al., NeurIPS 2023 (verified) |
| EASE | Split-100 | ViT-B/16 IN-21K | — | — | No | Zhou et al., CVPR 2024 [numbers in paper, unverified by me] |

**Important caveat on FeCAM vs. RanPAC numbers:** FeCAM's reported 70.9% is on 5 tasks with a *non-identical* protocol; RanPAC's 92.2% uses the full Split-CIFAR-100 with a 10-task protocol. Directly comparing these numbers without reading both protocols carefully is misleading. The *broad* message is that the 2023 prototype-based methods with frozen pretrained backbones sit in the high-80s to low-90s on CIFAR-100 class-IL.

### 5.4 Representative SOTA — train-from-scratch CNN era (pre-ViT regime)

`[Numbers in this section are from my general memory of the DER++ and iCaRL literature; I have not re-verified specific numbers in this session and mark them uncertain.]`

| Method | Split | Backbone | Avg Acc (%) | Exemplars |
|---|---|---|---|---|
| Naive (joint upper bound) | 10×10 | ResNet-32 | ~70 | N/A |
| iCaRL | 10×10 | ResNet-32 | ~50 | 2000 per-class buffer [uncertain] |
| DER++ | 10×10 | ResNet-32 | ~65 | 500-example buffer [uncertain] |
| GEM / A-GEM | 10×10 | ResNet-32 | ~55 | [uncertain] |
| LwF | 10×10 | ResNet-32 | ~30 | None |
| EWC | 10×10 | ResNet-32 | ~25 | None |

**Lesson:** from-scratch Class-IL on CIFAR-100 is dramatically harder than pretrained-ViT Class-IL. A ~40-point gap separates the two regimes.

### 5.5 What MoB's CIFAR experiments need to declare

1. **Regime**: train-from-scratch CNN or pretrained ViT-B/16? Different literature, different baselines.
2. **Task split**: 5/10/20 — affects comparability.
3. **Protocol**: Class-IL with no task ID at test, or Task-IL?
4. **Metric choice**: Avg Acc *and* Forgetting (both required; Avg Acc alone is insufficient per community convention).
5. **Baseline set**: at minimum DER++ and SimpleCIL. If pretrained ViT, add FeCAM and RanPAC. If prompt methods are in scope, add L2P and DualPrompt.

---

## 6. Gaps in the Canon — What the Established Literature Does Not Answer for MoB

1. **Distance-based routing at MoE scale is unstudied.** The MoE canon uses learned dot-product or hash; the CL prototype canon uses distance-based classification but not distance-based *sub-model selection*. MoB sits at this intersection and has no established reference for "how does Mahalanobis-routed MoE behave during training at 4096-dim features."

2. **Per-expert Fisher accumulation is an unresolved regularization pattern.** EWC (Kirkpatrick 2017) uses per-task Fisher; Online-EWC (Schwarz 2018) uses a running sum. MoB uses per-expert Fisher — which is neither. There is no published analysis of how expert-owned Fisher interacts with *online* prototype updates.

3. **Training-time vs. inference-time routing asymmetry is not a studied failure mode.** The MoE canon assumes training and inference routing are the same function (either learned-gate-trained-and-used, or hash-fixed). MoB's current open problem — training-time collapse despite eval-time routing working — does not match any documented MoE failure mode I know of. This may be a novel pathology specific to *statistical-score* routing.

4. **Expert multiplexing when #experts < #tasks is architecturally understudied.** PNN avoids it; PackNet prevents it; Mixtral has no task concept. MoB's intentional 4:5 ratio creates a "must-overload-one-expert" constraint that the published continual-learning-with-MoE literature does not explicitly analyze `[uncertain: may exist in recent 2024 workshop papers I did not catch]`.

5. **Covariance estimation at LLM-layer scale (d=4096) has no published MoB-analog.** FeCAM demonstrates per-class Σ with d~768 (ViT-B); scaling per-expert Σ to 4096 or 12288 (GPT-3-scale hidden dim) requires low-rank factorization that FeCAM does not examine. The statistics literature (Ledoit-Wolf shrinkage 2004; Graphical Lasso 2008) is the relevant reference set, but it is not in the MoE or CL canon.

6. **Prototype drift mitigation by online updates lacks peer-reviewed analysis.** MoB's online prototype update is plausible but not in the established canon as a standalone CL mechanism. Closest references (iCaRL exemplar replay, EASE feature synthesis) take different approaches.

7. **Auction-algorithm routing beyond linear assignment (BASE Layers) is unexplored.** BASE uses the Bertsekas 1992 auction on learned scores. MoB uses argmin on unlearned statistical scores. The broader auction-theory literature (VCG, second-price, combinatorial auctions) has not been applied to MoE routing in any paper I located. This is genuinely open territory.

---

## 7. Full Reference List

### Continual Learning — Regularization
- Kirkpatrick, J. et al. "Overcoming catastrophic forgetting in neural networks." *PNAS* 114(13):3521–3526, 2017. arXiv:1612.00796.
- Schwarz, J. et al. "Progress & Compress: A scalable framework for continual learning." *ICML* 2018. arXiv:1805.06370.
- Zenke, F., Poole, B., Ganguli, S. "Continual Learning Through Synaptic Intelligence." *ICML* 2017. arXiv:1703.04200.
- Aljundi, R. et al. "Memory Aware Synapses: Learning what (not) to forget." *ECCV* 2018. arXiv:1711.09601.
- Chaudhry, A. et al. "Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence." *ECCV* 2018. `[venue verified, full arxiv ID unverified in this session]`
- Rannen, A. et al. "Encoder Based Lifelong Learning." *ICCV* 2017. `[uncertain]`
- Huszár, F. "Note on the quadratic penalties in elastic weight consolidation." *PNAS* 2018. arXiv:1712.03847.

### Continual Learning — Replay
- Rebuffi, S.-A., Kolesnikov, A., Sperl, G., Lampert, C. H. "iCaRL: Incremental Classifier and Representation Learning." *CVPR* 2017. arXiv:1611.07725.
- Lopez-Paz, D., Ranzato, M. "Gradient Episodic Memory for Continual Learning." *NeurIPS* 2017. arXiv:1706.08840.
- Chaudhry, A. et al. "Efficient Lifelong Learning with A-GEM." *ICLR* 2019. arXiv:1812.00420.
- Buzzega, P. et al. "Dark Experience for General Continual Learning: a Strong, Simple Baseline." *NeurIPS* 2020. arXiv:2004.07211.
- Riemer, M. et al. "Learning to Learn without Forgetting by Maximizing Transfer and Minimizing Interference." *ICLR* 2019. arXiv:1810.11910.
- Chaudhry, A. et al. "On Tiny Episodic Memories in Continual Learning." arXiv:1902.10486.

### Continual Learning — Parameter Isolation
- Rusu, A. A. et al. "Progressive Neural Networks." arXiv:1606.04671, 2016.
- Mallya, A., Lazebnik, S. "PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning." *CVPR* 2018. arXiv:1711.05769.
- Fernando, C. et al. "PathNet: Evolution Channels Gradient Descent in Super Neural Networks." arXiv:1701.08734, 2017.
- Serra, J. et al. "Overcoming Catastrophic Forgetting with Hard Attention to the Task." *ICML* 2018. arXiv:1801.01423.
- Mallya, A., Davis, D., Lazebnik, S. "Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights." *ECCV* 2018. arXiv:1801.06519.
- Wortsman, M. et al. "Supermasks in Superposition." *NeurIPS* 2020. arXiv:2006.14769. `[venue uncertain]`

### Continual Learning — Functional Regularization
- Li, Z., Hoiem, D. "Learning without Forgetting." *ECCV* 2016 / *TPAMI* 2017. arXiv:1606.09282.
- Jung, H. et al. "Less-Forgetful Learning for Domain Expansion in Deep Neural Networks." `[uncertain venue/ID]`

### Continual Learning — Taxonomy
- van de Ven, G. M., Tolias, A. S. "Three scenarios for continual learning." arXiv:1904.07734, 2019.
- van de Ven, G. M., Tuytelaars, T., Tolias, A. S. "Three types of incremental learning." *Nature Machine Intelligence* 4:1185–1197, 2022. doi:10.1038/s42256-022-00568-3.

### Prototype / Distance-based CL
- Lee, K., Lee, K., Lee, H., Shin, J. "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." *NeurIPS* 2018. arXiv:1807.03888.
- Goswami, D., Liu, Y., Twardowski, B., van de Weijer, J. "FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning." *NeurIPS* 2023. arXiv:2309.14062.
- Zhang, G. et al. "SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model." *ICCV* 2023. arXiv:2303.05118.
- McDonnell, M. D., Gong, D., Parveneh, A., Abbasnejad, E., van den Hengel, A. "RanPAC: Random Projections and Pre-trained Models for Continual Learning." *NeurIPS* 2023. arXiv:2307.02251.
- Zhou, D.-W., Sun, H.-L., Ye, H.-J., Zhan, D.-C. "Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning (EASE)." *CVPR* 2024. arXiv:2403.12030.
- Zhou, D.-W., Sun, H.-L. et al. "Continual Learning with Pre-Trained Models: A Survey." *IJCAI* 2024. arXiv:2401.16386. (Discusses SimpleCIL and ADAM baselines.)

### MoE Routing Canon
- Shazeer, N. et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR* 2017. arXiv:1701.06538.
- Lepikhin, D. et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." *ICLR* 2021. arXiv:2006.16668.
- Fedus, W., Zoph, B., Shazeer, N. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *JMLR* 23:1–40, 2022. arXiv:2101.03961.
- Du, N. et al. "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts." *ICML* 2022. arXiv:2112.06905.
- Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., Zettlemoyer, L. "BASE Layers: Simplifying Training of Large, Sparse Models." *ICML* 2021. arXiv:2103.16716.
- Roller, S., Sukhbaatar, S., Szlam, A., Weston, J. "Hash Layers For Large Sparse Models." *NeurIPS* 2021. arXiv:2106.04426.
- Zhou, Y. et al. "Mixture-of-Experts with Expert Choice Routing." *NeurIPS* 2022. arXiv:2202.09368.
- Puigcerver, J., Riquelme, C., Mustafa, B., Houlsby, N. "From Sparse to Soft Mixtures of Experts." *ICLR* 2024. arXiv:2308.00951.
- Jiang, A. Q. et al. "Mixtral of Experts." arXiv:2401.04088, 2024.
- Dai, D. et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models." *ACL* 2024. arXiv:2401.06066.

### Auction-algorithm foundation (cited by BASE Layers)
- Bertsekas, D. P. "Auction algorithms for network flow problems: A tutorial introduction." *Computational Optimization and Applications* 1(1):7–66, 1992.

### Covariance estimation (relevant but outside the CL/MoE canon)
- Ledoit, O., Wolf, M. "A well-conditioned estimator for large-dimensional covariance matrices." *J. Multivariate Analysis* 88(2):365–411, 2004.
- Friedman, J., Hastie, T., Tibshirani, R. "Sparse inverse covariance estimation with the graphical lasso." *Biostatistics* 9(3):432–441, 2008.

---

*End of established-literature map. Further work should survey (a) 2024–2025 continual-MoE papers specifically, (b) auction-theory applications beyond BASE Layers, (c) low-rank covariance methods for the 4096-dim scaling question. Those three directions are not in the established canon and require a separate survey scoped to emerging work.*
