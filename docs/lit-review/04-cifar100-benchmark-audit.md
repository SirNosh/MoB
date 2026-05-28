# CIFAR-100 as a Continual-Learning Benchmark — Audit for MoB

**Author:** Cypher (data-centric audit)
**Date:** 2026-04-18
**Scope:** Map the community's use of CIFAR-100 for CL. Do not prescribe MoB experiments yet.
**Confidence convention:** `[HIGH]` = multiple primary sources agree, `[MED]` = one primary source + survey, `[LOW]` = folklore / not directly verified in this audit.

---

## 1. Executive Summary

- CIFAR-100 in 2025 is **not one benchmark**; it is a family of at least six distinct protocols (Split-CIFAR-100-10T / 20T / 5T, B0, B50, and GCIL). Reviewers check which one you ran. `[HIGH]`
- The **10-task Split-CIFAR-100 (10 classes/task)** is the de facto default in the pretrained-ViT era; **B0/B50 with 5–10 classes per step** is the default in the rehearsal-CNN era. MoB must report on at least one of each. `[HIGH]`
- The **CIFAR-100 test set contains ~10% near-duplicates** with training images (ciFAIR finding). Published gains of <~1% on the standard test set are statistically meaningless without the ciFAIR retest. `[HIGH]`
- Using **ViT pretrained on ImageNet-21k** is essentially the default backbone, and it is semantically adjacent to CIFAR-100 (shared animal/vehicle/household taxonomies). This is a **leakage confound** that the community tolerates but reviewers increasingly ask you to acknowledge. `[HIGH]`
- The **NCM-on-frozen-ViT baseline (SimpleCIL)** and **Mahalanobis-prototype baseline (FeCAM)** are the bars to clear. FeCAM is **architecturally identical** to MoB's v2 routing module (Mahalanobis to class prototypes). MoB must be prepared to explain what the auction buys on top of FeCAM. `[HIGH]`
- **32x32 native resolution** is an active choice: frozen-ViT protocols upsample to 224x224 and accept the interpolation bias; train-from-scratch ResNet protocols keep 32x32 with a modified stem (ResNet-32). Mixing them in one table is a methodological error. `[HIGH]`
- **Task-IL vs Class-IL confusion** is the single most common cheese move in the literature: Class-IL is the hard setting, Task-IL gives task-ID at test time. MoB's auction is stateless, so Class-IL is the natural setting — report it as such. `[HIGH]`
- Mandatory metrics: **final Average Accuracy (A_T)** and **Average Forgetting (F_T)**. Diagnostic but common: BWT, FWT, per-task trajectory, intransigence. `[HIGH]`
- Memory-buffer size is the **single biggest apples-to-oranges axis** in replay comparisons. Canonical buffers: 500, 2000, 5120 exemplars for CIFAR-100. `[HIGH]`
- With 4 experts, the cleanest 5-task mapping of CIFAR-100 is **5 tasks x 20 classes** or **4 tasks x 25 classes + held-out**; both are non-standard. See §6 and Open Questions. `[MED]`

---

## 2. Dataset Composition and Known Biases

### 2.1 Structure

- **Size:** 60,000 color images total; 50,000 train / 10,000 test. `[HIGH]`
- **Classes:** 100 fine labels, 600 images/class (500 train + 100 test). Perfectly class-balanced in both splits. `[HIGH]`
- **Superclasses:** 100 fine classes group into **20 coarse superclasses** (5 fine classes per superclass). E.g., superclass "fish" = {aquarium_fish, flatfish, ray, shark, trout}. `[HIGH]`
- **Resolution:** 32x32x3, uint8. `[HIGH]`
- **Source:** Subset of Tiny Images (which was retracted for offensive content, but CIFAR-100 remains a clean curated subset). `[MED]`

### 2.2 Known data-quality issues

- **Test/train duplication (ciFAIR):** Barz & Denzler (2020) found **10% of the CIFAR-100 test set has near-duplicates in the training set**. They released **ciFAIR-100**, a duplicate-free retest set. Few CL papers use it. This is a pervasive bias that inflates every reported accuracy number by a small but non-zero amount. `[HIGH]`
- **Label noise (CIFAR-100N):** Wei et al. (2021; arXiv 2110.12088) re-annotated CIFAR-100 via Mechanical Turk. **Real-world human label noise on CIFAR-100 is ~40.2% noise rate at the fine level, ~25.6% at the coarse level** (numbers from their paper; verify before citing). Noise is **instance-dependent**, not class-dependent, which is the harder case for CL because forgetting signals can be confounded with label ambiguity. `[MED — numbers need direct verification]`
- **Superclass semantic overlap:** Fine classes within one superclass are genuinely visually similar (e.g., "boy" vs "girl" vs "man" vs "woman" vs "baby" all in "people"). This **artificially inflates forgetting** in random-ordered Class-IL splits: if task T_0 holds "boy" and task T_4 holds "girl", the representation naturally drifts toward whichever is last-seen. Semantic-ordered splits behave very differently from random-ordered splits. `[HIGH]`

### 2.3 32x32 resolution interacting with pretrained encoders

- Modern CL protocols with **ViT-B/16 pretrained on ImageNet-21k** expect 224x224 input. The standard workaround is **bilinear upsample 32 -> 224**, a 7x blow-up. `[HIGH]`
- **Upsampling bias (under-studied):** At 224x224 a CIFAR image is 49x blurrier than a native ImageNet image. The ViT is operating far outside its training distribution on texture, not on semantics — but it still works because ImageNet-21k class semantics generalize. This is often unstated in papers. `[MED]`
- **Implication for prototype methods:** Mahalanobis distance in ViT feature space on upsampled CIFAR-100 may be dominated by upsampling artifacts rather than semantic content. Worth probing as an ablation (cf. MoB v2 routing). `[MED]`

---

## 3. Standard CL Protocols on CIFAR-100

Six protocols dominate. They are **not interchangeable**.

### 3.1 Split-CIFAR-100 (equal-split, no base)

| Variant | # Tasks | Classes/Task | Typical era |
|---------|---------|--------------|-------------|
| **5-task** | 5 | 20 | Older ResNet papers, task-IL |
| **10-task** (default) | 10 | 10 | Current default; used by L2P, DualPrompt, CODA-Prompt, SLCA |
| **20-task** | 20 | 5 | Harder; used for stress tests |
| **50-task** | 50 | 2 | Rare; reported in HiDe-Prompt and 2PCL for stress |

- **Class order** is the shared random seed `[0..99]` shuffled with a fixed seed (commonly seed=1993 in L2P/DualPrompt repositories). Always report the seed. `[HIGH]`
- **Heads:** single head for Class-IL, per-task heads for Task-IL. `[HIGH]`

### 3.2 CIFAR-100-B0 and B50 (base + increments)

- **B0 (base 0):** Start from zero knowledge. Train all 100 classes gradually, in steps of 5, 10, or 20 classes. **Fixed memory buffer of 2000 exemplars** is the canonical setting in iCaRL, FOSTER, DER. `[HIGH]`
- **B50 (base 50):** Start by training 50 classes offline (the "base"). Then incrementally add the remaining 50 classes, in steps of 2, 5, or 10 classes. **20 exemplars per class** is the canonical memory budget. `[HIGH]`
- **Why both exist:** B0 tests "cold-start" continual learning (no pretrained head). B50 tests "continual fine-tuning" of an already-well-initialized model. B50 is closer to the practical LLM-continual-learning scenario. `[MED]`

### 3.3 GCIL (Generalized Class-IL, Mi et al. 2020)

- **Premise:** Real data streams don't arrive in tidy disjoint task boundaries. Classes **repeat** across phases with **varying frequencies**. Mi et al. use 20 phases of 1000 images each, with class appearance governed by a stochastic process. `[HIGH]`
- **Why it matters for MoB:** MoB assumes task boundaries exist (Fisher updates are triggered at boundaries in the task-aware variant). GCIL stress-tests that assumption. The continual-MoB variant (shift-detection) is philosophically aligned with GCIL. `[MED]`
- **Adoption:** GCIL on CIFAR-100 is cited in ~20–30% of recent papers but not in the prompt-based-ViT canon. It is **the benchmark reviewers use to test whether your method is boundary-free**. `[MED]`

### 3.4 Online / single-epoch protocols

- **Online CIFAR-100:** Each sample seen once. Used by SCR, OCM, online-CL papers. Accuracy is 20–40% lower than offline multi-epoch on the same method. `[MED]`
- **Offline multi-epoch** is the canonical setting. Typical epoch budget: 50–100 epochs per task for ResNet-from-scratch; 5–20 epochs per task for ViT fine-tuning variants. `[HIGH]`

### 3.5 Three-scenarios taxonomy (van de Ven & Tolias, 2022)

The nature.com paper formalized three scenarios (Nature Machine Intelligence, Mar 2022):

| Scenario | Task-ID at test? | Output head | Difficulty |
|----------|------------------|-------------|------------|
| **Task-IL** | Yes | Multi-head (per task) | Easiest |
| **Domain-IL** | No | Single head, same classes each task | Medium |
| **Class-IL** | No | Single head, new classes each task | Hardest |

- On CIFAR-100, the **default now is Class-IL**. Task-IL numbers on CIFAR-100 are only legible when explicitly labeled. `[HIGH]`

### 3.6 Which protocol is the "real" 2025 benchmark?

- **Pretrained-ViT era (dominant since 2022):** **10-task Class-IL Split-CIFAR-100 with ViT-B/16 (ImageNet-21k pretrained)** is the default. L2P / DualPrompt / CODA-Prompt / SLCA / HiDe-Prompt / RanPAC / EASE / SimpleCIL all report this setting. Numbers cluster in **80–92% final average accuracy**. `[HIGH]`
- **From-scratch CNN era (still published for fairness):** **CIFAR-100-B0 with ResNet-32, 10 steps, 2000-exemplar memory** (iCaRL-style) or **CIFAR-100-B50 with ResNet-18, 10 steps** (DER-style). Numbers cluster in **50–75% final average accuracy**. `[HIGH]`

---

## 4. Evaluation Metrics Convention

Let `a_{i,j}` = accuracy on task `i` after training on task `j`. `T` = total number of tasks.

### 4.1 Mandatory (must report)

- **Final Average Accuracy:** `A_T = (1/T) * sum_i a_{i,T}`. Primary headline metric. `[HIGH]`
- **Average Forgetting (Chaudhry et al., 2018):** `F_T = (1/(T-1)) * sum_{i<T} max_{j<T}(a_{i,j}) - a_{i,T}`. Measures peak-minus-final per task. `[HIGH]`

### 4.2 Strongly recommended (diagnostic, common)

- **Backward Transfer (BWT, Lopez-Paz & Ranzato 2017):** `BWT = (1/(T-1)) * sum_{i<T} (a_{i,T} - a_{i,i})`. Signed; negative = forgetting, positive = consolidation. Different quantity from `F_T`. `[HIGH]`
- **Forward Transfer (FWT):** `FWT = (1/(T-1)) * sum_{i>1} (a_{i,i-1} - b_i)` where `b_i` is random-init accuracy. Rarely reported on CIFAR-100 because baseline `b_i` is noisy. `[MED]`
- **Per-task trajectory:** plot `a_{i,j}` for each task `i` over `j = i..T`. Diagnostic for forgetting dynamics. `[HIGH]`

### 4.3 Specialized (nice-to-have)

- **Intransigence (Chaudhry RWalk, 2018):** difference between joint-training accuracy and continual accuracy on the most recent task. Measures *how much new-task performance is sacrificed*. `[MED]`
- **Stability gap (Mattdl et al., ICLR 2023):** transient accuracy drop at the start of each new task, measured with fine-grained in-task evaluation. Growing in adoption. `[MED]`
- **Average Incremental Accuracy (iCaRL convention):** `(1/T) * sum_j (1/j) * sum_{i<=j} a_{i,j}` — average over all evaluation checkpoints, not just final. Still seen in B0/B50 papers. `[HIGH]`

### 4.4 What reviewers will reject

- Reporting only "final-task accuracy" (`a_{T,T}`) without retrospective metrics. `[HIGH]`
- Reporting Task-IL accuracy under the Class-IL label. `[HIGH]`
- Mixing different memory budgets in a single comparison table. `[HIGH]`

---

## 5. SOTA Leaderboard (Verified and Provisional)

**CRITICAL:** Numbers below are from paper excerpts in this audit's searches. Cross-check against the original paper before citing in MoB.

### 5.1 Pretrained ViT-B/16 era (10-task Split-CIFAR-100, Class-IL, ImageNet-21k backbone)

| Method | Final Acc (%) | Venue | arXiv | Notes |
|--------|---------------|-------|-------|-------|
| L2P | ~83 | CVPR 2022 | 2112.08654 | Prompt pool. `[MED]` |
| DualPrompt | 83.05 ± 1.16 | ECCV 2022 | 2204.04799 | G-prompt + E-prompt. `[HIGH]` |
| CODA-Prompt | 86.25 ± 0.74 | CVPR 2023 | 2211.13218 | Decomposed attention prompts. `[HIGH]` |
| SLCA | ~90 | ICCV 2023 | 2303.05118 | Slow learner + classifier alignment. `[MED]` |
| HiDe-Prompt | ~92 | NeurIPS 2023 | 2310.07234 | Hierarchical decomp. `[MED]` |
| RanPAC | 92.2 | NeurIPS 2023 | 2307.02251 | Random projection + class prototypes, **no rehearsal**. `[HIGH — cited in search]` |
| SimpleCIL | ~87 | arXiv 2023 | 2303.07338 | Frozen ViT + prototype classifier. **The "must beat" baseline.** `[MED]` |
| FeCAM | ~86 | NeurIPS 2023 | 2309.14062 | **Mahalanobis** + frozen features. **Directly overlaps MoB v2 routing.** `[MED]` |
| EASE | ~91 | CVPR 2024 | 2403.12030 | Expandable subspaces. `[LOW — not directly verified]` |
| O-LoRA | ~85 | EMNLP 2023 | 2310.14152 | Orthogonal LoRA subspaces (originally NLP; transferred). `[LOW]` |
| SLCA++ | ~92 | arXiv 2024 | 2408.08295 | Successor to SLCA, closes gap to joint training <2%. `[MED]` |
| VQ-Prompt | >CODA | NeurIPS 2024 | 2410.20444 | Vector quantization prompts. `[MED]` |

**Ceiling reference (joint training, ViT-B/16 on CIFAR-100):** ~93–94%. `[MED]`
**Naive sequential fine-tuning (no CL):** ~20% (catastrophic forgetting). `[HIGH]`

### 5.2 Train-from-scratch CNN era (ResNet-18 / ResNet-32)

| Method | Setting | Final Acc (%) | Buffer | Venue | arXiv |
|--------|---------|---------------|--------|-------|-------|
| iCaRL | B0, 10 steps, ResNet-32 | ~50 | 2000 | CVPR 2017 | 1611.07725 `[HIGH]` |
| GEM | 5-task, ResNet-18 | ~55–60 | Episodic | NeurIPS 2017 | 1706.08840 `[MED]` |
| A-GEM | 5-task, ResNet-18 | ~55 | Episodic | ICLR 2019 | 1812.00420 `[MED]` |
| ER (basic replay) | 10-task, ResNet-18 | 40–60 | 500–5120 | - | - `[MED]` |
| DER++ | 10-task, ResNet-18 | ~65 | 500 | NeurIPS 2020 | 2004.04709 `[MED]` |
| DER (dynamic expand) | B0, 10 steps, ResNet-32 | ~67 | 2000 | CVPR 2021 | 2103.16788 `[MED]` |
| FOSTER | B0, 10 steps, ResNet-32 | ~72 | 2000 | ECCV 2022 | 2204.04662 `[MED]` |
| DyTox | 10-task, ConvNet | ~67 | 1000 | CVPR 2022 | 2111.11326 `[LOW]` |
| X-DER | 10-task, ResNet-18 | ~70 | 500 | TPAMI 2022 | 2201.00766 `[LOW]` |

**Ceiling reference (joint training, ResNet-32 on CIFAR-100):** ~75%. `[HIGH]`

### 5.3 Caveats on the leaderboard

- **Numbers shift by 2–5 points across repos** depending on class-order seed, learning-rate schedule, and data augmentation choices. A 2-point gap is not a publishable contribution unless you share the exact seed.
- **FeCAM is particularly important for MoB:** it uses Mahalanobis distance from class means in a frozen feature space. MoB v2's routing uses Mahalanobis from class prototypes. **Functionally these are the same classifier.** The contribution of MoB must come from (a) routing to specialists, not classifying, and (b) the forget-cost term in the auction. Expect reviewers to demand: "Does MoB beat FeCAM per-class-accuracy? If no, what does the auction add?"

---

## 6. Known Pitfalls and Cheese Moves

### 6.1 Memory-buffer comparisons

- Canonical buffer sizes reported on CIFAR-100: **200, 500, 1000, 2000, 5120**. Methods are not directly comparable at different buffer sizes. A buffer of 5120 on CIFAR-100 = 51.2 images/class = 10.2% of training data. `[HIGH]`
- **Rehearsal-free methods (L2P, DualPrompt, CODA-Prompt, RanPAC, FeCAM, SimpleCIL)** report zero buffer. MoB currently has no rehearsal, which aligns it with this family. `[HIGH]`
- **Cheese move:** Using a 5120 buffer and comparing against iCaRL's 2000 buffer without flagging it. Always compare at matched buffer. `[HIGH]`

### 6.2 Pretraining leakage (the big one)

- ViT-B/16 ImageNet-21k pretraining **absolutely sees CIFAR-100-adjacent concepts**. ImageNet-21k has ~21,000 classes including "shark", "flatfish", "aquarium_fish", "maple_tree", etc. — many are near-synonyms of CIFAR-100 fine labels. `[HIGH]`
- The community's implicit stance: "Everyone uses the same ViT-B/16, so leakage is constant across methods." **This is methodologically weak.** The correct comparison for a novel method like MoB is either (a) a controlled ablation on from-scratch ResNet, or (b) ImageNet-21k-excluded-CIFAR pretraining (no public checkpoint exists; this is a known gap). `[MED]`
- **Cheese move to avoid:** Claiming "CL ability" from a frozen pretrained ViT when most accuracy comes from the pretraining itself. The gap between **random-init ViT + prompt** and **ImageNet-21k-pretrained ViT + prompt** is 40+ points on Split-CIFAR-100. The CL method contributes <10 points. Framing matters. `[HIGH]`

### 6.3 Task-ID leakage

- Task-IL is sometimes reported as Class-IL by methods that infer task ID from prompt-selection signals (e.g., L2P's prompt-selection loss uses training task labels). Whether this counts as "task-free" is contested. `[MED]`
- MoB's auction at test time uses **pseudo-labels from each expert**, not ground-truth task IDs. This is **genuinely task-free**, and MoB should advertise it prominently. `[HIGH]`

### 6.4 Per-task vs stream evaluation

- **Per-task eval:** evaluate after each task finishes. Standard. Produces `a_{i,j}` matrix. `[HIGH]`
- **Stream eval (Mattdl 2023):** evaluate every N batches during training. Reveals the **stability gap** — transient forgetting at task boundaries even for methods that recover by end-of-task. Not yet universal but gaining adoption. MoB's Fisher-update-at-boundary design is specifically vulnerable to this metric. `[MED]`

### 6.5 Coarse vs fine label confound

- **CIFAR-100 at the 20-superclass level is much easier** (~85% accuracy instead of ~70% with the same model on fine labels). Papers occasionally report superclass accuracy for CL and don't mark it clearly. `[MED]`
- **CIFAR-100-SC (superclass as task):** a protocol where each of the 20 superclasses becomes one task of 5 fine classes. This is **semantically structured**, not random, and is harder than random-ordered Split-CIFAR-100 because the classes within a task are more confusable. Papers using CIFAR-100-SC (e.g., HyperTransformer, some MoE-CL work) typically report lower numbers. `[MED]`

### 6.6 Class-order seed sensitivity

- Final accuracy variance across 5 different class-order seeds is **1.5–3 points absolute**. A single-seed comparison is not statistically honest. Report mean ± std over at least 3 seeds. `[HIGH]`

---

## 7. What MoB Needs to Be Legible on CIFAR-100

### 7.1 Protocols MoB must run

1. **Split-CIFAR-100, 10 tasks, Class-IL, single-head** — the default. Non-negotiable.
2. **Split-CIFAR-100, 20 tasks, Class-IL** — stress test.
3. **CIFAR-100-B0 with memory buffer = 0** — positions MoB in the rehearsal-free family alongside FeCAM/RanPAC/SimpleCIL.
4. (Optional, high-value) **GCIL-style CIFAR-100** — demonstrates boundary-free operation, which is the strongest selling point of the continual-MoB shift-detection variant.

### 7.2 Backbone decision (not prescription — audit of what the community expects)

- **Pretrained ViT-B/16 (ImageNet-21k)** is the default for any 2024+ CL paper. Skipping it = "not comparable to the SOTA bar". `[HIGH]`
- **ResNet-18 or ResNet-32 trained from scratch** is the fairness check. Skipping it = "your method only works because of the backbone". `[HIGH]`
- **Both must be reported** for a paper in 2025–2026. `[HIGH]`

### 7.3 Mandatory baselines (minimum reviewer-legible set)

| Class | Baseline | Why |
|-------|----------|-----|
| **Trivial** | Naive sequential fine-tuning | Shows the forgetting floor. |
| **Trivial** | Joint training | Shows the ceiling. |
| **Prototype-NCM** | SimpleCIL (frozen ViT + NCM) | **The 2024+ "must-beat" bar.** |
| **Prototype-Mahalanobis** | FeCAM | **Directly overlaps MoB's routing; must be included.** |
| **Rehearsal-free prompt** | L2P and/or CODA-Prompt | Canonical prompt-family baseline. |
| **Rehearsal** | DER++ or ER at matched buffer | Shows what a simple replay gets you. |
| **Regularization** | EWC (since MoB uses Fisher) | Isolates the auction's contribution over pure EWC. |
| **Expansion** | DER or FOSTER | Other expand-per-task families MoB must distinguish from. |

### 7.4 Mandatory metrics (report all four)

1. Final Average Accuracy `A_T`, mean ± std over >=3 seeds.
2. Average Forgetting `F_T`.
3. Backward Transfer `BWT`.
4. Per-task accuracy trajectory plot.

Nice-to-have: intransigence; stability-gap measurement; per-expert utilization histogram (MoB-specific diagnostic).

### 7.5 Mapping 4 experts x 5 tasks onto CIFAR-100

MoB's current Split-MNIST config is 5 tasks of 2 classes each = 10 classes across 4 experts. Translating to CIFAR-100:

| Option | Tasks | Classes/Task | Total Classes | Community legibility | Note |
|--------|-------|--------------|---------------|----------------------|------|
| **A: 5T x 20C** | 5 | 20 | 100 | Medium — matches MoB's N=5 design exactly | Non-standard split but clean; one expert handles 2 tasks (the forgetting challenge) |
| **B: 10T x 10C, 4 experts** | 10 | 10 | 100 | **High — matches community default** | Each expert handles ~2.5 tasks; stresses auction harder |
| **C: 20T x 5C, 4 experts** | 20 | 5 | 100 | Medium — community stress-test | Each expert handles 5 tasks; may saturate |
| **D: 5T x 20C on CIFAR-100-SC** | 5 | 20 | 100 | Low — but novel | Semantic tasks; 4 superclasses per task |
| **E: 4T x 25C + holdout** | 4 | 25 | 100 | Low — unusual | One-task-per-expert purity; no forgetting pressure |

- **Option A** preserves the Split-MNIST design's "one expert must handle 2 tasks = the continual learning challenge" (per project memory). This is faithful to MoB's intent.
- **Option B** is the community-legible choice and stresses the auction hardest (more routing decisions per expert).
- **Option C** is a genuine ablation of routing scaling.
- **Option D** is semantically coherent and may reveal whether the Mahalanobis router clusters by superclass naturally.

This is a **PI decision** (see Open Questions).

---

## 8. Minimum Credibility Checklist for MoB on CIFAR-100

Reviewers in 2025–2026 will expect all of the following in a single table:

- [ ] **Protocol label** ("Split-CIFAR-100 10-task Class-IL, ViT-B/16 IN21K").
- [ ] **Class-order seeds** reported (ideally 3+, report mean ± std).
- [ ] **Memory budget** stated explicitly (0 for MoB current).
- [ ] **Backbone + pretraining source** stated explicitly.
- [ ] **Metric set**: `A_T`, `F_T`, `BWT`, trajectory plot.
- [ ] **Baselines**: SimpleCIL, FeCAM, L2P or CODA-Prompt, DER++ at matched buffer, EWC, naive FT, joint upper bound.
- [ ] **At least one from-scratch ResNet result** to isolate method from backbone contribution.
- [ ] **Per-expert utilization diagnostic** (MoB-specific, differentiating).
- [ ] **Ablation on auction terms** (alpha, beta, or without forgetting cost).
- [ ] **Acknowledge ImageNet-21k pretraining overlap** in limitations.

Missing any of items 1–7 ≈ desk-reject risk at top venues. Items 8–10 are how MoB earns trust for the novel contribution.

---

## 9. Open Questions for Dev

These are decisions only the PI should make. Cypher flags them; does not resolve them.

1. **Task count.** MoB's design intent (one expert handles >=2 tasks = the challenge) argues for **Option A (5T x 20C)**. Community legibility argues for **Option B (10T x 10C)**. Running both is expensive. Which is the thesis claim?
2. **Backbone philosophy.** The long-term goal is LLM MoE routing. **Frozen ViT-B/16** is methodologically closest to "routing over frozen foundation-model features". **From-scratch ResNet-18** is the historical CL bar. Is the LLM-scaling narrative the primary frame, or is generality across backbones the claim?
3. **FeCAM head-to-head.** FeCAM is MoB v2 routing minus the auction. If MoB v2 does not beat FeCAM, is the auction's value in (a) router selection, or (b) expert specialization, or (c) the forget-cost term? The paper needs a clear story; the benchmark should isolate it.
4. **Rehearsal stance.** MoB is currently rehearsal-free. Adding a small buffer (500 exemplars) would bring it into direct comparison with DER++/ER and may raise the ceiling 5–10 points. Pure stance or competitive?
5. **GCIL inclusion.** GCIL-on-CIFAR-100 is where the continual-MoB shift-detection variant could shine above the task-aware variant. Running GCIL is ~30% additional compute. Worth it for the differentiation?
6. **Duplicate acknowledgment.** Use ciFAIR retest in final camera-ready? No one else does, so it's a credibility differentiator but also an "apples-to-something-else" risk.
7. **Class ordering.** Random (canonical) vs. semantic/superclass-structured (harder, more informative for routing analysis) vs. both?

---

## 10. Uncertainty Ledger

| Claim | Confidence | Why |
|-------|------------|-----|
| Split-CIFAR-100-10T is the default | HIGH | Multiple primary sources (L2P, DualPrompt, CODA-Prompt, SLCA) agree |
| ciFAIR 10% duplication | HIGH | Direct from ciFAIR authors (cvjena.github.io/cifair) |
| CIFAR-100N noise rates | MED | Numbers cited from search, not directly fetched; verify before citing |
| Specific leaderboard accuracies | MED | Some from direct paper excerpts (e.g., CODA-P 86.25); others approximated |
| EASE, O-LoRA numbers | LOW | Not directly verified in this audit |
| Upsampling-artifact bias magnitude | MED | Mechanism is sound; no paper directly quantifies it |
| GCIL adoption fraction | LOW | Based on informal impression, not a citation count |

---

## Sources

- [ciFAIR duplicate-free test set](https://cvjena.github.io/cifair/) — Barz & Denzler, test-set near-duplicate analysis.
- [CIFAR-100N / CIFAR-N](https://github.com/UCSC-REAL/cifar-10-100n) — Wei et al. 2021, arXiv [2110.12088](https://arxiv.org/abs/2110.12088).
- [Three scenarios of CL (van de Ven & Tolias)](https://www.nature.com/articles/s42256-022-00568-3) — Nature Machine Intelligence 2022.
- [L2P (Learning to Prompt)](https://github.com/google-research/l2p) — CVPR 2022, arXiv [2112.08654](https://arxiv.org/abs/2112.08654).
- [DualPrompt](https://arxiv.org/abs/2204.04799) — ECCV 2022.
- [CODA-Prompt (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.pdf) — arXiv [2211.13218](https://arxiv.org/abs/2211.13218).
- [SLCA](https://arxiv.org/abs/2303.05118) — ICCV 2023.
- [SLCA++](https://arxiv.org/abs/2408.08295) — arXiv 2024.
- [HiDe-Prompt](https://arxiv.org/abs/2310.07234) — NeurIPS 2023.
- [RanPAC](https://openreview.net/forum?id=aec58UfBzA) — NeurIPS 2023, arXiv [2307.02251](https://arxiv.org/abs/2307.02251).
- [FeCAM](https://arxiv.org/abs/2309.14062) — NeurIPS 2023.
- [FOSTER](https://arxiv.org/abs/2204.04662) — ECCV 2022.
- [iCaRL](https://arxiv.org/abs/1611.07725) — CVPR 2017.
- [DER (dynamically expandable)](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.pdf) — CVPR 2021.
- [GCIL (Mi, Kong, Lin)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Mi_Generalized_Class_Incremental_Learning_CVPRW_2020_paper.pdf) — CVPRW 2020.
- [Continuum CIFAR-100 CL docs](https://continuum.readthedocs.io/en/latest/tutorials/scenarios_suites/CIFAR100.html).
- [Avalanche CIFAR-100 benchmark](https://avalanche-api.continualai.org/en/v0.2.1/_modules/avalanche/benchmarks/classic/ccifar100.html).
- [GT-RIPL Continual-Learning-Benchmark (Split-CIFAR-100 scripts)](https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/scripts/split_CIFAR100_incremental_class.sh).
- [Stability gap (Mattdl, ICLR 2023 Spotlight)](https://github.com/Mattdl/ContinualEvaluation).
- [Paperswithcode CIFAR-100 CIL leaderboards](https://paperswithcode.com/sota/incremental-learning-on-cifar-100-50-classes-3).
- [VQ-Prompt NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/3baf4eeffad860ca9c54aeab632716b4-Paper-Conference.pdf).
