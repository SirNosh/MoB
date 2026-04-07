# MoB (Mixture of Bidders) — Project Handover

**Date**: 2026-04-03
**Author**: Devya (with Claude Code assistance)

---

## Table of Contents

1. [Objective](#1-objective)
2. [Project Timeline](#2-project-timeline)
3. [Current State — Theory & Implementation](#3-current-state--theory--implementation)
4. [Codebase State](#4-codebase-state)
5. [Current Results (v2 Experiment Suite)](#5-current-results-v2-experiment-suite)
6. [Next Steps & Open Problems](#6-next-steps--open-problems)

---

## 1. Objective

### The Problem

Standard Mixture-of-Experts (MoE) architectures in large language models use a **learned gating network** to route tokens to experts:

```
tokens → Router(W_g · x) → softmax → top-k → weighted expert sum → output
         ^^^^^^^^^^^^^^^^
         LEARNED (collapses, needs auxiliary loss, forgets)
```

This learned router suffers from three fundamental problems:

1. **Expert collapse**: The router learns to send everything to a few experts, wasting capacity. This requires an auxiliary load-balancing loss with a hand-tuned coefficient — a major practical headache (Fedus et al., 2022; Lepikhin et al., 2021).

2. **Gater forgetting**: In continual learning settings, the router itself forgets how to route. Even if Expert A perfectly remembers digits 0-1, a router trained on later tasks may no longer send those digits to Expert A. The routing knowledge is stored in learned parameters that get overwritten.

3. **Auxiliary loss sensitivity**: The load-balancing coefficient is fragile. Too high → experts become interchangeable (no specialization). Too low → collapse returns. There is no principled way to set it.

### The MoB Solution

**MoB (Mixture of Bidders)** replaces the learned router with a **stateless auction mechanism**:

```
tokens → per-token distance bids → auction → top-k winners → weighted sum → output
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         STATELESS (no collapse, no auxiliary loss, no forgetting)
```

Each expert computes a **bid** for each batch/sample based on two costs:

```
bid = α × execution_cost + β × forgetting_cost
```

- **Execution cost**: How well can this expert handle this input? Measured via Mahalanobis distance to learned class centroids in feature space. Low distance = expert is already good at this type of data = low bid.
- **Forgetting cost**: How much would training on this input damage existing knowledge? Measured via EWC (Elastic Weight Consolidation) — gradient interference with Fisher-weighted important parameters. High interference = high cost = high bid.

**Winner = argmin(bids)**. The expert with the lowest combined cost wins the auction and trains on the batch.

### Why Auctions?

The auction mechanism is **stateless** — it has zero learned parameters. This means:

- **Immune to forgetting**: Routing is computed fresh every batch from current expert states. No parameters to overwrite.
- **Naturally load-balancing**: As an expert accumulates knowledge, its forgetting cost rises for unrelated data, pushing new work to less-loaded experts. No auxiliary loss needed.
- **Deterministic and reproducible**: Same expert states + same input = same routing decision. No stochastic gating.
- **Allocatively efficient**: Always selects the expert with minimum total cost.

### The Long-Term Goal

Scale auction-based routing from the current Split-MNIST proof-of-concept through CIFAR-10 to **LLM-scale MoE transformer layers**, replacing the learned `W_g · x → softmax → top-k` router entirely. The path is:

```
MNIST (current)  →  CIFAR-10  →  LLM MoE Layer
4 SimpleCNN experts   4-8 ResNet experts   8-128 FFN experts within transformer
128-dim features      512-dim features     4096-dim features
Full covariance       Full covariance      Low-rank covariance approximation
```

What stays identical at every scale: the auction mechanism (`winner = argmin(bids)`), Mahalanobis prototype distance routing, stateless routing with zero learned router parameters, and emergent load balancing via forgetting cost. What changes: model architecture, feature dimensionality, covariance estimation (low-rank at LLM scale), and integration into the transformer forward pass.

### Design Constraint: 4 Experts, 5 Tasks

The entire experimental setup uses **4 experts for 5 tasks**. This is intentional and models a fundamental property of real-world knowledge: **related domains should share an expert**.

At LLM scale, consider an expert that learns geometry. That same expert should also handle computer graphics, 3D modeling, and spatial reasoning — these are domain-adjacent topics where shared representations are beneficial, not harmful. The expert doesn't just "tolerate" multiple domains; it actively benefits from the cross-pollination of related knowledge within its parameters. An expert that understands geometric primitives will be better at 3D modeling than one that doesn't.

The 4:5 ratio forces at least one expert to handle 2 tasks simultaneously. This is **both a continual learning challenge and a routing challenge**:

- **Continual learning**: The overloaded expert must learn a second task without forgetting the first. EWC protects important parameters while allowing plasticity for new knowledge.
- **Routing**: The auction must correctly route both Task 2 and Task 5 data to the same expert, even though they are different distributions. The expert's prototypes must cover both tasks, and the distance metric must still work when an expert has heterogeneous centroids.

This is not a confound to remove — it IS the problem MoB is designed to solve. Never increase num_experts to match num_tasks. The overloaded expert is the proof that auction routing + EWC can handle the real-world scenario where experts accumulate related competencies over time.

---

## 2. Project Timeline

### Phase 1: Core MoB with Pseudo-Label Routing (February 2026)

**What was built**: The complete MoB framework — auction mechanism, EWC bidding, expert pool, shift detection, and 5 baselines (GatedMoE+EWC, Monolithic EWC, A-GEM, Experience Replay, PNN). All routing used **pseudo-label** execution cost: `exec_cost = CE(logits, argmax(logits))`.

**Key milestones**:
- Optuna hyperparameter search: 700+ configurations across 100 trials × 5 seeds for both task-aware and continual MoB.
- Best results: **MoB-Online 90.22%** (λ=971.27, α=0.5278, β=0.6333, shift_threshold=2.58), **MoB-TaskAware 79.03%** (λ=277.54, α=0.3549, β=0.4151).
- GatedMoE+EWC collapsed to 19.86%, empirically validating the "gater forgetting" hypothesis.
- Fisher clamping fix (min=0.1) discovered: resolved 18x variance in Fisher magnitude across expert initializations; improved overloaded expert retention from ~0% to >87%.
- Optimizer reset strategy implemented.

**Published**: GitHub repository at **https://github.com/SirNosh/MoB** — contains the Phase 1 codebase. The README on GitHub documents the pseudo-label routing results and benchmark tables. All benchmark numbers in the README (79.03%, 90.22%, etc.) are from this phase using pseudo-label routing with Optuna-tuned hyperparameters.

### Intermediate Pre-Print Paper (February-March 2026)

An intermediate paper was written documenting the Phase 1 work:

> **"MoB: Mixture of Bidders — An Auction-Based Expert Routing Framework for Continual Learning"**
> Dev Vyas, Georgia State University

**What the paper covers**:
- Formal definition of gater forgetting as a distinct MoE failure mode
- Mathematical formulation of the bidding mechanism (Equations 2-5), EWC loss (Eq. 7), Fisher update (Eq. 8-9), shift detection (Eq. 10-11)
- Full benchmark comparison against 5 baselines on Split-MNIST
- 700+ configuration ablation studies with two-way interaction analysis
- Computational efficiency analysis (FLOPs, VRAM, throughput)

**What the paper explicitly identifies as future work** (Section 8):
- *"Prototype-based bidding using Mahalanobis distance in feature space, replacing loss-based execution cost with geometric similarity to learned cluster centroids"*
- CIFAR-100 and CORe50 evaluation
- Capacity-aware bidding
- Hybrid MoB + replay

**Important**: All numbers in the paper match the GitHub README. The paper predates the Mahalanobis implementation. It is available locally at `C:\Users\devya\Downloads\MoB.pdf` (and `mob_paper.tex`).

### Phase 2: Mahalanobis Prototype Routing Implementation (March 2026)

**Motivation**: The pseudo-label routing suffered from the "confidently wrong" problem (see Section 3.2). Experts predicted everything as their trained classes with ~99% confidence, collapsing the execution cost signal. The paper identified Mahalanobis distance routing as the solution.

**What was implemented**:
- `contibualmob/prototype_store.py`: Per-expert PrototypeStore with incremental centroid accumulation, global covariance tracking, Mahalanobis/Euclidean distance computation.
- `forward_features()` method added to all model architectures (SimpleCNN, LeNet5, MLP) — returns (penultimate_features, logits).
- Prototype routing integration in `contibualmob/pool.py` and `tests/run_mob_only.py`.
- Bid formula changed from `α × CE(pseudo_labels)/2.5` to `α × mahalanobis_distance/10.0`.
- `progress_report.md` written March 19, documenting initial prototype routing results and known issues.

**Initial experiments** (v1 suite, stored in `results/experiments/`): Used λ_ewc=5.0 (catastrophically weak — carried over from an earlier config). Results were misleading. All training-time prototype routing warmup variants produced identical 19.5% accuracy due to a bug (centroids not available during training).

### Phase 3: Bug Fixes & v2 Experiment Suite (Late March - April 3, 2026)

**Bugs discovered and fixed**:

1. **Incremental centroid bug** (root cause of Exp3 failure): `PrototypeStore.centroids` was empty until `finalize()` was called at task boundaries. The `has_prototypes` check required non-empty centroids, so it was always `False` during training. Prototype routing silently fell back to label routing for ALL experts during training. **Fix**: Added incremental centroid computation in `update()` method.

2. **Bid scale mismatch** (4 separate instances across 2 files): When some experts used prototype distances (~0-1 range) and others used label-based costs (different scale), the auction compared incompatible numbers. Experts without prototypes got `distance_score = 100.0` with default fallback rather than switching to a different cost formula. **Fix**: All experts in the same auction now use the identical bid formula.

3. **Raw forgetting cost as bid** in `evaluate_all()`: Used raw value (0-500k) while prototype path used normalized bid (~0-1). **Fix**: Consistent normalization.

4. **Near-zero CE as bid** in `evaluate_all_per_sample()`: `F.cross_entropy(logits, pseudo_labels)` produces near-zero for confident models, making no-prototype experts always win. **Fix**: High default distance tensor (100.0).

**Other changes**:
- "VCG" renamed to "PerBatchAuction" everywhere (~15 files). The auctions are single-item second-price, not full Vickrey-Clarke-Groves.
- λ_ewc default changed from 5.0 to 1000 for v2 experiments.
- v2 experiment suite designed: 47 experiments across 11 phases, output to `results/experiments_v2/`.

**v2 experiments run**: April 3, 2026. Best result: **87.34%** (seed 123, per-sample k=1, prototype eval routing, label-based training routing). Training-time prototype routing still fails (30-35%) — this is now confirmed as a systematic routing collapse issue, not a bug.

### Summary: Two Codebases, Two Routing Strategies

| | GitHub Repo | Local Codebase |
|---|---|---|
| **URL** | https://github.com/SirNosh/MoB | `C:\MoB Final` |
| **Routing** | Pseudo-label (Generation 1) | Prototype/Mahalanobis (Generation 2) |
| **Best Task-Aware** | 79.03% (Optuna-tuned) | 87.34% (v2, seed 123) |
| **Best Online** | 90.22% (Optuna-tuned) | 78.83% (v2, not Optuna-tuned) |
| **EWC λ** | 277.54 (task-aware), 971.27 (online) | 1000 (default, not optimized) |
| **Paper** | Documented in pre-print | Extends paper's future work |
| **Status** | Stable, published | Active development |

**Note on Online MoB**: The GitHub/paper result (90.22%) used Optuna-tuned hyperparameters (shift_threshold=2.58). The v2 continual MoB uses shift_threshold=50.0 (not tuned). An Optuna search over the v2 prototype routing could potentially recover or exceed the 90.22% figure.

---

## 3. Current State — Theory & Implementation

### 3.1 Routing Strategies (Two Generations)

#### Generation 1: Pseudo-Label Routing (Original)

```
bid = α × CE(logits, argmax(logits)) + β × EWC_gradient_cost
```

The expert's own predictions (pseudo-labels) are used as stand-ins for ground truth. Execution cost = cross-entropy of the expert's logits against its own argmax predictions.

**Problem — "Confidently Wrong"**: After training, experts predict everything as their trained classes with ~99% confidence. Cross-entropy against the expert's own predictions is near-zero regardless of input. This collapses the execution cost signal, degrading routing to near-random. Result: ~60-70% accuracy.

**Status**: Implemented in `mob/` package. Still available as a baseline (`--routing_strategy pseudo_label`). The README benchmark results (79.03% task-aware, 90.22% online) use this strategy with Optuna-tuned hyperparameters.

#### Generation 2: Prototype (Mahalanobis Distance) Routing (Current)

```
bid = α × Mahalanobis_distance_to_nearest_centroid + β × EWC_gradient_cost
```

This is the core technical contribution of the current codebase — replacing the broken pseudo-label execution cost with a **geometric distance metric in feature space**. It was explicitly identified as future work in the Phase 1 paper and implemented during Phase 2.

##### Why We Need It: The "Confidently Wrong" Problem

Pseudo-label routing computes `exec_cost = CE(logits, argmax(logits))`. After an expert trains on digits 2-3, it predicts everything (including digits 8-9) as class 2 or 3 with ~99% confidence. The cross-entropy of a model against its own argmax predictions is near-zero regardless of input — the model is "confidently wrong" about out-of-distribution data. This collapses the execution cost signal: all experts report near-zero exec_cost for all inputs, and routing degrades to depend solely on forgetting cost (near-random for untrained experts).

The solution: measure **how close the input is to what the expert has actually seen** in feature space, rather than how confident the expert is in its prediction. An expert trained on digits 2-3 will have centroids in the region of feature space where digits 2-3 cluster. Digits 8-9 will be far from those centroids — high distance = high bid = expert correctly loses the auction.

##### Why Mahalanobis Distance Specifically

**Euclidean distance** treats all feature dimensions equally:

```
d_E(x, μ) = ||x - μ||₂ = sqrt(Σᵢ (xᵢ - μᵢ)²)
```

This is problematic because neural network features have non-uniform scale and are often correlated. If feature dimension A has 10x the variance of dimension B, Euclidean distance is dominated by A. Two points that are genuinely similar in the learned representation may appear far apart simply because one high-variance dimension differs.

**Mahalanobis distance** normalizes by the inverse covariance, effectively computing Euclidean distance in the **whitened** (decorrelated, unit-variance) feature space:

```
d_M(x, μ) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))
```

Where `Σ⁻¹` is the inverse covariance matrix of the feature distribution. This makes the distance invariant to linear transformations of the feature space — rotating, scaling, or correlating features doesn't change the distance ranking. Points that are genuinely close in the learned representation have low Mahalanobis distance, regardless of how the features are scaled or correlated.

**Intuition**: Mahalanobis distance is the natural "how many standard deviations away" measure in multi-dimensional space. A point 2σ away along a high-variance axis is less surprising than a point 2σ away along a low-variance axis. Mahalanobis captures this correctly; Euclidean does not.

##### The PrototypeStore: End-to-End Pipeline

Each expert has its own `PrototypeStore` (file: `contibualmob/prototype_store.py`). The store accumulates feature statistics during training and provides distance-based routing at inference.

**Step 1: Feature Extraction** (`mob/models.py`)

All models implement `forward_features()` which returns both the penultimate feature vector and the logits:

```python
# SimpleCNN: Input (B, 1, 28, 28) → Conv blocks → Flatten → FC1 → FC2
features = F.relu(self.fc1(flattened))  # (B, 128) — penultimate layer
logits = self.fc2(dropout(features))     # (B, 10) — output layer
return features, logits
```

The 128-dimensional penultimate layer is the **prototype space**. This is where all distance computations happen. The choice of penultimate layer is deliberate: it captures the model's learned representation of the input before the final classification head squashes everything into class probabilities.

**Step 2: Prototype Accumulation** (`PrototypeStore.update()`)

During training, every time a winning expert processes a batch, its prototype store accumulates:

```python
# Per-class running sums and counts (for centroids)
for class c in batch_labels:
    class_sum[c] += sum(features[labels == c])     # (128,) per class
    class_count[c] += count(labels == c)

# Global covariance accumulator (shared across all classes)
cov_sum += features.T @ features                    # (128, 128) outer product
cov_count += batch_size
```

**Incremental centroids** are recomputed after each batch:

```python
centroids[c] = class_sum[c] / class_count[c]       # Available immediately
```

This is critical: centroids are available during training (not just after finalize), enabling training-time prototype routing. However, only Euclidean distance is available at this point — the inverse covariance requires finalize().

**Step 3: Finalization** (`PrototypeStore.finalize()`)

Called at task boundaries (task-aware MoB) or shift detection (continual MoB). Computes the inverse covariance matrix:

```python
# Global mean across all classes
global_mean = sum(class_sums) / sum(class_counts)

# Covariance: E[XXᵀ] - μμᵀ + εI
cov = (cov_sum / cov_count) - outer(global_mean, global_mean) + 1e-4 * I

# Inverse covariance (with pseudo-inverse fallback for singular matrices)
inv_cov = inv(cov)  # or pinv(cov) if singular
```

**Condition**: `inv_cov` is only computed if `cov_count >= 256` (MIN_SAMPLES_FOR_MAHALANOBIS). This threshold ensures stable covariance estimation — with fewer samples, the 128×128 covariance matrix may be poorly conditioned.

**Regularization**: `ε = 1e-4` (Tikhonov regularization) is added to the diagonal to prevent singularity and improve numerical stability.

**Step 4: Distance Computation** (`PrototypeStore.compute_routing_score()`)

For a batch of features `(B, 128)` against `C` class centroids:

```python
diff = features[:, None, :] - centroids[None, :, :]  # (B, C, 128)

if inv_cov is not None:  # Mahalanobis mode
    transformed = diff @ inv_cov                       # (B, C, 128) — whitening
    distances = sqrt(sum(transformed * diff, dim=-1))  # (B, C)
else:  # Euclidean fallback
    distances = ||diff||₂                              # (B, C)

min_distances = distances.min(dim=1).values            # (B,) — nearest centroid
routing_score = min_distances.mean()                   # Scalar for batch routing
```

For per-sample routing, `compute_per_sample_distances()` returns the `(B,)` tensor directly without averaging.

**Step 5: Bidding** (replaces execution cost)

```python
# Old (pseudo-label): bid = α × CE(logits, pseudo_labels)/2.5 + β × log1p(forget)/10
# New (prototype):    bid = α × mahalanobis_distance/10.0   + β × log1p(forget)/10

norm_distance = distance_score / 10.0    # Mahalanobis distances typically 0-20
norm_forget = math.log1p(raw_forget) / 10.0
bid = α × norm_distance + β × norm_forget
```

Experts without prototypes (not yet trained on any task) get `distance_score = 100.0` — a high default that ensures they lose auctions until they accumulate prototypes.

##### Distance Mode Timeline During Training

```
Batch 0     → Expert wins first auction → prototype_store created (lazy init)
Batch 1-N   → update() accumulates per-class sums + covariance
              → Incremental centroids available → Euclidean distance routing possible
Task boundary (or shift detected)
            → finalize() computes inv_cov
              → Mahalanobis distance now available for future routing
Next task   → Expert continues accumulating (prototypes GROW, not reset)
              → Mahalanobis distance uses all classes seen so far
```

Key design decision: **prototypes grow across tasks/shifts, not reset**. An expert that handles digits 2-3 and later digits 8-9 will have centroids for all four classes. This matches the domain-adjacency goal: the expert accumulates competence across related distributions.

**Status**: Implemented in `contibualmob/prototype_store.py`. Used by default in all v2 experiments (`--routing_strategy prototype`). Best v2 result: **87.34%** (seed 123, per-sample k=1, label-based training routing).

### 3.2 Elastic Weight Consolidation (EWC)

EWC is the forgetting-prevention mechanism within each expert. After an expert completes a task (or a distribution shift is detected), a **diagonal Fisher Information Matrix** is computed over the expert's parameters. This Fisher matrix encodes which parameters are "important" for the tasks the expert has learned.

**The EWC loss** added during training:

```
L_total = L_task + (λ/2) × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)²
```

Where `Fᵢ` = Fisher importance of parameter i, `θ*ᵢ` = optimal parameter value from previous tasks, `λ` = regularization strength.

**Online EWC** (Schwarz et al., 2018): Fisher and optimal parameters use exponential moving averages to prevent unbounded accumulation across tasks:

```
F_total = γ × F_old + (1-γ) × F_new     (γ = 0.9)
θ*_total = γ × θ*_old + (1-γ) × θ_current
```

**Input-dependent forgetting cost** (used in bidding):

```
forgetting_cost = Σᵢ Fᵢ × (∂L/∂θᵢ)²
```

This measures gradient interference — how much the current batch's gradients would conflict with important parameters. It is NOT the same as the EWC penalty (which measures parameter drift). The forgetting cost is input-dependent: it varies per batch, providing a per-sample routing signal.

#### Critical EWC Fixes

1. **Fisher clamping** (`min=0.1`): Without clamping, some parameters get near-zero Fisher values, leaving them completely unprotected. Creating multiple models changes random state, giving some experts "bad" initializations where Fisher max varies 18x between experts. Clamping to 0.1 ensures all parameters get at least basic L2-like protection. This fix improved the overloaded expert from 0% to 87% retention on its first task.

2. **Fisher normalization** (`mean=1.0`, `ε=1e-30`): Raw Fisher values can vary by orders of magnitude. Normalizing to mean=1.0 ensures λ_ewc has consistent meaning across experts and tasks.

3. **Optimizer reset**: Adam optimizer accumulates momentum from previous tasks. When an expert switches to a new distribution, stale momentum hurts learning. Task-aware MoB resets winning experts' optimizers at task END (after Fisher update). Continual MoB resets on shift detection.

### 3.3 The Auction Mechanism

**Class**: `PerBatchAuction` (in `mob/auction.py` and `contibualmob/auction.py`)

Single-item, lowest-bid-wins auction:

```python
winner = argmin(bids)
payment = second_lowest_bid  # second-price rule
```

The second-price payment is logged for diagnostics but doesn't affect training. The auction is a simple argmin — no learned parameters, no state. It was previously named "VCG" but has been renamed since it's a single-item second-price auction, not a full Vickrey-Clarke-Groves mechanism.

**Also includes**: `SealedBidProtocol` — an optional two-phase commit-reveal protocol using SHA-256. Designed for distributed settings where experts might try to game bids. Not used in current experiments.

### 3.4 Bid Normalization

Raw execution costs and forgetting costs live on completely different scales:
- Execution cost (cross-entropy): 0 to ~2.5
- Forgetting cost (Fisher-weighted gradient interference): 0 to 500,000+

Without normalization, forgetting cost would dominate every bid. The normalization scheme:

```python
norm_exec = raw_exec / 2.5                    # Maps CE to ~[0, 1]
norm_forget = math.log1p(raw_forget) / 10.0   # log compression for huge range
bid = α × norm_exec + β × norm_forget
```

For prototype routing, the execution cost is replaced by distance:

```python
norm_distance = mahalanobis_distance / 10.0   # Mahalanobis distances typically 0-20
norm_forget = math.log1p(raw_forget) / 10.0
bid = α × norm_distance + β × norm_forget
```

**Critical invariant**: ALL experts in the same auction must use the same bid formula. Experts without prototypes (haven't completed a task yet) get a default `distance_score = 100.0` (high distance = unlikely to win) rather than falling back to a different cost formula. This was a bug that was fixed in the v2 codebase — v1 experiments had scale mismatches where some experts used normalized prototype distances (~0-1) while others used raw forgetting costs (~0-500k).

### 3.5 Per-Sample vs Per-Batch Routing

**Per-batch routing** (original): All samples in a batch are routed to the same expert. The bid is computed from batch-mean distances. This is simpler but suboptimal for mixed batches (e.g., a batch containing digits from multiple tasks).

**Per-sample routing** (Experiment 1): Each sample is independently routed to its nearest expert. This is the analog of per-token routing in transformer MoE (Mixtral, Switch Transformer). Implementation: compute a `(batch_size, num_experts)` distance matrix, select the top-k experts per sample.

**Top-k combination** (k=2): For each sample, combine the top-2 experts' logits weighted by softmax of negative distances:

```python
weights = softmax(-distances[top_k_indices] / temperature)
output = Σ w_i × logits_i
```

Temperature controls sharpness: low τ ≈ winner-take-all (k=1 behavior), high τ ≈ uniform average.

### 3.6 Distance-Only Bidding

At evaluation time, Fisher matrices are frozen. The forgetting cost is effectively constant per expert regardless of input — it adds no per-sample routing signal. Therefore, at eval time:

```
bid = Mahalanobis_distance_only  (drop forgetting cost)
```

This reduces bid computation from N × (forward pass + gradient computation) to N × (features @ inv_cov @ features.T) — a single matrix multiplication per expert. At LLM scale, this makes MoB bid cost competitive with learned routers.

**v2 result**: Distance-only per-batch eval matches full bidding exactly (85.64% both). Per-sample distance-only shows a ~6% drop (74.67% vs 80.43%), suggesting forgetting cost still contributes routing signal at per-sample granularity.

### 3.7 Training-Time Prototype Routing (Experiment 3 — Open Problem)

The most important experiment for the LLM scaling story. In LLM pretraining, there are no task labels — routing must work from input features alone.

**Approach**:
1. **Warmup phase** (N batches): Use standard label-based bidding. Experts specialize, prototypes accumulate.
2. **After warmup**: Switch to `bid = distance_to_prototype + forget_cost`. Labels still used for TRAINING the winner (supervised CE loss), just not for ROUTING decisions.

**Current status**: This consistently fails across all seeds and warmup values (30-35% accuracy, routing collapse to 1-2 experts). See Section 5 for detailed analysis. This is the primary open problem.

### 3.8 Shift Detection (Continual MoB)

For task-free learning, `ShiftDetector` identifies distribution changes via EMA cost tracking:

```python
ema_cost = α × ema_cost + (1-α) × current_cost  # α = 0.99 (slow adaptation)
is_shift = cost > max(ema_cost, 0.5) × threshold_multiplier
```

On detected shift: finalize prototypes, update Fisher matrices, optionally reset optimizers. A cooldown period (50 batches) prevents spurious re-detection.

**Observed behavior**: With α=0.3/β=0.7 label routing, the detector correctly identifies 4 shifts (matching the 5-task structure). With default α=0.5/β=0.5, it detects 19 shifts (over-sensitive). The threshold_multiplier and α/β configuration significantly affect detection quality.

### 3.9 Evaluation Routing

At test time, ground-truth labels are unavailable. Evaluation uses:

1. **Full bid mode**: Pseudo-labels (argmax of logits) + forgetting cost → bid per expert → lowest wins
2. **Distance-only mode**: Mahalanobis distance to nearest centroid → lowest distance wins
3. **Per-sample mode**: Each sample independently routed (optionally top-k combined)

The prototype routing strategy at eval is preferred because it avoids the "confidently wrong" problem of pseudo-label routing.

### 3.10 The Scaling Vision: MNIST → CIFAR → LLM

| Component | MNIST (current) | CIFAR-10 (next) | LLM MoE Layer (goal) |
|-----------|-----------------|------------------|----------------------|
| "Expert" | SimpleCNN | ResNet-18 | FFN block within transformer layer |
| Features | 128-dim (penultimate FC) | 512-dim (ResNet penultimate) | 4096-dim (transformer hidden state) |
| Routing granularity | Per-sample or per-batch | Per-sample | Per-token |
| Prototype centroids | 128-dim, ~10 classes | 512-dim, ~10 classes | 4096-dim, domain-based clusters |
| Covariance | 128×128 full inverse | 512×512 full inverse | 4096×4096 → **low-rank approximation needed** |
| Expert count | 4 | 4-8 | 8-128 |
| EWC scope | Over full model | Over full model | Over FFN weights only |

**What's fundamentally new for LLM scale**:
1. Low-rank covariance approximation (full 4096×4096 inverse is prohibitive)
2. Integration into transformer forward pass (expert = FFN, not standalone model)
3. "Class" centroids → "domain/pattern" centroids (unsupervised clustering during training)
4. Backprop through winning expert's FFN only

**What stays identical (core contribution)**:
1. `winner = argmin(bids)` — same math at any scale
2. `d(x, centroid)` via Mahalanobis — same formula, higher dimensionality
3. Stateless routing with zero learned router parameters
4. Emergent load balancing via forgetting cost
5. No auxiliary loss

---

## 4. Codebase State

### 4.1 Package Architecture

The codebase has two parallel packages with shared logic:

```
mob/                    # Task-aware MoB (explicit task boundaries)
contibualmob/           # Continual/online MoB (task-free, shift detection)
tests/                  # Runners, test scripts, baselines, analysis tools
results/                # All experimental results (old and new)
```

`mob/` and `contibualmob/` share identical implementations for: `auction.py`, `bidding.py`, `models.py`, `utils.py`, `baselines.py`. The key difference is `contibualmob/` adds `prototype_store.py` for Mahalanobis routing and `pool.py`/`expert.py` have shift detection + prototype integration.

**Important — two codebases exist**:
- **GitHub** (https://github.com/SirNosh/MoB): Phase 1 code with pseudo-label routing only. No Mahalanobis, no PrototypeStore, no `forward_features()`. The README results (79.03%, 90.22%) come from this code.
- **Local** (`C:\MoB Final`): Phase 2-3 code with Mahalanobis additions. The v2 experiment results (87.34% best) come from this code.

**The pre-print paper** (`MoB.pdf` / `mob_paper.tex` in Downloads): Documents Phase 1 work. All benchmark numbers match GitHub. Mahalanobis is explicitly named as future work in Section 8.

### 4.2 File-by-File Descriptions

#### Core Packages

| File | Purpose | Key Classes/Functions | Notes |
|------|---------|----------------------|-------|
| **`mob/__init__.py`** | Package init + public API exports | Exports: `PerBatchAuction`, `SealedBidProtocol`, `ExecutionCostEstimator`, `EWCForgettingEstimator`, `MoBExpert`, `ExpertPool`, model factories, baselines, utilities | |
| **`mob/auction.py`** | Auction mechanism | `PerBatchAuction` (lowest-bid-wins, second-price payment, history tracking), `SealedBidProtocol` (commit-reveal for distributed settings) | Uses `np.partition()` for O(n) second-price. Stateless — no learned parameters. |
| **`mob/bidding.py`** | Cost estimators for bids | `ExecutionCostEstimator` (CE loss), `EWCForgettingEstimator` (diagonal Fisher + online EWC + input-dependent forgetting cost) | Fisher clamping min=0.1, normalization mean=1.0, ε=1e-30. FISHER_DECAY=0.9. |
| **`mob/models.py`** | Neural network architectures | `SimpleCNN` (128-dim features, ~421K params), `LeNet5` (84-dim), `MLP` (configurable). All support `forward_features()` returning (features, logits). | `width_multiplier` for fair baseline comparison. Dynamic FC init in SimpleCNN. |
| **`mob/expert.py`** | Individual expert agent | `MoBExpert`: bid computation (`α × norm_exec + β × norm_forget`), training with EWC penalty, Fisher updates, statistics tracking | Bid normalization: exec/2.5, log1p(forget)/10.0 |
| **`mob/pool.py`** | Expert pool management | `ExpertPool`: `collect_bids()`, `train_winner()`, `update_after_task()`, `evaluate_all()` (pseudo-label auction routing) | Stateless routing — no gater network. |
| **`mob/baselines.py`** | Comparison methods | `NaiveFineTuning`, `RandomAssignment`, `MonolithicEWC`, `GatedMoE` — each with `train_on_task()`, `update_after_task()`, `evaluate_all()` | 4 baselines for validating MoB effectiveness. |
| **`mob/bid_diagnostics.py`** | Logging infrastructure | `BidLogger`: per-batch bid logging, per-digit routing analysis, load balance metrics (entropy, Gini), prototype state snapshots | JSON output. Comprehensive training + eval diagnostics. |
| **`mob/utils.py`** | Utilities | `set_seed()`, `setup_logging()`, `count_parameters()`, `get_device()`, etc. | Deterministic seeding for reproducibility. |
| **`contibualmob/__init__.py`** | Package init | Same as mob/ plus `PrototypeStore` export | |
| **`contibualmob/prototype_store.py`** | **Core innovation**: Per-expert class prototype storage | `PrototypeStore`: `update()` (incremental centroids + covariance accumulation), `finalize()` (compute inv_cov for Mahalanobis), `compute_routing_score()` (distance to nearest centroid) | MIN_SAMPLES=256 for Mahalanobis; Euclidean fallback. Incremental centroids available during training. |
| **`contibualmob/expert.py`** | Expert with prototype integration | `MoBExpert`: adds `prototype_store` (lazy-init), `last_won_global_batch` (idle tracking) | Backward-compatible extension of mob/expert.py |
| **`contibualmob/pool.py`** | Expert pool with shift detection | `ShiftDetector` (EMA cost tracking, cooldown), `ExpertPool`: adds `train_routing` param ('label'/'prototype'), optimizer reset on shift, idle expert reset | `collect_bids()` has prototype routing branch with consistent bid scale (default distance=100.0 for experts without prototypes). |
| **`contibualmob/bid_diagnostics.py`** | Extended logging | Superset of mob/ version: adds `log_prototype_finalize()`, prototype state diagnostics, distance separation analysis | Warns if separation < 10% (near-random). |
| **`contibualmob/auction.py`** | Identical to mob/auction.py | | Shared logic |
| **`contibualmob/bidding.py`** | Identical to mob/bidding.py | | Shared logic |
| **`contibualmob/models.py`** | Identical to mob/models.py | | Shared logic |
| **`contibualmob/baselines.py`** | Identical to mob/baselines.py | | Shared logic |
| **`contibualmob/utils.py`** | Identical to mob/utils.py | | Shared logic |

#### Test & Runner Scripts

| File | Purpose | Notes |
|------|---------|-------|
| **`tests/run_mob_only.py`** | **Primary task-aware MoB runner** | Contains a local `MoBExpertLocal` class with LwF support. Supports all CLI flags: `--routing_strategy`, `--eval_bid_mode`, `--per_sample`, `--top_k`, `--temperature`, `--train_routing`, `--train_warmup`, `--save_bids`, `--reset_optimizer`, `--experiment_name`, etc. Imports from both `mob/` and `contibualmob/prototype_store`. |
| **`tests/run_continual_mob.py`** | **Primary continual MoB runner** | Continuous data stream (ConcatDataset). Shift detection, selective consolidation, per-digit diagnostics. Imports from `contibualmob/`. |
| **`tests/test_mnist.py`** | Basic MoB experiment on Split-MNIST | Uses `mob/` package only. Simpler training loop without prototype routing or advanced features. Includes `create_split_mnist()` with replay, `compute_specialization_metrics()`, `plot_specialization()`. |
| **`tests/test_baselines.py`** | Baseline comparison framework | Defines `create_split_mnist()` (reused by both runners). Runs all 4 baselines. |
| **`tests/test_components.py`** | Unit tests for core components | Tests imports, auction mechanics, expert initialization. |
| **`tests/test_ironclad.py`** | Robustness/reliability tests | |
| **`tests/hyperparameter_search.py`** | Optuna-based hyperparameter tuning | Bayesian optimization over α, β, λ_ewc, lr, etc. |
| **`tests/analyze_mob_bids.py`** | Post-hoc bid trace analysis | Reads JSON bid logs and generates analysis. |
| **`tests/analyze_ablation.py`** | Ablation study analysis | |
| **`tests/formula_comparison.py`** | Bid formula comparison | |
| **`tests/benchmark_resources.py`** | Resource usage benchmarking | |
| **`tests/run_gated_moe_ewc.py`** | GatedMoE+EWC baseline runner | |
| **`tests/run_monolithic_ewc.py`** | Monolithic EWC baseline runner | |
| **`tests/run_agem_baseline.py`** | A-GEM baseline runner | |
| **`tests/run_er_baseline.py`** | Experience Replay baseline runner | |
| **`tests/run_pnn_baseline.py`** | Progressive Neural Networks runner | |
| **`tests/check resources/`** | Reference implementations | Contains copies of main runners + `sanity_check_ewc.py` for cross-validation. |

#### Root Files

| File | Purpose | Notes |
|------|---------|-------|
| **`README.md`** | Full project documentation | Theory, architecture, implementation, baselines, benchmark results. **Note**: Benchmark results use pseudo-label routing (Generation 1), not prototype routing. |
| **`EXPERIMENTS.md`** | Experiment guide for v1 suite | Documents 4 experiment phases. Points to `results/experiments/` (v1 output directory). **Partially outdated**: v2 experiments have different naming and more phases. |
| **`progress_report.md`** | Status snapshot from 2026-03-19 | Documents prototype routing implementation, known issues, and initial results. **Outdated**: Pre-dates bug fixes and v2 experiments. |
| **`run_all_experiments.sh`** | v2 experiment suite | 47 experiments across 11 phases. Output: `results/experiments_v2/`. Default λ=1000, 4 experts, 5 tasks. |

### 4.3 Results Directory Map

```
results/
├── experiments_v2/           ← CURRENT (April 3, 2026). Use these.
│   ├── base_*.json           # Phase 1: Baselines
│   ├── distonly_*.json       # Phase 2: Distance-only eval
│   ├── trainproto_*.json     # Phase 3: Training-time prototype routing
│   ├── labelfree_*.json      # Phase 4: Fully label-free pipeline
│   ├── ab_*.json             # Phase 5: Alpha/beta ablation
│   ├── ewc_*.json            # Phase 6: EWC lambda ablation
│   ├── fscale_*.json         # Phase 7: Forgetting cost scale
│   ├── epochs_*.json         # Phase 8: Epochs ablation
│   ├── combined_*.json       # Phase 9: Combined best configs
│   ├── ms_*.json             # Phase 10: Multi-seed robustness
│   └── cmob_*.txt/.json      # Phase 11: Continual MoB variants
│
├── experiments/              ← OUTDATED (v1, March 30). All used λ_ewc=5.0 (catastrophically weak).
│   └── *.json                # v1 experiments — DO NOT USE for comparison
│
├── ablation_plots/           ← SUPPORTING (Feb 24). Optuna ablation visualizations.
│   └── *.png                 # 46 plots: parameter distributions, heatmaps, sensitivity
│
├── benchmark_results.json    ← OUTDATED (Feb 24). Pseudo-label routing benchmarks.
│                               README references these. Different routing strategy than v2.
│
├── optuna_search_*.json      ← INFORMATIONAL. Hyperparameter search results.
│                               MoB best: λ=277.54, α=0.3549, β=0.4151
│                               Continual best: λ=971.27, α=0.5278, β=0.6333
│
├── mob_results_seed_42.json  ← OUTDATED (Mar 20). Early MoB run, 31.88% accuracy.
├── mob_bids_seed_42.json     ← OUTDATED. Large bid trace file (~3.1 MB).
├── mob_bids_prototype_seed_42.json  ← OUTDATED. Early prototype routing bids.
├── continual_mob_bids_prototype_seed_42.json ← OUTDATED. Early continual bids.
├── continual_mob_summary_42.txt     ← OUTDATED. 25.41% accuracy.
├── agem_seed_42.json         ← OUTDATED (Feb 8). A-GEM baseline: 56.90%.
├── monolithic_ewc_seed_42.json ← OUTDATED (Feb 8). Monolithic: 19.90%.
├── gated_moe_ewc_seed_42.json ← OUTDATED (Feb 8). GatedMoE: 35.31%.
└── pnn_seed_42.json          ← OUTDATED (Feb 4). PNN: 67.42% agnostic / 99.86% oracle.
```

**Rule of thumb**: Only `results/experiments_v2/` contains current, trustworthy results. Everything else is historical.

### 4.4 Each v2 Result File Contains

**`{name}_results.json`**: Config (alpha, beta, lambda_ewc, etc.), avg_accuracy, forgetting, final_accuracies (per-task array), task_accuracies (per-task array at training time).

**`{name}_bids.json`**: Full training trace (per-batch bids, costs, winner), eval summary (per-expert costs, per-digit routing), load balance (entropy, Gini), prototype state snapshots.

**`{name}_summary.txt`** (continual MoB only): Average accuracy, detected shift positions.

### 4.5 Outdated / Deprecated Code

| Item | Status | Reason |
|------|--------|--------|
| `EXPERIMENTS.md` | Partially outdated | Describes v1 experiments (4 phases, `results/experiments/`). v2 has 11 phases with different naming. |
| `progress_report.md` | Outdated | Written 2026-03-19, pre-dates bug fixes and v2 experiments. |
| `results/experiments/` | Outdated | v1 results with λ=5.0 (catastrophically weak EWC). Do not use. |
| `results/benchmark_results.json` | Different context | Pseudo-label routing results from Feb 2026. Not comparable to v2 prototype routing. |
| `tests/check resources/` | Reference only | Duplicated runner code for cross-validation, not actively maintained. |
| `experiment_log.txt` | Build artifact | Stdout from experiment runs. |
| `mob.zip` | Archive | Compressed copy of the codebase. |

### 4.6 Cross-Package Dependencies

```
tests/run_mob_only.py
  ├── imports from mob/ (auction, bidding, models, diagnostics, utils)
  └── imports from contibualmob/prototype_store  (cross-package: prototype routing support)

tests/run_continual_mob.py
  └── imports from contibualmob/ (pool, auction, diagnostics, utils)

tests/test_baselines.py
  ├── defines create_split_mnist()  ← reused by both runners
  └── imports from mob/ (baselines, models)
```

---

## 5. Current Results (v2 Experiment Suite)

### 5.1 Context

The v2 experiment suite was run on April 3, 2026. All experiments use **prototype (Mahalanobis distance) routing** as the evaluation strategy (except `base_pseudolabel` which uses the original pseudo-label routing for comparison). Default config: λ_ewc=1000, 4 experts, 5 tasks, epochs=4, batch_size=32, seed=42 (with multi-seed runs at seeds 123, 456).

**Important distinction from README benchmarks**: The README reports results using **pseudo-label routing** with **Optuna-tuned hyperparameters** (e.g., λ=277.54, α=0.3549). The v2 experiments use **prototype routing** with λ=1000 and systematic ablations. These are different routing strategies and should not be directly compared as "improvement" or "regression."

### 5.2 Complete Results Table

#### Phase 1: Baselines (Seed 42)

| Experiment | Routing (train/eval) | Per-Sample | Top-k | Avg Accuracy | Forgetting | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|-----------|----------------------|------------|-------|-------------|-----------|--------|--------|--------|--------|--------|
| base_pseudolabel | label / pseudo-label | No | 1 | 69.29% | 17.91% | 99.91% | 29.04% | 98.24% | 99.90% | 19.37% |
| base_prototype_perbatch | label / prototype | No | 1 | **85.64%** | **2.95%** | 99.91% | 87.17% | 99.95% | 99.90% | 41.30% |
| base_persample_k1 | label / prototype | Yes | 1 | 80.43% | 9.86% | 99.81% | 61.95% | 98.13% | 99.40% | 42.86% |
| base_persample_k2_t0.5 | label / prototype | Yes | 2 | 62.41% | 22.14% | 99.81% | 14.45% | 96.96% | 98.94% | 1.87% |
| base_persample_k2_t1.0 | label / prototype | Yes | 2 | 60.22% | 24.68% | 99.81% | 9.06% | 92.74% | 98.39% | 1.11% |

**Key findings**:
- Prototype routing (+16.35% over pseudo-label) validates the Mahalanobis distance approach as superior to pseudo-label routing.
- Per-batch routing (85.64%) outperforms per-sample routing (80.43%) on seed 42. This is counterintuitive — per-sample should be better for mixed batches. May be seed-dependent (see Phase 10).
- **k=2 hurts significantly** (60-62% vs 80-86%). The second expert adds noise rather than useful signal on MNIST's clean digit separation. The temperature parameter doesn't rescue it.
- Tasks 1, 3, 4 consistently achieve >98%. The bottleneck is Tasks 2 and 5 — one expert must handle both.

#### Phase 2: Distance-Only Evaluation (Seed 42)

| Experiment | Per-Sample | Top-k | Avg Accuracy | Forgetting | vs Full Bid Equivalent |
|-----------|------------|-------|-------------|-----------|----------------------|
| distonly_perbatch | No | 1 | **85.64%** | **2.95%** | = base_prototype_perbatch (identical) |
| distonly_persample_k1 | Yes | 1 | 74.67% | 19.31% | -5.76% vs base_persample_k1 |
| distonly_persample_k2 | Yes | 2 | 74.85% | 17.45% | +12.44% vs base_persample_k2_t0.5 |

**Key findings**:
- **Per-batch distance-only = full bidding** (85.64% both). This validates the theory: at eval time, forgetting cost is constant per expert and adds no per-sample routing signal. For per-batch routing, distance alone is sufficient.
- Per-sample distance-only drops ~6% vs full bidding. Forgetting cost may provide useful tie-breaking when per-sample distances are close.
- This is a positive result for LLM scaling: per-batch distance-only routing is computationally cheap (N matmuls) and loses nothing.

#### Phase 3: Training-Time Prototype Routing — OPEN PROBLEM (Seed 42)

| Experiment | Warmup | Avg Accuracy | Forgetting | Expert Wins |
|-----------|--------|-------------|-----------|-------------|
| trainproto_w0 | 0 | 19.43% | 95.25% | Only 2 experts active |
| trainproto_w500 | 500 | 30.71% | 73.91% | Expert 0: ~79% of wins |
| trainproto_w1000 | 1000 | 30.71% | 73.91% | Expert 0: ~79% of wins |
| trainproto_w1500 | 1500 | 30.71% | 73.91% | Expert 0: ~79% of wins |

**This is the most critical failure in the v2 results.** Training-time prototype routing collapses: one expert captures ~79% of all batches, and 3 tasks drop to 0% accuracy. The warmup doesn't help beyond 500 batches — all warmup≥500 produce identical results.

**Diagnosis**: After warmup, routing switches from label-based (which produces balanced expert usage, entropy ~0.96) to prototype-based. The prototype distances apparently create a positive feedback loop: Expert 0 wins → trains on more data → accumulates more centroids → has lower distances to more inputs → wins even more. The other experts starve and never update their prototypes.

**Root cause hypothesis**: The incremental centroids computed during training (Euclidean distance, no Mahalanobis yet) may not provide enough separation between experts. The first expert to accumulate centroids for a class dominates that class permanently. Unlike label-based routing where exec_cost provides a natural "this expert is bad at this" signal, Euclidean distance to accumulated centroids doesn't penalize an expert for being overloaded.

**This is the primary open problem for the LLM scaling story.** Without training-time label-free routing, MoB cannot replace learned routers during LLM pretraining.

#### Phase 4: Fully Label-Free Pipeline (Seed 42)

| Experiment | Config | Avg Accuracy | Forgetting |
|-----------|--------|-------------|-----------|
| labelfree_perbatch | proto train + distance eval | 30.71% | 73.91% |
| labelfree_k1 | proto train + distance eval, per-sample | 30.71% | 73.91% |
| labelfree_k2 | proto train + distance eval, per-sample k=2 | 30.71% | 73.91% |

All three converge to the same poor result. The failure originates at training time (Phase 3), not evaluation. Per-sample/k=2/perbatch variations at eval make no difference because the experts are already poorly trained due to routing collapse during training.

#### Phase 5: Alpha/Beta Ablation (Seed 42)

| Experiment | α | β | Avg Accuracy | Forgetting | Notes |
|-----------|---|---|-------------|-----------|-------|
| ab_a0.7_b0.3 | 0.7 | 0.3 | **80.81%** | 10.95% | Favoring exec cost |
| ab_a0.3_b0.7 | 0.3 | 0.7 | 77.48% | 11.03% | Favoring forget cost |
| ab_a0.3_b0.7_trainproto | 0.3 | 0.7 | 34.96% | 75.54% | Still fails with proto training |

**Key findings**:
- α=0.7, β=0.3 slightly outperforms α=0.3, β=0.7 (+3.33%) on seed 42. Exec cost weight matters more for per-sample routing accuracy.
- However, **for continual MoB** (Phase 11), α=0.3/β=0.7 produces the best result (78.83%). Higher forget cost weight helps in the task-free setting where shift detection matters.
- Alpha/beta tuning cannot rescue training-time prototype routing.

#### Phase 6: EWC Lambda Ablation (Seed 42)

| Experiment | λ_ewc | Avg Accuracy | Forgetting | Task 2 | Task 5 |
|-----------|-------|-------------|-----------|--------|--------|
| ewc_l100 | 100 | 69.23% | 24.51% | 4.41% | 45.44% |
| ewc_l500 | 500 | 76.36% | 13.53% | 46.77% | 37.22% |
| **λ=1000 (default)** | **1000** | **80.43%** | **9.86%** | **61.95%** | **42.86%** |
| ewc_l2000 | 2000 | 44.00% | 55.04% | 0.00% | 41.40% |
| ewc_l5000 | 5000 | 29.69% | 74.89% | 0.00% | 49.32% |

**Key findings**:
- Clear **U-shaped curve**: λ too low → forgetting dominates; λ too high → learning is over-constrained.
- Optimal range: **500-1000**. The Optuna-tuned value (277.54) from the README was for pseudo-label routing; prototype routing works best at higher λ.
- λ=2000+ causes catastrophic failure: Tasks 2 and 3 drop to 0% because EWC constraints prevent the overloaded expert from learning new classes.
- λ=100 gives nearly random performance on Task 2 (4.41%) — too little protection.

#### Phase 7: Forgetting Cost Scale (Seed 42)

| Experiment | Scale | Avg Accuracy | Forgetting |
|-----------|-------|-------------|-----------|
| fscale_0.5 | 0.5 | 80.43% | 9.86% |
| fscale_2.0 | 2.0 | 80.43% | 9.86% |
| fscale_3.0 | 3.0 | 80.43% | 9.86% |

**All identical.** The forgetting_cost_scale parameter has zero effect on the v2 results. This suggests the log1p normalization already compresses the forgetting cost range sufficiently, and the scale multiplier doesn't change the relative ranking between experts. This parameter can likely be removed or fixed at 1.0.

#### Phase 8: Epochs Ablation (Seed 42)

| Experiment | Epochs | Avg Accuracy | Forgetting | Task 2 | Task 5 |
|-----------|--------|-------------|-----------|--------|--------|
| epochs_2 | 2 | **84.60%** | **6.83%** | 77.87% | 52.65% |
| **epochs_4 (default)** | **4** | **80.43%** | **9.86%** | **61.95%** | **42.86%** |
| epochs_8 | 8 | 76.60% | 17.62% | 30.17% | 54.16% |

**Key findings**:
- **Fewer epochs = better accuracy + less forgetting**. 2 epochs (84.60%, 6.83% forgetting) beats 4 epochs (80.43%, 9.86%).
- More epochs causes more forgetting: Task 2 degrades from 77.87% → 61.95% → 30.17% as epochs increase. The overloaded expert overwrites earlier knowledge with more passes.
- The sweet spot is 2-4 epochs. This aligns with continual learning literature — brief exposure per task reduces interference.

#### Phase 9: Combined Best Configurations (Seed 42)

| Experiment | Config | Avg Accuracy | Forgetting |
|-----------|--------|-------------|-----------|
| combined_best | a0.3/b0.7, label train, full eval | 77.48% | 11.03% |
| combined_best_distonly | a0.3/b0.7, label train, distance eval | 74.67% | 19.31% |
| combined_best_fscale2 | a0.3/b0.7, fscale=2.0 | 77.48% | 11.03% |
| combined_best_labelfree | a0.3/b0.7, proto train, distance eval | 34.96% | 75.54% |
| combined_full | a0.3/b0.7, proto train, dist eval, fscale=2.0 | 34.96% | 75.54% |

**Key findings**:
- combined_best matches ab_a0.3_b0.7 exactly (same config).
- fscale has no effect (confirmed again).
- Any configuration with `train_routing=prototype` fails (~35% accuracy). The training-time routing collapse dominates all other hyperparameter choices.

#### Phase 10: Multi-Seed Robustness (Seeds 123, 456)

| Experiment | Seed | Avg Accuracy | Forgetting | Task 2 | Task 5 |
|-----------|------|-------------|-----------|--------|--------|
| ms_persample_s123 | 123 | **87.34%** | 10.92% | 59.79% | 81.39% |
| ms_persample_s456 | 456 | 87.10% | **5.58%** | 82.37% | 59.15% |
| ms_combined_s123 | 123 | 86.40% | 11.21% | 56.42% | 77.86% |
| ms_combined_s456 | 456 | 86.82% | 6.12% | 78.06% | 59.91% |
| ms_trainproto_s123 | 123 | 32.80% | 70.20% | 93.34% | 70.65% |
| ms_trainproto_s456 | 456 | 30.32% | 77.20% | 87.81% | 63.79% |
| ms_labelfree_s123 | 123 | 32.80% | 70.20% | 93.34% | 70.65% |
| ms_labelfree_s456 | 456 | 30.32% | 77.20% | 87.81% | 63.79% |

**Key findings**:
- **Seeds 123 and 456 significantly outperform seed 42** on label-routing configs (87.1-87.3% vs 80.4%). Seed 42 appears to produce a particularly unfortunate expert initialization.
- Seed 456 has exceptionally low forgetting (5.58-6.12%), suggesting favorable Fisher initialization.
- **Training-time prototype routing fails across all seeds** (30-33%). This confirms it's a systematic problem, not a seed artifact.
- **Best overall result: 87.34%** (seed 123, per-sample k=1, α=0.5, β=0.5, λ=1000).

#### Phase 11: Continual MoB (Task-Free Learning)

| Experiment | Config | Avg Accuracy | Shifts Detected |
|-----------|--------|-------------|-----------------|
| cmob_pseudolabel | Pseudo-label routing | 59.40% | 19 (over-sensitive) |
| cmob_prototype | Prototype eval routing | 65.35% | 19 |
| cmob_persample_k1 | Per-sample prototype | 63.99% | 19 |
| cmob_distonly | Distance-only | 55.63% | 19 |
| **cmob_a0.3_b0.7** | **α=0.3, β=0.7** | **78.83%** | **4 (correct)** |
| cmob_ms_s123 | Multi-seed s123 | 77.44% | 4 |
| cmob_ms_s456 | Multi-seed s456 | 78.58% | 4 |
| cmob_trainproto_w500 | Proto train, w500 | 4.97% | 1 (collapses early) |
| cmob_trainproto_w1000 | Proto train, w1000 | 4.97% | 1 |
| cmob_ms_trainproto_s123 | Proto train, s123 | 11.81% | 1 |
| cmob_ms_trainproto_s456 | Proto train, s456 | 10.71% | 1 |

**Key findings**:
- **α=0.3, β=0.7 is critical for continual MoB** (78.83% vs 65.35% default). Higher forgetting cost weight produces better shift detection (4 correct shifts vs 19 spurious). The forgetting cost EMA is what drives shift detection — weighting it higher makes the EMA more responsive to genuine distribution changes and less noisy.
- Continual MoB is ~8-9% below task-aware MoB (78.83% vs 87.34%). Expected — task-free learning is harder.
- **Training-time prototype routing catastrophically fails in continual setting** (4.97-11.81%). Even worse than task-aware (30-35%) because the routing collapse prevents shift detection from ever triggering (only 1 shift detected).
- Multi-seed continual results are robust: 77.44-78.83% across seeds 42, 123, 456 (with α=0.3/β=0.7).

### 5.3 Expert Load Balancing

Across all successful (label-routing) experiments:
- **Normalized entropy**: ~0.96 out of max 1.39 (for 4 experts). Good balance.
- **Expert distribution**: Roughly 19%, 20%, 21%, 40% — one expert consistently captures more load than others. This is expected: the expert that handles 2 tasks naturally wins more auctions.
- **Gini coefficient**: ~0.11 (low inequality) for label-routing configs.

For training-time prototype routing:
- **Normalized entropy**: 0.37 — severe collapse.
- **Expert distribution**: One expert captures ~79% of batches.

This contrast validates MoB's load-balancing claim for label-based routing but highlights the failure mode of prototype-based training routing.

### 5.4 Per-Task Accuracy Patterns

The consistent pattern across all successful experiments:

| Task | Classes | Typical Final Accuracy | Pattern |
|------|---------|----------------------|---------|
| Task 1 | 0, 1 | 98-99% | Nearly perfect — easy digits, dedicated expert |
| Task 2 | 2, 3 | 30-87% (high variance) | **The bottleneck** — shares expert with Task 5 |
| Task 3 | 4, 5 | 95-99% | Strong — dedicated expert |
| Task 4 | 6, 7 | 97-99% | Strong — dedicated expert |
| Task 5 | 8, 9 | 19-81% (high variance) | **The bottleneck** — shares expert with Task 2 |

Tasks 2 and 5 are handled by the same overloaded expert. The accuracy on these tasks is the primary differentiator between configurations. The overloaded expert must balance EWC protection (retaining Task 2 knowledge) with plasticity (learning Task 5). This is the core continual learning challenge.

### 5.5 Summary of Key Findings

| Finding | Evidence | Implication |
|---------|----------|------------|
| Prototype routing >> pseudo-label routing | 85.64% vs 69.29% (+16.35%) | Mahalanobis distance is a more reliable routing signal than output confidence |
| Distance-only eval = full bid (per-batch) | 85.64% both | LLM-viable: cheap eval routing with no accuracy loss |
| Training-time prototype routing fails | 30-35% across all seeds/warmup | **Open problem** — blocks LLM scaling story |
| λ_ewc sweet spot is 500-1000 | U-shaped curve; 2000+ catastrophic | Higher than pseudo-label optimal (277) |
| 2 epochs > 4 epochs > 8 epochs | 84.60% > 80.43% > 76.60% | Brief exposure reduces interference |
| k=2 hurts on MNIST | 60-62% vs 80-86% | Clean digit separation doesn't benefit from expert blending |
| Forgetting cost scale has no effect | 0.5/2.0/3.0 all identical | Parameter can be removed |
| α=0.3/β=0.7 best for continual MoB | 78.83% vs 65.35% default | Higher forget cost weight improves shift detection |
| Seed variance is significant | 87.3% (s123) vs 80.4% (s42) | Multi-seed reporting essential; seed 42 is pessimistic |
| Load balancing emerges naturally | Entropy ~0.96, Gini ~0.11 | Validates no-auxiliary-loss claim for label routing |

---

## 6. Next Steps & Open Problems

### 6.1 Critical: Fix Training-Time Prototype Routing

This is the **single most important open problem**. Without label-free training routing, MoB cannot replace learned routers in LLM pretraining (where no task labels exist).

**What's happening**: After warmup, one expert accumulates the most centroids → has lowest distances to most inputs → wins most auctions → trains on more data → accumulates more centroids → positive feedback loop → routing collapse.

**Possible investigation directions**:

1. **Bid normalization per expert**: Normalize each expert's distances by their own centroid spread, so an expert with many centroids doesn't automatically have lower absolute distances.

2. **Entropy regularization on routing**: Add a soft penalty when one expert's win rate exceeds 1/N, without using a learned auxiliary loss. Could be implemented as a bid discount for underutilized experts.

3. **Centroid competition**: When two experts have centroids for the same class, only keep the one with lower distance (more specialized). Prevents centroid duplication.

4. **Temperature annealing**: Start with high temperature (uniform routing) and gradually sharpen, allowing all experts to build prototypes before competition begins.

5. **Investigate why label-based routing avoids collapse**: The exec_cost (CE loss) naturally provides a "this expert is bad at this" signal that distance doesn't. An expert bad at class X has high CE loss on class X → high exec_cost → loses auction. But an expert that has never seen class X has no centroids for X → gets default distance 100.0 → also loses. The asymmetry may be that CE loss grows smoothly as the expert becomes less suitable, while distance jumps from 0 (has centroids) to 100 (no centroids).

### 6.2 Better Hyperparameter Defaults

Current v2 results suggest:
- **λ_ewc=1000** is good but not Optuna-tuned for prototype routing. A focused search over [500, 2000] could find the optimum.
- **epochs=2** outperforms epochs=4 on seed 42. Validate across seeds.
- **forgetting_cost_scale** has no effect — remove the parameter or investigate why.
- **α/β** depends on setting: α=0.7/β=0.3 for task-aware, α=0.3/β=0.7 for continual. Consider making this automatic.

### 6.3 Reduce Seed Variance

Seed 42 gives 80.43% while seed 123 gives 87.34% — a 7% gap from initialization alone. This likely stems from the Fisher clamping sensitivity documented in the README. Investigate:
- Whether wider Fisher clamp bounds (e.g., min=0.5 instead of 0.1) reduce seed sensitivity.
- Whether weight initialization strategy matters (e.g., initializing all experts identically, then diverging through training).

### 6.4 CIFAR-10 Scaling

The next milestone before LLM integration:
- Replace `SimpleCNN` with `ResNet-18` (512-dim features).
- Create `create_split_cifar10()` data loader.
- Increase `MIN_SAMPLES_FOR_MAHALANOBIS` (512-dim needs more samples for stable covariance).
- Everything else (auction, bidding, EWC, prototype store) should work unchanged.
- k=2 may become useful on CIFAR (harder classification than MNIST, benefit from expert combination).

### 6.5 LLM MoE Integration (Long-Term)

Key engineering challenges:
1. **Low-rank covariance**: 4096×4096 full inverse is prohibitive. Need low-rank approximation (e.g., top-k eigenvalues + diagonal residual).
2. **Transformer integration**: Expert = FFN block, features = transformer hidden states. Need to hook into the MoE layer's forward pass.
3. **Unsupervised centroids**: No class labels in LLM pretraining. Centroids must emerge from unsupervised clustering of hidden states (e.g., online k-means on features, or let classes = token types discovered via routing patterns).
4. **Gradient-free routing**: Even distance-only routing requires N matmuls per token. At 128 experts, this may still be too expensive. Investigate approximate nearest-centroid search (e.g., product quantization, locality-sensitive hashing).

### 6.6 Update Documentation

- `EXPERIMENTS.md` should be updated to reflect v2 experiment structure and naming.
- `progress_report.md` is stale — either update or remove.
- `README.md` benchmark results reflect pseudo-label routing. Consider adding a v2 prototype routing benchmark section.
