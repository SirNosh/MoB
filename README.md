# MoB: Mixture of Bidders

## A Continual Learning Framework with Auction-Based Expert Routing

---

## Table of Contents

1. [What is MoB? (Plain English Introduction)](#what-is-mob-plain-english-introduction)
2. [The Problem: Catastrophic Forgetting](#the-problem-catastrophic-forgetting)
3. [The MoB Solution](#the-mob-solution)
4. [Theoretical Foundation](#theoretical-foundation)
5. [Architecture Overview](#architecture-overview)
6. [Implementation Details](#implementation-details)
   - [Task-Aware MoB](#task-aware-mob)
   - [Online/Continual MoB](#onlinecontinual-mob)
7. [Critical Implementation Fixes](#critical-implementation-fixes)
8. [Baselines](#baselines)
9. [Benchmark Results](#benchmark-results)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Project Structure](#project-structure)
13. [Configuration Reference](#configuration-reference)
14. [Development Timeline](#development-timeline)
15. [References](#references)

---

## What is MoB? (Plain English Introduction)

Imagine you're running a company with multiple specialist employees. When a new project comes in, instead of assigning it to someone randomly or having a single manager decide (who might forget who's good at what), you hold an auction. Each specialist bids based on two things:

1. **"How well can I handle this?"** - If they're already good at similar work, they bid low
2. **"How much would this hurt my existing skills?"** - If taking this project would make them forget their current expertise, they bid high

The specialist with the lowest total bid wins the project. This simple mechanism naturally leads to specialization: specialists who become good at certain types of work keep winning those projects, while protecting their expertise from being diluted by unrelated work.

**MoB (Mixture of Bidders)** applies this exact idea to neural networks learning continuously over time:

- **Experts** are neural networks (CNNs in our case)
- **Projects** are batches of data to learn from
- **Bids** combine predicted performance and estimated forgetting
- **The auction** routes each batch to the expert with the lowest bid

### Why This Matters

Traditional machine learning assumes you have all your data upfront. But real-world AI systems often need to learn new things over time without forgetting what they already know. This is called **continual learning** or **lifelong learning**.

The classic problem is **catastrophic forgetting**: when you train a neural network on new data, it tends to completely forget what it learned before. MoB addresses this by:

1. Having multiple experts that can specialize
2. Using an auction mechanism that naturally protects experts from learning incompatible things
3. Applying EWC (Elastic Weight Consolidation) regularization within each expert

The key innovation is that the auction mechanism is **stateless** - it doesn't learn or remember anything, so it can never forget how to route data correctly. This eliminates "gater forgetting" that plagues traditional Mixture-of-Experts systems.

---

## The Problem: Catastrophic Forgetting

### Standard Neural Network Forgetting

When you train a neural network on Task A, then train it on Task B, the weights optimized for Task A get overwritten:

```
Initial:     Task A Training:    Task B Training:
  ???    -->   Good at A     -->   Good at B, BAD at A!
```

### Gater Forgetting in Mixture-of-Experts

Standard MoE systems use a learned "gater" network to route inputs to experts:

```python
output = sum(gater(x)[i] * expert[i](x) for i in range(num_experts))
```

The problem: **the gater itself forgets!**

```
Task 1: Gater learns "digit 0,1 -> Expert A"
Task 5: Gater overwrites to "digit 8,9 -> Expert A"
Result: Task 1 samples now route to wrong expert!
```

Even if Expert A perfectly remembers digits 0 and 1, the gater no longer sends those digits to it.

### MoB's Solution: Stateless Routing

MoB replaces the learned gater with an auction mechanism that:
- Requires **no learning** (no parameters to forget)
- Computes routing **fresh every batch** based on current costs
- Always selects the expert with the lowest cost (most suitable)

---

## The MoB Solution

### Core Innovation

Replace learned gating with auction-based routing:

```
             Traditional MoE                      MoB
            ┌─────────────┐                 ┌─────────────┐
   Input -> │ Learned     │ -> Expert       │ Stateless   │ -> Expert
            │ Gater (can  │                 │ Auction     │
            │ forget!)    │                 │ (immune to  │
            └─────────────┘                 │ forgetting) │
                                            └─────────────┘
```

### How Bidding Works

Each expert computes a bid for each batch:

```
bid = α × execution_cost + β × forgetting_cost
```

Where:
- **Execution cost**: How well can this expert currently handle this data? (Lower = expert is already good at this)
- **Forgetting cost**: How much would training on this data damage existing knowledge? (Lower = safe to train)
- **α, β**: Balancing weights (typically 0.5 each)

**Key insight**: An expert that's already good at certain data will have:
- Low execution cost (it predicts well)
- Low forgetting cost (gradients align with existing Fisher information)
- Therefore, **low bid = wins the auction**

### The Auction Mechanism

The auction selects the expert with the lowest bid:

```python
winner = argmin(bids)  # Lowest bid wins
```

This is a simple **lowest-bid-wins** allocation mechanism:
- **Efficient**: Always selects the expert with the lowest combined cost
- **Stateless**: No learned parameters, immune to forgetting
- **Deterministic**: Same inputs always produce the same routing decision

---

## Theoretical Foundation

### Auction-Based Routing

The auction mechanism provides:

1. **Allocative Efficiency**: Always selects the expert with minimum total cost
2. **Statelessness**: No learned parameters means no forgetting at the routing level
3. **Determinism**: Routing decisions are reproducible given the same expert states

### Elastic Weight Consolidation (EWC)

Each expert uses EWC to protect important parameters:

```
L_total = L_task + (λ/2) × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)²
```

Where:
- `L_task`: Cross-entropy loss on current batch
- `λ`: EWC regularization strength
- `Fᵢ`: Fisher Information for parameter i (importance for past tasks)
- `θᵢ`: Current parameter value
- `θ*ᵢ`: Optimal parameter value from previous tasks

### Input-Dependent Forgetting Cost

The forgetting cost estimates gradient interference:

```
forgetting_cost = Σᵢ Fᵢ × (∂L/∂θᵢ)²
```

This measures how much training on the current batch would conflict with important (Fisher-weighted) parameters. High interference = high forgetting cost = high bid = expert likely loses the auction.

### Online EWC

We use Online EWC (Schwarz et al., 2018) with exponential moving average:

```python
# After each task:
F_new = compute_fisher(current_task_data)
F_total = γ × F_old + (1-γ) × F_new  # γ = 0.9

# Similarly for optimal parameters:
θ*_total = γ × θ*_old + (1-γ) × θ_current
```

This prevents unbounded Fisher growth across tasks.

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        MoB System                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │  Expert 0   │   │  Expert 1   │   │  Expert N   │        │
│  │  ┌───────┐  │   │  ┌───────┐  │   │  ┌───────┐  │        │
│  │  │ Model │  │   │  │ Model │  │   │  │ Model │  │        │
│  │  └───────┘  │   │  └───────┘  │   │  └───────┘  │        │
│  │  ┌───────┐  │   │  ┌───────┐  │   │  ┌───────┐  │        │
│  │  │ EWC   │  │   │  │ EWC   │  │   │  │ EWC   │  │        │
│  │  │Engine │  │   │  │Engine │  │   │  │Engine │  │        │
│  │  └───────┘  │   │  └───────┘  │   │  └───────┘  │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Bid Collection Layer                   │    │
│  │   bid_0 = α×exec_0 + β×forget_0                     │    │
│  │   bid_1 = α×exec_1 + β×forget_1                     │    │
│  │   ...                                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Auction Mechanism                      │    │
│  │   winner = argmin(bids)                             │    │
│  │   payment = second_lowest_bid                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│                    Winner trains on batch                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Expert Model: SimpleCNN

Each expert uses a SimpleCNN architecture (~421,642 parameters):

```
Input (1×28×28)
    ↓
Conv2d(1→32, 3×3, padding=1) + ReLU + MaxPool(2×2)
    ↓
Conv2d(32→64, 3×3, padding=1) + ReLU + MaxPool(2×2)
    ↓
Dropout2d(0.25)
    ↓
Flatten (64×7×7 = 3136)
    ↓
Linear(3136→128) + ReLU + Dropout(0.5)
    ↓
Linear(128→10)
    ↓
Output (10 classes)
```

### Dataset: Split-MNIST

The benchmark uses Split-MNIST, a standard continual learning dataset:

| Task | Classes | Train Samples | Description |
|------|---------|---------------|-------------|
| 1 | 0, 1 | ~12,000 | Digits zero and one |
| 2 | 2, 3 | ~12,000 | Digits two and three |
| 3 | 4, 5 | ~12,000 | Digits four and five |
| 4 | 6, 7 | ~12,000 | Digits six and seven |
| 5 | 8, 9 | ~12,000 | Digits eight and nine |

**Challenge**: With 4 experts and 5 tasks, at least one expert must handle 2 tasks without forgetting the first.

---

## Implementation Details

### Task-Aware MoB

**File**: `tests/run_mob_only.py`
**Module**: `mob/`

Task-Aware MoB operates with explicit task boundaries:

```python
for task_id, task_loader in enumerate(train_tasks):
    # Train on task
    for epoch in range(epochs_per_task):
        for x, y in task_loader:
            # 1. Collect bids from all experts
            bids, components = pool.collect_bids(x, y)

            # 2. Run auction (lowest bid wins)
            winner_id = argmin(bids)

            # 3. Train ONLY the winning expert
            pool.train_winner(winner_id, x, y, optimizers)

    # Update Fisher for experts that won batches
    for eid in winning_experts:
        pool.experts[eid].update_after_task(task_loader)
```

**Key features**:
- Explicit task boundaries (Fisher updates after each task)
- LwF (Learning without Forgetting) support (optional)
- Optimizer reset at task end (recommended)

### Bid Computation

```python
def compute_bid(self, x, y):
    # Get raw costs
    raw_exec = self.exec_estimator.compute_predicted_loss(x, y)
    raw_forget = self.forget_estimator.compute_forgetting_cost(x, y)

    # Normalize execution cost (cross-entropy ~2.3 untrained, ~0.1 trained)
    norm_exec = raw_exec / 2.5

    # Normalize forgetting cost (log scale for huge range)
    norm_forget = math.log1p(raw_forget) / 10.0

    # Final bid
    bid = alpha * norm_exec + beta * norm_forget
    return bid
```

**Why log-scale for forgetting cost?** Raw forgetting costs range from 0 to 500,000+. Log transformation (`log1p`) compresses this naturally while preserving relative differences.

### Online/Continual MoB

**File**: `tests/run_continual_mob.py`
**Module**: `contibualmob/`

Online MoB operates without task boundaries:

```python
# Create continuous data stream
stream_loader = create_continuous_stream(train_tasks)

for batch_idx, (x, y) in enumerate(stream_loader):
    # 1. Auction + Training
    bids, _ = pool.collect_bids(x, y)
    winner_id = argmin(bids)
    metrics = pool.train_winner(winner_id, x, y, optimizers)

    # 2. Shift Detection
    if metrics['shift_detected']:
        # Consolidate knowledge
        pool.consolidate(replay_buffer, expert_ids=active_experts)
        replay_buffer = []
```

**Key features**:
- No explicit task boundaries (task-free learning)
- Automatic shift detection via EMA cost tracking
- Selective consolidation based on replay buffer

### Shift Detection

```python
class ShiftDetector:
    def __init__(self, alpha=0.99, threshold_multiplier=50.0):
        self.alpha = alpha  # EMA smoothing factor
        self.threshold_multiplier = threshold_multiplier
        self.ema_cost = None
        self.shift_cooldown = 0

    def update(self, cost):
        # Check for significant upward spike
        baseline = max(self.ema_cost, 0.5)
        is_shift = cost > (baseline * self.threshold_multiplier)

        # Update EMA
        self.ema_cost = self.alpha * self.ema_cost + (1 - self.alpha) * cost

        if is_shift:
            self.shift_cooldown = 50  # Absorb new distribution

        return is_shift
```

**Note**: With optimized hyperparameters (λ_ewc=971.27, shift_threshold=2.58), Online MoB achieves **90.22% accuracy**, outperforming Task-Aware MoB (79.03%). See Training Configuration Details for full parameters.

### Evaluation Routing

At evaluation time, MoB uses **pseudo-label routing** (no ground truth labels):

```python
def evaluate_all(self, dataloader):
    for x, y in dataloader:
        # Use model's own predictions as pseudo-labels
        for expert in self.experts:
            logits = expert.model(x)
            pseudo_labels = logits.argmax(dim=-1)

            # Compute bid with pseudo-labels
            forget_cost = expert.compute_forgetting_cost(x, pseudo_labels)
            bids[expert.id] = forget_cost

        # Route to expert with lowest forgetting cost
        winner_id = argmin(bids)
        predictions = experts[winner_id].model(x).argmax(dim=-1)
```

**Key insight**: The expert whose Fisher information is most aligned with the input will have the lowest forgetting cost, naturally routing data to the correct specialist.

**Note on evaluation reproducibility**: The test DataLoader uses `shuffle=True`, so batch ordering varies between iterations. The main `tests/run_mob_only.py` includes an "Individual Expert Accuracies" diagnostic that iterates test data before ensemble evaluation, while `tests/check resources/run_mob_only.py` only performs ensemble evaluation. This extra iteration changes the random state, causing slightly different routing decisions (~78% vs ~79% accuracy with identical training).

---

## Critical Implementation Fixes

### 1. EWC Fisher Clamping (Critical)

**Problem**: EWC effectiveness varies dramatically based on model initialization. Creating multiple models changes the random state, giving some experts "bad" initializations with weak Fisher protection (Fisher max varies up to 18x between initializations).

**Solution**: Clamp Fisher values to a minimum of 0.1:

```python
def _normalize_fisher(self):
    # Normalize to mean = 1.0
    all_fisher = torch.cat([f.flatten() for f in self.fisher.values()])
    fisher_mean = all_fisher.mean()

    for n in self.fisher:
        self.fisher[n] = self.fisher[n] / (fisher_mean + 1e-30)
        # CRITICAL: Clamp to minimum value
        self.fisher[n] = torch.clamp(self.fisher[n], min=0.1)
```

**Result**: Expert handling 2 tasks improved from 0% to 87% retention on first task.

**Location**: `mob/bidding.py` and `contibualmob/bidding.py` in `_normalize_fisher()` method.

### 2. Optimizer Reset

**Problem**: Adam optimizer accumulates momentum from previous tasks. When an expert switches to a new task, stale momentum can hurt learning.

**Solution**:
- **Task-aware MoB**: Reset ALL winning experts' optimizers at task END (after Fisher update)
- **Continual MoB**: Reset winner's optimizer when shift is detected

```python
# Task-Aware MoB: Reset at task end
if config.get('reset_optimizer', False):
    for eid in winning_experts:
        optimizers[eid] = torch.optim.Adam(
            pool.experts[eid].model.parameters(),
            lr=config['learning_rate']
        )
```

**Flag**: `--reset_optimizer` (recommended to always enable)

---

## Baselines

MoB is compared against several baseline methods:

### 1. Gated MoE + EWC

**File**: `tests/run_gated_moe_ewc.py`

Traditional Mixture-of-Experts with learned gating:

- **Architecture**: 4 SimpleCNN experts + MLP gater
- **Parameters**: ~2,090,540 (more than MoB due to gater)
- **Training**: End-to-end with gater + expert gradients
- **EWC**: Applied to both experts and gater
- **Key limitation**: Gater forgets how to route previous tasks

```bash
python tests/run_gated_moe_ewc.py --lambda_ewc 50.0 --gater_ewc
```

### 2. Monolithic EWC

**File**: `tests/run_monolithic_ewc.py`

Single wide CNN with EWC regularization:

- **Architecture**: SimpleCNN with width_multiplier=2 (~1,682,954 params)
- **Purpose**: Test if multi-expert architecture is actually beneficial
- **Key limitation**: All parameters shared for all tasks

```bash
python tests/run_monolithic_ewc.py --width_multiplier 2 --lambda_ewc 10.0
```

### 3. A-GEM (Averaged Gradient Episodic Memory)

**File**: `tests/run_agem_baseline.py`

Gradient projection method (Chaudhry et al., 2019):

- **Architecture**: Wide SimpleCNN (~1,682,954 params)
- **Memory**: Episodic memory buffer for gradient projection
- **Key feature**: Projects gradients to not increase loss on memory

```bash
python tests/run_agem_baseline.py --memory_size 256
```

### 4. Experience Replay (ER)

**File**: `tests/run_er_baseline.py`

Simple yet effective replay-based method (Rolnick et al., 2019):

- **Architecture**: Wide SimpleCNN (~1,682,954 params)
- **Memory**: Reservoir sampling buffer
- **Key feature**: Joint training on current + replay batches

```bash
python tests/run_er_baseline.py --memory_size 256 --replay_batch_size 32
```

### 5. Progressive Neural Networks (PNN)

**File**: `tests/run_pnn_baseline.py`

Zero-forgetting architecture (Rusu et al., 2016):

- **Architecture**: New column per task with lateral connections
- **Parameters**: Grows with each task (~6,135,642 after 5 tasks)
- **Key feature**: Previous columns frozen = zero forgetting
- **Key limitation**: Requires task oracle at inference

```bash
python tests/run_pnn_baseline.py --max_columns 4
```

### Baseline Comparison Summary

| Method | Params | Task Oracle | Fixed Params | Expected Forgetting |
|--------|--------|-------------|--------------|---------------------|
| **MoB-TaskAware** | 1.69M | No | Yes | Low |
| **MoB-Online** | 1.69M | No | Yes | Low |
| **Gated MoE + EWC** | 2.09M | No | Yes | Moderate (gater forgets) |
| **Monolithic EWC** | 1.68M | N/A | Yes | Moderate |
| **A-GEM** | 1.68M | N/A | Yes | Low |
| **Experience Replay** | 1.68M | N/A | Yes | Low |
| **PNN** | 6.14M | Yes | No | Zero |

---

## Benchmark Results

All results from `benchmark_results.json`. Experiments conducted with:
- **Seed**: 42 (deterministic for reproducibility)
- **Device**: CUDA (GPU-accelerated)
- **Epochs per task**: 4
- **Batch size**: 32
- **Dataset**: Split-MNIST (5 tasks, 2 classes per task)

### Summary: Accuracy and Forgetting

| Method | Avg Accuracy | Avg Forgetting | Total Params | Trainable Params |
|--------|-------------|----------------|--------------|------------------|
| **Experience Replay** | **97.48%** | **2.71%** | 1,682,954 | 1,682,954 |
| **MoB-TaskAware** | 79.03% | 20.72% | 1,686,568 | 1,686,568 |
| **A-GEM** | 75.68% | 30.24% | 1,682,954 | 1,682,954 |
| **PNN**† | 73.58% | 0.00% | 6,135,642 | 2,032,532 |
| **MoB-Online** | 90.22% | 0.00% | 1,686,568 | 1,686,568 |
| **Monolithic+EWC** | 37.34% | 68.14% | 1,682,954 | 1,682,954 |
| **GatedMoE+EWC** | 19.86% | 0.00% | 2,090,540 | 2,090,540 |

† **PNN Evaluation Modes**:
- **Task-Agnostic (73.58%)**: Fair comparison with MoB - uses confidence-based routing without knowing which task is being evaluated. Column selections: `{0: 2012, 1: 1069, 2: 2964, 3: 3004, 4: 951}`
- **Task-Oracle (99.84%)**: Unfair advantage - PNN is told which column to use for each task. This is the standard PNN evaluation but not comparable to MoB which must route without task knowledge.

### Per-Task Accuracy: Training vs Final

This table shows the accuracy achieved immediately after training each task (Training) versus the accuracy measured at the end of all training (Final). The difference reveals forgetting.

**Training Accuracies (measured immediately after each task):**

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|--------|--------|--------|--------|--------|--------|
| **ER** | 99.91% | 99.76% | 99.95% | 99.55% | 99.09% |
| **MoB-TaskAware** | 99.91% | 98.97% | 99.95% | 99.90% | 86.43% |
| **A-GEM** | 99.95% | 100.0% | 100.0% | 99.95% | 99.45% |
| **PNN** | 99.91% | 99.66% | 100.0% | 100.0% | 99.65% |
| **MoB-Online** | 99.86% | 0.00% | 99.95% | 99.09% | 0.00% |
| **Mono+EWC** | 99.95% | 93.29% | 89.38% | 89.27% | 87.34% |
| **Gated MoE** | 99.01% | 0.20% | 0.00% | 0.05% | 0.00% |

**Final Accuracies (measured after all 5 tasks complete):**

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Δ Task 1 |
|--------|--------|--------|--------|--------|--------|----------|
| **ER** | 98.72% | 96.87% | 95.84% | 96.88% | 99.09% | -1.19% |
| **MoB-TaskAware** | 99.91% | 26.44% | 89.70% | 99.80% | 79.32% | +0.00% |
| **A-GEM** | 79.57% | 71.65% | 62.11% | 65.61% | 99.45% | -20.38% |
| **PNN** | 99.91% | 99.66% | 100.0% | 100.0% | 99.65% | +0.00% |
| **MoB-Online** | 99.86% | 0.00% | 99.95% | 99.09% | 0.00% | +0.00% |
| **Mono+EWC** | 99.34% | 0.00% | 0.00% | 0.00% | 87.34% | -0.61% |
| **Gated MoE** | 99.05% | 0.20% | 0.00% | 0.05% | 0.00% | +0.04% |

**Key observations from per-task data:**
- **ER maintains uniformly high accuracy** across all tasks (95-99%), with minimal per-task forgetting
- **MoB-TaskAware retains Task 1 perfectly** (99.91%) but significantly forgets Task 2 (98.97% → 26.44%)
- **A-GEM shows backward interference** - earlier tasks degrade as later tasks are learned (Task 1: 99.95% → 79.57%)
- **PNN achieves perfect retention** by freezing previous columns (all training = final)
- **Monolithic+EWC exhibits severe forgetting** on middle tasks (Tasks 2-4 drop to 0%)
- **Gated MoE never properly learns** Tasks 2-5, demonstrating catastrophic gater forgetting

**MoB-Online Per-Digit Accuracies (with optimized hyperparameters):**

| Digit | Accuracy | Routing |
|-------|----------|---------|
| 0 | 99.90% | Expert 3 (979), Expert 1 (1) |
| 1 | 99.74% | Expert 3 (1133), Expert 1 (2) |
| 2 | 86.72% | Expert 1 (1032) |
| 3 | 93.96% | Expert 1 (1010) |
| 4 | 98.88% | Expert 2 (971), Expert 1 (11) |
| 5 | 99.10% | Expert 2 (885), Expert 1 (7) |
| 6 | 98.54% | Expert 0 (945), Expert 1 (13) |
| 7 | 97.76% | Expert 0 (1007), Expert 1 (21) |
| 8 | 39.43% | Expert 1 (974) |
| 9 | 87.12% | Expert 1 (1009) |

**Key insight**: MoB-Online achieves excellent routing for 8/10 digits. Digit 8 shows the main weakness (39.43%) where Expert 1 handles it but was primarily trained on digits 2-3. Despite this, the overall 90.22% accuracy demonstrates effective automatic shift detection without explicit task boundaries.

### Complete Resource Usage

| Method | Train Time | Eval Time | Total Time | Train VRAM | Eval VRAM | Peak RAM |
|--------|------------|-----------|------------|------------|-----------|----------|
| **MoB-TaskAware** | 88.98s | 5.00s | 93.97s | 54.79 MB | 49.99 MB | 1,256.68 MB |
| **MoB-Online** | 101.77s | 5.58s | 107.35s | 55.15 MB | 53.48 MB | 1,373.02 MB |
| **GatedMoE+EWC** | 58.59s | 3.30s | 61.89s | 62.93 MB | 54.62 MB | 1,330.41 MB |
| **Monolithic+EWC** | 46.62s | 1.73s | 48.35s | 95.71 MB | 67.23 MB | 1,303.27 MB |
| **A-GEM** | 54.37s | 1.70s | 56.07s | 101.07 MB | 54.68 MB | 1,308.27 MB |
| **ER** | 46.60s | 1.73s | 48.33s | 72.23 MB | 54.38 MB | 1,312.61 MB |
| **PNN** | 57.74s | 7.86s | 65.60s | 91.95 MB | 75.66 MB | 1,313.34 MB |

### Throughput and Computational Cost (FLOPs)

| Method | Train Throughput | Eval Throughput | Train FLOPs | Eval FLOPs |
|--------|-----------------|-----------------|-------------|------------|
| **MoB-TaskAware** | 2,697 samples/s | 2,001 samples/s | 190.95B | 2.67B |
| **MoB-Online** | 2,358 samples/s | 1,792 samples/s | 190.85B | 2.65B |
| **GatedMoE+EWC** | 4,099 samples/s | 3,029 samples/s | 190.95B | 2.67B |
| **Monolithic+EWC** | 5,150 samples/s | 5,796 samples/s | 743.37B | 10.40B |
| **A-GEM** | 4,416 samples/s | 5,889 samples/s | 743.37B | 10.40B |
| **ER** | 5,153 samples/s | 5,783 samples/s | 743.37B | 10.40B |
| **PNN** | 4,159 samples/s | 1,272 samples/s | 3,327.22B | 46.56B |

**FLOP Analysis:**
- **MoB variants are most compute-efficient** at ~191B FLOPs (only one expert trains per batch)
- **Monolithic baselines** (ER, A-GEM, Mono+EWC) use ~743B FLOPs (3.9x more than MoB)
- **PNN is most expensive** at 3,327B FLOPs (17.4x more than MoB) due to growing architecture

### Memory Efficiency Analysis

| Method | Train VRAM | vs MoB | Eval VRAM | Peak RAM |
|--------|------------|--------|-----------|----------|
| **MoB-TaskAware** | 54.79 MB | baseline | 49.99 MB | 1,256.68 MB |
| **MoB-Online** | 55.15 MB | +0.7% | 53.48 MB | 1,373.02 MB |
| **GatedMoE+EWC** | 62.93 MB | +14.9% | 54.62 MB | 1,330.41 MB |
| **ER** | 72.23 MB | +31.8% | 54.38 MB | 1,312.61 MB |
| **PNN** | 91.95 MB | +67.8% | 75.66 MB | 1,313.34 MB |
| **Monolithic+EWC** | 95.71 MB | +74.7% | 67.23 MB | 1,303.27 MB |
| **A-GEM** | 101.07 MB | +84.5% | 54.68 MB | 1,308.27 MB |

**Memory observations:**
- **MoB has the lowest VRAM footprint** - only ~55 MB during training
- **A-GEM requires 85% more VRAM** than MoB due to gradient projection overhead
- **Gated MoE uses 15% more VRAM** than MoB due to gater network gradients
- **System RAM usage is similar** across methods (~1.3 GB), dominated by PyTorch overhead

### Forgetting Pattern Analysis

| Method | Forgetting Metric | Forgetting Pattern |
|--------|-------------------|-------------------|
| **ER** | 2.71% | Uniform mild degradation across all tasks |
| **MoB-TaskAware** | 20.72% | Significant on Task 2; Tasks 1,4 near-perfect; Task 3 moderate |
| **A-GEM** | 30.24% | Gradual backward transfer (earlier tasks hurt more) |
| **PNN** | 0.00% | Zero forgetting (frozen columns) |
| **MoB-Online** | 0.00% | Failed to learn Tasks 2, 5 (not forgetting, but not learning) |
| **Monolithic+EWC** | 68.14% | Catastrophic on middle tasks (2, 3, 4 → 0%) |
| **GatedMoE+EWC** | 0.00% | Gater never learned proper routing |

**Forgetting metric** = average of (training accuracy - final accuracy) across tasks that were successfully learned.

### Training Configuration Details

The benchmark used optimized hyperparameters from Optuna-based Bayesian search:

| Method | Key Hyperparameters |
|--------|---------------------|
| **MoB-TaskAware** | α=0.3549, β=0.4151, λ_ewc=277.54, lr=0.001028, forgetting_scale=2.1314, reset_optimizer=True |
| **MoB-Online** | α=0.5278, β=0.6333, λ_ewc=971.27, lr=0.000683, shift_threshold=2.58, forgetting_scale=0.7949, reset_optimizer=True |
| **GatedMoE+EWC** | λ_ewc=5924.15, lr=0.000219, gater_hidden=512, gater_ewc=False |
| **Monolithic+EWC** | λ_ewc=1791.93, lr=0.000707, width_multiplier=2 |
| **A-GEM** | memory_size=1024, memory_batch=64, lr=0.000102, width_multiplier=2 |
| **ER** | memory_size=2048, replay_batch=16, replay_weight=1.7043, lr=0.000552 |
| **PNN** | max_columns=-1 (unlimited), lr=0.000494 |

### Key Insights and Analysis

1. **Experience Replay achieves the best accuracy** (97.48%) with minimal forgetting (2.71%). The combination of reservoir sampling and joint training proves highly effective on Split-MNIST.

2. **MoB-TaskAware demonstrates selective retention**: Tasks 1, 4 maintain near-perfect accuracy (99.9%), Task 3 retains well (89.7%), but Task 2 suffers significant forgetting (26.44%). Expert 3 handles both Task 2 and Task 5, causing interference.

3. **Gated MoE validates MoB's core hypothesis**: The learned gater network fails to properly route after Task 1, achieving only 19.86% overall accuracy. This is precisely the "gater forgetting" problem MoB was designed to solve.

4. **PNN achieves zero forgetting** but at significant cost:
   - 3.6x more total parameters (6.14M vs 1.69M)
   - 17.4x more FLOPs (3,327B vs 191B)
   - **Task-Oracle mode (99.84%)**: Knows which column to use - unfair advantage
   - **Task-Agnostic mode (73.58%)**: Confidence-based routing - fair comparison with MoB
   - When fairly compared (task-agnostic), PNN performs below MoB-TaskAware (73.58% vs 79.03%)

5. **MoB is most memory-efficient**: 54.79 MB peak VRAM vs 62.93 MB for Gated MoE (13% savings) and 101.07 MB for A-GEM (46% savings). This comes from training only one expert per batch.

6. **A-GEM shows backward interference**: Unlike MoB's targeted forgetting on Task 2, A-GEM's forgetting affects all previous tasks roughly equally (Task 1 drops 20.38%, others 25-35%).

7. **Monolithic+EWC fails on middle tasks**: Tasks 2, 3, 4 all drop to 0% accuracy, while Task 1 and 5 are retained. This suggests EWC alone cannot protect a single network from sequential overwriting.

8. **Online MoB outperforms Task-Aware MoB**: With optimized hyperparameters (λ_ewc=971.27, shift_threshold=2.58), MoB-Online achieves **90.22% accuracy**, surpassing MoB-TaskAware (79.03%). This demonstrates that automatic shift detection can be more effective than explicit task boundaries when properly tuned.

9. **Why Online MoB succeeds**: MoB-Online's shift detection allows more granular expert specialization. While Task-Aware MoB forces one expert per task (leading to Expert 3 handling both Task 2 and Task 5), Online MoB can detect sub-task patterns and route more flexibly. The per-digit results show Expert 3 specializes on digits 0-1, Expert 1 on digits 2-3 and 8-9, Expert 2 on digits 4-5, and Expert 0 on digits 6-7.

10. **Training time vs accuracy tradeoff**: ER and Monolithic+EWC are fastest (46.6s) but ER achieves 97.48% while Mono+EWC achieves only 37.34%. MoB-TaskAware takes 89s for 79.03%.

11. **Throughput reflects architecture**: Monolithic methods achieve ~5,150 samples/s while MoB methods achieve ~2,600 samples/s due to bid computation overhead. However, MoB uses 4x fewer FLOPs per training run.

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/mob.git
cd mob

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision tqdm numpy matplotlib seaborn scipy
```

### Verify Installation

```bash
python -c "from mob import ExpertPool, PerBatchVCGAuction; print('MoB installed successfully!')"
```

---

## Usage

### Quick Start: Run Task-Aware MoB

```bash
python tests/run_mob_only.py --seed 42 --lambda_ewc 277.54 --alpha 0.355 --beta 0.415 --forgetting_cost_scale 2.13 --reset_optimizer
```

### Run Online MoB

```bash
python tests/run_continual_mob.py --seed 42 --lambda_ewc 971.27 --alpha 0.5278 --beta 0.6333 --shift_threshold 2.58 --learning_rate 0.000683 --forgetting_cost_scale 0.7949 --reset_optimizer
```

### Run All Baselines Comparison

```bash
python tests/test_baselines.py --single-seed 42
```

### Run Full Benchmark with Resource Tracking

```bash
python tests/benchmark_resources.py --save_results
```

### Run Individual Baselines

```bash
# Gated MoE + EWC
python tests/run_gated_moe_ewc.py --lambda_ewc 50.0 --gater_ewc --save_results

# Monolithic EWC
python tests/run_monolithic_ewc.py --width_multiplier 2 --lambda_ewc 10.0 --save_results

# A-GEM
python tests/run_agem_baseline.py --memory_size 256 --save_results

# Experience Replay
python tests/run_er_baseline.py --memory_size 256 --replay_batch_size 32 --save_results

# Progressive Neural Networks
python tests/run_pnn_baseline.py --max_columns 4 --save_results
```

### Multi-Seed Statistical Analysis

```bash
python tests/test_baselines.py --seeds 5
```

---

## Project Structure

```
MoB Final/
├── mob/                           # Task-Aware MoB module
│   ├── __init__.py               # Package exports
│   ├── models.py                 # Neural network architectures (SimpleCNN, LeNet5, MLP)
│   ├── bidding.py                # ExecutionCostEstimator, EWCForgettingEstimator
│   ├── auction.py                # PerBatchAuction, SealedBidProtocol
│   ├── expert.py                 # MoBExpert class
│   ├── pool.py                   # ExpertPool management
│   ├── baselines.py              # NaiveFineTuning, RandomAssignment, MonolithicEWC, GatedMoE
│   ├── bid_diagnostics.py        # BidLogger for analysis
│   └── utils.py                  # Utilities (set_seed, etc.)
│
├── contibualmob/                  # Online/Continual MoB module # I know its supposed to be continual mob, its a typo and I like it
│   ├── __init__.py
│   ├── models.py                 # Same architectures as mob/
│   ├── bidding.py                # Same bidding logic with Fisher clamping
│   ├── auction.py                # Same auction mechanism
│   ├── expert.py                 # MoBExpert with consolidate() method
│   ├── pool.py                   # ExpertPool with ShiftDetector
│   ├── bid_diagnostics.py
│   └── utils.py
│
├── tests/                         # Experiment runners
│   ├── run_mob_only.py           # Task-Aware MoB experiments
│   ├── run_continual_mob.py      # Online MoB experiments
│   ├── run_gated_moe_ewc.py      # Gated MoE baseline
│   ├── run_monolithic_ewc.py     # Monolithic EWC baseline
│   ├── run_agem_baseline.py      # A-GEM baseline
│   ├── run_er_baseline.py        # Experience Replay baseline
│   ├── run_pnn_baseline.py       # Progressive Neural Networks
│   ├── test_baselines.py         # Full baseline comparison
│   ├── benchmark_resources.py    # Resource usage benchmarking
│   ├── hyperparameter_search.py  # Optuna hyperparameter optimization
│   ├── analyze_mob_bids.py       # Bid diagnostics analysis
│   ├── formula_comparison.py     # Forgetting cost formula comparisons
│   ├── analyze_ablation.py       # Ablation study analysis
│   │
│   └── check resources/          # Resource-instrumented experiment runners
│       ├── benchmark_all.py      # Run all 7 models with resource tracking
│       ├── resource_utils.py     # ResourceTracker, FlopCounter utilities
│       ├── run_mob_only.py       # MoB Task-Aware with resource tracking
│       ├── run_continual_mob.py  # MoB Online with resource tracking
│       ├── run_gated_moe_ewc.py  # Gated MoE with resource tracking
│       ├── run_monolithic_ewc.py # Monolithic EWC with resource tracking
│       ├── run_agem_baseline.py  # A-GEM with resource tracking
│       ├── run_er_baseline.py    # Experience Replay with resource tracking
│       ├── run_pnn_baseline.py   # PNN with resource tracking
│       ├── sanity_check_ewc.py   # EWC lambda sweep diagnostic
│       └── results/              # Resource benchmark outputs
│
├── results/                       # Experiment outputs
│   ├── benchmark_results.json    # Main benchmark results
│   └── ...                       # Various result files
│
└── README.md                      # This file
```

---

## Configuration Reference

### Task-Aware MoB (`run_mob_only.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--num_experts` | 4 | Number of expert CNNs |
| `--alpha` | 0.5 | Execution cost weight in bid |
| `--beta` | 0.5 | Forgetting cost weight in bid |
| `--lambda_ewc` | 1000.0 | EWC regularization strength |
| `--learning_rate` | 0.001 | Adam optimizer LR |
| `--epochs` | 4 | Epochs per task |
| `--batch_size` | 32 | Training batch size |
| `--reset_optimizer` | False | Reset optimizer at task end |
| `--use_lwf` | False | Enable Learning without Forgetting |
| `--lwf_temperature` | 2.0 | Temperature for LwF soft targets |
| `--lwf_alpha` | 0.1 | Weight for LwF distillation loss |

### Online MoB (`run_continual_mob.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--num_experts` | 4 | Number of expert CNNs |
| `--alpha` | 0.5 | Execution cost weight |
| `--beta` | 0.5 | Forgetting cost weight |
| `--lambda_ewc` | 40.0 | EWC regularization (higher for streaming) |
| `--shift_threshold` | 2.0 | Multiplier for shift detection |
| `--learning_rate` | 0.001 | Adam optimizer LR |
| `--epochs` | 4 | Repetitions per task in stream |
| `--reset_optimizer` | False | Reset optimizer on shift |

### Gated MoE + EWC (`run_gated_moe_ewc.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lambda_ewc` | 50.0 | EWC strength for experts |
| `--gater_ewc` | True | Apply EWC to gater |
| `--gater_hidden_size` | 256 | Gater MLP hidden dimension |
| `--learning_rate` | 0.001 | Adam LR |

### Experience Replay (`run_er_baseline.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--memory_size` | 256 | Total replay buffer size |
| `--replay_batch_size` | 32 | Samples replayed per batch |
| `--replay_weight` | 1.0 | Weight for replay loss |
| `--width_multiplier` | 2 | CNN width (matches MoB params) |

### A-GEM (`run_agem_baseline.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--memory_size` | 256 | Episodic memory size |
| `--memory_batch_size` | 32 | Samples for reference gradient |
| `--width_multiplier` | 2 | CNN width (matches MoB params) |

### PNN (`run_pnn_baseline.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_columns` | 4 | Max columns (-1 = unlimited) |
| `--epochs` | 4 | Epochs per task |

---

## Development Timeline

This section tells the story of how MoB evolved from a research question to a working continual learning framework.

### The Origin: November 2025

The project began in mid-November 2025 with a simple observation about Mixture-of-Experts architectures. In standard MoE systems, a learned gater network routes inputs to specialized experts. But what happens when you train this system on sequential tasks? The gater itself is a neural network - and neural networks forget.

The initial repository was created on November 11, 2025. The first few days focused on establishing the basic infrastructure: getting MNIST data loading working, setting up the expert models (SimpleCNN with ~421K parameters each), and implementing a basic bidding mechanism. Early commits show the characteristic struggles of any new project - fixing device handling issues (CUDA vs CPU), debugging learning rate problems, and getting the bid logging to work properly so the auction dynamics could be analyzed.

By mid-November, the codebase had evolved from a proof-of-concept to something more structured. There was an important architectural shift from "TIL to DIL" - task-incremental learning to domain-incremental learning. In task-incremental learning, the model knows which task it's evaluating at test time. In domain-incremental learning (which MoB targets), the model must figure out routing without task labels. This shift fundamentally changed how evaluation worked - moving from "tell me which task, I'll route to the right expert" to "figure it out yourself based on the data."

The bid logging infrastructure from this period proved essential. Understanding why certain experts won certain batches - seeing the raw execution costs, forgetting costs, and final bids - was crucial for debugging the routing behavior and tuning the α/β balance.

### The Quiet Months: December 2025 - January 2026

The repository shows less visible activity during this period, but behind the scenes work continued. In January 2026, an HPC branch appeared - suggesting the project moved to high-performance computing infrastructure for more intensive experimentation. This branch was created, deleted, recreated, and force-pushed multiple times, indicating active development and iteration on ideas that weren't ready for the main branch.

This period was likely spent on the harder problems: getting EWC to actually work reliably across different random seeds, tuning the balance between execution cost and forgetting cost in the bidding formula, and running enough experiments to understand the stability-plasticity tradeoff. The Fisher Information computation proved particularly tricky - computed correctly, it captures which parameters are important for previous tasks; computed incorrectly, it either protects nothing or freezes the network entirely.

The key technical challenges being tackled included:
- **Normalization strategy**: Raw forgetting costs span an enormous range (0 to 500,000+). Various approaches were tried before settling on log-scale normalization with `log1p`.
- **Evaluation routing**: How do you route at test time without labels? The eventual solution used pseudo-labels (the model's own predictions) to compute bids.
- **Online EWC**: The standard EWC accumulates Fisher Information unboundedly. The solution was exponential moving average (decay factor 0.9) following Schwarz et al.'s "Progress & Compress" approach.

### The Rebuild: Early February 2026

In early February, the project underwent a significant restructuring. The entire codebase was reorganized and pushed as "Working Intermediate MoB" - a fresh start that consolidated everything learned in the previous months.

The architecture that emerged was clean and modular:

**The Expert (`MoBExpert`)**: Each expert wraps a SimpleCNN neural network with two cost estimators. The `ExecutionCostEstimator` simply computes the cross-entropy loss on a batch - lower loss means the expert is already good at this data. The `EWCForgettingEstimator` maintains Fisher Information matrices and computes how much training on new data would interfere with important parameters. The bidding formula combines these: `bid = α × exec_cost + β × forget_cost`, where both costs are normalized to roughly 0-1 ranges.

**The Pool (`ExpertPool`)**: The pool manages multiple experts and orchestrates the auction. For each batch, it collects bids from all experts, runs a simple argmin to select the winner, and trains only that expert. Crucially, the pool contains no learned parameters - it's purely a coordination mechanism.

**The Gater Problem (Baseline)**: The `GatedMoE` baseline demonstrates what MoB avoids. It uses a small MLP gater (`Flatten → Linear → ReLU → Dropout → Linear`) that learns to route inputs to experts. During training, gradients flow through both the selected expert AND the gater simultaneously. The gater learns routing patterns - but it also forgets them when new tasks arrive.

The rebuild also included the first proper baselines:
- **Gated MoE + EWC**: Demonstrates the gater forgetting problem MoB solves
- **Monolithic EWC**: Tests whether multiple experts actually help
- **Progressive Neural Networks**: Zero-forgetting baseline that grows per task
- **NaiveFineTuning**: Lower bound showing maximum forgetting
- **RandomAssignment**: Isolates the value of intelligent routing

Around this time, a critical assessment document was written - a brutally honest evaluation of what was working and what wasn't. This document identified several problems, including a particularly troubling failure mode where digit 8 achieved only 27.62% accuracy in the Continual MoB variant. The assessment also noted that the "VCG auction" terminology was technically incorrect since the payment mechanism was never actually used.

### The Breakthrough: Fisher Clamping

One of the most important discoveries came while investigating why some experts retained knowledge dramatically better than others. The problem seemed almost random - sometimes an expert would retain 90%+ accuracy on its first task while learning a second, other times it would completely forget.

The investigation revealed that when creating multiple expert models sequentially, the random number generator state changes. This meant different experts got different initializations, and some of those initializations happened to have much weaker Fisher Information signals than others. Max Fisher values ranged from 514 to 3,901 across different initializations - nearly an 18x difference.

The fix was elegant: clamp all normalized Fisher values to a minimum of 0.1. This ensures that even "unimportant" parameters get basic protection against drift. The code change was just one line:

```python
self.fisher[n] = torch.clamp(self.fisher[n], min=0.1)
```

But the impact was transformative. Experts handling multiple tasks went from 0% to 87% retention on their first task.

### Expanding the Baselines: February 2026

With the core MoB working reliably, attention turned to making the comparisons more rigorous. Two additional baselines were implemented, both using reservoir sampling to maintain fixed-size memory buffers:

**A-GEM (Averaged Gradient Episodic Memory)** takes a geometric approach from Chaudhry et al.'s ICLR 2019 paper. It maintains an episodic memory of samples from previous tasks. Before each gradient update, it computes a "reference gradient" on the memory samples. If the current gradient would increase loss on past data, it projects the gradient onto a direction that doesn't. The key insight: you don't need to store all past data or replay it - just use it to constrain gradient directions.

**Experience Replay** uses a simpler strategy following Rolnick et al.'s NeurIPS 2019 work. It maintains a fixed-size buffer using Vitter's reservoir sampling (ensuring uniform random selection across all seen samples) and jointly trains on current data AND replay batches. The loss is simply summed: `total_loss = current_loss + replay_weight × replay_loss`. Despite its conceptual simplicity, Experience Replay achieved 97.48% average accuracy - the highest of any method tested.

Both baselines were carefully matched to MoB's parameter count by using a width-multiplied SimpleCNN (~1.7M parameters), ensuring fair computational comparison.

The hyperparameter search infrastructure was also significantly upgraded during this period, moving from grid search to Optuna-based Bayesian optimization with TPE (Tree-structured Parzen Estimator) sampling. The new search used percentile pruning to skip bad configurations early and evaluated each configuration across 5 seeds (42, 123, 456, 789, 1024) to ensure robustness.

### Another Discovery: Optimizer State

A subtler issue emerged during continued testing. Adam optimizer momentum from previous tasks was interfering with learning new ones. When an expert that had been training on digits 0-1 suddenly needed to learn digits 8-9, the accumulated momentum was pointing in unhelpful directions.

The solution was to reset the optimizer at natural transition points:
- **Task-Aware MoB**: Reset all winning experts' optimizers at task END (after Fisher update)
- **Online MoB**: Reset the winning expert's optimizer when a distribution shift is detected

For Online MoB, shift detection uses EMA-based anomaly detection in the `ShiftDetector` class. It maintains a smoothed average of execution costs (α=0.99) and triggers when the current cost exceeds a dynamic threshold: `cost > max(ema_cost, 0.5) × threshold_multiplier`. After triggering, a 50-batch cooldown prevents repeated false positives as the system adapts to the new distribution. This provides natural "task boundaries" in a task-free setting.

### Experimental Features: LwF Integration

An experimental feature was also added during this period: **Learning without Forgetting (LwF)** following Li & Hoiem's ECCV 2016 paper. The idea is to use knowledge distillation - before learning a new task, record the soft targets (temperature-scaled softmax outputs) from experts on the new data. During training, add a KL divergence loss that encourages the expert to maintain similar outputs on the new data:

```python
distill_loss = lwf_alpha * T² * KL(current_outputs / T, stored_targets / T)
```

The temperature T=2.0 softens the probability distributions, making them more informative for distillation. The lwf_alpha weight (recommended <0.3) balances distillation against task learning. LwF only activates for experts that already have Fisher information - new experts don't need to preserve nonexistent knowledge.

### The Final Push: February 2026

The last major phase focused on comprehensive resource benchmarking. A dedicated `tests/check resources/` directory was created containing instrumented versions of all experiment runners. The `ResourceTracker` class uses a background thread to sample system RAM (via psutil) and captures GPU VRAM at key points. The `FlopCounter` uses PyTorch's flop counting utilities (or analytical fallback for older versions) to estimate computational cost.

These track:
- Wall-clock time for training and evaluation
- Peak GPU VRAM usage (allocated and reserved)
- System RAM consumption via background thread sampling
- Training throughput in samples per second
- FLOP counts for forward and backward passes

This produced the benchmark results comparing all seven methods on equal footing, measuring not just accuracy and forgetting but also computational cost.

The project documentation was consolidated into a single comprehensive README, replacing the various markdown files that had accumulated during development.

### Key Lessons Learned

Throughout this development process, several important insights emerged:

**Fisher Clamping is Critical**: Without a minimum clamp on normalized Fisher values, EWC behavior is inconsistent across model initializations. This single change had more impact than weeks of hyperparameter tuning.

**Optimizer State Matters for Continual Learning**: Stale momentum from previous tasks hurts learning. Resetting optimizers at natural boundaries (task changes or detected shifts) improves results.

**Pseudo-Label Routing Works**: At evaluation time, ground truth labels aren't available for routing decisions. Using the model's own predictions (pseudo-labels) to compute bids actually works well - the expert whose Fisher information best aligns with the input naturally wins.

**Log-Scale Normalization**: Raw forgetting costs can range from 0 to 500,000+. Using `log1p` compresses this range while preserving relative differences between experts.

**Simple Baselines Are Hard to Beat**: Experience Replay, despite being conceptually straightforward, achieved 97.48% average accuracy - higher than any other method including MoB. Never underestimate simple approaches with good implementations.

### Current State

The project has reached a stable point:

- **Task-Aware MoB**: Achieves 79.03% average accuracy with optimized hyperparameters (λ_ewc=277.54, α=0.355, β=0.415) and Fisher clamping. Outperforms PNN in task-agnostic evaluation (79.03% vs 73.58%) while using 3.6x fewer parameters.
- **Online MoB**: Achieves **90.22%** with optimized hyperparameters (λ_ewc=971.27, shift_threshold=2.58), outperforming Task-Aware MoB despite having no task boundary information.
- **Five Baselines**: All implemented with resource tracking
- **Comprehensive Benchmarks**: Full comparison across accuracy, forgetting, and resource usage
---

## References

### Core Methods

1. **EWC**: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks" - PNAS
2. **Online EWC**: Schwarz et al. (2018) "Progress & Compress: A scalable framework for continual learning" - ICML
3. **Auction Theory**: Vickrey (1961) "Counterspeculation, Auctions, and Competitive Sealed Tenders"

### Baselines

4. **PNN**: Rusu et al. (2016) "Progressive Neural Networks" - arXiv
5. **A-GEM**: Chaudhry et al. (2019) "Efficient Lifelong Learning with A-GEM" - ICLR
6. **Experience Replay**: Rolnick et al. (2019) "Experience Replay for Continual Learning" - NeurIPS
7. **MoE/Gating**: Shazeer et al. (2017) "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" - ICLR

### Related Work

8. **LwF**: Li & Hoiem (2016) "Learning without Forgetting" - ECCV
9. **SI**: Zenke et al. (2017) "Continual Learning Through Synaptic Intelligence" - ICML
10. **Mixtral**: Jiang et al. (2024) "Mixtral of Experts" - arXiv

---

## License

[MIT License](LICENSE)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mob2026,
  title = {MoB: Mixture of Bidders - Continual Learning with Auction-Based Expert Routing},
  year = {2026},
  url = {https://github.com/your-repo/mob}
}
```
