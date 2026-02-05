# Baselines Implementation Details

## Overview

This document describes the baseline implementations used to evaluate MoB (Mixture of Bidders). All baselines are designed for fair comparison on the Split-MNIST continual learning benchmark.

### Baseline Summary

| Baseline | File | Task-Aware Training | Task-Aware Eval | Fixed Params | Expected Forgetting |
|----------|------|---------------------|-----------------|--------------|---------------------|
| **Gated MoE + EWC** | `run_gated_moe_ewc.py` | ✓ | ✗ (gater routes) | ✓ | Moderate (gater forgets) |
| **Monolithic EWC** | `run_monolithic_ewc.py` | ✓ | N/A (single model) | ✓ | Moderate (capacity limit) |
| **Progressive NN** | `run_pnn_baseline.py` | ✓ | ✓ (task oracle!) | ✗ (grows) | ~Zero (by construction) |
| **Task-Aware MoB** | `run_mob_only.py` | ✓ | ✗ (auction routes) | ✓ | Low (stateless routing) |
| **Continual MoB** | `run_continual_mob.py` | ✗ (task-free) | ✗ (auction routes) | ✓ | Low (shift detection) |

---

## Running the Baselines

### 1. Gated MoE + EWC (Best Known Config)

```bash
python tests/run_gated_moe_ewc.py --seed 42 --lambda_ewc 50.0 --learning_rate 0.001 --epochs 4 --gater_hidden_size 256 --save_results
```

**Key settings:**
- `lambda_ewc=50.0`: High regularization to protect both experts and gater
- Gater EWC enabled by default (use `--disable_gater_ewc` to turn off)

---

### 2. Monolithic EWC (Fair Param Count)

```bash
python tests/run_monolithic_ewc.py --seed 42 --width_multiplier 4 --lambda_ewc 10.0 --learning_rate 0.001 --epochs 4 --save_results
```

**Key settings:**
- `width_multiplier=4`: Makes CNN 4× wider to match MoB's 4-expert parameter count
- `lambda_ewc=10.0`: Standard EWC strength (tune if needed)

---

### 3. Progressive Neural Networks

#### 3a. Fair Comparison (Capped at 4 columns = MoB param count)

```bash
python tests/run_pnn_baseline.py --seed 42 --max_columns 4 --epochs 4 --save_results
```

**Key settings:**
- `max_columns=4`: Caps columns to match MoB's 4 experts
- Task 5 reuses column 3 (hybrid approach)
- Reports BOTH task-oracle and task-agnostic accuracy

#### 3b. Full PNN (Unlimited columns - upper bound)

```bash
python tests/run_pnn_baseline.py --seed 42 --max_columns -1 --epochs 4 --save_results
```

**Key settings:**
- `max_columns=-1`: Traditional PNN with unlimited columns
- More parameters, but tests if PNN can beat MoB with any amount of capacity

---

### 4. MoB (For Comparison)

#### 4a. Task-Aware MoB

```bash
python tests/run_mob_only.py --seed 42 --alpha 0.5 --beta 0.5 --lambda_ewc 10.0 --learning_rate 0.001 --epochs 4
```

**Key settings:**
- `alpha=0.5, beta=0.5`: Equal weight to execution and forgetting costs
- `lambda_ewc=10.0`: Standard EWC strength for Task-Aware mode
- Uses explicit task boundaries with Fisher updates after each task

#### 4b. Continual MoB (Task-Free / Online)

```bash
python tests/run_continual_mob.py --seed 42 --alpha 0.5 --beta 0.5 --lambda_ewc 40.0 --shift_threshold 2.0 --epochs 4
```

**Key settings:**
- `lambda_ewc=40.0`: Higher EWC strength for online setting (no explicit task boundaries)
- `shift_threshold=2.0`: Sensitivity for automatic shift detection
- Shift detection triggers Fisher consolidation when distribution change is detected

---

### Quick Comparison Script

Run all baselines sequentially:

```bash
# Gated MoE + EWC
python tests/run_gated_moe_ewc.py --seed 42 --lambda_ewc 50.0 --save_results

# Monolithic EWC
python tests/run_monolithic_ewc.py --seed 42 --width_multiplier 4 --lambda_ewc 10.0 --save_results

# PNN (Capped)
python tests/run_pnn_baseline.py --seed 42 --max_columns 4 --save_results

# PNN (Unlimited)
python tests/run_pnn_baseline.py --seed 42 --max_columns -1 --save_results

# Task-Aware MoB
python tests/run_mob_only.py --seed 42 --lambda_ewc 10.0

# Continual MoB (Online)
python tests/run_continual_mob.py --seed 42 --lambda_ewc 40.0 --shift_threshold 2.0
```

Results are saved to `results/` directory as JSON files.

# 1. Gated MoE + EWC (`run_gated_moe_ewc.py`)

## Purpose

Tests the standard Mixture-of-Experts approach with a **learned gater**. The key hypothesis is that the gater itself suffers from catastrophic forgetting, causing routing degradation over time.

## Architecture

### StandardGater (MLP)

```
Input (28×28 = 784)
    ↓
Flatten
    ↓
Linear(784 → 256) + ReLU + Dropout(0.3)
    ↓
Linear(256 → num_experts)
    ↓
Softmax → Routing Probabilities
```

**Parameters:** ~201,732

### Expert Model (SimpleCNN × 4)

Each expert uses the same SimpleCNN as MoB:

```
Input (1×28×28)
    ↓
Conv2d(1→32, 3×3) + ReLU + MaxPool(2×2)
    ↓
Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)
    ↓
Dropout2d(0.25)
    ↓
Flatten (3136)
    ↓
Linear(3136→128) + ReLU + Dropout(0.5)
    ↓
Linear(128→10)
```

**Parameters per expert:** ~421,642

### Total Parameters

```
Experts: 4 × 421,642 = 1,686,568
Gater:   201,732
Total:   1,888,300
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 4 | Number of expert CNNs |
| `lambda_ewc` | 50.0 | EWC regularization strength |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `epochs` | 4 | Epochs per task |
| `gater_hidden_size` | 256 | Gater MLP hidden dimension |
| `gater_ewc` | True | Apply EWC to gater too |

## Training Pipeline

### Per-Task Training Loop

```python
for task_id, task_loader in enumerate(train_tasks):
    for epoch in range(epochs_per_task):
        for x, y in task_loader:
            # 1. Gater produces routing probabilities
            gating_logits = gater(x)
            gating_probs = softmax(gating_logits)
            
            # 2. Batch-level top-1 routing (mode of argmax)
            winner_ids = gating_probs.argmax(dim=-1)
            winner_id = winner_ids.mode().values.item()
            
            # 3. Forward through selected expert
            expert_output = expert_models[winner_id](x)
            
            # 4. Compute losses
            task_loss = cross_entropy(expert_output, y)
            ewc_penalty = expert_ewc[winner_id].penalty()
            gater_ewc_penalty = gater_ewc_estimator.penalty()  # If enabled
            
            total_loss = task_loss + ewc_penalty + gater_ewc_penalty
            
            # 5. End-to-end backprop (trains BOTH gater and expert)
            total_loss.backward()
            gater_optimizer.step()
            expert_optimizers[winner_id].step()
    
    # Update Fisher for trained experts and gater
    update_fisher_after_task(task_loader)
```

**Key difference from MoB:** Gradients flow through the gater, training it end-to-end on the task loss.

### Gater EWC (Optional)

When `--gater_ewc` is enabled, the gater's Fisher Information is computed using **self-supervised targets**:

```python
def _update_gater_fisher(dataloader):
    for x, y in dataloader:
        gating_logits = gater(x)
        
        # Use gater's OWN predictions as targets
        targets = gating_logits.argmax(dim=-1).detach()
        log_probs = log_softmax(gating_logits, dim=-1)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        
        # Accumulate squared gradients as Fisher
        fisher[n] += grad.pow(2) * batch_size
```

This preserves the gater's routing decisions from previous tasks.

## Evaluation (Task-Agnostic)

```python
def evaluate_all(dataloader):
    for x, y in dataloader:
        # Gater decides routing (no task oracle)
        gating_probs = softmax(gater(x))
        routes = gating_probs.argmax(dim=-1)  # Per-sample routing
        
        # Get predictions from routed experts
        for sample_idx in range(batch_size):
            expert_id = routes[sample_idx].item()
            pred = expert_outputs[expert_id][sample_idx].argmax()
```

**Important:** The gater routes based on learned patterns, NOT task ID.

## Expected Behavior

- **Gater Forgetting:** The gater learns to route Task 1 to Expert 0. When trained on Task 2, the gater's weights change, potentially disrupting Task 1 routing.
- **EWC Mitigation:** Gater EWC helps but may not fully prevent routing degradation.
- **Expert Preservation:** Individual experts are protected by EWC.

---

# 2. Monolithic EWC (`run_monolithic_ewc.py`)

## Purpose

Tests whether a **single large model** with EWC can match MoB's performance. This baseline has the SAME total parameter count as MoB (4 experts).

The key question: **Is the multi-expert architecture actually beneficial, or would a single large model with EWC work just as well?**

## Architecture

### Wide SimpleCNN (width_multiplier=4)

```
Input (1×28×28)
    ↓
Conv2d(1→128, 3×3) + ReLU + MaxPool(2×2)     # 4× wider: 32×4=128
    ↓
Conv2d(128→256, 3×3) + ReLU + MaxPool(2×2)   # 4× wider: 64×4=256
    ↓
Dropout2d(0.25)
    ↓
Flatten (256×7×7 = 12,544)
    ↓
Linear(12544→512) + ReLU + Dropout(0.5)       # 4× wider: 128×4=512
    ↓
Linear(512→10)
```

**Total Parameters:** ~6,425,354 (approximately 4× SimpleCNN)

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width_multiplier` | 4 | CNN width multiplier (matches 4 experts) |
| `lambda_ewc` | 10.0 | EWC regularization strength |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `epochs` | 4 | Epochs per task |

## Training Pipeline

### Per-Task Training Loop

```python
for task_id, task_loader in enumerate(train_tasks):
    for epoch in range(epochs_per_task):
        for x, y in task_loader:
            # 1. Forward through single model
            logits = model(x)
            
            # 2. Compute losses
            task_loss = cross_entropy(logits, y)
            ewc_penalty = ewc_estimator.penalty()
            
            total_loss = task_loss + ewc_penalty
            
            # 3. Backprop
            total_loss.backward()
            optimizer.step()
    
    # Update Fisher after each task
    ewc_estimator.update_fisher(task_loader, num_samples=200)
```

**Key difference from MoB:** No routing - all parameters are used for all inputs.

## Evaluation (Single Model)

```python
def evaluate(dataloader):
    for x, y in dataloader:
        logits = model(x)
        preds = logits.argmax(dim=-1)
```

No routing needed - just forward through the single model.

## Expected Behavior

- **Capacity Sharing:** All parameters contribute to all tasks (good for transfer, bad for isolation)
- **EWC Protection:** Fisher-weighted penalty protects important parameters
- **Interference:** New task gradients may still interfere with old knowledge
- **No Specialization:** Unlike MoB, cannot have task-specific experts

## Comparison with MoB

| Aspect | Monolithic EWC | MoB |
|--------|----------------|-----|
| **Routing** | None | VCG Auction |
| **Capacity Isolation** | Shared | Per-expert |
| **Task Specialization** | Implicit (EWC) | Explicit (routing) |
| **Routing Forgetting** | N/A | Zero (stateless) |

---

# 3. Progressive Neural Networks (`run_pnn_baseline.py`)

## Purpose

Provides a **zero-forgetting upper bound**. PNN is the gold standard for preventing catastrophic forgetting but has significant limitations.

Paper: Rusu et al., 2016 - "Progressive Neural Networks"

## Architecture

### PNN Column (SimpleCNN + Lateral Connections)

Each column mirrors SimpleCNN architecture:

```
Input (1×28×28)
    ↓
Conv2d(1→32, 3×3) + ReLU + MaxPool(2×2)
    ↓
Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)
    ↓
Dropout2d(0.25)
    ↓
Flatten (3136)
    ↓                         ┌──────────────────────────┐
    ├── FC1 (3136→128) ←──────┤ Lateral from prev cols   │
    ↓                         │ (3136 × num_prev → 128)  │
ReLU + Dropout(0.5)           └──────────────────────────┘
    ↓                         ┌──────────────────────────┐
    ├── FC2 (128→10) ←────────┤ Lateral from prev cols   │
    ↓                         │ (128 × num_prev → 10)    │
Output (10 classes)           └──────────────────────────┘
```

### Lateral Connections

For column `k`, lateral connections aggregate features from columns `0, 1, ..., k-1`:

```python
# FC1 with lateral
fc1_out = fc1(conv_out)
if lateral_fc1 is not None:
    lateral_input = concat([prev_cols[i].conv_out for i in range(k)])
    fc1_out = fc1_out + lateral_fc1(lateral_input)

# FC2 with lateral
logits = fc2(fc1_out)
if lateral_fc2 is not None:
    lateral_input = concat([prev_cols[i].fc1_out for i in range(k)])
    logits = logits + lateral_fc2(lateral_input)
```

### Parameter Growth

| After Task | Columns | Parameters (approx) |
|------------|---------|---------------------|
| Task 1 | 1 | ~421,642 |
| Task 2 | 2 | ~875,394 (lateral adds ~32K) |
| Task 3 | 3 | ~1,361,256 |
| Task 4 | 4 | ~1,879,228 |
| Task 5 | 5 | ~2,429,310 |

Parameters grow super-linearly due to lateral connections.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_columns` | 4 | Max columns: 4=match MoB params, -1=unlimited (full PNN) |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `epochs` | 4 | Epochs per task |

No EWC needed - freezing prevents forgetting.

**Note:** With `max_columns=4`, the 5th task reuses the last column (like a hybrid approach). Use `--max_columns -1` for unlimited columns (traditional PNN).

## Training Pipeline

### Per-Task Training Loop

```python
for task_id, task_loader in enumerate(train_tasks):
    # 1. Check max_columns limit
    if max_columns > 0 and len(columns) >= max_columns:
        # Reuse last column (don't freeze it)
        column = columns[-1]
    else:
        # Freeze all existing columns
        for col in columns:
            for param in col.parameters():
                param.requires_grad = False
        
        # 2. Add new column with lateral connections
        new_column = PNNColumn(
            column_id=len(columns),
            previous_columns=list(columns)
        )
        columns.append(new_column)
    
    # 3. Train current column
    optimizer = Adam(filter(lambda p: p.requires_grad, column.parameters()))
    
    for epoch in range(epochs_per_task):
        for x, y in task_loader:
            logits, _, _ = column(x)
            loss = cross_entropy(logits, y)
            
            loss.backward()
            optimizer.step()
```

**Key property:** Previous columns are FROZEN - no gradients, no forgetting.

## Evaluation Modes

### 1. Task Oracle Mode (Traditional PNN)

```python
def evaluate_task(dataloader, task_id):
    # REQUIRES knowing which task this is!
    column_id = task_to_column[task_id]
    column = columns[column_id]
    
    for x, y in dataloader:
        logits, _, _ = column(x)
        preds = logits.argmax(dim=-1)
```

**CRITICAL LIMITATION:** Requires task ID at inference (unfair advantage over MoB).

### 2. Task-Agnostic Mode (Fair Comparison with MoB)

```python
def evaluate_task_agnostic(dataloader):
    for x, y in dataloader:
        # Run through ALL columns
        all_logits = []
        all_confidences = []
        
        for col in columns:
            logits, _, _ = col(x)
            all_logits.append(logits)
            
            # Confidence = max softmax probability
            probs = softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
            all_confidences.append(confidence)
        
        # Per-sample: select column with HIGHEST confidence
        confidences = stack(all_confidences)
        selected_columns = confidences.argmax(dim=0)
        
        # Get predictions from selected columns
        for sample_idx in range(batch_size):
            col_id = selected_columns[sample_idx]
            pred = all_logits[col_id][sample_idx].argmax()
```

**This mode is comparable to MoB** - no task oracle, uses confidence-based routing.

## Expected Behavior

- **Zero Forgetting:** Previous columns are frozen, so Task 1 accuracy never degrades
- **Knowledge Transfer:** Lateral connections enable feature reuse
- **Parameter Growth:** Not suitable for long sequences (unless max_columns is set)
- **Two Evaluation Modes:** Task-oracle gives upper bound; task-agnostic is fair comparison

## Comparison with MoB

| Aspect | PNN (Task Oracle) | PNN (Task Agnostic) | MoB |
|--------|-------------------|---------------------|-----|
| **Forgetting** | Zero | Near-Zero | Low (EWC) |
| **Task Oracle** | ✓ Required | ✗ Confidence routing | ✗ Auction |
| **Parameter Count** | Grows (or capped) | Grows (or capped) | Fixed |
| **Practical Deployment** | Limited | Better | Best |

---

# Comparison Summary

## Parameter Counts (5 Tasks)

| Baseline | Total Parameters | Notes |
|----------|-----------------|-------|
| **Gated MoE + EWC** | ~1,888,300 | 4 experts + gater |
| **Monolithic EWC** | ~6,425,354 | Wide CNN (4× width) |
| **PNN (capped)** | ~4,103,110 | 4 columns (matches MoB experts) |
| **PNN (unlimited)** | ~2,429,310 | After 5 tasks (grows) |
| **Task-Aware MoB** | ~1,686,568 | 4 experts (no gater) |
| **Continual MoB** | ~1,686,568 | 4 experts (no gater) |

## Training Comparison

| Baseline | Task Boundaries | Fisher Updates | Routing During Training |
|----------|-----------------|----------------|------------------------|
| **Gated MoE + EWC** | ✓ Known | After each task | Learned gater |
| **Monolithic EWC** | ✓ Known | After each task | None (single model) |
| **PNN** | ✓ Known | None (freezing) | None (column per task) |
| **Task-Aware MoB** | ✓ Known | After each task | VCG Auction (stateless) |
| **Continual MoB** | ✗ Unknown | On shift detection | VCG Auction (stateless) |

## Evaluation Comparison

| Baseline | Task Oracle | Routing Mechanism | Forgetting Source |
|----------|-------------|-------------------|-------------------|
| **Gated MoE + EWC** | ✗ | Learned gater | Gater forgetting |
| **Monolithic EWC** | N/A | None | Parameter interference |
| **PNN (oracle)** | ✓ | Task ID → Column | None (by construction) |
| **PNN (agnostic)** | ✗ | Confidence-based | None (by construction) |
| **Task-Aware MoB** | ✗ | Auction (stateless) | Expert forgetting (EWC mitigates) |
| **Continual MoB** | ✗ | Auction (stateless) | Expert forgetting (EWC mitigates) |

## Expected Ranking

Based on theoretical properties:

```
Zero Forgetting: PNN > MoB > Gated MoE ≈ Monolithic EWC
Task Agnostic:   MoB = Gated MoE = Monolithic EWC > PNN (requires oracle)
Fixed Params:    MoB = Gated MoE = Monolithic EWC > PNN (grows)
Task-Free:       Continual MoB > Task-Aware MoB = others (require boundaries)
Practical:       Continual MoB > Task-Aware MoB > Monolithic EWC > Gated MoE > PNN
```

---

## References

- **EWC:** Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks" (PNAS)
- **Online EWC:** Schwarz et al. (2018) "Progress & Compress" (ICML)
- **PNN:** Rusu et al. (2016) "Progressive Neural Networks" (arXiv)
- **MoE/Gating:** Shazeer et al. (2017) "Outrageously Large Neural Networks" (ICLR)
- **Mixtral:** Jiang et al. (2024) "Mixtral of Experts" (arXiv)
