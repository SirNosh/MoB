# MoB (Mixture of Bidders) - Implementation Details

## Overview

MoB is a continual learning framework that replaces learned gating in Mixture-of-Experts (MoE) architectures with a VCG (Vickrey-Clarke-Groves) auction mechanism. The key insight is that auction-based routing is **stateless** and therefore immune to catastrophic forgetting at the routing level.

This codebase contains two primary experiment runners:

| Script | Module | Mode | Description |
|--------|--------|------|-------------|
| `tests/run_mob_only.py` | `mob/` | Task-Aware | Sequential task training with explicit task boundaries |
| `tests/run_continual_mob.py` | `contibualmob/` | Task-Free | Continuous data stream with automatic shift detection |

---

## Architecture

### Expert Model: SimpleCNN

Each expert uses a SimpleCNN architecture defined in `mob/models.py`:

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

**Key features:**
- Dynamic FC layer initialization based on input size
- Supports `width_multiplier` parameter for fair baseline comparisons
- ~421,642 parameters per expert (at width_multiplier=1)

---

## Dataset: Split-MNIST

Both experiments use **Split-MNIST**, a standard continual learning benchmark:

| Task | Classes | Description |
|------|---------|-------------|
| Task 1 | 0, 1 | Digits zero and one |
| Task 2 | 2, 3 | Digits two and three |
| Task 3 | 4, 5 | Digits four and five |
| Task 4 | 6, 7 | Digits six and seven |
| Task 5 | 8, 9 | Digits eight and nine |

**Properties:**
- **Total samples:** 60,000 training, 10,000 test
- **Per task:** ~12,000 training samples (~6,000 per class)
- **Image size:** 28×28 grayscale
- **Challenge:** Tasks are presented sequentially; model must retain knowledge of all tasks

---

# Part 1: `run_mob_only.py` (Task-Aware MoB)

## Module Structure

Uses local classes defined within the script plus imports from `mob/`:

```
mob/
├── models.py        # SimpleCNN, LeNet5, MLP architectures
├── bidding.py       # ExecutionCostEstimator, EWCForgettingEstimator
├── auction.py       # PerBatchVCGAuction (VCG mechanism)
├── expert.py        # MoBExpert (baseline version, unused in run_mob_only)
├── pool.py          # ExpertPool (baseline version, unused in run_mob_only)
├── bid_diagnostics.py  # BidLogger for analysis
└── utils.py         # set_seed()

tests/run_mob_only.py  # Contains MoBExpertLocal, ExpertPoolLocal (used here)
```

**Why Local Classes?**

The script defines `MoBExpertLocal` and `ExpertPoolLocal` classes that are copies of the original classes. This allows modifying MoB-specific features (e.g., forgetting-cost routing, LwF) without affecting baseline comparisons in `test_baselines.py`.

---

## Configuration & Hyperparameters

| Parameter | Default | CLI Argument | Description |
|-----------|---------|--------------|-------------|
| `num_experts` | 4 | `--num_experts` | Number of expert neural networks |
| `num_tasks` | 5 | - | Number of sequential tasks (Split-MNIST) |
| `batch_size` | 32 | `--batch_size` | Samples per training batch |
| `epochs_per_task` | 4 | `--epochs` | Training epochs for each task |
| `learning_rate` | 0.001 | `--learning_rate` | Adam optimizer learning rate |
| `alpha` (α) | 0.5 | `--alpha` | Weight for normalized execution cost |
| `beta` (β) | 0.5 | `--beta` | Weight for normalized forgetting cost |
| `lambda_ewc` (λ) | 10.0 | `--lambda_ewc` | EWC regularization strength |
| `forgetting_cost_scale` | 1.0 | `--forgetting_cost_scale` | Scaling factor for forgetting cost in bidding |
| `seed` | 42 | `--seed` | Random seed for reproducibility |

### LwF (Learning without Forgetting) Parameters

| Parameter | Default | CLI Argument | Description |
|-----------|---------|--------------|-------------|
| `use_lwf` | False | `--use_lwf` | Enable knowledge distillation |
| `lwf_temperature` | 2.0 | `--lwf_temperature` | Temperature for soft targets |
| `lwf_alpha` | 0.1 | `--lwf_alpha` | Weight for LwF distillation loss (recommended < 0.3) |

---

## Core Components

### 1. MoBExpertLocal Class

Located in `tests/run_mob_only.py` (lines 56-280). Encapsulates an expert neural network with bidding and training logic.

**Initialization:**
```python
MoBExpertLocal(
    expert_id: int,
    model: nn.Module,
    alpha: float,           # Execution cost weight
    beta: float,            # Forgetting cost weight
    lambda_ewc: float,      # EWC regularization strength
    forgetting_cost_scale: float = 1.0,
    device: torch.device = None,
    use_lwf: bool = False,
    lwf_temperature: float = 2.0,
    lwf_alpha: float = 0.1
)
```

**Key Attributes:**
- `model`: The SimpleCNN neural network
- `exec_estimator`: `ExecutionCostEstimator` instance
- `forget_estimator`: `EWCForgettingEstimator` instance
- `batches_won`: Total batches won across all tasks
- `batches_won_this_task`: Batches won in current task
- `lwf_soft_targets`: Dictionary mapping batch_idx → soft targets tensor

---

### 2. Bid Computation

Each expert computes a bid using **raw scaled normalization** (defined in `compute_bid()` method):

```python
def compute_bid(self, x, y):
    raw_exec = self.exec_estimator.compute_predicted_loss(x, y)
    raw_forget = self.forget_estimator.compute_forgetting_cost(x, y)
    
    # EXECUTION: Raw scaled (cross-entropy ~2.3 untrained, ~0.1 trained)
    norm_exec = raw_exec / 2.5  # Scale to ~0-1 range
    
    # FORGETTING: Log-scale normalization
    # log(1 + x) maps: 0→0, 100→4.6, 10000→9.2, 100000→11.5
    log_forget = math.log1p(raw_forget)
    norm_forget = log_forget / 10.0
    
    bid = alpha * norm_exec + beta * norm_forget
    return bid, components
```

**Formula:**
```
final_bid = α × (raw_exec / 2.5) + β × (log1p(raw_forget) / 10.0)
```

**Why This Normalization:**
- **Lower exec = wins:** Expert that learned the data wins the batch (correct incentive)
- **No averaging:** Winner's advantage is NOT erased by per-expert statistics
- **Log-scale forget:** Compresses explosive range (0 to 500,000+) naturally
- **VCG Independence:** Each expert's bid depends only on its own costs

---

### 3. ExecutionCostEstimator (mob/bidding.py)

```python
def compute_predicted_loss(self, x, y):
    self.model.eval()
    with torch.no_grad():
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, reduction='mean')
    return loss.item()
```

**Interpretation:**
- Measures how well the expert currently handles this data
- Low loss → Expert is already good at this → Low bid → Competitive
- High loss → Expert would struggle → High bid → Uncompetitive

---

### 4. EWCForgettingEstimator (mob/bidding.py)

**Key Constants:**
- `FISHER_DECAY = 0.9` (γ for Online EWC)

**Attributes:**
- `fisher`: Dictionary of Fisher Information matrices per parameter
- `optimal_params`: Dictionary of optimal parameter values (θ*)
- `num_tasks_consolidated`: Count of tasks consolidated

**Input-Dependent Forgetting Cost:**
```python
def compute_forgetting_cost(self, x, y):
    if not self.fisher:
        return 0.0  # No Fisher = no knowledge to protect

    # Compute gradient for THIS specific batch
    self.model.train()
    self.model.zero_grad()
    logits = self.model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    # Measure gradient interference with important parameters
    interference = 0.0
    for n, p in self.model.named_parameters():
        if n in self.fisher and p.grad is not None:
            interference += (self.fisher[n] * p.grad.pow(2)).sum().item()

    self.model.zero_grad()
    return self.forgetting_cost_scale * interference
```

**Formula:**
```
forgetting_cost = forgetting_cost_scale × Σᵢ Fᵢ × (∂L/∂θᵢ)²
```

---

### 5. Fisher Information Update (Online EWC)

Called at the end of each task for experts that won batches:

```python
def update_fisher(self, dataloader, num_samples=200):
    # 1. Compute current task Fisher
    current_fisher = {n: zeros_like(p) for n, p in model.named_parameters()}
    
    for x, y in dataloader:  # Up to num_samples
        self.model.zero_grad()
        logits = self.model(x)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(log_probs, y)  # Empirical Fisher
        loss.backward()
        
        for n, p in self.model.named_parameters():
            current_fisher[n] += p.grad.pow(2) * batch_size
    
    # 2. Normalize by samples
    current_fisher[n] /= samples_seen
    
    # 3. Online EWC: Exponential moving average
    gamma = 0.9
    self.fisher[n] = γ × self.fisher[n] + (1-γ) × current_fisher[n]
    self.optimal_params[n] = γ × self.optimal_params[n] + (1-γ) × current_params[n]
    
    # 4. CRITICAL: Normalize Fisher to mean = 1.0
    self._normalize_fisher()
```

**Fisher Normalization:**
```python
def _normalize_fisher(self):
    all_fisher = concat([f.flatten() for f in self.fisher.values()])
    fisher_mean = all_fisher.mean()
    
    if fisher_mean > 1e-8:
        for n in self.fisher:
            self.fisher[n] = self.fisher[n] / fisher_mean
```

**Why Normalize Fisher?**
- Makes `lambda_ewc` work in reasonable range (1.0 - 10.0)
- Consistent EWC strength across model architectures
- Prevents numerical issues from tiny/huge Fisher values

---

### 6. EWC Penalty During Training

```python
def penalty(self):
    if not self.fisher:
        return 0.0
    
    penalty = 0.0
    for n, p in self.model.named_parameters():
        if n in self.fisher:
            diff_sq = (p - self.optimal_params[n]).pow(2)
            penalty += (self.fisher[n] * diff_sq).sum()
    
    return (self.lambda_ewc / 2.0) * penalty
```

**Formula:**
```
L_EWC = (λ/2) × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)²
```

---

### 7. VCG Auction (mob/auction.py)

```python
class PerBatchVCGAuction:
    def run_auction(self, bids):
        # 1. Allocation: Find the winner (minimum bid)
        winner = int(np.argmin(bids))
        
        # 2. Payment: Second-lowest bid (VCG mechanism)
        if self.num_experts > 1:
            payment = float(np.partition(bids, 1)[1])
        else:
            payment = bids[winner]
        
        return winner, payment, metrics
```

**VCG Properties:**
- **Truthful:** Experts have no incentive to misreport their true costs
- **Efficient:** Selects the socially optimal expert (lowest cost)
- **Stateless:** No learned parameters - immune to forgetting

---

### 8. Learning without Forgetting (LwF) Support

**Optional feature** in `MoBExpertLocal`. When enabled:

**1. Record Soft Targets (Before new task):**
```python
def record_lwf_soft_targets(self, dataloader, max_batches=None):
    """Record what model outputs on new task data BEFORE training."""
    self.model.eval()
    for batch_idx, (x, y) in enumerate(dataloader):
        logits = self.model(x)
        soft_targets = F.softmax(logits / self.lwf_temperature, dim=1)
        self.lwf_soft_targets[batch_idx] = soft_targets.cpu()
```

**2. Compute LwF Loss (During training):**
```python
def compute_lwf_loss(self, current_logits, batch_idx):
    """KL divergence between current and stored soft targets."""
    old_soft_targets = self.lwf_soft_targets[batch_idx]
    current_soft = F.log_softmax(current_logits / self.lwf_temperature, dim=1)
    lwf_loss = F.kl_div(current_soft, old_soft_targets, reduction='batchmean')
    lwf_loss = lwf_loss * (self.lwf_temperature ** 2)  # Scale by T²
    return lwf_loss
```

**Key insight:** Only trained experts (those with Fisher) record soft targets. Untrained experts have nothing worth preserving.

---

## Training Pipeline

### Complete Training Loop

```python
def run_experiment(train_tasks, test_tasks, config):
    # 1. Setup
    pool = ExpertPoolLocal(num_experts=4, expert_config, device)
    auction = PerBatchVCGAuction(num_experts=4)
    optimizers = [Adam(expert.model.parameters()) for expert in pool.experts]
    bid_logger = BidLogger(num_experts=4, alpha, beta)

    # 2. Training Loop (per task)
    for task_id, task_loader in enumerate(train_tasks):
        # Reset per-task statistics
        for expert in pool.experts:
            expert.reset_task_statistics()
        
        # LwF: Record soft targets for trained experts BEFORE training
        if config['use_lwf']:
            trained_experts = [e for e in pool.experts if e.has_fisher()]
            for expert in trained_experts:
                expert.record_lwf_soft_targets(task_loader, max_batches=100)
                expert.reset_lwf_for_new_task()
        
        winners_this_task = {}
        
        for epoch in range(epochs_per_task):  # 4 epochs
            for x, y in task_loader:
                # Collect bids from all experts
                bids, components = pool.collect_bids(x, y)
                
                # Run VCG auction (lowest bid wins)
                winner_id, payment, _ = auction.run_auction(bids)
                winners_this_task[winner_id] = winners_this_task.get(winner_id, 0) + 1
                
                # Log bid
                bid_logger.log_batch(batch_idx, bids, components, winner_id, task_id)
                
                # Train ONLY the winning expert
                pool.train_winner(winner_id, x, y, optimizers)
        
        # Update Fisher for experts that won batches
        for expert_id in winners_this_task.keys():
            pool.experts[expert_id].update_after_task(task_loader, num_samples=200)
```

### Expert Training with EWC (and optional LwF)

```python
def train_on_batch(self, x, y, optimizer):
    self.model.train()
    self.batches_won += 1
    self.batches_won_this_task += 1

    optimizer.zero_grad()
    logits = self.model(x)
    task_loss = F.cross_entropy(logits, y)
    ewc_penalty = self.forget_estimator.penalty()
    
    # LwF: Add distillation loss
    lwf_loss = torch.tensor(0.0)
    if self.use_lwf and self.lwf_soft_targets:
        lwf_loss = self.compute_lwf_loss(logits, self.lwf_batch_counter)
        self.lwf_batch_counter += 1

    # Total loss
    total_loss = task_loss + ewc_penalty + self.lwf_alpha * lwf_loss
    
    total_loss.backward()
    optimizer.step()
```

**Total Loss Formula:**
```
L_total = L_task + L_EWC + α_lwf × L_LwF
        = CrossEntropy + (λ/2) × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)² + α_lwf × KL_div
```

---

## Evaluation (Inference)

Uses **forgetting-cost-based routing** with pseudo-labels:

```python
def evaluate_all(self, dataloader):
    for x, y in dataloader:
        batch_forget_costs = np.zeros(self.num_experts)
        batch_logits = []
        
        for i, expert in enumerate(self.experts):
            logits = expert.model(x)
            batch_logits.append(logits)
            
            # Use expert's own predictions as pseudo-labels
            pseudo_labels = logits.argmax(dim=-1).detach()
            forget_cost = expert.forget_estimator.compute_forgetting_cost(x, pseudo_labels)
            batch_forget_costs[i] = forget_cost
        
        # Select expert with LOWEST forgetting cost
        winner_id = np.argmin(batch_forget_costs)
        predictions = batch_logits[winner_id].argmax(dim=-1)
```

**Why Lowest Forgetting Cost?**

| Scenario | Forgetting Cost | Explanation |
|----------|-----------------|-------------|
| **Trained expert on its own data** | **LOW** | Fisher is aligned with these gradients |
| **Trained expert on other data** | **HIGH** | Gradient interference with existing knowledge |
| **Untrained expert** | **ZERO** | No Fisher information yet |

---

## Event Timeline

| Event | When | What Happens |
|-------|------|--------------|
| **Bid Computation** | Every batch | All experts compute `α×norm_exec + β×norm_forget` |
| **Auction** | Every batch | Winner = argmin(bids), Payment = 2nd lowest |
| **Training** | Every batch | Only winner trains with `L_task + L_EWC + L_LwF` |
| **LwF Recording** | Start of each task | Trained experts record soft targets on new task data |
| **Fisher Update** | End of each task | Only winning experts update Fisher |
| **Optimal Params Update** | End of each task | Moving average with Fisher |

---

# Part 2: `run_continual_mob.py` (Task-Free Continual MoB)

## Module Structure

Uses classes from `contibualmob/`:

```
contibualmob/
├── models.py        # Same as mob/models.py
├── bidding.py       # Same as mob/bidding.py
├── auction.py       # Same as mob/auction.py
├── expert.py        # MoBExpert with consolidate() method
├── pool.py          # ExpertPool with ShiftDetector
├── bid_diagnostics.py  # BidLogger
└── utils.py         # set_seed()
```

---

## Configuration & Hyperparameters

| Parameter | Default | CLI Argument | Description |
|-----------|---------|--------------|-------------|
| `num_experts` | 4 | `--num_experts` | Number of expert neural networks |
| `num_tasks` | 5 | - | Number of latent contexts (Split-MNIST tasks) |
| `batch_size` | 32 | - | Samples per training batch |
| `epochs_per_task` | 4 | `--epochs` | Repetitions of each task in stream |
| `learning_rate` | 0.001 | `--learning_rate` | Adam optimizer learning rate |
| `alpha` (α) | 0.5 | `--alpha` | Weight for normalized execution cost |
| `beta` (β) | 0.5 | `--beta` | Weight for normalized forgetting cost |
| `lambda_ewc` (λ) | 40.0 | `--lambda_ewc` | EWC regularization strength (higher default!) |
| `shift_threshold` | 2.0 | `--shift_threshold` | Multiplier for shift detection |
| `seed` | 42 | `--seed` | Random seed for reproducibility |

> **Note:** `lambda_ewc` defaults to 40.0 here vs 10.0 in `run_mob_only.py`

---

## Key Differences from Task-Aware MoB

| Aspect | `run_mob_only.py` | `run_continual_mob.py` |
|--------|-------------------|------------------------|
| **Task Boundaries** | Explicit | None (task-free) |
| **Data Presentation** | Sequential tasks | Continuous stream |
| **Fisher Update Trigger** | End of each task | Shift detection |
| **Shift Detection** | None | ShiftDetector class |
| **Evaluation Routing** | Forgetting-cost (argmin) | Auction-based (argmin bids) |
| **Module Used** | Local classes | `contibualmob/` |
| **LwF Support** | Yes (optional) | No |
| **Per-Digit Stats** | No | Yes |

---

## Core Components

### 1. ShiftDetector (contibualmob/pool.py)

Detects distribution shifts using Exponential Moving Average (EMA) of execution cost:

```python
class ShiftDetector:
    def __init__(self, alpha: float = 0.99, threshold_multiplier: float = 50.0):
        self.alpha = alpha  # Smoothing factor (high = slow adaptation)
        self.threshold_multiplier = threshold_multiplier
        self.ema_cost = None
        self.shift_cooldown = 0

    def update(self, cost: float) -> bool:
        """Returns True if a significant upward spike is detected."""
        # If cooldown is active, allow new task to stabilize
        if self.shift_cooldown > 0:
            self.shift_cooldown -= 1
            self.ema_cost = 0.5 * self.ema_cost + 0.5 * cost  # Fast adaptation
            return False

        if self.ema_cost is None:
            self.ema_cost = cost
            return False

        # Check for spike with minimum floor (avoid noise at low losses)
        baseline = max(self.ema_cost, 0.5)
        is_shift = cost > (baseline * self.threshold_multiplier)

        # Update EMA
        self.ema_cost = self.alpha * self.ema_cost + (1 - self.alpha) * cost
        
        if is_shift:
            self.shift_cooldown = 50  # Absorb new distribution
            self.ema_cost = cost  # Jump to new level
            
        return is_shift
```

**Shift Detection Formula:**
```
is_shift = current_cost > max(EMA_cost, 0.5) × threshold_multiplier
```

**Parameters:**
- `alpha = 0.99`: High value = slow adaptation to normal fluctuations
- `threshold_multiplier = 50.0` (default), configurable via `--shift_threshold`
- `shift_cooldown = 50`: Batches to wait after shift before detecting again

---

### 2. MoBExpert with Consolidation (contibualmob/expert.py)

Key difference from `mob/expert.py`: has `consolidate()` method:

```python
def consolidate(self, dataloader, num_samples: int = 200):
    """
    Consolidates knowledge by updating EWC parameters (Fisher Matrix).
    Triggered when a distribution shift is detected.
    """
    self.forget_estimator.update_fisher(dataloader, num_samples=num_samples)
```

Also, logging is different (every 500 batches globally vs first 3 per task):
```python
if self.batches_won % 500 == 0:
    print(f"[Expert {self.expert_id}] Global Batch {self.batches_won}: ...")
```

---

### 3. ExpertPool with Shift Detection (contibualmob/pool.py)

```python
class ExpertPool:
    def __init__(self, num_experts, expert_config, device=None, use_shift_detection=False):
        self.shift_detector = ShiftDetector() if use_shift_detection else None
        # ... create experts ...
    
    def train_winner(self, winner_id, x, y, optimizers):
        winner = self.experts[winner_id]
        
        # Check for distribution shift BEFORE training
        shift_detected = False
        if self.shift_detector:
            current_loss = winner.exec_estimator.compute_predicted_loss(x, y)
            shift_detected = self.shift_detector.update(current_loss)
        
        metrics = winner.train_on_batch(x, y, optimizers[winner_id])
        metrics['shift_detected'] = shift_detected
        return metrics
    
    def consolidate(self, dataloader, num_samples=200, expert_ids=None):
        """Consolidates knowledge for specific experts."""
        targets = expert_ids if expert_ids is not None else range(len(self.experts))
        for i in targets:
            self.experts[i].consolidate(dataloader, num_samples=num_samples)
```

---

### 4. Evaluation with Auction-Based Routing

Unlike `run_mob_only.py` which uses forgetting-cost routing, this uses auction bids:

```python
def evaluate_all(self, dataloader):
    for x, y in dataloader:
        batch_logits = []
        
        # 1. Compute logits for all experts
        for expert in self.experts:
            expert.model.eval()
            with torch.no_grad():
                batch_logits.append(expert.model(x))
        
        # 2. Get auction bids
        bids, _ = self.collect_bids(x, y)
        
        # 3. Winner = Lowest Bid (auction logic)
        auction_winner_id = np.argmin(bids)
        
        # 4. Use winner's predictions
        winning_preds = batch_logits[auction_winner_id].argmax(dim=-1)
```

---

## Training Pipeline

### Continuous Data Stream Construction

```python
def run_continual_experiment(train_tasks, test_tasks, config):
    # Create continuous stream from task dataloaders
    epochs_per_task = config['epochs_per_task']
    stream_datasets = []
    
    for task_loader in train_tasks:
        ds = task_loader.dataset
        for _ in range(epochs_per_task):
            stream_datasets.append(ds)
    
    # Concatenate into single stream (NO SHUFFLING!)
    full_stream_dataset = torch.utils.data.ConcatDataset(stream_datasets)
    stream_loader = torch.utils.data.DataLoader(
        full_stream_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # Preserve task order for continual learning
        num_workers=0
    )
```

**Stream Structure (with epochs_per_task=4, num_tasks=5):**
```
[Task1 × 4] → [Task2 × 4] → [Task3 × 4] → [Task4 × 4] → [Task5 × 4]
```

---

### Complete Training Loop

```python
# Setup
pool = ExpertPool(num_experts, expert_config, device, use_shift_detection=True)
pool.shift_detector.threshold_multiplier = config['shift_threshold']
auction = PerBatchVCGAuction(num_experts)
optimizers = [Adam(expert.model.parameters()) for expert in pool.experts]

# Replay buffer for Fisher updates
replay_buffer = []
MAX_BUFFER_SIZE = 500

# Digit-granular statistics
expert_digit_dist = {i: {d: 0 for d in range(10)} for i in range(num_experts)}
online_digit_stats = {i: {d: {'correct': 0, 'total': 0} for d in range(10)} for i in range(num_experts)}

# Stream processing loop
for batch_idx, (x, y) in enumerate(stream_loader):
    # 1. Auction Phase
    bids, components = pool.collect_bids(x, y)
    winner_id, payment, _ = auction.run_auction(bids)
    
    # 2. Buffer Management
    replay_buffer.append((x, y, winner_id))
    if len(replay_buffer) > MAX_BUFFER_SIZE:
        replay_buffer.pop(0)
    
    # 3. Online Accuracy Check (before training)
    pool.experts[winner_id].model.eval()
    with torch.no_grad():
        logits = pool.experts[winner_id].model(x)
        preds = logits.argmax(dim=1)
    
    for i, label in enumerate(y):
        digit = label.item()
        is_correct = (preds[i] == label).item()
        expert_digit_dist[winner_id][digit] += 1
        online_digit_stats[winner_id][digit]['total'] += 1
        if is_correct:
            online_digit_stats[winner_id][digit]['correct'] += 1
    
    # 4. Training Phase
    metrics = pool.train_winner(winner_id, x, y, optimizers)
    
    # 5. Shift Detection & Consolidation
    if metrics.get('shift_detected', False):
        print(f">>> SHIFT DETECTED at Batch {batch_idx}")
        
        if replay_buffer:
            buffer_loader = [(bx, by) for bx, by, _ in replay_buffer]
            buffer_winners = [w for _, _, w in replay_buffer]
            
            # Filter: Only consolidate experts with >1% of buffer
            from collections import Counter
            counts = Counter(buffer_winners)
            threshold = len(replay_buffer) * 0.01
            active_experts = [eid for eid, count in counts.items() if count > threshold]
            
            pool.consolidate(buffer_loader, num_samples=200, expert_ids=active_experts)
        
        replay_buffer = []  # Clear buffer after consolidation
```

---

### Selective Consolidation Logic

When a shift is detected:

1. **Count expert wins in buffer:**
   ```python
   counts = Counter(buffer_winners)
   # e.g., {0: 250, 1: 150, 2: 80, 3: 20}
   ```

2. **Filter by threshold (>1% of buffer):**
   ```python
   threshold = 500 * 0.01 = 5
   active_experts = [eid for eid, count in counts.items() if count > 5]
   # e.g., [0, 1, 2, 3] (all pass if they have > 5 wins)
   ```

3. **Consolidate only active experts:**
   ```python
   pool.consolidate(buffer_loader, num_samples=200, expert_ids=active_experts)
   ```

**Purpose:** Prevents locking in "random" weights for experts that won only 1-2 batches by chance.

---

## Evaluation Pipeline

### Per-Digit Final Evaluation

```python
# Create unified test set (all digits)
all_test_data = torch.utils.data.ConcatDataset([t.dataset for t in test_tasks])
test_loader = torch.utils.data.DataLoader(all_test_data, batch_size=batch_size)

digit_eval_stats = {d: {'correct': 0, 'total': 0, 'routed_to': defaultdict(int)} for d in range(10)}

for x, y in test_loader:
    # 1. Auction Routing
    bids, _ = pool.collect_bids(x, y)
    winner_id = np.argmin(bids)  # AUCTION LOGIC
    
    # 2. Prediction
    winner_model = pool.experts[winner_id].model
    with torch.no_grad():
        logits = winner_model(x)
        preds = logits.argmax(dim=1)
    
    # 3. Record Per-Sample Results
    for i, label in enumerate(y):
        digit = label.item()
        correct = (preds[i] == label).item()
        
        digit_eval_stats[digit]['total'] += 1
        if correct:
            digit_eval_stats[digit]['correct'] += 1
        digit_eval_stats[digit]['routed_to'][winner_id] += 1
```

### Reporting Format

**Online Training Accuracy (per expert per digit):**
```
Expert 0:
  Digit 0: 95.23% (500 samples)
  Digit 1: 94.87% (480 samples)
  ...

Expert 1:
  Digit 2: 92.15% (520 samples)
  ...
```

**Test Accuracy & Routing:**
```
Digit 0: 97.45% | Routed: [E0:312, E1:5]
Digit 1: 96.82% | Routed: [E0:298, E2:12]
...
Overall Average Accuracy: 94.23%
```

---

## Key Data Structures

### Replay Buffer

```python
replay_buffer: List[Tuple[Tensor, Tensor, int]]
# Format: [(x_batch, y_batch, winner_id), ...]
# Max size: 500 entries
```

**Purpose:** Store recent batches for Fisher computation when shift is detected.

### Digit Statistics

```python
# Routing distribution: Which expert handles which digit?
expert_digit_dist[expert_id][digit] = count

# Online accuracy: How well does expert learn each digit during training?
online_digit_stats[expert_id][digit] = {'correct': int, 'total': int}

# Final evaluation: Per-digit test accuracy and routing
digit_eval_stats[digit] = {
    'correct': int,
    'total': int,
    'routed_to': {expert_id: count, ...}
}
```

---

## Comparison Summary

| Feature | run_mob_only.py | run_continual_mob.py |
|---------|-----------------|----------------------|
| **Task Boundaries** | Explicit (task_id known) | Task-Free (no boundaries) |
| **Fisher Update** | End of each task | On shift detection |
| **Shift Detection** | None | EMA-based ShiftDetector |
| **LwF Support** | Yes | No |
| **Replay Buffer** | No | Yes (500 batches) |
| **Evaluation Routing** | Forgetting-cost (pseudo-labels) | Auction bids |
| **Logging** | First 3 batches/task | Every 500 global batches |
| **Statistics** | Per-task | Per-digit granular |
| **EWC Lambda Default** | 10.0 | 40.0 |

---

## References

- **EWC:** Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks" (PNAS)
- **Online EWC:** Schwarz et al. (2018) "Progress & Compress" (ICML)
- **VCG Auction:** Vickrey (1961) "Counterspeculation, Auctions, and Competitive Sealed Tenders"
- **LwF:** Li & Hoiem (2016) "Learning without Forgetting" (ECCV)
