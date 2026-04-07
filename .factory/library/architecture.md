# MoB Architecture — Continual Learning Package (`contibualmob/`)

## System Components and Relationships

```
ExpertPool                         PerBatchAuction
├── ShiftDetector                  └── Vickrey second-price sealed-bid
├── MoBExpert[0..N-1]
│   ├── model (CNN/MLP)
│   ├── ExecutionCostEstimator     ─ cross-entropy on batch
│   ├── EWCForgettingEstimator     ─ Fisher-weighted param change
│   └── PrototypeStore (lazy)      ─ per-class feature centroids
└── collect_bids / train_winner / consolidate
```

| Component | File | Role |
|---|---|---|
| **ExpertPool** | `contibualmob/pool.py` | Owns the list of experts, orchestrates bid collection, winner training, consolidation, and evaluation. No centralized gater—the auction *is* the router. |
| **MoBExpert** | `contibualmob/expert.py` | One independent agent: holds a model, execution-cost estimator, EWC forgetting estimator, and (optionally) a PrototypeStore. Computes its own bid. |
| **PerBatchAuction** | `contibualmob/auction.py` | Single-item Vickrey auction. Takes a bid vector, returns `argmin(bids)` as winner and second-lowest bid as payment. Stateless per round. |
| **PrototypeStore** | `contibualmob/prototype_store.py` | Per-expert store of class-centroid feature vectors. Accumulates running sums during training; finalizes Mahalanobis inverse covariance at consolidation. Used for distance-based routing. |
| **ShiftDetector** | `contibualmob/pool.py` (inner class) | EMA-based spike detector on execution cost. When `cost > max(ema, 0.5) × threshold_multiplier`, declares a distribution shift and enters cooldown. |
| **ExecutionCostEstimator** | `contibualmob/bidding.py` | Wraps `F.cross_entropy(model(x), y)` to give the expert's current loss on a batch. |
| **EWCForgettingEstimator** | `contibualmob/bidding.py` | Computes Fisher Information Matrix after each consolidation. Provides `compute_forgetting_cost()` (gradient–Fisher dot product) and `penalty()` (EWC regularization term added to training loss). |

## Data Flow — One Training Batch

```
Input batch (x, y)
        │
        ▼
1. ExpertPool.collect_bids(x, y, train_routing)
   ├── For each expert i:
   │   ├── [label routing]     exec_cost = CE(model(x), y)
   │   │                       forget_cost = Fisher·grad overlap
   │   │                       bid = α·(exec/2.5) + β·(log1p(forget)/10)
   │   └── [prototype routing] distance = min_c ‖feat – centroid_c‖
   │                           bid = α·(distance/10) + β·(log1p(forget)/10)
   └── Returns bids[N], components[N]
        │
        ▼
2. PerBatchAuction.run_auction(bids)
   └── winner = argmin(bids), payment = second-lowest bid
        │
        ▼
3. ExpertPool.train_winner(winner_id, x, y, optimizers)
   ├── ShiftDetector.update(current_loss) → shift_detected?
   ├── (Optional) Reset optimizer if shift detected & expert has Fisher
   └── MoBExpert.train_on_batch(x, y, optimizer)
       ├── forward_features(x) → features, logits
       ├── PrototypeStore.update(features, y)   ← accumulate centroids
       ├── task_loss = CE(logits, y)
       ├── ewc_penalty = Σ F_i·(θ_i − θ*_i)²
       ├── total_loss = task_loss + ewc_penalty
       └── backward + optimizer.step
        │
        ▼
4. (On shift or task boundary) ExpertPool.consolidate(dataloader)
   ├── EWCForgettingEstimator.update_fisher()   ← recompute Fisher
   └── PrototypeStore.finalize()                ← compute inv_cov for Mahalanobis
```

## Key Invariants

1. **Uniform bid formula.** Every expert uses the *same* bid equation:
   `bid = α × norm_exec + β × norm_forget`. No expert has a structural advantage—differentiation comes solely from each expert's learned parameters and history.

2. **Prototypes grow, never reset.** `PrototypeStore.finalize()` does *not* clear `class_sum` / `class_count`. Centroids accumulate across distribution shifts, preserving knowledge of all previously seen classes.

3. **EWC protects important parameters.** The Fisher Information Matrix captures which parameters matter for previously learned tasks. The EWC penalty `Σ F_i·(θ_i − θ*_i)²` is added to every training loss, preventing catastrophic forgetting of old tasks when learning new ones.

4. **Auction is truthful.** The Vickrey (second-price) mechanism makes truthful bidding a dominant strategy. Each expert's bid reflects its genuine cost, not strategic manipulation.

## The Routing Collapse Mechanism (Prototype Routing)

When `train_routing='prototype'`, routing depends on `PrototypeStore.compute_routing_score(features)`:

```python
distance_score = min over centroids of ‖features − centroid‖
bid = α × (distance_score / 10.0) + β × (log1p(forget) / 10.0)
```

### Why collapse happens

1. **100.0 default cliff.** An expert with *no* prototypes returns `distance_score = 100.0`, producing `norm_distance = 10.0`. An expert with *any* centroid returns distance ≈ 1–5, producing `norm_distance ≈ 0.1–0.5`. This ~20× gap means the first expert to win a batch *always* wins all subsequent batches, because having any centroid is vastly better than the 100.0 default.

2. **Immediate centroid expansion.** `PrototypeStore.update()` adds new class centroids on the very first batch. So the first winner instantly covers multiple classes, making its `min()` over centroids even lower for any future input class.

3. **min() over centroids advantage.** The routing score is `min_c ‖feat − centroid_c‖`. Each new class the winner learns adds another centroid, strictly decreasing its future routing score. Other experts with fewer (or no) centroids can never close the gap.

4. **Self-reinforcing loop.** Winner trains → gains centroids → lower distance → wins again → trains more → gains more centroids. No natural break in this cycle.

### Result
One expert monopolizes all batches (99–100% win rate). Other experts remain untrained with 0 wins. Load balance collapses to ~0.0.

## Three Fix Approaches and Integration Points

### Fix 1: Conscience Bias (DeSieno-style load balancing)

**Concept:** Add a bias term to bids based on historical win rates. Experts that win too often get penalized; underused experts get a bonus.

**Integration point:** Inside `ExpertPool.collect_bids()`, after computing raw bid:
```python
# Track per-expert win counts
conscience_bias = compute_conscience_penalty(expert_win_rates)
bid = raw_bid + conscience_bias[i]
```

**Where it breaks the loop:** Directly counteracts the self-reinforcing cycle by making the monopolist's bid artificially higher, giving other experts a chance to win and build their own prototypes.

### Fix 2: Prototype Seeding

**Concept:** Initialize all experts' PrototypeStores with a small set of feature centroids *before* training begins, eliminating the 100.0 default cliff.

**Integration point:** In `ExpertPool.__init__()` or a new `seed_prototypes()` method, run a small warmup forward pass through each expert and populate `PrototypeStore.class_sum/class_count/centroids`:
```python
for expert in self.experts:
    features = expert.model.forward_features(warmup_batch)
    expert.prototype_store = PrototypeStore(feature_dim, device)
    expert.prototype_store.update(features, warmup_labels)
```

**Where it breaks the loop:** Removes the 100.0 → ~2.0 cliff. All experts start with comparable prototype distances, so the first batch doesn't create an insurmountable advantage.

### Fix 3: Temperature Annealing on Prototype Distance

**Concept:** Apply a temperature parameter to the prototype distance that starts high (flattening differences between experts) and decreases over training (allowing natural specialization).

**Integration point:** In `ExpertPool.collect_bids()` prototype routing branch:
```python
temperature = compute_temperature(current_batch, total_batches)  # e.g. 10→1
norm_distance = distance_score / (10.0 * temperature)
```

**Where it breaks the loop:** Early in training, high temperature compresses distance differences between experts to near-zero, making routing effectively random. As temperature decreases, experts that have actually learned relevant features gradually gain routing advantage—but by then, multiple experts have had a chance to build prototypes.
