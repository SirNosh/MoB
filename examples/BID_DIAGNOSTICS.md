# Bid Diagnostics Guide

Comprehensive guide for diagnosing potential issues in the MoB bidding mechanism.

## Problem: How do I know if my bidding is working correctly?

The MoB architecture relies on experts bidding for batches based on two components:
- **α (alpha) × Execution Cost**: Predicted loss on the batch
- **β (beta) × Forgetting Cost**: EWC penalty for updating parameters

If your MoB system is underperforming, the bidding mechanism might have issues. This guide shows you how to diagnose them.

---

## Common Issues to Check

### 1. Is α (PredictedLoss) signal being ignored?

**Symptom**: All experts bid the same regardless of batch content

**Why it happens**:
- α is too small compared to β
- Execution cost estimator is broken
- All experts have similar predictions (not specialized)

**How to check**:
```python
bid_logger.print_diagnostics()
# Look for: "Mean execution cost near zero" warning
```

**Fix**:
- Increase α relative to β (e.g., `alpha=0.7, beta=0.3`)
- Verify execution cost varies across batches

---

### 2. Is β (ForgettingCost) too high preventing learning?

**Symptom**: Experts refuse to learn new tasks, accuracy stagnates

**Why it happens**:
- β is too large
- λ_EWC (EWC regularization) is too high
- Fisher matrix has exploded

**How to check**:
```python
bid_logger.print_diagnostics()
# Look for: "Forgetting cost is 100x execution cost!" warning
```

**Fix**:
- Reduce β (e.g., `beta=0.3` instead of `beta=0.5`)
- Reduce λ_EWC (e.g., `lambda_ewc=1000` instead of `lambda_ewc=5000`)
- Check Fisher matrix statistics (should be ~1e-6 to 1e-3)

---

### 3. Are bids exploding or vanishing?

**Symptom**: NaN losses, training crashes, or all bids identical

**Why it happens**:
- Numerical instability in loss computation
- EWC penalty too large
- Learning rate too high

**How to check**:
```python
bid_logger.print_diagnostics()
# Look for: "Bids are exploding!" or "Bids contain NaN" warnings
```

**Fix**:
- Reduce learning rate
- Reduce λ_EWC
- Check for gradient clipping needs

---

## Usage Examples

### Example 1: Quick Diagnostic Check

Run the diagnostic example script:

```bash
python examples/diagnose_bids.py
```

This will:
1. Train MoB for 2 tasks on Split-MNIST
2. Log all bid components
3. Print comprehensive diagnostics
4. Save logs and visualizations

**Output**:
- `bid_diagnostics_results.json`: Full bid logs
- `bid_components.png`: Visualization of exec_cost, forget_cost, bids over time

---

### Example 2: Integrate into Your Experiment

```python
from mob import ExpertPool, PerBatchVCGAuction, BidLogger

# Create bid logger
bid_logger = BidLogger(num_experts=4, log_file="my_bids.json")

# During training loop
for batch_idx, (x, y) in enumerate(train_loader):
    # Collect bids
    bids, components = pool.collect_bids(x, y)

    # Run auction
    winner_id = auction.run_auction(bids)

    # LOG BIDS (critical!)
    bid_logger.log_batch(
        batch_idx=batch_idx,
        bids=bids,
        components=components,
        winner_id=winner_id,
        task_id=task_id
    )

    # Train winner
    pool.train_winner(winner_id, x, y, optimizers)

# After training, print diagnostics
bid_logger.print_diagnostics()
bid_logger.save_logs("final_bid_logs.json")
bid_logger.plot_bid_components("bid_plot.png")
```

---

### Example 3: Analyze Specific Batches

```python
# Print detailed breakdown for a specific batch
bid_logger.print_batch_details(batch_idx=0)
```

**Output**:
```
================================================================================
BATCH 0 DETAILED BREAKDOWN
Task ID: 0
================================================================================

    Expert   Exec Cost  Forget Cost      α      β          Bid   Winner
--------------------------------------------------------------------------------
         0     0.234567     0.000000   0.50   0.50     0.117283        ✓
         1     0.456789     0.000000   0.50   0.50     0.228394
         2     0.345678     0.000000   0.50   0.50     0.172839
         3     0.567890     0.000000   0.50   0.50     0.283945
================================================================================
```

---

### Example 4: Analyze Last N Batches Only

Useful for checking if bidding changes after Fisher update:

```python
# Analyze only the last 100 batches (e.g., after Task 1 Fisher update)
bid_logger.print_diagnostics(last_n_batches=100)
```

---

## Interpreting Diagnostics Output

### Section 1: Alpha Signal Check

```
[1] ALPHA SIGNAL CHECK (Execution Cost)
--------------------------------------------------------------------------------
  Expert 0:
    Mean: 0.234567 ± 0.045123
    Range: [0.123456, 0.345678]
```

**What to look for**:
- ✅ **Good**: Mean > 0.1, variance > 0.01
- ⚠️  **Warning**: Mean < 0.001 (signal too weak)
- 🔴 **Critical**: All experts have identical exec costs (no specialization)

---

### Section 2: Beta Signal Check

```
[2] BETA SIGNAL CHECK (Forgetting Cost)
--------------------------------------------------------------------------------
  Expert 0:
    Mean: 23.456789 ± 5.123456
    Range: [12.345678, 45.678901]
    Forget/Exec Ratio: 100.15x
    🔴 CRITICAL: Forgetting cost is 100x execution cost!
       This prevents experts from learning new tasks.
       Consider: reducing β, reducing λ_EWC, or increasing α
```

**What to look for**:
- ✅ **Good**: Ratio between 0.1x and 10x
- ⚠️  **Warning**: Ratio > 10x (learning hindered)
- 🔴 **Critical**: Ratio > 100x (learning blocked)
- ⚠️  **Warning**: Ratio < 0.01x (EWC not preventing forgetting)

---

### Section 3: Bid Magnitude Check

```
[3] BID MAGNITUDE CHECK
--------------------------------------------------------------------------------
  Expert 0:
    Mean: 12.345678 ± 2.567890
    Range: [5.678901, 23.456789]
```

**What to look for**:
- ✅ **Good**: Mean between 0.01 and 100
- ⚠️  **Warning**: Mean < 1e-6 (vanishing)
- ⚠️  **Warning**: Max > 1000 (very large)
- 🔴 **Critical**: Max > 1e6 (exploding)
- 🔴 **Critical**: NaN or Inf detected

---

### Section 4: Expert Win Distribution

```
[4] EXPERT WIN DISTRIBUTION
--------------------------------------------------------------------------------
  Expert 0: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░  40.2% (120/299)
  Expert 1: ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  25.8% (77/299)
  Expert 2: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  20.4% (61/299)
  Expert 3: █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  13.6% (41/299)
```

**What to look for**:
- ✅ **Good**: Some specialization (variance > 0.01)
- ⚠️  **Warning**: One expert dominates (>80%)
- ⚠️  **Warning**: Uniform distribution (no specialization)

---

## Troubleshooting Guide

### Problem: "Forgetting cost is 100x execution cost!"

**Solution 1**: Reduce β
```python
config = {
    'alpha': 0.7,
    'beta': 0.3,  # Changed from 0.5
    'lambda_ewc': 5000
}
```

**Solution 2**: Reduce λ_EWC
```python
config = {
    'alpha': 0.5,
    'beta': 0.5,
    'lambda_ewc': 1000  # Changed from 5000
}
```

**Solution 3**: Increase α
```python
config = {
    'alpha': 0.8,  # Changed from 0.5
    'beta': 0.2,
    'lambda_ewc': 5000
}
```

---

### Problem: "Execution cost near zero"

**Causes**:
1. Model is already perfect (unlikely)
2. Execution cost estimator is broken
3. All experts make identical predictions

**Solutions**:
- Check that model is actually training
- Verify execution cost = cross_entropy_loss
- Ensure experts are initialized differently

---

### Problem: "Bids are exploding!"

**Solutions**:
- Reduce learning rate: `learning_rate=0.0001` (from 0.001)
- Reduce λ_EWC: `lambda_ewc=1000` (from 5000)
- Add gradient clipping:
  ```python
  torch.nn.utils.clip_grad_norm_(expert.model.parameters(), max_norm=1.0)
  ```

---

### Problem: One expert dominates (>80% win rate)

**Causes**:
1. Expert initialized better by chance
2. First task trained one expert heavily
3. Auction dynamics broken

**Solutions**:
- Use multiple random seeds (see multi-seed experiments)
- Check that replay mechanism is working
- Verify all experts get similar training data

---

## API Reference

### BidLogger

#### `__init__(num_experts, log_file=None)`
Create a new bid logger.

**Parameters**:
- `num_experts` (int): Number of experts in the system
- `log_file` (str, optional): Path to auto-save logs every 10 batches

#### `log_batch(batch_idx, bids, components, winner_id, task_id=None)`
Log all bid information for a single batch.

**Parameters**:
- `batch_idx` (int): Index of the current batch
- `bids` (np.ndarray): Array of final bids from all experts
- `components` (list[dict]): Bid component breakdowns from each expert
- `winner_id` (int): ID of the winning expert
- `task_id` (int, optional): Current task ID for multi-task tracking

#### `print_diagnostics(last_n_batches=None)`
Print comprehensive diagnostics.

**Parameters**:
- `last_n_batches` (int, optional): Only analyze last N batches. If None, analyzes all.

#### `print_batch_details(batch_idx)`
Print detailed breakdown for a specific batch.

**Parameters**:
- `batch_idx` (int): Index of the batch to analyze

#### `save_logs(filepath)`
Save all logs to JSON file.

**Parameters**:
- `filepath` (str): Path to save the log file

#### `load_logs(filepath)`
Load logs from JSON file.

**Parameters**:
- `filepath` (str): Path to the log file

#### `plot_bid_components(save_path=None)`
Create visualization of bid components over time.

**Parameters**:
- `save_path` (str, optional): Path to save plot (PNG). If None, shows interactively.

**Requires**: matplotlib

---

## Files Created

When you run experiments with bid logging:

1. **`mob_bid_diagnostics.json`**: Full bid logs from test_baselines.py
2. **`mob_bid_components.png`**: Visualization of exec_cost, forget_cost, bids
3. **`bid_diagnostics_results.json`**: Logs from diagnose_bids.py example

---

## FAQ

**Q: Should I always enable bid logging?**

A: Enable it when:
- Debugging poor MoB performance
- Validating new hyperparameters
- Writing papers (to show bidding works correctly)

Disable it for:
- Large-scale experiments (memory overhead)
- Production systems (performance overhead)

**Q: How much memory does bid logging use?**

A: Approximately 1 KB per batch. For 1000 batches, ~1 MB.

**Q: Can I analyze bids during training?**

A: Yes! Call `bid_logger.print_diagnostics()` after each task to monitor in real-time.

**Q: What if I don't see the warnings I expect?**

A: The diagnostics are calibrated for typical settings. Adjust thresholds in the code if needed.

---

## Related Documentation

- **Phase 1 Validation**: `PHASE1_VALIDATION.md`
- **MoB Specification**: `MoB.md`
- **Baseline Comparisons**: `Phase1Baseline.md`

---

**Summary**: Use `BidLogger` to diagnose bidding issues. Run `examples/diagnose_bids.py` for a quick check. The most common issue is β (forgetting cost) being too high relative to α (execution cost).
