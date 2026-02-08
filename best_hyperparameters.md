# Best Hyperparameters

This document records the best hyperparameters found for each model configuration.

**Last Updated**: February 8, 2026

## Common Configuration
*   **Number of Experts**: 4 (for MoE-based models)
*   **Number of Tasks**: 5 (Split-MNIST)
*   **Batch Size**: 32
*   **Epochs per Task**: 4
*   **Target Parameter Count**: ~1.7M (matched across all models)

---

## 1. MoB (Mixture of Bidders) - Task Aware
*Script*: `tests/run_mob_only.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| Alpha | 0.5 | Weight for execution cost |
| Beta | 0.5 | Weight for forgetting cost |
| Lambda EWC | 10.0 | EWC regularization strength |
| Learning Rate | 0.001 | Adam optimizer |
| Forgetting Cost Scale | 1.0 | Scaling for forgetting cost |
| Use LwF | False | Learning without Forgetting |

---

## 2. Gated MoE + EWC
*Script*: `tests/run_gated_moe_ewc.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| Lambda EWC | 50.0 | **Needs re-tuning for 1.7M params** |
| Gater EWC | True | Apply EWC to gater network |
| Gater Hidden Size | 256 | MLP hidden layer size |
| Learning Rate | 0.001 | Adam optimizer |

**Note**: Current values are from 300k param version. Run hyperparameter search for 1.7M version.

---

## 3. Continual MoB (Task-Free)
*Script*: `tests/run_continual_mob.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| Lambda EWC | 40.0 | Higher than task-aware |
| Alpha | 0.5 | Weight for execution cost |
| Beta | 0.5 | Weight for forgetting cost |
| Shift Threshold | 2.0 | Multiplier for shift detection |
| Learning Rate | 0.001 | Adam optimizer |

---

## 4. Monolithic EWC (Single Large Model)
*Script*: `tests/run_monolithic_ewc.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| Lambda EWC | **TBD** | See plasticity-stability tradeoff below |
| Width Multiplier | 2 | Matches MoB ~1.7M params |
| Learning Rate | 0.001 | Adam optimizer |

### Plasticity-Stability Tradeoff (Observed)
| Lambda EWC | Task 1 Acc | Task 5 Acc | Behavior |
|------------|------------|------------|----------|
| 100 | ~0% | ~98% | Very plastic (forgets everything) |
| 10,000 | ~87% | ~23% | Very stable (resists new learning) |

**Search Range**: 10 - 50,000 (log scale)

---

## 5. A-GEM (Averaged Gradient Episodic Memory)
*Script*: `tests/run_agem_baseline.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| Memory Size | 256 | Total episodic memory samples |
| Memory Batch Size | 32 | Samples for reference gradient |
| Width Multiplier | 2 | Matches MoB ~1.7M params |
| Learning Rate | 0.001 | Adam optimizer |

---

## 6. Experience Replay (ER)
*Script*: `tests/run_er_baseline.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| Memory Size | 256 | Total replay buffer size |
| Replay Batch Size | 32 | Samples replayed per batch |
| Replay Weight | 1.0 | Weight for replay loss (1.0 = equal) |
| Width Multiplier | 2 | Matches MoB ~1.7M params |
| Learning Rate | 0.001 | Adam optimizer |

---

## Hyperparameter Search

Run the hyperparameter search with Optuna (Bayesian optimization) and multi-seed evaluation:

```bash
# Full search (all models, 5 seeds, 100 trials per model)
python tests/hyperparameter_search.py

# Quick search (2 seeds, 20 trials)
python tests/hyperparameter_search.py --quick

# Single model search
python tests/hyperparameter_search.py --model mob
python tests/hyperparameter_search.py --model monolithic
python tests/hyperparameter_search.py --model er

# Custom settings
python tests/hyperparameter_search.py --model mob --n_trials 50 --verbose
```

### Search Strategy
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Pruner**: Percentile pruning after first seed (saves ~80% compute on bad configs)
- **Seeds**: 5 seeds per configuration (42, 123, 456, 789, 1024)
- **Metric**: Mean accuracy across all seeds

---

## Model Comparison (Parameter Counts)

| Model | Parameters | Notes |
|-------|------------|-------|
| MoB (4 experts) | ~1.7M | 4 x SimpleCNN |
| Gated MoE | ~1.9M | 4 experts + gater overhead |
| Monolithic EWC | ~1.7M | Single wide CNN (width_multiplier=2) |
| A-GEM | ~1.7M | Single wide CNN + memory buffer |
| Experience Replay | ~1.7M | Single wide CNN + replay buffer |
