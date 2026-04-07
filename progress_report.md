# MoB (Mixture of Bidders) — Progress Report
**Date**: 2026-03-19

---

## Project Overview

MoB replaces the **learned router** in Mixture-of-Experts (MoE) architectures with a **stateless auction** + **Mahalanobis prototype-distance routing**. The goal is to eliminate expert collapse, auxiliary load-balancing losses, and gater forgetting — the three major problems with learned MoE routers.

**Target**: Replace `W_g · x → softmax → top-k` inside transformer MoE layers with auction-based routing that scales from MNIST → CIFAR → LLM.

---

## Current Implementation Status

### What is Built and Working

| Component | File | Status |
|-----------|------|--------|
| Auction (single-winner, per-batch) | `contibualmob/auction.py` | Working |
| Expert bidding (exec_cost + forget_cost) | `contibualmob/bidding.py` | Working |
| EWC forgetting estimator (diagonal Fisher) | `contibualmob/bidding.py` | Working (Fisher clamp fix at 0.1) |
| Expert pool management | `contibualmob/pool.py` | Working |
| Expert training + consolidation | `contibualmob/expert.py` | Working |
| Shift detection (EMA-based) | `contibualmob/pool.py` | Working |
| Prototype Store (class centroids + Mahalanobis) | `contibualmob/prototype_store.py` | Working |
| `forward_features()` on all models | `contibualmob/models.py`, `mob/models.py` | Working |
| Prototype routing at eval time | `contibualmob/pool.py`, both runners | Working |
| Bid diagnostics with prototype logging | `contibualmob/bid_diagnostics.py` | Working |
| Task-aware MoB runner | `tests/run_mob_only.py` | Working |
| Continual (online) MoB runner | `tests/run_continual_mob.py` | Working |
| 4 baselines (Naive, Random, MonolithicEWC, GatedMoE) | `contibualmob/baselines.py` | Working |

### Model Architectures

All support `forward()` and `forward_features()` (returns penultimate features + logits):
- **SimpleCNN**: 128-dim features (Conv→Conv→FC(128)→FC(10))
- **LeNet5**: 84-dim features
- **MLP**: configurable hidden layers

### Routing Strategies

1. **Pseudo-label routing** (original): `bid = α × CE(logits, argmax(logits)) + β × EWC_cost`
   - Problem: "confidently wrong" — experts predict everything as their trained classes with 99% confidence, making exec_cost near-zero for any input. Routing degrades to near-random.
   - Result: ~60% accuracy (vs ~90% with ground-truth labels)

2. **Prototype routing** (new, working): `bid = α × Mahalanobis_distance + β × EWC_cost`
   - Each expert accumulates per-class centroids in feature space during training
   - At eval: routes by distance to nearest centroid (no labels needed)
   - Immune to overconfident logits — measures feature-space proximity, not output confidence
   - Result: **86.7% accuracy** (best, λ_ewc=5.0)

---

## Current Results

### Split-MNIST (5 tasks × 2 digits, 4 experts)

#### Prototype Routing Results (per-digit accuracy)

| Digit | Expert | λ=40 | λ=5.0 | λ=1.0 |
|-------|--------|------|-------|-------|
| 0 | E3 | ~99% | 99.9% | 91.7% |
| 1 | E3 | ~99% | 99.7% | 92.8% |
| 2 | E1 | ~98% | 98.4% | 27.0% |
| 3 | E1 | ~98% | 97.7% | 27.9% |
| 4 | E2 | ~98% | 98.9% | 95.4% |
| 5 | E2 | ~98% | 99.1% | 95.7% |
| 6 | E0 | ~97% | 97.0% | 93.6% |
| 7 | E0 | ~95% | 96.1% | 93.0% |
| 8 | E1 | 0% | 16.3% | 95.9% |
| 9 | E1 | ~50% | 62.2% | 97.2% |
| **Overall** | | **79.35%** | **86.7%** | **80.72%** |

**Routing quality**: 24-26% distance separation (winner distance is 24-26% lower than loser average).

#### Baseline Comparison

| Method | Accuracy |
|--------|----------|
| MoB + Prototype Routing (λ=5.0) | **86.7%** |
| MoB + Pseudo-label Routing (λ=40) | 79.35% |
| Continual MoB (λ=1.0) | 80.72% |
| Gated MoE + EWC | 35.31% |
| Monolithic EWC | 19.90% |

### Known Issue: Overloaded Expert

With 4 experts and 5 tasks, one expert (E1) must handle two tasks ({2,3} and {8,9}). EWC creates an impossible tradeoff:
- **High λ (40, 5)**: Protects {2,3} weights → blocks learning {8,9}
- **Low λ (1)**: Learns {8,9} → catastrophically forgets {2,3}

This is NOT a routing failure — prototype routing correctly sends digits 8,9 to E1. It's a **single-expert capacity limitation** under EWC. The routing mechanism works; the expert can't serve both tasks simultaneously.

---

## What We're Trying Next and Why

### Goal: Validate MoB routing for LLM-scale MoE integration

In a standard MoE transformer layer, a learned router (`W_g · x → softmax → top-k`) selects which expert FFNs process each token. This router suffers from:
1. **Expert collapse** (self-reinforcing bias toward few experts)
2. **Auxiliary loss hell** (load-balancing loss coefficient is fragile)
3. **Gater forgetting** (router itself forgets in continual learning)

MoB replaces this with auction-based prototype routing. To prove it scales, we need to validate on MNIST features that map directly to LLM requirements:

### Experiment 1: Per-Sample Top-k Expert Combination

**What**: Route each sample independently (not per-batch) and combine top-k experts' outputs.

**Why**: In MoE transformers, each token routes independently to top-k experts (Mixtral uses k=2). Currently MoB routes all samples in a batch to one expert. Per-sample top-k is the direct analog of per-token MoE routing.

**Expected outcome**: Per-sample k=1 should beat per-batch k=1 because mixed-digit batches currently force all samples to one expert. k=2 provides smoother outputs for borderline cases.

### Experiment 2: Distance-Only Bidding

**What**: At eval time, bid = Mahalanobis distance only (drop EWC forget cost).

**Why**: In LLM MoE, the router is a single matmul — extremely cheap. If MoB bids require gradient computation per expert, it's too expensive. Distance-only bids reduce to matmuls, making MoB cost-competitive with learned routers.

**Expected outcome**: <1% accuracy drop. At eval time, forget cost is effectively constant per expert (static Fisher), adding no per-sample routing signal. Distance carries all the routing information.

### Experiment 3: Training-Time Prototype Routing

**What**: After warmup (~1000 batches), route training batches by prototype distance instead of label-based exec_cost.

**Why**: In MoE training, the router operates without task labels. MoB must work during training, not just eval. Labels are still used for the winner's supervised loss, just not for routing decisions.

**Expected outcome**: After warmup, prototype routing should agree with label-based routing >80% of the time. Centroids encode class identity, so distance implicitly captures "which expert knows this input."

### Experiment 4: Emergent Load Balancing Analysis

**What**: Quantify expert utilization (entropy, Gini coefficient) and compare to auxiliary-loss MoE.

**Why**: Core MoB claim — load balance emerges naturally without auxiliary losses. Forgetting cost acts as a natural load balancer: experts with more knowledge bid higher, pushing new work to less-loaded experts.

**Expected outcome**: Utilization entropy >1.0 (of max 1.39 for 4 experts), Gini <0.3, with zero auxiliary loss.

---

## Scaling Path

```
MNIST (now)  →  CIFAR-10  →  LLM MoE Layer
SimpleCNN       ResNet-18     FFN block in transformer
128-dim feat    512-dim feat  4096-dim hidden states
4 experts       4-8 experts   8-128 experts
Per-batch       Per-sample    Per-token
```

**What stays the same across all scales** (core contribution):
- Auction: `winner = argmin(bids)`
- Prototype distance: `d(x, centroid)` via Mahalanobis
- Stateless routing: zero learned router parameters
- DSIC truthful bidding
- Emergent load balancing
- No auxiliary loss

**What changes**:
- Model architecture (CNN → ResNet → FFN)
- Feature dimensionality (128 → 512 → 4096)
- Covariance structure (full → potentially low-rank at 4096-dim)
- Routing granularity (per-batch → per-sample → per-token)
