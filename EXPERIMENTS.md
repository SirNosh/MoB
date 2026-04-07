# MoB Routing Experiments — Guide

## How to Run

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh 2>&1 | tee experiment_log.txt
```

All results land in `results/experiments/`. Each experiment produces two files:
- `{name}_results.json` — accuracy, forgetting, config, per-task breakdown
- `{name}_bids.json` — full bid traces, per-digit routing, load balance metrics, prototype snapshots

Continual MoB experiments produce `{name}_summary.txt` + `{name}_bids.json`.

---

## Experiment Map

### Phase 1: Baselines (2 runs)

| Name | What it is |
|------|-----------|
| `mob_baseline_pseudolabel` | Original MoB eval — bid = exec_cost + forget_cost, all samples in batch go to same expert |
| `mob_baseline_prototype` | Prototype routing — bid = Mahalanobis distance + forget_cost, still per-batch |

**What to compare**: These two should be close. If prototype is significantly worse, the centroids aren't capturing class structure well. If it's better, distance is a more reliable signal than pseudo-label cross-entropy.

**Where to look**: `avg_accuracy` and `forgetting` in `_results.json`. Per-task breakdown in `final_accuracies` array (index 0 = digits 0,1 ... index 4 = digits 8,9).

---

### Phase 2: Per-Sample Top-k Routing — Experiment 1 (4 runs)

| Name | Config | What it tests |
|------|--------|--------------|
| `exp1_persample_k1` | k=1, per-sample | Each sample picks its own best expert (vs batch-mean baseline) |
| `exp1_persample_k2_temp0.5` | k=2, temp=0.5 | Two experts combined, sharp weighting (nearly winner-take-all) |
| `exp1_persample_k2_temp1.0` | k=2, temp=1.0 | Two experts combined, moderate weighting |
| `exp1_persample_k2_temp2.0` | k=2, temp=2.0 | Two experts combined, soft weighting (closer to uniform) |

**Why this matters**: In real MoE transformers (Mixtral, Switch), each token is routed independently to top-k experts. This tests whether MoB can do the same. Per-batch routing forces all 32 samples to the same expert, which hurts when a batch contains digits from different tasks.

**What to expect**:
- `exp1_persample_k1` should beat `mob_baseline_prototype` — mixed batches now split correctly across experts
- On MNIST with clean expert separation, k=1 is probably near-optimal already
- k=2 may help borderline samples (digits that look ambiguous) but could also hurt if the second expert is unrelated
- Temperature controls the blend: temp=0.5 is almost winner-take-all, temp=2.0 gives the second expert significant weight

**Where to look**:
- `avg_accuracy` comparison across all 4
- `expert_sample_rates` in results — should show each expert getting ~20-25% of samples (not 100% to one)
- `top2_agreement` (k=2 only) — how often both experts predict the same class. High agreement means the second expert adds no new information. Low agreement means the blend is doing real work.

**Key question**: Does per-sample k=1 already saturate accuracy, or does k=2 add value?

---

### Phase 3: Distance-Only Bidding — Experiment 2 (3 runs)

| Name | Config | What it tests |
|------|--------|--------------|
| `exp2_distance_only_perbatch` | Per-batch, distance only | Drop forget_cost from eval bid entirely |
| `exp2_distance_only_persample_k1` | Per-sample k=1, distance only | Cheapest possible routing: just Mahalanobis distance |
| `exp2_distance_only_persample_k2` | Per-sample k=2, distance only | k=2 with minimal computation |

**Why this matters**: Full bids require computing EWC forgetting cost per expert per batch — that's N gradient computations. Distance-only is just N matrix multiplications (one Mahalanobis distance per expert). At LLM scale, this difference determines whether MoB is viable as a router replacement.

**What to expect**:
- Distance-only should be **very close** to full-bid accuracy (< 1% drop)
- Key insight: at eval time, Fisher matrices are frozen. Forgetting cost is effectively constant per expert regardless of input — it adds no per-sample routing signal. Only distance varies per sample.
- If distance-only is much worse, it means forget_cost is actually contributing routing information (worth investigating why)

**Where to look**:
- Compare `exp2_distance_only_persample_k1` vs `exp1_persample_k1` — same setup, only difference is bid mode
- Compare `exp2_distance_only_perbatch` vs `mob_baseline_prototype` — same setup, only difference is bid mode
- If the accuracy gap is < 0.5%, distance-only routing is validated for LLM use

**Computational cost context**:
```
Full bid:         N experts x (forward pass + gradient computation)  ~expensive
Distance-only:    N experts x (features @ inv_cov @ features.T)      ~single matmul each
Learned router:   1 x (W_g @ hidden_state)                          ~single matmul total
```

---

### Phase 4: Training-Time Prototype Routing — Experiment 3 (3 runs)

| Name | Config | What it tests |
|------|--------|--------------|
| `exp3_train_proto_warmup500` | Switch at batch 500 | Early switch — prototypes still rough |
| `exp3_train_proto_warmup1000` | Switch at batch 1000 | Mid switch — prototypes have ~1 task of data |
| `exp3_train_proto_warmup2000` | Switch at batch 2000 | Late switch — prototypes well-formed |

**Why this matters**: This is the most important experiment for the LLM story. In LLM pretraining, there are no task labels — you can't use label-based exec_cost to route. The router must work from the input alone. This experiment proves MoB can route during training using only feature-space distances.

**How it works**:
1. For the first N warmup batches: normal label-based bidding (experts specialize, prototypes accumulate)
2. After warmup: bidding switches to `bid = distance_to_prototype + forget_cost` (no labels in the routing decision)
3. Labels are still used to train the winning expert (supervised CE loss) — just not for routing

**What to expect**:
- `warmup2000` should be close to `mob_baseline_prototype` — by batch 2000, prototypes are well-formed
- `warmup500` will likely be worse — prototypes are rough, routing is noisier, experts may get wrong batches
- `warmup1000` is the sweet spot to look for
- None will match label-based routing exactly (labels are stronger signal), but >90% routing agreement is the target

**Where to look**:
- `avg_accuracy` vs baselines — how much accuracy is lost by removing labels from routing?
- Per-task `final_accuracies` — does any particular task get hurt disproportionately?
- The console output prints when the switch happens: `>>> Switching to PROTOTYPE training routing at batch N`

**Key question**: How much warmup is needed before prototype routing is reliable? This directly maps to "how much supervised fine-tuning before MoB can route unsupervised."

---

### Phase 5: Hyperparameter Sweep (5 runs)

| Name | Config | What it tests |
|------|--------|--------------|
| `sweep_ewc1.0` | lambda_ewc=1.0 | Weak EWC — experts forget freely |
| `sweep_ewc50.0` | lambda_ewc=50.0 | Moderate EWC |
| `sweep_ewc1000.0` | lambda_ewc=1000.0 | Strong EWC — heavy forgetting penalty |
| `sweep_alpha0.7_beta0.3` | alpha=0.7, beta=0.3 | Routing prefers low exec_cost (competence) |
| `sweep_alpha0.3_beta0.7` | alpha=0.3, beta=0.7 | Routing prefers low forget_cost (safety) |

**What to expect**:

*EWC sweep*: lambda_ewc controls how much an expert resists forgetting old tasks.
- `ewc1.0`: Experts will be good at their current task but forget everything else. High accuracy on last task, low on earlier tasks. High `forgetting` metric.
- `ewc1000.0`: Experts resist change strongly. The overloaded expert (handling 2 tasks) may fail to learn the second task because EWC locks weights too hard. Look for low accuracy on task 5 specifically.
- `ewc50.0`: Middle ground. Compare to the `exp1_persample_k1` baseline (ewc=5.0).

*Alpha/beta sweep*: Controls the exec vs forget tradeoff in the bid formula `bid = alpha * exec_cost + beta * forget_cost`.
- `alpha0.7_beta0.3`: Routing cares more about "who can do this well" — experts that are good at the input win more often, even if it risks forgetting
- `alpha0.3_beta0.7`: Routing cares more about "who won't forget" — safer routing, but may send batches to less competent experts

**Where to look**:
- Compare all 5 to `exp1_persample_k1` (the lambda=5.0, alpha=0.5 baseline)
- `forgetting` metric (higher = more catastrophic forgetting)
- `expert_task_wins` in results — does the win distribution change?

---

### Phase 6: Continual MoB (5 runs)

| Name | Config | What it tests |
|------|--------|--------------|
| `cmob_baseline_pseudolabel` | Pseudo-label, per-batch | Continual MoB baseline (task-free, shift detection) |
| `cmob_baseline_prototype` | Prototype, per-batch | Prototype routing in task-free setting |
| `cmob_persample_k1` | Prototype, per-sample k=1 | Per-sample routing in task-free setting |
| `cmob_distance_only` | Distance-only, per-sample k=1 | Cheapest routing in task-free setting |
| `cmob_train_proto` | Prototype training routing | Full prototype routing during both train and eval |

**Continual MoB vs task-aware MoB**: The task-aware runner (phases 1-5) knows when tasks switch and runs Fisher/prototype updates at task boundaries. Continual MoB sees a continuous data stream and must detect distribution shifts on its own. It's harder — accuracy will be lower.

**What to expect**:
- `cmob_baseline_pseudolabel` is the reference — compare all others to this
- `cmob_persample_k1` should improve over `cmob_baseline_prototype` for the same reason as phase 2: mixed batches get split correctly
- `cmob_distance_only` tests whether the distance-only finding holds in the harder continual setting
- `cmob_train_proto` is the most aggressive — prototype routing during training in a task-free stream

**Where to look**: The `_summary.txt` file has the overall accuracy. The `_bids.json` has shift detection events (`prototype_state_log`), load balance metrics, and per-digit routing.

---

## How to Read the Output Files

### `_results.json` (task-aware MoB only)

```json
{
  "avg_accuracy": 0.8670,        // <-- THE key number. Higher = better.
  "forgetting": 0.0523,          // How much earlier tasks degrade. Lower = better.
  "task_accuracies": [0.99, ...], // Accuracy right after training each task (peak)
  "final_accuracies": [0.95, ...] // Accuracy at the END on each task (after all tasks)
  // forgetting = mean(task_accuracies[i] - final_accuracies[i]) for tasks 1..4
}
```

**Reading `final_accuracies`**: Index maps to task:
- `[0]` = digits 0,1 (task 1, trained first — most vulnerable to forgetting)
- `[1]` = digits 2,3
- `[2]` = digits 4,5
- `[3]` = digits 6,7
- `[4]` = digits 8,9 (task 5, trained last — always fresh, the "overloaded" expert task)

The overloaded expert problem: With 4 experts and 5 tasks, one expert must handle 2 tasks. That expert's accuracy on its first task is the real test of continual learning.

### `_bids.json`

```json
{
  "training_summary": {
    "expert_wins": [352, 381, 396, 747],  // How many batches each expert won
    "load_balance": {
      "entropy": 1.33,              // Shannon entropy of win distribution
      "max_entropy": 1.39,          // Perfect balance = log(4) = 1.386
      "normalized_entropy": 0.96,   // entropy / max_entropy. >0.7 = good.
      "gini": -0.09                 // Gini coefficient. <0.3 = good.
    },
    "per_expert": {
      "expert_0": {
        "wins": 352,
        "win_rate": 0.1877,
        "exec_cost": {"mean": 3.64, "std": ...},   // How well expert predicts
        "forget_cost": {"mean": 106200, "std": ...}, // How much it'd forget
        "bid": {"mean": 0.98, "std": ...}
      }
    }
  },
  "eval_summary": {
    "load_balance": { ... },       // Same metrics for eval routing
    "per_digit_routing": {
      "0": {
        "total": 980,
        "correct": 960,
        "accuracy": 0.9796,
        "routing": {"0": 0, "1": 0, "2": 970, "3": 10}  // Which expert got digit 0
      }
    }
  }
}
```

### Key things to look for in `_bids.json`:

**1. Load balance** (`training_summary.load_balance`):
- `normalized_entropy` > 0.7 = experts share the work reasonably
- `normalized_entropy` < 0.4 = one expert dominates (routing collapse)
- Compare across experiments: does per-sample routing improve balance?

**2. Per-digit routing** (`eval_summary.per_digit_routing`):
- Each digit should route primarily to one expert
- If a digit routes to the wrong expert (one that didn't train on it), accuracy for that digit will be low
- Look for routing consistency: digit 0 and digit 1 should go to the same expert (they were trained together as task 1)

**3. Forgetting cost trajectory** (`per_expert.forget_cost.mean`):
- Experts with more accumulated knowledge have higher forget_cost
- This is the mechanism behind emergent load balancing: high forget_cost = high bid = loses auction = new work goes elsewhere
- Compare across experts: the expert handling 2 tasks should have the highest forget_cost

---

## What Success Looks Like

### Experiment 1 (Per-sample routing)
- `exp1_persample_k1` accuracy > `mob_baseline_prototype` accuracy
- The improvement comes from correctly splitting mixed batches — check per-digit routing for cleaner expert assignment

### Experiment 2 (Distance-only)
- `exp2_distance_only_*` accuracy within 1% of corresponding full-bid run
- This validates that forget_cost adds no useful per-sample routing signal at eval time

### Experiment 3 (Training-time routing)
- `exp3_train_proto_warmup2000` accuracy within 5% of `mob_baseline_prototype`
- This proves MoB can route without labels — the key requirement for LLM MoE

### Experiment 4 (Load balancing)
- `normalized_entropy` > 0.7 across all experiments
- No auxiliary loss was used — balance emerged from the auction mechanism alone
- Compare to GatedMoE results in existing baseline data (if available)

### Hyperparameter sensitivity
- Accuracy should be relatively stable across lambda_ewc={5, 50} and alpha/beta ratios
- If one setting dominates all others, the system is too sensitive to hyperparameters (bad for the LLM scaling story)

---

## Mapping Results to the LLM Scaling Story

| MNIST Result | What it proves for LLM MoE |
|---|---|
| Per-sample k=1 works | Per-token routing works at any feature dimension |
| k=2 adds value on ambiguous inputs | Mixtral-style top-2 routing is viable with auction semantics |
| Distance-only matches full bid | Router cost is competitive with learned `W_g @ x` |
| Training-time routing works after warmup | MoB can route during LLM pretraining (unsupervised) |
| Entropy > 1.0 without auxiliary loss | No load-balance hyperparameter tuning needed at scale |
| Results stable across lambda_ewc | EWC strength is not a fragile hyperparameter |

The MNIST experiments validate the **mechanisms**. The actual numbers (86% vs 90%) matter less than whether the mechanisms work correctly. If per-sample routing, distance-only bids, and training-time routing all work on MNIST, they'll work on CIFAR and LLM features too — the math is dimension-agnostic.
