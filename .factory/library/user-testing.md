# MoB Validation Surface

## Overview

- **Surface:** CLI-based Python scripts (no web UI, no API server).
- **Testing tool:** Run experiment scripts, then inspect output JSON/TXT files for metrics.
- **Framework:** `pytest` for unit tests; manual script execution for integration / full experiments.

## Running Experiments

### Task-aware MoB (`mob/`)

```powershell
cd "C:\MoB Final"
python tests/run_mob_only.py --seed 42 --num_experts 4 --epochs 3
```

### Continual MoB (`contibualmob/`)

```powershell
cd "C:\MoB Final"
python tests/run_continual_mob.py --seed 42 --num_experts 4 --train_routing prototype
```

Common flags: `--seed`, `--num_experts`, `--epochs`, `--train_routing` (`label` or `prototype`), `--eval_routing` (`pseudo_label` or `prototype`).

### Unit Tests

```powershell
cd "C:\MoB Final"
pytest tests/test_components.py tests/test_baselines.py -v
```

## Validating Results

Experiment output lands in `results/` (or `results/experiments_v3/` for v3 runs).

### Key files to inspect

| File pattern | Contents |
|---|---|
| `*_results_seed_*.json` | Full structured metrics — per-task accuracy, forgetting, load balance, bid diagnostics |
| `*_summary_*.txt` | Human-readable summary |

### Key metrics to check

| Metric | Where in JSON | Healthy range | What collapse looks like |
|---|---|---|---|
| `avg_accuracy` | Top-level | 0.85–0.95 | < 0.60 (single expert can't generalize) |
| `forgetting` | Top-level | 0.0–0.10 | > 0.30 (EWC not protecting old tasks) |
| `load_balance` | `load_balance` or per-task stats | 0.6–1.0 (1.0 = perfectly uniform) | < 0.1 (one expert wins everything) |
| Per-expert win counts | `expert_wins` or bid diagnostics | Roughly equal across experts | One expert at 95%+, others at 0–2% |

### Quick validation checklist

1. **Load balance > 0.3** — confirms routing collapse is not occurring.
2. **avg_accuracy > 0.80** — confirms experts are learning effectively.
3. **forgetting < 0.15** — confirms EWC is working and old knowledge is retained.
4. **All experts have wins > 0** — confirms no expert is permanently idle.

## Resource Cost

| Metric | Value |
|---|---|
| Time per full experiment | ~90–110 seconds |
| GPU VRAM per experiment | ~2 GB |
| GPU available | RTX 4070 Super, 12 GB VRAM |
| Max concurrent experiments | **2** (each uses ~2 GB; leave headroom for OS/driver) |
| CPU usage | Moderate (data loading is CPU-bound) |
| Disk per result set | < 1 MB (JSON + TXT + optional PNG) |

## Experiment Workflow

```
1. Edit source in contibualmob/ or mob/
2. Run pytest to catch regressions:
   pytest tests/test_components.py -v
3. Run full experiment:
   python tests/run_continual_mob.py --seed 42 --train_routing prototype
4. Read output:
   - results/*_summary_*.txt for quick check
   - results/*_results_*.json for detailed metrics
5. Compare load_balance, avg_accuracy, forgetting against baseline
```
