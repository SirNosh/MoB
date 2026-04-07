# MoB Development Environment

## Hardware

| Resource | Spec |
|---|---|
| GPU | NVIDIA RTX 4070 Super (12 GB VRAM) |
| RAM | 33.5 GB |
| CPUs | 16 |
| CUDA | Enabled (cu121) |

## Software

| Component | Version / Detail |
|---|---|
| OS | Windows 10 (build 10.0.26200) |
| Shell | **PowerShell** — use `;` for command chaining, not `&&` |
| Python | 3.11 (system install, no virtual environment) |
| PyTorch | 2.5.1+cu121 |
| Key packages | `torch`, `torchvision`, `numpy`, `tqdm`, `matplotlib`, `scipy` |
| Test runner | `pytest` |
| Git | 2.46.0 |

### Shell Notes

- PowerShell does not support `&&` chaining. Use `;` instead:
  ```powershell
  cd "C:\MoB Final" ; python tests/run_mob_only.py
  ```
- Paths with spaces must be quoted: `"C:\MoB Final"`.
- `rg` (ripgrep), `gh` (GitHub CLI), `wget`, and `ffmpeg` are **not** installed.

## Repository Layout

```
C:\MoB Final\
├── mob/                    # Task-aware MoB (knows task boundaries)
│   ├── auction.py
│   ├── bidding.py
│   ├── expert.py
│   ├── models.py
│   └── pool.py
├── contibualmob/           # Continual/online MoB (task-free, with shift detection)
│   ├── auction.py
│   ├── bidding.py
│   ├── expert.py
│   ├── models.py
│   ├── pool.py
│   ├── prototype_store.py
│   └── bid_diagnostics.py
├── tests/
│   ├── run_mob_only.py           # Experiment runner for mob/
│   ├── run_continual_mob.py      # Experiment runner for contibualmob/
│   ├── test_baselines.py         # pytest unit tests
│   ├── test_components.py        # pytest unit tests
│   └── check resources/          # Additional experiment scripts
├── results/                      # Output directory
│   └── experiments_v3/           # Current experiment results destination
├── data/                         # Dataset cache (auto-downloaded MNIST etc.)
└── README.md
```

### Two Parallel Packages

| Package | Purpose | Key Difference |
|---|---|---|
| `mob/` | Task-aware MoB | Receives explicit task boundaries; consolidates between tasks |
| `contibualmob/` | Continual/online MoB | No task boundaries; uses `ShiftDetector` + prototype routing for task-free operation |

Both share the same auction mechanism and bid formula. `contibualmob/` adds `PrototypeStore` and `ShiftDetector` on top.

## Experiment Output

Results are written to `results/experiments_v3/` as:
- JSON files (`*_results_seed_*.json`) — structured metrics (accuracy, forgetting, load balance)
- TXT files (`*_summary_*.txt`) — human-readable summaries
- PNG files — plots (ablation, accuracy curves)
