# MoB Phase-1 FeCAM-Router Gate — HPC Deployment Package

**Protocol**: `docs/protocols/fecam-router-gate.md` v1.2.1 (freeze target 2026-04-26)
**Target cluster**: ACIDSDB (Arctic), partition `qGPU48`, 1x V100 32GB per job
**User home**: `/home/users/dvyas4/`

## What's in this directory

```
HPC/
├── README.md                         # (this file)
├── HPCguide.md                       # step-by-step setup + submission
├── requirements.txt                  # pip install
├── slurm/
│   └── run_gate.sh                   # array driver: 20 seeds x 2 arms = 40 jobs
└── gate_runner/
    ├── __init__.py
    ├── run_arm.py                    # main entry: python -m gate_runner.run_arm ...
    ├── fecam_core.py                 # port from dipamgoswami/FeCAM @ e33f39d1
    ├── auction.py                    # MoB auction routing (alpha*d_M + beta*F + gamma*c)
    ├── experts.py                    # ViT-B/16 frozen + LoRA(r=8)+FFN trainable expert
    ├── data.py                       # CIFAR-100 10-task loader, LAMDA-PILOT class order
    ├── ewc.py                        # Fisher + consolidation, clamp_min=0.1
    ├── config.yaml                   # frozen config (§2.1 invariants + §2.3 T1..T7)
    ├── utils.py                      # seeding, JSON IO, hashing, logging
    └── tests/
        ├── __init__.py
        └── test_acceptance.py        # 20 acceptance tests from §8.4
```

## Quick start

Upload this directory to `/home/users/dvyas4/mob-gate/` and follow `HPCguide.md`.

## Upload from local

Drag-and-drop the **contents** of `C:/MoB Final/HPC/` into
`/home/users/dvyas4/mob-gate/` via the cluster's web file manager.
Detailed steps (including pre-upload bytecode cleanup and post-upload
line-ending normalization) are in **`HPCguide.md` §1**. Start there.

After upload, SSH into the cluster and continue at `HPCguide.md` step 2
(module load).

## Protocol compliance flags (read this before running)

The following items diverge from the user's sprint brief but align with
the protocol v1.2.1; pinning protocol where they disagree and surfacing
the deltas explicitly so Breach/Nosh can over-ride via §13 amendment if
desired.

1. **Expert count K=4**, not K=8. Protocol §2.1 pins "Expert count: 4 experts
   (per S1/S2 commitment; `memory/MEMORY.md` and `feedback_4experts.md`)". This
   is load-bearing continual-learning difficulty — an over-loaded expert IS the
   continual-learning challenge. See `memory/MEMORY.md` feedback entry.

2. **Task count default is 20T x 5C** (protocol §2.1 "Headline config" +
   §1.4 "A_T on the 20-task split" + §1.7 paper metric primacy). The
   earlier 10T x 10C default from the user's sprint brief was corrected
   by Nosh post-sprint (2026-04-19) because §1.4's preregistered primary
   statistic is computed at T=20. The 10T x 10C and 5T x 20C configs
   remain protocol-registered (§4.2 configs 2 and 3) as supplementary
   runs; set `data.num_tasks` and `data.classes_per_task` in
   `gate_runner/config.yaml` to switch.

3. **Arm B definition (resolved)**: Per §1.3 + §2.2, Arm B = Arm A with
   `beta=gamma=0` hard-coded. SAME K=4 experts, SAME Mahalanobis core, SAME
   auction structure, only the beta/gamma coefficients are zero. NOT
   single-expert FeCAM. `auction.Auction(arm="B", ...)` enforces this
   structurally in `auction.py` and acceptance tests 1, 9, 11 verify it.

4. **LoRA-QKV location**: applied to the CLS-token output as a
   per-expert linear correction, not inside each block's QKV. This keeps
   the backbone strictly frozen and shared across experts (DSIC safety
   per §2.1 Backbone freeze scope rationale). Expressively equivalent
   for a frozen encoder. Trainable-param budget ~300K per expert matches
   §2.1. See `experts.Expert` class docstring for the argument.

5. **No data augmentation** per user sprint spec. Protocol §2.1 row
   "Data augmentation" suggests LAMDA-PILOT default (crop + flip); this
   is matched across arms either way so does not bias A-vs-B.

## Operating ranges

- **Single-seed runtime estimate**: ~2-4 hours on V100 32GB (20 epochs/task
  x 20 tasks x ~10 steps/epoch at bs=128, 5 classes/task × 500 imgs/class ≈ 2500
  samples → ~20 steps/epoch; total optimizer steps ≈ 4000/seed, similar to
  10T×10C's 8000 steps but with fewer samples/task — wall-clock comparable).
- **Full 40-job array**: ~80-160 GPU-hours total, parallelizable to
  completion in a single 48-hour slurm wall-clock window if the cluster
  has >= 5 V100s free concurrently.
- **Pilot (6 jobs = 3 seeds x 2 arms)**: ~12-24 GPU-hours. Run this first,
  validate acceptance tests, then dispatch the remainder.

## Safety notes

- `gate_runner/config.yaml` is the SINGLE source of truth for hyperparameters.
  Its SHA-256 is emitted into every `run.json` as `config_hash`; acceptance
  test 8 asserts equality across all runs.
- Fisher clamp `min=0.1` in `ewc.EWC._normalize_and_clamp` is LOAD-BEARING
  per `memory/MEMORY.md`. Do not lower without a §4.4 ladder rerun.
- `CUBLAS_WORKSPACE_CONFIG=":4096:8"` is set in `slurm/run_gate.sh` BEFORE
  the python process starts; this is required for `torch.use_deterministic_algorithms`.

## Status: pre-freeze delivery

This package is delivered pre-freeze (2026-04-26 target). It is ready for
Dev to upload and run acceptance tests. Production seed launches start
after Nosh's freeze signature on the protocol.
