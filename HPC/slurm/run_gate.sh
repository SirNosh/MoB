#!/bin/bash
# MoB Phase-1 FeCAM-Router gate — ACIDSDB (Arctic) slurm array driver
# Protocol: docs/protocols/fecam-router-gate.md v1.2.1
#
# Array layout:
#   40 jobs total = 20 seeds (42..61) x 2 arms (A, B)
#   index N -> seed = 42 + (N / 2)
#              arm  = A if (N % 2 == 0) else B
# The seed plan §4.1 prereg is {42..51}; §4.3 D3-backup extends to {42..61}
# only if the pilot's sigma_upper lands in the (1.30, 1.75] band. Default
# submission covers BOTH ranges; pilot gate (Dev adjudicates) decides which
# indices to actually run.
#
# Pilot: indices 0..5  (seeds 42, 43, 44 x arms A, B)  -> run first
# Full run (n=10):   indices 6..19  (seeds 45..51 x arms A, B)
# Full run backup (n=20):  indices 20..39 (seeds 52..61 x arms A, B)

#SBATCH --job-name=mob-fecam-gate
#SBATCH --partition=qGPU48
#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/gate_%A_%a.out
#SBATCH --error=logs/gate_%A_%a.err

set -euo pipefail

# --- Modules ---
module purge
module load cuda/12.1
module load python/3.10.14

# --- Project paths ---
PROJECT_DIR="${PROJECT_DIR:-/home/users/dvyas4/mob-gate}"
cd "$PROJECT_DIR"

# --- Virtualenv ---
VENV="$PROJECT_DIR/mob_env"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found at $VENV. Run HPCguide.md step 3 first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- Ensure output dirs ---
mkdir -p logs results/gate

# --- Array index -> (seed, arm) ---
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( 42 + IDX / 2 ))
if (( IDX % 2 == 0 )); then
    ARM="A"
else
    ARM="B"
fi

# --- cuBLAS determinism (must be set before any CUDA launch, per §4.3) ---
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# --- Sanity: GPU visible? ---
python - <<'PY'
import torch, sys
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_count={torch.cuda.device_count()} name={torch.cuda.get_device_name(0)}")
else:
    print("NO GPU VISIBLE — aborting", file=sys.stderr)
    sys.exit(1)
PY

OUT="results/gate/seed_${SEED}_arm_${ARM}.json"

echo "============================================================"
echo "ARRAY_IDX=$IDX  SEED=$SEED  ARM=$ARM"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "OUT=$OUT"
echo "T0: $(date -Iseconds)"
echo "============================================================"

T0=$(date +%s)
python -m gate_runner.run_arm \
    --arm "$ARM" \
    --seed "$SEED" \
    --config gate_runner/config.yaml \
    --out "$OUT" \
    --data_root "$PROJECT_DIR/data" \
    --device cuda
T1=$(date +%s)

echo "============================================================"
echo "COMPLETE  wall_clock_sec=$((T1 - T0))"
echo "============================================================"
