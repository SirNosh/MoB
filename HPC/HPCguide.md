# HPC Setup and Submission Guide — ACIDSDB (Arctic)

**Protocol**: `docs/protocols/fecam-router-gate.md` v1.2.1
**Cluster user**: `dvyas4`
**Target dir**: `/home/users/dvyas4/mob-gate/`

This guide walks from cold-clone to pilot results. Each numbered step is a
single shell operation; read before pasting.

---

## 1. Upload (drag-and-drop via cluster web portal)

The `HPC/` directory is self-contained. Drag-and-drop the **contents** of
`C:/MoB Final/HPC/` into `/home/users/dvyas4/mob-gate/` via the cluster's
web file manager. Drop the contents (not the `HPC/` folder itself) so the
target tree is:

```
/home/users/dvyas4/mob-gate/
├── README.md
├── HPCguide.md
├── requirements.txt
├── slurm/
│   └── run_gate.sh
└── gate_runner/
    ├── __init__.py
    ├── run_arm.py
    ├── fecam_core.py
    ├── auction.py
    ├── experts.py
    ├── data.py
    ├── ewc.py
    ├── config.yaml
    ├── utils.py
    └── tests/
        ├── __init__.py
        └── test_acceptance.py
```

**Before upload** (one-time on local Windows machine): delete any
accidentally-generated pytest/bytecode artifacts — they'd confuse the
pytest run on the cluster. From the repo root:

```powershell
# PowerShell
Get-ChildItem -Path "C:\MoB Final\HPC\" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path "C:\MoB Final\HPC\" -Recurse -Filter "*.pyc" | Remove-Item -Force
Get-ChildItem -Path "C:\MoB Final\HPC\" -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force
```

Then drag-and-drop. Expected upload size: ~60 KB (this is all source; no
data, no model weights, no venv — those are built on the cluster in
steps 3 and 5).

**After upload**, SSH in and verify the layout:

```bash
ssh dvyas4@<cluster-login-host>
cd /home/users/dvyas4/mob-gate/
ls -la
# expect: README.md HPCguide.md requirements.txt slurm/ gate_runner/

# confirm the sh script has unix line endings (drag-and-drop from Windows
# can sometimes convert to CRLF; slurm scripts need LF-only)
file slurm/run_gate.sh
# expected: "ASCII text" (NOT "ASCII text, with CRLF line terminators")

# if CRLF was introduced by the web uploader, normalize:
# sed -i 's/\r$//' slurm/run_gate.sh
```

**Line-ending check is important.** If `slurm/run_gate.sh` picked up CRLF
from Windows, slurm will fail to launch it with an opaque shell error.
`sed -i 's/\r$//' slurm/run_gate.sh` fixes it in-place. Similarly if you
ever see odd Python import behavior after a Windows-to-cluster upload,
`dos2unix gate_runner/*.py` is a safe one-shot normalizer (apt-install
may be needed on the cluster; `sed -i 's/\r$//' <file>` works without it).

---

## 2. Load modules

On a login node (not inside a slurm job yet):

```bash
module purge
module load cuda/12.1
module load python/3.10.14

# sanity
python3 --version   # Python 3.10.14
nvcc --version      # release 12.1
```

---

## 3. Create + activate virtualenv

```bash
cd /home/users/dvyas4/mob-gate
python3 -m venv mob_env
source mob_env/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Install takes ~5-10 minutes; torch wheels are large. If pip complains about
torch not finding a CUDA 12.1-compatible wheel, force explicit index:

```bash
pip install torch>=2.0 torchvision>=0.15 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 4. Verify GPU + torch determinism

Run on a login node (CPU-only check first), then grab an interactive GPU
node to confirm CUDA.

```bash
# CPU sanity
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

# GPU sanity (interactive slurm)
srun -p qGPU48 --gres=gpu:V100:1 --mem=8G --time=00:15:00 --pty bash
module load cuda/12.1 python/3.10.14
source mob_env/bin/activate
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"mem={torch.cuda.mem_get_info()}")
import os; os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
x = torch.randn(1024, 1024, device="cuda"); y = x @ x
print(f"mm ok, y.sum()={y.sum().item():.3f}")
PY
exit   # leave interactive slurm
```

---

## 5. Download CIFAR-100 once

On a login node:

```bash
python - <<'PY'
from torchvision import datasets
_ = datasets.CIFAR100(root="/home/users/dvyas4/mob-gate/data", train=True, download=True)
_ = datasets.CIFAR100(root="/home/users/dvyas4/mob-gate/data", train=False, download=True)
print("CIFAR-100 ready")
PY
```

This populates `./data/cifar-100-python/` and avoids each slurm job racing
to download. The array script passes `--data_root $PROJECT_DIR/data`.

---

## 6. Run acceptance tests (pre-pilot)

All 20 tests from §8.4. Most are [PRE-RUN] synthetic; a few are [POST-RUN]
and skip cleanly until pilot artifacts exist.

```bash
cd /home/users/dvyas4/mob-gate
source mob_env/bin/activate
pytest gate_runner/tests/ -v
```

Expected output: 15+ tests pass, remaining 4-5 skipped (post-run).
Any FAIL blocks pilot launch per §7.2 criterion 4. See
`test_acceptance.py` docstring for the pre-/post-run classification.

---

## 7. Submit the pilot (indices 0..5 = 3 seeds x 2 arms)

```bash
sbatch --array=0-5 slurm/run_gate.sh
```

Monitor:

```bash
squeue -u dvyas4
# progress of a specific job
tail -f logs/gate_<JOBID>_0.out
# all logs
tail -n 20 logs/gate_*_*.out
```

Each pilot job wall-clock should be ~2-4 hours on V100 32GB. With 3 seeds
x 2 arms = 6 jobs, cluster parallelism will bring the pilot home in a
single wall-clock pass if nodes are available.

---

## 8. Pilot adjudication

When the pilot completes, six JSON files exist under `results/gate/`:

```
results/gate/seed_42_arm_A.json
results/gate/seed_42_arm_B.json
results/gate/seed_43_arm_A.json
results/gate/seed_43_arm_B.json
results/gate/seed_44_arm_A.json
results/gate/seed_44_arm_B.json
```

Re-run the acceptance tests — now the POST-RUN ones fire on the real
artifacts:

```bash
pytest gate_runner/tests/ -v
```

Critical post-run gates (see §7.2):
- `test_arms_disagree_on_at_least_one_percent_of_pilot_inputs` — if this
  fails, beta/gamma are mechanistically inactive; HALT and escalate to
  Nosh per §7.2 criterion 7.
- `test_fisher_cv_inclusion` — if CV(log F) > 0.5, execute §4.4 branch-(a)
  clamp-ladder rerun (Fisher clamp `{0.1, 0.3, 1.0, 3.0}`).

---

## 9. Full run submission (after pilot passes)

**Default (n=10 per arm)**: indices 6..19 (seeds 45..51 x arms A, B).

```bash
sbatch --array=6-19 slurm/run_gate.sh
```

**D3 backup (n=20 per arm; only if pilot sigma_upper in (1.30, 1.75])**:
add indices 20..39 (seeds 52..61 x arms A, B) via:

```bash
sbatch --array=20-39 slurm/run_gate.sh
```

Full run wall-clock: ~14 remaining jobs (or 34 under D3 backup) x 2-4h each,
parallelizable across available V100 nodes.

---

## 10. Results download (drag-and-drop from cluster web portal)

Download via the same web file manager used in step 1, or bundle into a
tarball first for a cleaner single-file download. On the cluster:

```bash
cd /home/users/dvyas4/mob-gate
tar czf results_gate_$(date -I).tar.gz results/gate/ logs/
ls -lh results_gate_*.tar.gz
```

Then drag-and-drop `results_gate_<date>.tar.gz` from the cluster web
portal down to your local machine. Expected size: ~1–5 MB (JSONs are
small; slurm logs dominate).

On local Windows, extract into the repo:

```powershell
# Create the target subtree if missing
New-Item -Path "C:\MoB Final\results\gate" -ItemType Directory -Force
New-Item -Path "C:\MoB Final\HPC\logs"      -ItemType Directory -Force

# tar is available on Windows 10+ / 11 out of the box
cd "C:\MoB Final"
tar xzf path\to\results_gate_<date>.tar.gz
# Contents land under .\results\gate\ and .\logs\ — move logs to HPC\logs\:
Move-Item -Path ".\logs\*" -Destination ".\HPC\logs\" -Force
Remove-Item ".\logs" -Force
```

Every run produces `seed_<s>_arm_<A|B>.json` with per-task accuracy,
routing distribution, Fisher diagnostics, backbone hash, and timing.

**Drop the downloaded results into** `C:/MoB Final/results/gate/` and ping
Nosh; KAY/O pre-adjudication fires on the pilot 6 JSONs before the full
array launches.

---

## 11. Post-run analysis (Nosh-facing handoff)

After download, hand the JSONs to Nosh for the §6 BCa bootstrap analysis:

```
results/gate/seed_{42..51}_arm_A.json
results/gate/seed_{42..51}_arm_B.json
```

Breach's analysis pipeline (not in this package) consumes these and
produces `results/gate/summary.json` with L_95, U_95, gate_status
(PASS/TIE/FAIL) per §1.2.

---

## Troubleshooting

- **`CUDA error: CUBLAS_STATUS_NOT_INITIALIZED`** — `CUBLAS_WORKSPACE_CONFIG`
  is likely unset. `run_gate.sh` exports it before python launch. If you
  invoke python manually, export it in your shell first.
- **`ImportError: timm`** — make sure the venv is activated and
  `pip install -r requirements.txt` completed.
- **Determinism warnings at runtime** (e.g. "upsample_bicubic2d_backward
  does not have a deterministic implementation") — these are WARN_ONLY per
  §4.3. Logged; not fatal.
- **Job wall-clock exceeded 06:00:00** — raise `--time` in
  `slurm/run_gate.sh` (partition `qGPU48` allows up to 48h) or tune the
  data loader `num_workers` / pin_memory for throughput.
- **Out of VRAM on V100 32GB** — lower `eval_batch_size` in config.yaml
  first (eval loops collect more activations at once than training).
  Training batch size 128 fits comfortably in 32GB for ViT-B/16 + 4 experts.
