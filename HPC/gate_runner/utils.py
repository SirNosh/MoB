"""Utilities: deterministic seeding, JSON IO, hashing, logging.

Scope: infrastructure only. No torch.nn.Modules, no FeCAM math. Kept tiny.

Protocol refs:
    docs/protocols/fecam-router-gate.md v1.2.1
    §4.3 Determinism protocol  (five-source seeding + cuBLAS env var)
    §8.3 run.json schema       (output format)
    §8.4 tests 5, 8, 11, 13    (determinism + hash artifacts these helpers feed)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Union

import numpy as np
import torch
import yaml


# ------------------------------------------------------------------
# Deterministic seeding (§4.3)
# ------------------------------------------------------------------
def set_all_seeds(seed: int, *, cudnn_deterministic: bool = True,
                  cudnn_benchmark: bool = False,
                  use_deterministic_algorithms: bool = True,
                  warn_only: bool = True) -> None:
    """Seed every source listed in §4.3.

    Order matters: cuBLAS env var must be set BEFORE the first CUDA kernel
    launch to take effect. Callers should invoke this at the very top of
    main() before any CUDA work (including model construction on GPU).
    """
    # cuBLAS deterministic workspace (must come first)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark
    try:
        torch.use_deterministic_algorithms(
            use_deterministic_algorithms, warn_only=warn_only
        )
    except Exception as e:  # older torch may not expose warn_only
        logging.warning(f"use_deterministic_algorithms not fully applied: {e!r}")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn; §2.1 DataLoader worker seeding invariant.

    Each worker derives its seed from torch.initial_seed() (which PyTorch
    already offsets by worker_id from the base generator), and seeds
    numpy.random and python random from it. This matches the standard
    PyTorch idiom cited in v1.1 §2.1 and verified by §8.4 test 14.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------------
# JSON IO (with numpy-array serialization)
# ------------------------------------------------------------------
class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=indent, cls=_NumpyJSONEncoder, sort_keys=False)


def read_json(path: Union[str, Path]) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Hashing (used for config_hash, backbone_hash, first_bid_hash, loss_at_t0_hash)
# ------------------------------------------------------------------
def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    """Hash the raw bytes of a tensor. Tensor is detached, moved to CPU,
    made contiguous, and cast to float64 for cross-device determinism."""
    arr = t.detach().cpu().contiguous().to(torch.float64).numpy()
    return sha256_bytes(arr.tobytes())


def sha256_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    """Backbone-hash artifact per §8.3. Iterates keys sorted for determinism."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            h.update(k.encode("utf-8"))
            h.update(v.detach().cpu().contiguous().to(torch.float64).numpy().tobytes())
    return h.hexdigest()


def canonical_yaml_hash(path: Union[str, Path]) -> str:
    """Config-hash artifact per §8.3 / §8.4 test 8 / test 18.

    Load the YAML, re-dump in canonical form (sorted keys, no aliases), hash.
    Two structurally-identical configs yield the same hash regardless of
    whitespace or key order in the source file.
    """
    with Path(path).open("r") as f:
        obj = yaml.safe_load(f)
    canonical = yaml.safe_dump(obj, sort_keys=True, default_flow_style=False)
    return sha256_bytes(canonical.encode("utf-8"))


# ------------------------------------------------------------------
# Logger (stdout only — slurm captures to .out)
# ------------------------------------------------------------------
def get_logger(name: str = "gate_runner", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ------------------------------------------------------------------
# run.json schema helpers (§8.3)
# ------------------------------------------------------------------
def new_run_record(arm: str, seed: int, protocol_version: str,
                   config_hash: str, gpu: str) -> Dict[str, Any]:
    """Initialize a run.json dict. Populated incrementally as the run
    progresses; written to disk at the end."""
    return {
        "protocol_id": "MOB-GATE-001",
        "protocol_version": protocol_version,
        "arm": arm,
        "seed": int(seed),
        "config_hash": config_hash,
        "gpu": gpu,
        "started_at": None,
        "completed_at": None,
        "wall_clock_sec": None,
        "per_task_acc": [],
        "final_acc": None,
        "aia": None,
        "forgetting": None,
        "fisher_diagnostics": {},
        "routing_distribution": [],
        "first_bid_hash": None,
        "loss_at_t0_hash": None,
        "backbone_hash": None,
        "caveats": [],
    }


def compute_aia(per_task_acc: Iterable) -> float:
    """Average Incremental Accuracy = mean over tasks of the per-task
    "mean acc of tasks seen so far after learning task t". Input is the
    triangular matrix [[a_{0,0}], [a_{0,1}, a_{1,1}], ...]. Each row is
    the "after-task-t" evaluation over tasks 0..t."""
    rows = list(per_task_acc)
    means = [float(np.mean(r)) for r in rows]
    return float(np.mean(means)) if means else 0.0


def compute_forgetting(per_task_acc: Iterable) -> float:
    """Chaudhry 2018 avg forgetting: 1/(T-1) * sum_{i<T} max_{t<T}(a_{i,t}) - a_{i,T}.
    Input triangular: rows[t][i] = accuracy on task i after learning task t,
    defined for i <= t.
    """
    rows = list(per_task_acc)
    T = len(rows)
    if T < 2:
        return 0.0
    fvals = []
    for i in range(T - 1):  # exclude last task
        best = max(rows[t][i] for t in range(i, T))
        final = rows[T - 1][i]
        fvals.append(best - final)
    return float(np.mean(fvals)) if fvals else 0.0
