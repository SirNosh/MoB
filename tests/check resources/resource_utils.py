"""
Resource Tracking Utilities for Benchmarking Continual Learning Models.

Provides consistent measurement of:
- Wall-clock time (total, per-snapshot intervals)
- GPU VRAM (peak allocated, peak reserved, model-only footprint)
- System RAM (peak RSS via psutil background thread)
- FLOPs (via torch.utils.flop_counter or analytical fallback)
- Throughput (samples/sec)
- Parameter count (total, trainable)

Usage:
    from resource_utils import ResourceTracker, FlopCounter, format_results

    tracker = ResourceTracker(device)
    tracker.start()
    # ... training loop ...
    tracker.snapshot("train_end", num_samples=N)
    # ... eval loop ...
    tracker.snapshot("eval_end", num_samples=M)
    results = tracker.stop()
"""

import time
import threading
import torch
import torch.nn as nn
from typing import Dict, Optional, Any


# ============================================================================
# RAM Tracking (psutil)
# ============================================================================

_PSUTIL_AVAILABLE = False
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass


class _RAMMonitor:
    """Background thread that samples RSS to capture peak RAM usage."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_rss_bytes = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not _PSUTIL_AVAILABLE:
            return
        self._running = True
        proc = psutil.Process()
        self.peak_rss_bytes = proc.memory_info().rss
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        proc = psutil.Process()
        while self._running:
            try:
                rss = proc.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> int:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak_rss_bytes


# ============================================================================
# FlopCounter
# ============================================================================

class FlopCounter:
    """
    Measure FLOPs for a single forward pass, optionally including backward.

    Uses torch.utils.flop_counter.FlopCounterMode (PyTorch 2.1+).
    Falls back to analytical estimate for SimpleCNN if unavailable.
    """

    @staticmethod
    def count_forward_flops(model: nn.Module, sample_input: torch.Tensor) -> int:
        """
        Count FLOPs for a single forward pass.

        Parameters
        ----------
        model : nn.Module
        sample_input : torch.Tensor
            A single batch (any batch size) of input data.

        Returns
        -------
        int  Total FLOPs for the forward pass.
        """
        try:
            from torch.utils.flop_counter import FlopCounterMode
            model.eval()
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                with torch.no_grad():
                    model(sample_input)
            total = flop_counter.get_total_flops()
            return int(total)
        except (ImportError, AttributeError, Exception):
            return FlopCounter._analytical_estimate(model, sample_input)

    @staticmethod
    def _analytical_estimate(model: nn.Module, sample_input: torch.Tensor) -> int:
        """Rough analytical FLOPs for a Conv-FC network on MNIST-sized input."""
        total = 0
        x = sample_input
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # FLOPs ~ 2 * Cout * Cin * Kh * Kw * Hout * Wout * batch
                batch = x.size(0)
                h_out = x.size(2)  # approximate (before this layer)
                w_out = x.size(3)
                flops = 2 * m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * h_out * w_out * batch
                total += flops
            elif isinstance(m, nn.Linear):
                batch = x.size(0) if x.dim() > 1 else 1
                flops = 2 * m.in_features * m.out_features * batch
                total += flops
        return total

    @staticmethod
    def estimate_total_flops(
        model: nn.Module,
        sample_input: torch.Tensor,
        num_train_batches: int,
        num_eval_batches: int,
        backward_multiplier: float = 2.0
    ) -> Dict[str, int]:
        """
        Estimate total FLOPs for an entire experiment run.

        Parameters
        ----------
        backward_multiplier : float
            Backward pass is typically ~2x forward FLOPs.

        Returns
        -------
        Dict with 'forward_per_batch', 'train_total', 'eval_total' keys.
        """
        fwd = FlopCounter.count_forward_flops(model, sample_input)
        train_per_batch = int(fwd * (1 + backward_multiplier))
        return {
            'forward_per_batch': fwd,
            'backward_per_batch': int(fwd * backward_multiplier),
            'train_per_batch': train_per_batch,
            'train_total': train_per_batch * num_train_batches,
            'eval_per_batch': fwd,
            'eval_total': fwd * num_eval_batches,
        }


# ============================================================================
# ResourceTracker
# ============================================================================

class ResourceTracker:
    """
    Track GPU VRAM, system RAM, wall-clock time, and throughput.

    Usage:
        tracker = ResourceTracker(device)
        tracker.start()
        # ... work ...
        tracker.snapshot("train_end", num_samples=6000)
        # ... more work ...
        tracker.snapshot("eval_end", num_samples=1000)
        results = tracker.stop()
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == 'cuda'

        self._start_time: Optional[float] = None
        self._last_snapshot_time: Optional[float] = None
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._ram_monitor = _RAMMonitor(interval=0.1)

        # Baseline GPU memory (recorded at start)
        self._baseline_vram_alloc = 0
        self._baseline_vram_reserved = 0

    def start(self):
        """Begin tracking. Call before training/eval."""
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._baseline_vram_alloc = torch.cuda.memory_allocated(self.device)
            self._baseline_vram_reserved = torch.cuda.memory_reserved(self.device)

        self._ram_monitor.start()
        self._start_time = time.perf_counter()
        self._last_snapshot_time = self._start_time

    def reset_peak(self):
        """Reset peak memory stats (call between train and eval)."""
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self, name: str, num_samples: int = 0):
        """
        Record a snapshot of current resource usage.

        Parameters
        ----------
        name : str
            Label for this snapshot (e.g. "train_end", "eval_end").
        num_samples : int
            Number of samples processed since last snapshot (for throughput).
        """
        now = time.perf_counter()
        elapsed_total = now - self._start_time
        elapsed_interval = now - self._last_snapshot_time

        snap: Dict[str, Any] = {
            'wall_time_total_s': elapsed_total,
            'wall_time_interval_s': elapsed_interval,
            'num_samples': num_samples,
            'throughput_samples_per_sec': num_samples / elapsed_interval if elapsed_interval > 0 and num_samples > 0 else 0.0,
        }

        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            snap['vram_peak_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            snap['vram_peak_reserved_mb'] = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            snap['vram_current_allocated_mb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        else:
            snap['vram_peak_allocated_mb'] = 0.0
            snap['vram_peak_reserved_mb'] = 0.0
            snap['vram_current_allocated_mb'] = 0.0

        if _PSUTIL_AVAILABLE:
            snap['ram_peak_rss_mb'] = self._ram_monitor.peak_rss_bytes / (1024 ** 2)
        else:
            snap['ram_peak_rss_mb'] = 0.0

        self._snapshots[name] = snap
        self._last_snapshot_time = now

    def stop(self) -> Dict[str, Any]:
        """
        Stop tracking and return all collected metrics.

        Returns
        -------
        Dict with 'snapshots', 'total_wall_time_s', 'baseline_vram_mb' keys.
        """
        end_time = time.perf_counter()
        peak_rss = self._ram_monitor.stop()

        result: Dict[str, Any] = {
            'total_wall_time_s': end_time - self._start_time if self._start_time else 0.0,
            'baseline_vram_allocated_mb': self._baseline_vram_alloc / (1024 ** 2),
            'baseline_vram_reserved_mb': self._baseline_vram_reserved / (1024 ** 2),
            'snapshots': dict(self._snapshots),
        }

        if _PSUTIL_AVAILABLE:
            result['final_peak_rss_mb'] = peak_rss / (1024 ** 2)
        else:
            result['final_peak_rss_mb'] = 0.0

        return result


# ============================================================================
# Parameter counting
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total, 'trainable_params': trainable}


def count_parameters_multi(models: list) -> Dict[str, int]:
    """Count parameters across multiple models (e.g., a pool of experts)."""
    total = sum(p.numel() for m in models for p in m.parameters())
    trainable = sum(p.numel() for m in models for p in m.parameters() if p.requires_grad)
    return {'total_params': total, 'trainable_params': trainable}


# ============================================================================
# Result formatting
# ============================================================================

def format_results(
    model_name: str,
    tracker_results: Dict[str, Any],
    flop_results: Optional[Dict[str, int]] = None,
    param_counts: Optional[Dict[str, int]] = None,
    accuracy: float = 0.0,
    forgetting: float = 0.0,
    task_accuracies: Optional[list] = None,
    final_accuracies: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format all resource metrics into a unified dict for comparison.

    Parameters
    ----------
    model_name : str
        Human-readable model name.
    tracker_results : dict
        Output of ResourceTracker.stop().
    flop_results : dict, optional
        Output of FlopCounter.estimate_total_flops().
    param_counts : dict, optional
        Output of count_parameters().
    accuracy : float
        Final average accuracy.
    forgetting : float
        Average forgetting metric.
    task_accuracies : list, optional
        Per-task accuracy measured right after training each task.
    final_accuracies : list, optional
        Per-task accuracy measured at the end after all tasks trained.
    extra : dict, optional
        Any extra model-specific metrics.

    Returns
    -------
    Dict with all metrics organized for comparison.
    """
    snaps = tracker_results.get('snapshots', {})
    train_snap = snaps.get('train_end', {})
    eval_snap = snaps.get('eval_end', {})

    result = {
        'model_name': model_name,
        'accuracy': accuracy,
        'forgetting': forgetting,

        # Per-task accuracies
        'task_accuracies': task_accuracies if task_accuracies else [],
        'final_accuracies': final_accuracies if final_accuracies else [],

        # Parameters
        'total_params': param_counts.get('total_params', 0) if param_counts else 0,
        'trainable_params': param_counts.get('trainable_params', 0) if param_counts else 0,

        # Training resources
        'train_time_s': train_snap.get('wall_time_interval_s', train_snap.get('wall_time_total_s', 0.0)),
        'train_vram_peak_mb': train_snap.get('vram_peak_allocated_mb', 0.0),
        'train_ram_peak_mb': train_snap.get('ram_peak_rss_mb', 0.0),
        'train_throughput': train_snap.get('throughput_samples_per_sec', 0.0),
        'train_flops': flop_results.get('train_total', 0) if flop_results else 0,

        # Eval resources
        'eval_time_s': eval_snap.get('wall_time_interval_s', 0.0),
        'eval_vram_peak_mb': eval_snap.get('vram_peak_allocated_mb', 0.0),
        'eval_throughput': eval_snap.get('throughput_samples_per_sec', 0.0),
        'eval_flops': flop_results.get('eval_total', 0) if flop_results else 0,

        # Totals
        'total_wall_time_s': tracker_results.get('total_wall_time_s', 0.0),
        'final_peak_rss_mb': tracker_results.get('final_peak_rss_mb', 0.0),
    }

    if extra:
        result.update(extra)

    return result


def print_comparison_table(all_results: list):
    """
    Print a formatted comparison table for all models.

    Parameters
    ----------
    all_results : list of dict
        Each dict is the output of format_results().
    """
    if not all_results:
        print("No results to display.")
        return

    # Column headers = model names
    names = [r['model_name'] for r in all_results]
    col_width = max(14, max(len(n) for n in names) + 2)

    def _fmt_val(val):
        if isinstance(val, float):
            if val >= 1e9:
                return f"{val/1e9:.2f}G"
            elif val >= 1e6:
                return f"{val/1e6:.2f}M"
            elif val >= 1e3:
                return f"{val/1e3:.1f}K"
            else:
                return f"{val:.2f}"
        elif isinstance(val, int):
            if val >= 1e9:
                return f"{val/1e9:.2f}G"
            elif val >= 1e6:
                return f"{val/1e6:.2f}M"
            elif val >= 1e3:
                return f"{val/1e3:.1f}K"
            else:
                return str(val)
        return str(val)

    rows = [
        ("Params (total)", 'total_params'),
        ("Params (trainable)", 'trainable_params'),
        ("Train Time (s)", 'train_time_s'),
        ("Train VRAM Peak (MB)", 'train_vram_peak_mb'),
        ("Train RAM Peak (MB)", 'train_ram_peak_mb'),
        ("Train FLOPs", 'train_flops'),
        ("Train Throughput (s/s)", 'train_throughput'),
        ("Eval Time (s)", 'eval_time_s'),
        ("Eval VRAM Peak (MB)", 'eval_vram_peak_mb'),
        ("Eval FLOPs", 'eval_flops'),
        ("Eval Throughput (s/s)", 'eval_throughput'),
        ("Total Time (s)", 'total_wall_time_s'),
        ("Peak RSS (MB)", 'final_peak_rss_mb'),
        ("Avg Accuracy", 'accuracy'),
        ("Avg Forgetting", 'forgetting'),
    ]

    # Determine number of tasks from final_accuracies
    num_tasks = max((len(r.get('final_accuracies', [])) for r in all_results), default=0)

    # Header
    header_label = "Metric".ljust(24)
    header_cols = "| ".join(n.center(col_width) for n in names)
    separator_label = "-" * 24
    separator_cols = "| ".join("-" * col_width for _ in names)

    print(f"\n{'='*80}")
    print("RESOURCE COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{header_label}| {header_cols}")
    print(f"{separator_label}| {separator_cols}")

    for label, key in rows:
        vals = [_fmt_val(r.get(key, 0)) for r in all_results]
        row_str = "| ".join(v.center(col_width) for v in vals)
        print(f"{label.ljust(24)}| {row_str}")

    # Per-task final accuracies
    if num_tasks > 0:
        print(f"{separator_label}| {separator_cols}")
        for t in range(num_tasks):
            label = f"Task {t+1} Acc (d{t*2},{t*2+1})"
            vals = []
            for r in all_results:
                fa = r.get('final_accuracies', [])
                if t < len(fa):
                    vals.append(f"{fa[t]:.4f}")
                else:
                    vals.append("N/A")
            row_str = "| ".join(v.center(col_width) for v in vals)
            print(f"{label.ljust(24)}| {row_str}")

    # Per-task forgetting (task_acc - final_acc)
    if num_tasks > 1:
        print(f"{separator_label}| {separator_cols}")
        for t in range(num_tasks - 1):
            label = f"Task {t+1} Forgetting"
            vals = []
            for r in all_results:
                ta = r.get('task_accuracies', [])
                fa = r.get('final_accuracies', [])
                if t < len(ta) and t < len(fa):
                    forg = max(0, ta[t] - fa[t])
                    vals.append(f"{forg:.4f}")
                else:
                    vals.append("N/A")
            row_str = "| ".join(v.center(col_width) for v in vals)
            print(f"{label.ljust(24)}| {row_str}")

    print(f"{'='*80}\n")
