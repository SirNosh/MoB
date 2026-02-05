"""
Resource Benchmarking: Three-Way Comparison
  1. Gated MoE + EWC (learnable gater, task boundaries)
  2. Task-Aware MoB (auction-based routing, task boundaries)
  3. Online MoB (auction-based routing, streaming/no task boundaries)

This file directly calls the run_experiment functions from the original files,
ensuring any changes to those files are automatically reflected in benchmarks.

Measures and compares:
1. Compute overhead (time, throughput)
2. Memory overhead (GPU peak, training overhead)
3. Accuracy and forgetting metrics

Usage:
    python tests/benchmark_resources.py
    python tests/benchmark_resources.py --runs 3  # Average over multiple runs
    python tests/benchmark_resources.py --skip_continual  # Skip Online MoB
"""

import os
import sys
import time
import json
import argparse
import gc
from typing import Dict

import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob.utils import set_seed
from tests.test_baselines import create_split_mnist

# =============================================================================
# IMPORT RUN_EXPERIMENT FUNCTIONS FROM ORIGINAL FILES
# This ensures any changes to those files are automatically reflected here.
# We ONLY import the run_experiment functions - no model classes.
# =============================================================================
from tests.run_mob_only import run_experiment as run_mob_experiment
from tests.run_gated_moe_ewc import run_experiment as run_gated_moe_experiment
from tests.run_continual_mob import run_continual_experiment


# =============================================================================
# MEMORY AND TIME TRACKING
# =============================================================================

class ResourceTracker:
    """Track time and GPU memory for an experiment with detailed breakdown."""

    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.baseline_memory = 0.0

    def reset_memory(self):
        if self.is_cuda:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            # Record baseline (PyTorch overhead)
            self.baseline_memory = self.get_current_memory_mb()

    def get_current_memory_mb(self):
        """Get currently allocated GPU memory in MB."""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device) / (1024**2)
        return 0.0

    def get_reserved_memory_mb(self):
        """Get reserved GPU memory (including cached) in MB."""
        if self.is_cuda:
            return torch.cuda.memory_reserved(self.device) / (1024**2)
        return 0.0

    def get_peak_memory_mb(self):
        """Get peak allocated GPU memory in MB."""
        if self.is_cuda:
            return torch.cuda.max_memory_allocated(self.device) / (1024**2)
        return 0.0

    def get_memory_snapshot(self):
        """Get detailed memory snapshot."""
        return {
            'current_allocated_mb': self.get_current_memory_mb(),
            'reserved_mb': self.get_reserved_memory_mb(),
            'peak_allocated_mb': self.get_peak_memory_mb(),
            'baseline_mb': self.baseline_memory,
            'overhead_mb': self.get_peak_memory_mb() - self.baseline_memory,
        }

    def start_timer(self):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def stop_timer(self, start_time):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter() - start_time


# =============================================================================
# BENCHMARK RUNNERS - Call original run_experiment functions
# =============================================================================

def benchmark_gated_moe(config: Dict, device: torch.device) -> Dict:
    """Benchmark Gated MoE + EWC by calling original run_experiment."""

    tracker = ResourceTracker(device)
    tracker.reset_memory()

    # Create datasets
    train_tasks = create_split_mnist(
        config['num_tasks'], train=True, batch_size=config['batch_size']
    )
    test_tasks = create_split_mnist(
        config['num_tasks'], train=False, batch_size=config['batch_size']
    )

    # Track memory before experiment starts
    memory_before = tracker.get_current_memory_mb()

    # Run experiment with timing (original file handles model creation internally)
    start_time = tracker.start_timer()
    results = run_gated_moe_experiment(train_tasks, test_tasks, config)
    total_time = tracker.stop_timer(start_time)

    # Get detailed memory snapshot
    memory_snapshot = tracker.get_memory_snapshot()
    memory_after = tracker.get_current_memory_mb()

    return {
        'model_name': 'Gated MoE + EWC',
        'total_time': total_time,
        'memory': memory_snapshot,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'avg_accuracy': results['avg_accuracy'],
        'forgetting': results['forgetting'],
        'final_accuracies': results['final_accuracies'],
        'task_accuracies': results['task_accuracies'],
    }


def benchmark_mob(config: Dict, device: torch.device) -> Dict:
    """Benchmark MoB by calling original run_experiment."""

    tracker = ResourceTracker(device)
    tracker.reset_memory()

    # Create datasets
    train_tasks = create_split_mnist(
        config['num_tasks'], train=True, batch_size=config['batch_size']
    )
    test_tasks = create_split_mnist(
        config['num_tasks'], train=False, batch_size=config['batch_size']
    )

    # Track memory before experiment starts
    memory_before = tracker.get_current_memory_mb()

    # Run experiment with timing (original file handles model creation internally)
    start_time = tracker.start_timer()
    results = run_mob_experiment(train_tasks, test_tasks, config)
    total_time = tracker.stop_timer(start_time)

    # Get detailed memory snapshot
    memory_snapshot = tracker.get_memory_snapshot()
    memory_after = tracker.get_current_memory_mb()

    return {
        'model_name': 'Task-Aware MoB',
        'total_time': total_time,
        'memory': memory_snapshot,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'avg_accuracy': results['avg_accuracy'],
        'forgetting': results['forgetting'],
        'final_accuracies': results['final_accuracies'],
        'task_accuracies': results['task_accuracies'],
    }


def benchmark_continual(config: Dict, device: torch.device) -> Dict:
    """Benchmark Continual MoB by calling original run_experiment."""

    tracker = ResourceTracker(device)
    tracker.reset_memory()

    # Create datasets
    train_tasks = create_split_mnist(
        config['num_tasks'], train=True, batch_size=config['batch_size']
    )
    test_tasks = create_split_mnist(
        config['num_tasks'], train=False, batch_size=config['batch_size']
    )

    # Track memory before experiment starts
    memory_before = tracker.get_current_memory_mb()

    # Run experiment with timing (original file handles model creation internally)
    start_time = tracker.start_timer()
    results = run_continual_experiment(train_tasks, test_tasks, config)
    total_time = tracker.stop_timer(start_time)

    # Get detailed memory snapshot
    memory_snapshot = tracker.get_memory_snapshot()
    memory_after = tracker.get_current_memory_mb()

    return {
        'model_name': 'Online MoB',
        'total_time': total_time,
        'memory': memory_snapshot,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'avg_accuracy': results['avg_accuracy'] / 100.0 if results['avg_accuracy'] > 1.0 else results['avg_accuracy'],  # Fix formatting issue
        'detected_shifts': results.get('detected_shifts', []),
    }


# =============================================================================
# COMPARISON REPORT
# =============================================================================

def compare_results(all_results: Dict[str, Dict]):
    """Generate detailed comparison report for all models."""

    model_names = list(all_results.keys())

    print("\n" + "="*100)
    print("RESOURCE COMPARISON: " + " vs ".join(model_names))
    print("="*100)

    header = f"{'Metric':<30}"
    for name in model_names:
        header += f" {name:>22}"

    # ===========================================================================
    # COMPUTE OVERHEAD
    # ===========================================================================
    print("\n" + "-"*100)
    print("COMPUTE OVERHEAD")
    print("-"*100)
    print(header)
    print("-"*100)

    row = f"{'Total Time':<30}"
    for name in model_names:
        row += f" {all_results[name]['total_time']:>21.2f}s"
    print(row)

    # Compute overhead metrics (assuming 5 tasks, 4 epochs, ~1875 batches per task = 9375 total)
    # This is approximate based on Split-MNIST with batch_size=32
    estimated_total_batches = 5 * 4 * 60  # 5 tasks * 4 epochs * ~60 batches

    row = f"{'Avg Time per Batch':<30}"
    for name in model_names:
        avg_time_per_batch = (all_results[name]['total_time'] / estimated_total_batches) * 1000  # ms
        row += f" {avg_time_per_batch:>20.2f}ms"
    print(row)

    row = f"{'Throughput (samples/sec)':<30}"
    for name in model_names:
        # batch_size=32, so samples/sec = 32 / time_per_batch
        time_per_batch = all_results[name]['total_time'] / estimated_total_batches
        throughput = 32.0 / time_per_batch if time_per_batch > 0 else 0
        row += f" {throughput:>21.1f}"
    print(row)

    # ===========================================================================
    # MEMORY OVERHEAD
    # ===========================================================================
    print("\n" + "-"*100)
    print("MEMORY OVERHEAD (GPU)")
    print("-"*100)
    print(header)
    print("-"*100)

    row = f"{'Peak Allocated':<30}"
    for name in model_names:
        peak = all_results[name]['memory']['peak_allocated_mb']
        row += f" {peak:>20.1f}MB"
    print(row)

    row = f"{'Reserved (with cache)':<30}"
    for name in model_names:
        reserved = all_results[name]['memory']['reserved_mb']
        row += f" {reserved:>20.1f}MB"
    print(row)

    row = f"{'Baseline (PyTorch)':<30}"
    for name in model_names:
        baseline = all_results[name]['memory']['baseline_mb']
        row += f" {baseline:>20.1f}MB"
    print(row)

    row = f"{'Model + Training Overhead':<30}"
    for name in model_names:
        overhead = all_results[name]['memory']['overhead_mb']
        row += f" {overhead:>20.1f}MB"
    print(row)

    row = f"{'Current (after training)':<30}"
    for name in model_names:
        current = all_results[name]['memory_after_mb']
        row += f" {current:>20.1f}MB"
    print(row)

    # ===========================================================================
    # ACCURACY METRICS
    # ===========================================================================
    print("\n" + "-"*100)
    print("ACCURACY METRICS")
    print("-"*100)
    print(header)
    print("-"*100)

    row = f"{'Average Accuracy':<30}"
    for name in model_names:
        acc = all_results[name].get('avg_accuracy', 0)
        row += f" {acc:>21.4f}"
    print(row)

    # Forgetting (if available)
    has_forgetting = any('forgetting' in all_results[name] for name in model_names)
    if has_forgetting:
        row = f"{'Average Forgetting':<30}"
        for name in model_names:
            forg = all_results[name].get('forgetting', 'N/A')
            if isinstance(forg, float):
                row += f" {forg:>21.4f}"
            else:
                row += f" {str(forg):>21}"
        print(row)

    # ===========================================================================
    # THREE-WAY COMPARISON
    # ===========================================================================
    print("\n" + "="*100)
    print("THREE-WAY COMPARISON")
    print("="*100)

    # Define comparison pairs
    comparisons = [
        ('Task-Aware MoB', 'Gated MoE + EWC'),
        ('Online MoB', 'Gated MoE + EWC'),
        ('Online MoB', 'Task-Aware MoB'),
    ]

    for model_a, model_b in comparisons:
        if model_a not in all_results or model_b not in all_results:
            continue

        results_a = all_results[model_a]
        results_b = all_results[model_b]

        print(f"\n{'-'*100}")
        print(f"{model_a} vs {model_b}")
        print(f"{'-'*100}")

        # Compute time comparison
        time_diff = 100 * (results_a['total_time'] / results_b['total_time'] - 1)
        time_verdict = "slower" if time_diff > 0 else "faster"
        print(f"  Compute Time:           {time_diff:>+7.1f}%  ({time_verdict})")

        # Throughput comparison
        throughput_a = (32.0 * estimated_total_batches) / results_a['total_time']
        throughput_b = (32.0 * estimated_total_batches) / results_b['total_time']
        throughput_diff = 100 * (throughput_a / throughput_b - 1)
        print(f"  Throughput:             {throughput_diff:>+7.1f}%  ({throughput_a:.1f} vs {throughput_b:.1f} samples/sec)")

        # Memory comparison
        peak_a = results_a['memory']['peak_allocated_mb']
        peak_b = results_b['memory']['peak_allocated_mb']
        if peak_b > 0:
            memory_diff = 100 * (peak_a / peak_b - 1)
            memory_verdict = "more" if memory_diff > 0 else "less"
            print(f"  Peak Memory:            {memory_diff:>+7.1f}%  ({memory_verdict} GPU memory)")
        else:
            print(f"  Peak Memory:                N/A")

        # Training overhead comparison
        overhead_a = results_a['memory']['overhead_mb']
        overhead_b = results_b['memory']['overhead_mb']
        if overhead_b > 0:
            overhead_diff = 100 * (overhead_a / overhead_b - 1)
            overhead_verdict = "more" if overhead_diff > 0 else "less"
            print(f"  Training Overhead:      {overhead_diff:>+7.1f}%  ({overhead_verdict} model+gradient memory)")
        else:
            print(f"  Training Overhead:          N/A")

        # Accuracy comparison
        acc_a = results_a.get('avg_accuracy', 0)
        acc_b = results_b.get('avg_accuracy', 0)
        acc_diff = acc_a - acc_b
        acc_verdict = "better" if acc_diff > 0 else "worse"
        print(f"  Accuracy:               {acc_diff:>+7.4f}  ({acc_verdict}, {acc_diff*100:+.2f} pp)")

        # Forgetting comparison (if available)
        if 'forgetting' in results_a and 'forgetting' in results_b:
            forg_a = results_a['forgetting']
            forg_b = results_b['forgetting']
            forg_diff = forg_a - forg_b
            forg_verdict = "worse" if forg_diff > 0 else "better"
            print(f"  Forgetting:             {forg_diff:>+7.4f}  ({forg_verdict})")

        # Key insights
        print(f"\n  Key Insight:")
        if model_a == 'Task-Aware MoB' and model_b == 'Gated MoE + EWC':
            if overhead_diff < 0:
                print(f"    → MoB saves {abs(overhead_diff):.1f}% memory (no gater gradients!)")
            if acc_diff > 0:
                print(f"    → MoB achieves {acc_diff*100:.1f}pp better accuracy via auction-based routing")
            if time_diff > 0:
                print(f"    → MoB is {time_diff:.1f}% slower due to bid computation overhead")
        elif model_a == 'Online MoB' and model_b == 'Gated MoE + EWC':
            if overhead_diff < 0:
                print(f"    → Online MoB saves {abs(overhead_diff):.1f}% memory (no gater, task-free)")
            if acc_diff > 0:
                print(f"    → Online MoB achieves {acc_diff*100:.1f}pp better accuracy in streaming setting")
        elif model_a == 'Online MoB' and model_b == 'Task-Aware MoB':
            shifts = results_a.get('detected_shifts', [])
            print(f"    → Online MoB detected {len(shifts)} distribution shifts")
            if acc_diff > 0:
                print(f"    → Shift detection improves accuracy by {acc_diff*100:.1f}pp")
            elif acc_diff < 0:
                print(f"    → Task boundaries help by {abs(acc_diff)*100:.1f}pp vs streaming")

    print("\n" + "="*100)
    print("NOTES:")
    print("  - Parameter counts are printed by the individual experiment scripts above")
    print("  - Memory 'overhead' = Peak memory - PyTorch baseline (includes model + gradients + buffers)")
    print("  - Throughput = Training samples processed per second")
    print("  - 'pp' = percentage points (absolute difference)")
    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description='Three-Way Benchmark: Gated MoE + EWC vs Task-Aware MoB vs Online MoB'
    )

    # Shared config
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)

    # Best hyperparameters from best_hyperparameters.md
    parser.add_argument('--lambda_ewc_moe', type=float, default=50.0,
                        help='EWC lambda for Gated MoE (default: 50.0)')
    parser.add_argument('--gater_ewc', action='store_true', default=True,
                        help='Apply EWC to gater in Gated MoE (default: True)')
    parser.add_argument('--no_gater_ewc', action='store_true',
                        help='Disable EWC on gater')

    parser.add_argument('--lambda_ewc_mob', type=float, default=10.0,
                        help='EWC lambda for Task-Aware MoB (default: 10.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha for MoB bidding (default: 0.5)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Beta for MoB bidding (default: 0.5)')

    parser.add_argument('--lambda_ewc_continual', type=float, default=40.0,
                        help='EWC lambda for Online MoB (default: 40.0)')
    parser.add_argument('--shift_threshold', type=float, default=2.0,
                        help='Shift detection threshold for Online MoB (default: 2.0)')

    # Run options
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs to average (default: 1)')
    parser.add_argument('--skip_continual', action='store_true',
                        help='Skip Online MoB benchmark')
    parser.add_argument('--save_results', action='store_true')

    args = parser.parse_args()

    # Handle gater_ewc flag
    gater_ewc = args.gater_ewc and not args.no_gater_ewc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*90)
    print("THREE-WAY RESOURCE BENCHMARK")
    print("  1. Gated MoE + EWC (learnable gater)")
    print("  2. Task-Aware MoB (auction routing)")
    print("  3. Online MoB (streaming, shift detection)")
    print("="*90)
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"\nConfiguration:")
    print(f"  Experts: {args.num_experts}")
    print(f"  Epochs per task: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # Build configs for each model
    gated_moe_config = {
        'num_experts': args.num_experts,
        'num_tasks': 5,
        'lambda_ewc': args.lambda_ewc_moe,
        'gater_ewc': gater_ewc,
        'gater_hidden_size': 256,
        'learning_rate': 0.001,
        'device': str(device),
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs,
    }

    mob_config = {
        'num_experts': args.num_experts,
        'num_tasks': 5,
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_ewc': args.lambda_ewc_mob,
        'learning_rate': 0.001,
        'forgetting_cost_scale': 1.0,
        'use_lwf': False,
        'lwf_temperature': 2.0,
        'lwf_alpha': 0.1,
        'device': str(device),
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs,
    }

    continual_config = {
        'seed': args.seed,
        'num_experts': args.num_experts,
        'num_tasks': 5,
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_ewc': args.lambda_ewc_continual,
        'learning_rate': 0.001,
        'shift_threshold': args.shift_threshold,
        'device': str(device),
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs,
    }

    all_results = {}

    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n{'='*90}")
            print(f"RUN {run+1}/{args.runs}")
            print(f"{'='*90}")

        # =================================================================
        # Benchmark Gated MoE + EWC
        # =================================================================
        set_seed(args.seed + run)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("BENCHMARKING: Gated MoE + EWC")
        print(f"{'='*60}")
        gated_results = benchmark_gated_moe(gated_moe_config, device)
        all_results['Gated MoE + EWC'] = gated_results
        print(f"  -> Time: {gated_results['total_time']:.2f}s")
        print(f"  -> Accuracy: {gated_results['avg_accuracy']:.4f}")
        print(f"  -> Peak Memory: {gated_results['memory']['peak_allocated_mb']:.1f}MB")
        print(f"  -> Model+Training Overhead: {gated_results['memory']['overhead_mb']:.1f}MB")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # =================================================================
        # Benchmark MoB
        # =================================================================
        set_seed(args.seed + run)

        print(f"\n{'='*60}")
        print("BENCHMARKING: Task-Aware MoB")
        print(f"{'='*60}")
        mob_results = benchmark_mob(mob_config, device)
        all_results['Task-Aware MoB'] = mob_results
        print(f"  -> Time: {mob_results['total_time']:.2f}s")
        print(f"  -> Accuracy: {mob_results['avg_accuracy']:.4f}")
        print(f"  -> Peak Memory: {mob_results['memory']['peak_allocated_mb']:.1f}MB")
        print(f"  -> Model+Training Overhead: {mob_results['memory']['overhead_mb']:.1f}MB")

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # =================================================================
        # Benchmark Continual MoB
        # =================================================================
        if not args.skip_continual:
            set_seed(args.seed + run)

            print(f"\n{'='*60}")
            print("BENCHMARKING: Online MoB")
            print(f"{'='*60}")
            continual_results = benchmark_continual(continual_config, device)
            all_results['Online MoB'] = continual_results
            print(f"  -> Time: {continual_results['total_time']:.2f}s")
            print(f"  -> Accuracy: {continual_results['avg_accuracy']:.4f}")
            print(f"  -> Peak Memory: {continual_results['memory']['peak_allocated_mb']:.1f}MB")
            print(f"  -> Model+Training Overhead: {continual_results['memory']['overhead_mb']:.1f}MB")

    # Compare results
    compare_results(all_results)

    # Save results
    if args.save_results:
        os.makedirs('results', exist_ok=True)

        combined_results = {
            'config': {
                'seed': args.seed,
                'num_experts': args.num_experts,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'device': str(device),
                'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
            }
        }

        for model_name, results in all_results.items():
            key = model_name.lower().replace(' ', '_').replace('+', '_')
            combined_results[key] = {
                'total_time': results['total_time'],
                'memory': results['memory'],
                'memory_before_mb': results['memory_before_mb'],
                'memory_after_mb': results['memory_after_mb'],
                'avg_accuracy': results.get('avg_accuracy', 0),
                'forgetting': results.get('forgetting', None),
            }

        filename = f"results/benchmark_comparison_seed_{args.seed}.json"
        with open(filename, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"\nResults saved to: {filename}")


if __name__ == '__main__':
    main()
