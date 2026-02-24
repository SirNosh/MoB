"""
Resource Benchmarking Orchestrator - Run All 7 Models and Compare

Runs all continual learning models sequentially with identical seed/config,
measures resource usage (time, VRAM, RAM, FLOPs, throughput, params),
and produces a unified comparison table + JSON output.

Usage:
    python tests/check\ resources/benchmark_all.py                 # Full benchmark
    python tests/check\ resources/benchmark_all.py --epochs 1      # Quick test
    python tests/check\ resources/benchmark_all.py --seed 123      # Different seed
    python tests/check\ resources/benchmark_all.py --skip pnn er   # Skip specific models
"""

import torch
import gc
import os
import sys
import json
import argparse
import time

# Add parent directory to path (for mob.*, tests.*, contibualmob.*)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Add check resources directory for resource_utils
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mob.utils import set_seed
from tests.test_baselines import create_split_mnist
from resource_utils import print_comparison_table

# Force tqdm to write to stdout so progress bars render correctly
# on Windows PowerShell (stderr gets line-buffered, breaking \r overwrites)
import functools
try:
    import tqdm as _tqdm_module
    _orig_init = _tqdm_module.tqdm.__init__
    @functools.wraps(_orig_init)
    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault('file', sys.stdout)
        _orig_init(self, *args, **kwargs)
    _tqdm_module.tqdm.__init__ = _patched_init
except ImportError:
    pass


_CHECK_RESOURCES_DIR = os.path.abspath(os.path.dirname(__file__))
_TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def clear_gpu():
    """Aggressively clear GPU VRAM between model runs for fair measurement."""
    # Unload imported model modules to free GPU tensor references
    modules_to_unload = [key for key in sys.modules if key.startswith('run_')]
    for mod_name in modules_to_unload:
        del sys.modules[mod_name]

    # Fix sys.path: model files insert tests/ at position 0 during import,
    # which shadows the check resources/ versions on re-import.
    # Remove all tests/ entries and ensure check resources/ stays at front.
    while _TESTS_DIR in sys.path:
        sys.path.remove(_TESTS_DIR)
    if sys.path[0] != _CHECK_RESOURCES_DIR:
        if _CHECK_RESOURCES_DIR in sys.path:
            sys.path.remove(_CHECK_RESOURCES_DIR)
        sys.path.insert(0, _CHECK_RESOURCES_DIR)

    # Force multiple GC passes to break reference cycles
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def run_mob_task_aware(train_tasks, test_tasks, config):
    """Run MoB Task-Aware experiment."""
    from run_mob_only import run_experiment
    mob_config = {
        'num_experts': 4,
        'num_tasks': 5,
        'alpha': 0.3549,
        'beta': 0.4151,
        'lambda_ewc': 277.54,
        'forgetting_cost_scale': 2.1314,
        'learning_rate': 0.001028,
        'device': config['device'],
        'batch_size': config['batch_size'],
        'epochs_per_task': config['epochs_per_task'],
        'use_lwf': False,
        # Optimizer reset at task END for better continual learning
        'reset_optimizer': True,
    }
    results = run_experiment(train_tasks, test_tasks, mob_config)
    return results.get('resource_metrics')


def run_mob_online(train_tasks, test_tasks, config):
    """Run Continual MoB (Online) experiment."""
    from run_continual_mob import run_continual_experiment
    mob_config = {
        'seed': config['seed'],
        'num_experts': 4,
        'num_tasks': 5,
        'alpha': 0.5966,
        'beta': 0.6606,
        'lambda_ewc': 112.16,
        'learning_rate': 0.006971,
        'device': config['device'],
        'batch_size': config['batch_size'],
        'epochs_per_task': config['epochs_per_task'],
        'shift_threshold': 4.05,
        # Optimizer reset on shift detection for better continual learning
        'reset_optimizer': True,
    }
    results = run_continual_experiment(train_tasks, test_tasks, mob_config)
    return results.get('resource_metrics')


def run_gated_moe(train_tasks, test_tasks, config):
    """Run Gated MoE + EWC experiment."""
    from run_gated_moe_ewc import run_experiment
    moe_config = {
        'num_experts': 4,
        'num_tasks': 5,
        'lambda_ewc': 5924.15,
        'learning_rate': 0.000219,
        'gater_ewc': False,
        'gater_hidden_size': 512,
        'device': config['device'],
        'batch_size': config['batch_size'],
        'epochs_per_task': config['epochs_per_task'],
    }
    results = run_experiment(train_tasks, test_tasks, moe_config)
    return results.get('resource_metrics')


def run_monolithic(train_tasks, test_tasks, config):
    """Run Monolithic EWC experiment."""
    from run_monolithic_ewc import run_experiment
    mono_config = {
        'num_tasks': 5,
        'num_experts': 4,
        'width_multiplier': 2,
        'lambda_ewc': 1791.93,
        'learning_rate': 0.000707,
        'device': config['device'],
        'batch_size': config['batch_size'],
        'epochs_per_task': config['epochs_per_task'],
    }
    results = run_experiment(train_tasks, test_tasks, mono_config)
    return results.get('resource_metrics')


def run_agem(train_tasks, test_tasks, config):
    """Run A-GEM experiment."""
    from run_agem_baseline import run_experiment
    agem_config = {
        'num_tasks': 5,
        'width_multiplier': 2,
        'memory_size': 1024,
        'memory_batch_size': 64,
        'learning_rate': 0.000102,
        'device': config['device'],
        'batch_size': config['batch_size'],
        'epochs_per_task': config['epochs_per_task'],
    }
    results = run_experiment(train_tasks, test_tasks, agem_config)
    return results.get('resource_metrics')


def run_er(train_tasks, test_tasks, config):
    """Run Experience Replay experiment."""
    from run_er_baseline import run_experiment
    er_config = {
        'num_tasks': 5,
        'width_multiplier': 2,
        'memory_size': 2048,
        'replay_batch_size': 16,
        'replay_weight': 1.7043,
        'learning_rate': 0.000552,
        'device': config['device'],
        'batch_size': config['batch_size'],
        'epochs_per_task': config['epochs_per_task'],
    }
    results = run_experiment(train_tasks, test_tasks, er_config)
    return results.get('resource_metrics')


def run_pnn(train_tasks, test_tasks, config):
    """Run Progressive Neural Networks experiment."""
    from run_pnn_baseline import run_experiment
    pnn_config = {
        'num_tasks': 5,
        'max_columns': -1,
        'learning_rate': 0.000494,
        'device': config['device'],
        'batch_size': config['batch_size'],
        'epochs_per_task': config['epochs_per_task'],
    }
    results = run_experiment(train_tasks, test_tasks, pnn_config)
    return results.get('resource_metrics')


# Registry of all models
MODEL_REGISTRY = {
    'mob_ta': ('MoB-TaskAware', run_mob_task_aware),
    'mob_online': ('MoB-Online', run_mob_online),
    'gated_moe': ('GatedMoE+EWC', run_gated_moe),
    'monolithic': ('Monolithic+EWC', run_monolithic),
    'agem': ('A-GEM', run_agem),
    'er': ('ER', run_er),
    'pnn': ('PNN', run_pnn),
}


def main():
    parser = argparse.ArgumentParser(description='Benchmark all continual learning models')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=4, help='Epochs per task')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip', nargs='*', default=[],
                        help=f'Models to skip. Options: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('--only', nargs='*', default=[],
                        help='Only run these models')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                        help='Output JSON path')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = {
        'seed': args.seed,
        'device': device,
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs,
    }

    print("=" * 80)
    print("RESOURCE BENCHMARKING - ALL MODELS")
    print("=" * 80)
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print(f"  Epochs per task: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # Determine which models to run
    if args.only:
        models_to_run = {k: v for k, v in MODEL_REGISTRY.items() if k in args.only}
    else:
        models_to_run = {k: v for k, v in MODEL_REGISTRY.items() if k not in args.skip}

    print(f"  Models: {list(models_to_run.keys())}")
    print("=" * 80)

    all_results = []

    for model_key, (model_name, run_fn) in models_to_run.items():
        print(f"\n{'#'*80}")
        print(f"# RUNNING: {model_name}")
        print(f"{'#'*80}")

        # Set seed for reproducibility
        set_seed(args.seed)

        # Fresh datasets for each model
        print("Creating Split-MNIST datasets...")
        train_tasks = create_split_mnist(5, train=True, batch_size=args.batch_size)
        test_tasks = create_split_mnist(5, train=False, batch_size=args.batch_size)

        # Clear GPU between runs
        clear_gpu()

        start = time.perf_counter()
        try:
            resource_metrics = run_fn(train_tasks, test_tasks, config)
            elapsed = time.perf_counter() - start

            if resource_metrics:
                all_results.append(resource_metrics)
                print(f"\n  [{model_name}] Completed in {elapsed:.1f}s")
            else:
                print(f"\n  [{model_name}] WARNING: No resource metrics returned")
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"\n  [{model_name}] FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            # Persist error to results so failures aren't lost
            all_results.append({
                'model_name': model_name,
                'error': str(e),
                'status': 'FAILED',
            })

        # Delete references to train/test data to free memory
        del train_tasks, test_tasks

        # Aggressively clear GPU VRAM and unload model modules
        clear_gpu()

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    if all_results:
        print_comparison_table(all_results)

        # Save to JSON
        output_path = args.output
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        # Make JSON-serializable
        serializable = []
        for r in all_results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    clean[k] = v
                elif isinstance(v, list):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            serializable.append(clean)

        with open(output_path, 'w') as f:
            json.dump({
                'seed': args.seed,
                'device': device,
                'epochs_per_task': args.epochs,
                'batch_size': args.batch_size,
                'results': serializable
            }, f, indent=2)

        print(f"Results saved to: {output_path}")
    else:
        print("\nNo results collected.")


if __name__ == '__main__':
    main()
