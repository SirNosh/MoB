"""
Hyperparameter Search for MoB Models

Searches over hyperparameter ranges to find optimal configurations.
Calls the original experiment files directly to ensure consistency.

Usage:
    python tests/hyperparameter_search.py                    # Full search
    python tests/hyperparameter_search.py --model mob        # Search MoB only
    python tests/hyperparameter_search.py --model gated_moe  # Search Gated MoE only
    python tests/hyperparameter_search.py --model continual  # Search Continual MoB only
    python tests/hyperparameter_search.py --quick            # Quick search (fewer combinations)
"""

import os
import sys
import json
import argparse
import itertools
from datetime import datetime
from typing import Dict, List, Any

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob.utils import set_seed
from tests.test_baselines import create_split_mnist

# =============================================================================
# IMPORT RUN_EXPERIMENT FUNCTIONS FROM ORIGINAL FILES
# =============================================================================
from tests.run_mob_only import run_experiment as run_mob_experiment
from tests.run_gated_moe_ewc import run_experiment as run_gated_moe_experiment
from tests.run_continual_mob import run_continual_experiment


# =============================================================================
# HYPERPARAMETER RANGES
# =============================================================================

def get_search_space(model_type: str, quick: bool = False) -> Dict[str, List[Any]]:
    """
    Get hyperparameter search space for each model type.

    Args:
        model_type: 'mob', 'gated_moe', or 'continual'
        quick: If True, use smaller search space for faster results
    """

    if model_type == 'gated_moe':
        if quick:
            return {
                'lambda_ewc': [10.0, 50.0, 100.0],
                'gater_ewc': [True, False],
                'learning_rate': [0.001],
                'gater_hidden_size': [256],
            }
        else:
            return {
                'lambda_ewc': [0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
                'gater_ewc': [True, False],
                'learning_rate': [0.0005, 0.001, 0.002],
                'gater_hidden_size': [128, 256, 512],
            }

    elif model_type == 'mob':
        if quick:
            return {
                'lambda_ewc': [5.0, 10.0, 25.0],
                'alpha': [0.5],
                'beta': [0.5],
                'learning_rate': [0.001],
                'forgetting_cost_scale': [1.0],
            }
        else:
            return {
                'lambda_ewc': [0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0],
                'alpha': [0.3, 0.5, 0.7],
                'beta': [0.3, 0.5, 0.7],
                'learning_rate': [0.0005, 0.001, 0.002],
                'forgetting_cost_scale': [0.5, 1.0, 2.0],
            }

    elif model_type == 'continual':
        if quick:
            return {
                'lambda_ewc': [20.0, 40.0, 60.0],
                'alpha': [0.5],
                'beta': [0.5],
                'shift_threshold': [2.0],
                'learning_rate': [0.001],
            }
        else:
            return {
                'lambda_ewc': [0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 125.0, 150.0],
                'alpha': [0.3, 0.5, 0.7],
                'beta': [0.3, 0.5, 0.7],
                'shift_threshold': [1.0, 1.5, 2.0, 2.5, 3.0],
                'learning_rate': [0.0005, 0.001, 0.002],
            }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_configs(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters."""
    keys = list(search_space.keys())
    values = list(search_space.values())

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


# =============================================================================
# EXPERIMENT RUNNERS (Call original files)
# =============================================================================

def run_gated_moe_with_config(config: Dict, base_config: Dict) -> Dict:
    """Run Gated MoE experiment with given hyperparameters."""

    # Merge base config with search config
    full_config = {
        'num_experts': base_config['num_experts'],
        'num_tasks': base_config['num_tasks'],
        'device': base_config['device'],
        'batch_size': base_config['batch_size'],
        'epochs_per_task': base_config['epochs_per_task'],
        'lambda_ewc': config['lambda_ewc'],
        'gater_ewc': config['gater_ewc'],
        'learning_rate': config['learning_rate'],
        'gater_hidden_size': config['gater_hidden_size'],
    }

    # Create datasets
    train_tasks = create_split_mnist(
        full_config['num_tasks'],
        train=True,
        batch_size=full_config['batch_size']
    )
    test_tasks = create_split_mnist(
        full_config['num_tasks'],
        train=False,
        batch_size=full_config['batch_size']
    )

    # Run experiment using original function
    results = run_gated_moe_experiment(train_tasks, test_tasks, full_config)

    return {
        'config': config,
        'avg_accuracy': results['avg_accuracy'],
        'forgetting': results['forgetting'],
        'final_accuracies': results['final_accuracies'],
        'task_accuracies': results['task_accuracies'],
    }


def run_mob_with_config(config: Dict, base_config: Dict) -> Dict:
    """Run MoB experiment with given hyperparameters."""

    # Merge base config with search config
    full_config = {
        'num_experts': base_config['num_experts'],
        'num_tasks': base_config['num_tasks'],
        'device': base_config['device'],
        'batch_size': base_config['batch_size'],
        'epochs_per_task': base_config['epochs_per_task'],
        'lambda_ewc': config['lambda_ewc'],
        'alpha': config['alpha'],
        'beta': config['beta'],
        'learning_rate': config['learning_rate'],
        'forgetting_cost_scale': config['forgetting_cost_scale'],
        # LwF disabled for search (can enable separately)
        'use_lwf': False,
        'lwf_temperature': 2.0,
        'lwf_alpha': 0.1,
    }

    # Create datasets
    train_tasks = create_split_mnist(
        full_config['num_tasks'],
        train=True,
        batch_size=full_config['batch_size']
    )
    test_tasks = create_split_mnist(
        full_config['num_tasks'],
        train=False,
        batch_size=full_config['batch_size']
    )

    # Run experiment using original function
    results = run_mob_experiment(train_tasks, test_tasks, full_config)

    return {
        'config': config,
        'avg_accuracy': results['avg_accuracy'],
        'forgetting': results['forgetting'],
        'final_accuracies': results['final_accuracies'],
        'task_accuracies': results['task_accuracies'],
    }


def run_continual_with_config(config: Dict, base_config: Dict) -> Dict:
    """Run Continual MoB experiment with given hyperparameters."""

    # Merge base config with search config
    full_config = {
        'seed': base_config['seed'],
        'num_experts': base_config['num_experts'],
        'num_tasks': base_config['num_tasks'],
        'device': base_config['device'],
        'batch_size': base_config['batch_size'],
        'epochs_per_task': base_config['epochs_per_task'],
        'lambda_ewc': config['lambda_ewc'],
        'alpha': config['alpha'],
        'beta': config['beta'],
        'learning_rate': config['learning_rate'],
        'shift_threshold': config['shift_threshold'],
    }

    # Create datasets
    train_tasks = create_split_mnist(
        full_config['num_tasks'],
        train=True,
        batch_size=full_config['batch_size']
    )
    test_tasks = create_split_mnist(
        full_config['num_tasks'],
        train=False,
        batch_size=full_config['batch_size']
    )

    # Run experiment using original function
    results = run_continual_experiment(train_tasks, test_tasks, full_config)

    return {
        'config': config,
        'avg_accuracy': results['avg_accuracy'],
        'detected_shifts': results.get('detected_shifts', []),
    }


# =============================================================================
# SEARCH ORCHESTRATION
# =============================================================================

def run_hyperparameter_search(
    model_type: str,
    base_config: Dict,
    quick: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """
    Run hyperparameter search for a model type.

    Args:
        model_type: 'mob', 'gated_moe', or 'continual'
        base_config: Base configuration (experts, tasks, etc.)
        quick: Use smaller search space
        verbose: Print progress

    Returns:
        List of results sorted by accuracy (best first)
    """

    # Get search space and generate configs
    search_space = get_search_space(model_type, quick=quick)
    configs = generate_configs(search_space)

    if verbose:
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER SEARCH: {model_type.upper()}")
        print(f"{'='*70}")
        print(f"Search space: {search_space}")
        print(f"Total configurations: {len(configs)}")
        print(f"{'='*70}\n")

    # Select runner function
    if model_type == 'gated_moe':
        runner = run_gated_moe_with_config
    elif model_type == 'mob':
        runner = run_mob_with_config
    elif model_type == 'continual':
        runner = run_continual_with_config
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Run all configurations
    results = []
    for i, config in enumerate(configs):
        if verbose:
            print(f"\n[{i+1}/{len(configs)}] Testing: {config}")

        try:
            set_seed(base_config['seed'])
            result = runner(config, base_config)
            results.append(result)

            if verbose:
                print(f"  -> Accuracy: {result['avg_accuracy']:.4f}")
                if 'forgetting' in result:
                    print(f"  -> Forgetting: {result['forgetting']:.4f}")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({
                'config': config,
                'avg_accuracy': 0.0,
                'error': str(e)
            })

    # Sort by accuracy (descending)
    results.sort(key=lambda x: x.get('avg_accuracy', 0), reverse=True)

    return results


def print_search_results(model_type: str, results: List[Dict], top_k: int = 5):
    """Print top-k results from hyperparameter search."""

    print(f"\n{'='*70}")
    print(f"TOP {top_k} RESULTS: {model_type.upper()}")
    print(f"{'='*70}")

    for i, result in enumerate(results[:top_k]):
        print(f"\n#{i+1} - Accuracy: {result['avg_accuracy']:.4f}")
        if 'forgetting' in result:
            print(f"     Forgetting: {result['forgetting']:.4f}")
        print(f"     Config: {result['config']}")

    print(f"\n{'='*70}")
    print("BEST CONFIGURATION:")
    print(f"{'='*70}")
    best = results[0]
    print(f"Accuracy: {best['avg_accuracy']:.4f}")
    for key, value in best['config'].items():
        print(f"  {key}: {value}")


def save_search_results(model_type: str, results: List[Dict], seed: int):
    """Save search results to JSON file."""
    os.makedirs('results', exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/hyperparam_search_{model_type}_seed{seed}_{timestamp}.json"

    output = {
        'model_type': model_type,
        'seed': seed,
        'timestamp': timestamp,
        'num_configs_tested': len(results),
        'best_config': results[0]['config'] if results else None,
        'best_accuracy': results[0]['avg_accuracy'] if results else None,
        'all_results': results
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Search for MoB Models')

    # Model selection
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'mob', 'gated_moe', 'continual'],
                        help='Which model to search (default: all)')

    # Base configuration
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)

    # Search options
    parser.add_argument('--quick', action='store_true',
                        help='Use smaller search space for faster results')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top results to display')
    parser.add_argument('--save', action='store_true',
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Base configuration (fixed across search)
    base_config = {
        'seed': args.seed,
        'num_experts': args.num_experts,
        'num_tasks': 5,
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs,
        'device': device,
    }

    print("="*70)
    print("HYPERPARAMETER SEARCH")
    print("="*70)
    print(f"\nDevice: {device}")
    print(f"Base Configuration:")
    for k, v in base_config.items():
        print(f"  {k}: {v}")
    print(f"Quick mode: {args.quick}")

    # Determine which models to search
    if args.model == 'all':
        models_to_search = ['gated_moe', 'mob', 'continual']
    else:
        models_to_search = [args.model]

    # Run searches
    all_results = {}

    for model_type in models_to_search:
        results = run_hyperparameter_search(
            model_type=model_type,
            base_config=base_config,
            quick=args.quick,
            verbose=True
        )

        all_results[model_type] = results
        print_search_results(model_type, results, top_k=args.top_k)

        if args.save:
            save_search_results(model_type, results, args.seed)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - BEST CONFIGURATIONS")
    print("="*70)

    for model_type, results in all_results.items():
        if results:
            best = results[0]
            print(f"\n{model_type.upper()}:")
            print(f"  Accuracy: {best['avg_accuracy']:.4f}")
            print(f"  Config: {best['config']}")


if __name__ == '__main__':
    main()
