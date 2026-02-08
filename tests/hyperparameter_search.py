"""
Hyperparameter Search for MoB Models using Bayesian Optimization (Optuna)

This script performs hyperparameter optimization using:
1. TPE (Tree-structured Parzen Estimator) for intelligent sampling
2. Multi-seed evaluation (5 seeds) for robust hyperparameter selection
3. Early pruning to skip bad configurations after first seed(s)

The approach ensures:
- Hyperparameters are validated across multiple seeds (not overfitted to one)
- Computational efficiency via early pruning of poor configurations
- Best results with confidence intervals across seeds

Usage:
    python tests/hyperparameter_search.py                      # Full search (all models)
    python tests/hyperparameter_search.py --model mob          # Search MoB only
    python tests/hyperparameter_search.py --model monolithic   # Search Monolithic EWC
    python tests/hyperparameter_search.py --model er           # Search Experience Replay
    python tests/hyperparameter_search.py --n_trials 50        # Number of Optuna trials
    python tests/hyperparameter_search.py --quick              # Quick mode (fewer trials, 2 seeds)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

# Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import PercentilePruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("WARNING: Optuna not installed. Install with: pip install optuna")
    print("Falling back to grid search mode.")

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
from tests.run_agem_baseline import run_experiment as run_agem_experiment
from tests.run_monolithic_ewc import run_experiment as run_monolithic_experiment
from tests.run_er_baseline import run_experiment as run_er_experiment
from tests.run_pnn_baseline import run_experiment as run_pnn_experiment


# =============================================================================
# CONFIGURATION
# =============================================================================

# Seeds to use for multi-seed evaluation
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
QUICK_SEEDS = [42, 123]

# Default number of Optuna trials per model
DEFAULT_N_TRIALS = 100
QUICK_N_TRIALS = 20

# Pruning threshold: prune if first seed accuracy is below this percentile
PRUNE_PERCENTILE = 25  # Bottom 25% get pruned after first seed


# =============================================================================
# HYPERPARAMETER SEARCH SPACES (for Optuna)
# =============================================================================

def suggest_mob_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """Suggest hyperparameters for MoB using Optuna."""
    return {
        'lambda_ewc': trial.suggest_float('lambda_ewc', 0.0, 10000.0),
        'alpha': trial.suggest_float('alpha', 0.2, 0.8),
        'beta': trial.suggest_float('beta', 0.2, 0.8),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'forgetting_cost_scale': trial.suggest_float('forgetting_cost_scale', 0.1, 5.0),
    }


def suggest_gated_moe_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """Suggest hyperparameters for Gated MoE using Optuna."""
    return {
        'lambda_ewc': trial.suggest_float('lambda_ewc', 0.0, 10000.0),
        'gater_ewc': trial.suggest_categorical('gater_ewc', [True, False]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'gater_hidden_size': trial.suggest_categorical('gater_hidden_size', [128, 256, 512]),
    }


def suggest_continual_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """Suggest hyperparameters for Continual MoB using Optuna."""
    return {
        'lambda_ewc': trial.suggest_float('lambda_ewc', 0.0, 10000.0),
        'alpha': trial.suggest_float('alpha', 0.2, 0.8),
        'beta': trial.suggest_float('beta', 0.2, 0.8),
        'shift_threshold': trial.suggest_float('shift_threshold', 0.5, 5.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
    }


def suggest_agem_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """Suggest hyperparameters for A-GEM using Optuna."""
    return {
        'memory_size': trial.suggest_categorical('memory_size', [128, 256, 512, 1024, 2048]),
        'memory_batch_size': trial.suggest_categorical('memory_batch_size', [16, 32, 64]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'width_multiplier': 2,  # Fixed to match MoB params
    }


def suggest_monolithic_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """
    Suggest hyperparameters for Monolithic EWC using Optuna.

    Note: Based on user feedback:
    - lambda_ewc=100 was very plastic (Task 1: 0%, Task 5: 98%)
    - lambda_ewc=10000 was very stable (Task 1: 87%, Task 5: 23%)
    So we search in a wide range from 10 to 50000.
    """
    return {
        'lambda_ewc': trial.suggest_float('lambda_ewc', 10.0, 50000.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'width_multiplier': 2,  # Fixed to match MoB params (~1.7M)
    }


def suggest_er_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """Suggest hyperparameters for Experience Replay using Optuna."""
    return {
        'memory_size': trial.suggest_categorical('memory_size', [128, 256, 512, 1024, 2048]),
        'replay_batch_size': trial.suggest_categorical('replay_batch_size', [16, 32, 64]),
        'replay_weight': trial.suggest_float('replay_weight', 0.1, 2.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'width_multiplier': 2,  # Fixed to match MoB params
    }


def suggest_pnn_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """
    Suggest hyperparameters for Progressive Neural Networks using Optuna.

    PNN has few hyperparameters since it uses freezing instead of regularization.
    Main tunable is max_columns (capacity limit).
    """
    return {
        'max_columns': trial.suggest_categorical('max_columns', [4, 5, -1]),  # 4=match MoB, -1=unlimited
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
    }


# =============================================================================
# EXPERIMENT RUNNERS (Call original files)
# =============================================================================

def run_single_experiment(
    model_type: str,
    config: Dict[str, Any],
    base_config: Dict[str, Any],
    seed: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single experiment with given config and seed.

    Returns dict with 'avg_accuracy', 'forgetting', and other metrics.
    """
    set_seed(seed)

    # Create datasets with this seed
    train_tasks = create_split_mnist(
        base_config['num_tasks'],
        train=True,
        batch_size=base_config['batch_size']
    )
    test_tasks = create_split_mnist(
        base_config['num_tasks'],
        train=False,
        batch_size=base_config['batch_size']
    )

    # Build full config
    full_config = {
        **base_config,
        **config,
        'seed': seed,
    }

    # Suppress output if not verbose
    if not verbose:
        import io
        import contextlib
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            results = _run_experiment_by_type(model_type, train_tasks, test_tasks, full_config)
    else:
        results = _run_experiment_by_type(model_type, train_tasks, test_tasks, full_config)

    return results


def _run_experiment_by_type(model_type: str, train_tasks, test_tasks, config) -> Dict:
    """Dispatch to appropriate experiment runner."""

    if model_type == 'mob':
        # Add MoB-specific defaults
        config.setdefault('use_lwf', False)
        config.setdefault('lwf_temperature', 2.0)
        config.setdefault('lwf_alpha', 0.1)
        return run_mob_experiment(train_tasks, test_tasks, config)

    elif model_type == 'gated_moe':
        return run_gated_moe_experiment(train_tasks, test_tasks, config)

    elif model_type == 'continual':
        return run_continual_experiment(train_tasks, test_tasks, config)

    elif model_type == 'agem':
        return run_agem_experiment(train_tasks, test_tasks, config)

    elif model_type == 'monolithic':
        return run_monolithic_experiment(train_tasks, test_tasks, config)

    elif model_type == 'er':
        return run_er_experiment(train_tasks, test_tasks, config)

    elif model_type == 'pnn':
        results = run_pnn_experiment(train_tasks, test_tasks, config)
        # PNN returns avg_accuracy_agnostic for fair comparison (no task oracle)
        results['avg_accuracy'] = results.get('avg_accuracy_agnostic', results.get('avg_accuracy_oracle', 0))
        return results

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_multi_seed_experiment(
    model_type: str,
    config: Dict[str, Any],
    base_config: Dict[str, Any],
    seeds: List[int],
    trial: Optional['optuna.Trial'] = None,
    verbose: bool = False
) -> Tuple[float, float, List[float]]:
    """
    Run experiment across multiple seeds with optional pruning.

    Returns:
        mean_accuracy: Average accuracy across seeds
        std_accuracy: Standard deviation across seeds
        all_accuracies: List of accuracies for each seed
    """
    accuracies = []

    for i, seed in enumerate(seeds):
        try:
            results = run_single_experiment(
                model_type, config, base_config, seed, verbose=verbose
            )
            acc = results.get('avg_accuracy', 0.0)
            accuracies.append(acc)

            # Report intermediate result for pruning
            if trial is not None:
                trial.report(np.mean(accuracies), i)

                # Check if should prune (after first seed)
                if i == 0 and trial.should_prune():
                    raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise
        except Exception as e:
            if verbose:
                print(f"  Seed {seed} failed: {e}")
            accuracies.append(0.0)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    return mean_acc, std_acc, accuracies


# =============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =============================================================================

def create_objective(
    model_type: str,
    base_config: Dict[str, Any],
    seeds: List[int],
    verbose: bool = False
):
    """Create Optuna objective function for a model type."""

    # Get appropriate parameter suggester
    param_suggesters = {
        'mob': suggest_mob_params,
        'gated_moe': suggest_gated_moe_params,
        'continual': suggest_continual_params,
        'agem': suggest_agem_params,
        'monolithic': suggest_monolithic_params,
        'er': suggest_er_params,
        'pnn': suggest_pnn_params,
    }

    suggest_params = param_suggesters[model_type]

    def objective(trial: 'optuna.Trial') -> float:
        """Objective function for Optuna optimization."""

        # Suggest hyperparameters
        config = suggest_params(trial)

        if verbose:
            print(f"\n[Trial {trial.number}] Testing: {config}")

        # Run multi-seed experiment
        mean_acc, std_acc, all_accs = run_multi_seed_experiment(
            model_type=model_type,
            config=config,
            base_config=base_config,
            seeds=seeds,
            trial=trial,
            verbose=verbose
        )

        if verbose:
            print(f"  -> Mean: {mean_acc:.4f} +/- {std_acc:.4f}")
            print(f"  -> All: {[f'{a:.4f}' for a in all_accs]}")

        # Store additional info
        trial.set_user_attr('std_accuracy', std_acc)
        trial.set_user_attr('all_accuracies', all_accs)
        trial.set_user_attr('config', config)

        return mean_acc

    return objective


# =============================================================================
# MAIN SEARCH FUNCTION
# =============================================================================

def run_optuna_search(
    model_type: str,
    base_config: Dict[str, Any],
    n_trials: int = DEFAULT_N_TRIALS,
    seeds: List[int] = None,
    verbose: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run Bayesian hyperparameter optimization using Optuna.

    Args:
        model_type: Type of model to optimize
        base_config: Base configuration
        n_trials: Number of Optuna trials
        seeds: List of seeds for multi-seed evaluation
        verbose: Print progress
        save_results: Save results to file

    Returns:
        Dict with best params, best score, and study results
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available. Install with: pip install optuna")

    if seeds is None:
        seeds = DEFAULT_SEEDS

    print(f"\n{'='*70}")
    print(f"OPTUNA HYPERPARAMETER SEARCH: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"Trials: {n_trials}")
    print(f"Seeds per trial: {len(seeds)} ({seeds})")
    print(f"Sampler: TPE (Tree-structured Parzen Estimator)")
    print(f"Pruner: Percentile ({PRUNE_PERCENTILE}th percentile)")
    print(f"{'='*70}\n")

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=PercentilePruner(
            percentile=PRUNE_PERCENTILE,
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=0,    # Can prune after first seed
        )
    )

    # Create objective
    objective = create_objective(model_type, base_config, seeds, verbose=verbose)

    # Suppress Optuna logging if not verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get results
    best_trial = study.best_trial

    print(f"\n{'='*70}")
    print(f"BEST RESULT: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"Mean Accuracy: {best_trial.value:.4f} +/- {best_trial.user_attrs.get('std_accuracy', 0):.4f}")
    print(f"All Seeds: {best_trial.user_attrs.get('all_accuracies', [])}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Compile results
    results = {
        'model_type': model_type,
        'best_accuracy': best_trial.value,
        'best_std': best_trial.user_attrs.get('std_accuracy', 0),
        'best_all_accuracies': best_trial.user_attrs.get('all_accuracies', []),
        'best_params': best_trial.params,
        'n_trials': n_trials,
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'seeds': seeds,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
                'std': t.user_attrs.get('std_accuracy'),
                'all_accs': t.user_attrs.get('all_accuracies'),
            }
            for t in study.trials
        ]
    }

    # Save results
    if save_results:
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/optuna_search_{model_type}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {filename}")

    return results


def run_all_searches(
    model_types: List[str],
    base_config: Dict[str, Any],
    n_trials: int,
    seeds: List[int],
    verbose: bool = True,
    save_results: bool = True
) -> Dict[str, Dict]:
    """Run hyperparameter search for multiple model types."""

    all_results = {}

    for model_type in model_types:
        try:
            results = run_optuna_search(
                model_type=model_type,
                base_config=base_config,
                n_trials=n_trials,
                seeds=seeds,
                verbose=verbose,
                save_results=save_results
            )
            all_results[model_type] = results
        except Exception as e:
            print(f"ERROR running {model_type}: {e}")
            all_results[model_type] = {'error': str(e)}

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL MODELS")
    print("="*70)

    for model_type, results in all_results.items():
        if 'error' in results:
            print(f"\n{model_type.upper()}: FAILED - {results['error']}")
        else:
            print(f"\n{model_type.upper()}:")
            print(f"  Best Accuracy: {results['best_accuracy']:.4f} +/- {results['best_std']:.4f}")
            print(f"  Trials: {results['n_trials']} ({results['n_pruned']} pruned)")
            print(f"  Best Params: {results['best_params']}")

    return all_results


# =============================================================================
# FALLBACK: GRID SEARCH (if Optuna not available)
# =============================================================================

def get_grid_search_space(model_type: str, quick: bool = False) -> Dict[str, List[Any]]:
    """Get grid search space when Optuna is not available."""

    if model_type == 'mob':
        if quick:
            return {
                'lambda_ewc': [5.0, 25.0, 100.0],
                'alpha': [0.5],
                'beta': [0.5],
                'learning_rate': [0.001],
                'forgetting_cost_scale': [1.0],
            }
        return {
            'lambda_ewc': [0.0, 10.0, 50.0, 100.0, 200.0],
            'alpha': [0.3, 0.5, 0.7],
            'beta': [0.3, 0.5, 0.7],
            'learning_rate': [0.0005, 0.001],
            'forgetting_cost_scale': [0.5, 1.0, 2.0],
        }

    elif model_type == 'gated_moe':
        if quick:
            return {
                'lambda_ewc': [10.0, 50.0, 100.0],
                'gater_ewc': [True],
                'learning_rate': [0.001],
                'gater_hidden_size': [256],
            }
        return {
            'lambda_ewc': [0.0, 25.0, 50.0, 100.0, 200.0],
            'gater_ewc': [True, False],
            'learning_rate': [0.0005, 0.001],
            'gater_hidden_size': [128, 256, 512],
        }

    elif model_type == 'continual':
        if quick:
            return {
                'lambda_ewc': [20.0, 40.0, 80.0],
                'alpha': [0.5],
                'beta': [0.5],
                'shift_threshold': [2.0],
                'learning_rate': [0.001],
            }
        return {
            'lambda_ewc': [10.0, 40.0, 80.0, 150.0],
            'alpha': [0.3, 0.5, 0.7],
            'beta': [0.3, 0.5, 0.7],
            'shift_threshold': [1.0, 2.0, 3.0],
            'learning_rate': [0.0005, 0.001],
        }

    elif model_type == 'agem':
        if quick:
            return {
                'memory_size': [256, 512],
                'memory_batch_size': [32],
                'learning_rate': [0.001],
                'width_multiplier': [2],
            }
        return {
            'memory_size': [256, 512, 1024, 2048],
            'memory_batch_size': [32, 64],
            'learning_rate': [0.0005, 0.001],
            'width_multiplier': [2],
        }

    elif model_type == 'monolithic':
        if quick:
            return {
                'lambda_ewc': [100.0, 1000.0, 10000.0],
                'learning_rate': [0.001],
                'width_multiplier': [2],
            }
        return {
            # Wide range based on user feedback (100=plastic, 10000=stable)
            'lambda_ewc': [50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 25000.0],
            'learning_rate': [0.0005, 0.001, 0.002],
            'width_multiplier': [2],
        }

    elif model_type == 'er':
        if quick:
            return {
                'memory_size': [256, 512],
                'replay_batch_size': [32],
                'replay_weight': [1.0],
                'learning_rate': [0.001],
                'width_multiplier': [2],
            }
        return {
            'memory_size': [256, 512, 1024, 2048],
            'replay_batch_size': [32, 64],
            'replay_weight': [0.5, 1.0, 2.0],
            'learning_rate': [0.0005, 0.001],
            'width_multiplier': [2],
        }

    elif model_type == 'pnn':
        if quick:
            return {
                'max_columns': [4],
                'learning_rate': [0.001],
            }
        return {
            'max_columns': [4, 5, -1],  # 4=match MoB, 5=one per task, -1=unlimited
            'learning_rate': [0.0005, 0.001, 0.002],
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_grid_search(
    model_type: str,
    base_config: Dict[str, Any],
    seeds: List[int],
    quick: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run grid search when Optuna is not available.
    Evaluates all combinations across all seeds.
    """
    import itertools

    search_space = get_grid_search_space(model_type, quick=quick)

    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    print(f"\n{'='*70}")
    print(f"GRID SEARCH: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"Configurations: {len(configs)}")
    print(f"Seeds per config: {len(seeds)}")
    print(f"Total experiments: {len(configs) * len(seeds)}")
    print(f"{'='*70}\n")

    results = []

    for i, config in enumerate(configs):
        if verbose:
            print(f"\n[{i+1}/{len(configs)}] Testing: {config}")

        try:
            mean_acc, std_acc, all_accs = run_multi_seed_experiment(
                model_type=model_type,
                config=config,
                base_config=base_config,
                seeds=seeds,
                trial=None,
                verbose=False
            )

            results.append({
                'config': config,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'all_accuracies': all_accs,
            })

            if verbose:
                print(f"  -> Mean: {mean_acc:.4f} +/- {std_acc:.4f}")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({
                'config': config,
                'mean_accuracy': 0.0,
                'error': str(e)
            })

    # Sort by mean accuracy
    results.sort(key=lambda x: x.get('mean_accuracy', 0), reverse=True)

    # Print best results
    print(f"\n{'='*70}")
    print(f"TOP 5 RESULTS: {model_type.upper()}")
    print(f"{'='*70}")

    for i, r in enumerate(results[:5]):
        print(f"\n#{i+1} - Mean: {r['mean_accuracy']:.4f} +/- {r.get('std_accuracy', 0):.4f}")
        print(f"     Config: {r['config']}")
        if 'all_accuracies' in r:
            print(f"     Seeds: {[f'{a:.4f}' for a in r['all_accuracies']]}")

    return {
        'model_type': model_type,
        'best_accuracy': results[0]['mean_accuracy'] if results else 0,
        'best_std': results[0].get('std_accuracy', 0) if results else 0,
        'best_params': results[0]['config'] if results else {},
        'all_results': results,
        'seeds': seeds,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter Search for MoB Models using Optuna'
    )

    # Model selection
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'mob', 'gated_moe', 'continual', 'agem', 'monolithic', 'er', 'pnn'],
                        help='Which model to search (default: all)')

    # Search configuration
    parser.add_argument('--n_trials', type=int, default=None,
                        help=f'Number of Optuna trials (default: {DEFAULT_N_TRIALS})')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer trials, 2 seeds')

    # Base configuration
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)

    # Output options
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save results to file')
    parser.add_argument('--grid', action='store_true',
                        help='Use grid search instead of Optuna')

    args = parser.parse_args()

    # Determine seeds and trials
    if args.quick:
        seeds = QUICK_SEEDS
        n_trials = args.n_trials or QUICK_N_TRIALS
    else:
        seeds = DEFAULT_SEEDS
        n_trials = args.n_trials or DEFAULT_N_TRIALS

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Base configuration
    base_config = {
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
    print(f"Mode: {'Optuna (Bayesian)' if OPTUNA_AVAILABLE and not args.grid else 'Grid Search'}")
    print(f"Seeds: {seeds}")
    if OPTUNA_AVAILABLE and not args.grid:
        print(f"Trials per model: {n_trials}")
    print(f"\nBase Configuration:")
    for k, v in base_config.items():
        print(f"  {k}: {v}")

    # Determine which models to search
    if args.model == 'all':
        model_types = ['mob', 'gated_moe', 'continual', 'agem', 'monolithic', 'er', 'pnn']
    else:
        model_types = [args.model]

    # Run searches
    if OPTUNA_AVAILABLE and not args.grid:
        all_results = run_all_searches(
            model_types=model_types,
            base_config=base_config,
            n_trials=n_trials,
            seeds=seeds,
            verbose=args.verbose,
            save_results=not args.no_save
        )
    else:
        # Fallback to grid search
        all_results = {}
        for model_type in model_types:
            results = run_grid_search(
                model_type=model_type,
                base_config=base_config,
                seeds=seeds,
                quick=args.quick,
                verbose=args.verbose
            )
            all_results[model_type] = results

            # Save results
            if not args.no_save:
                os.makedirs('results', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"results/grid_search_{model_type}_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Results saved to: {filename}")


if __name__ == '__main__':
    main()
