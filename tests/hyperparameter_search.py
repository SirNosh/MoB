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
    """
    Suggest hyperparameters for MoB (Task-Aware) using Optuna.

    Based on EWC sanity check results (sanity_check_ewc.py):
    - Single expert optimal λ_ewc ≈ 1000 for best avg accuracy (0.8989)
    - λ_ewc = 500-2500 range gives best stability-plasticity tradeoff
    - Fisher is normalized to mean=1.0, so λ values are architecture-independent

    MoB with 4 experts distributes batches via auction, so each expert sees
    fewer batches than a monolithic model → may need slightly lower λ.
    Search range: 100-5000 (log scale) with focus on 500-2500.
    """
    return {
        # Log-scale search centered around optimal (500-2500 from sanity check)
        'lambda_ewc': trial.suggest_float('lambda_ewc', 100.0, 5000.0, log=True),
        # Bidding weights (alpha=exec cost, beta=forgetting cost)
        'alpha': trial.suggest_float('alpha', 0.3, 0.7),
        'beta': trial.suggest_float('beta', 0.3, 0.7),
        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.005, log=True),
        # Scale factor for forgetting cost in bids (affects routing, not EWC penalty)
        'forgetting_cost_scale': trial.suggest_float('forgetting_cost_scale', 0.5, 3.0),
        # Optimizer reset at task END: always enabled for best continual learning
        'reset_optimizer': True,
    }


def suggest_gated_moe_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """
    Suggest hyperparameters for Gated MoE using Optuna.

    Gated MoE uses the same expert architecture as MoB, so λ_ewc range
    should be similar. Based on sanity check: optimal ~500-2500.
    """
    return {
        'lambda_ewc': trial.suggest_float('lambda_ewc', 100.0, 5000.0, log=True),
        'gater_ewc': trial.suggest_categorical('gater_ewc', [True, False]),
        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.005, log=True),
        'gater_hidden_size': trial.suggest_categorical('gater_hidden_size', [128, 256, 512]),
    }


def suggest_continual_params(trial: 'optuna.Trial') -> Dict[str, Any]:
    """
    Suggest hyperparameters for Continual MoB (Task-Free) using Optuna.

    Based on EWC sanity check results:
    - Single expert optimal λ_ewc ≈ 1000 (avg accuracy 0.8989)
    - For <5% forgetting: λ_ewc ≈ 2500 (avg accuracy 0.8548)

    Continual MoB uses shift detection to trigger consolidation, so Fisher
    updates are less frequent than task-aware MoB. May need higher λ to
    compensate for fewer consolidation opportunities.
    """
    return {
        # Higher range than task-aware MoB due to less frequent Fisher updates
        'lambda_ewc': trial.suggest_float('lambda_ewc', 250.0, 5000.0, log=True),
        # Bidding weights
        'alpha': trial.suggest_float('alpha', 0.3, 0.7),
        'beta': trial.suggest_float('beta', 0.3, 0.7),
        # Shift detection sensitivity (lower = more sensitive, more consolidations)
        'shift_threshold': trial.suggest_float('shift_threshold', 1.0, 4.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.005, log=True),
        # Forgetting cost scale for bidding
        'forgetting_cost_scale': trial.suggest_float('forgetting_cost_scale', 0.5, 3.0),
        # Optimizer reset on shift detection: always enabled for better continual learning
        'reset_optimizer': True,
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


def run_multi_seed_experiment_detailed(
    model_type: str,
    config: Dict[str, Any],
    base_config: Dict[str, Any],
    seeds: List[int],
    trial: Optional['optuna.Trial'] = None,
    verbose: bool = False
) -> Tuple[float, float, List[float], List[Dict]]:
    """
    Run experiment across multiple seeds with FULL results for per-expert analysis.

    This is used when optimize_per_expert=True to get detailed metrics
    for computing per-expert scores.

    Returns:
        mean_score: Mean per-expert composite score across seeds
        std_score: Standard deviation of scores
        all_scores: List of scores for each seed
        all_results: List of full per-expert analysis results
    """
    scores = []
    all_results = []

    for i, seed in enumerate(seeds):
        try:
            results = run_single_experiment(
                model_type, config, base_config, seed, verbose=verbose
            )

            # Compute per-expert score
            full_config = {**base_config, **config}
            per_expert_result = compute_per_expert_score(results, full_config, verbose=verbose)
            score = per_expert_result['composite_score']

            scores.append(score)
            all_results.append(per_expert_result)

            # Report intermediate result for pruning
            if trial is not None:
                trial.report(np.mean(scores), i)

                # Check if should prune (after first seed)
                if i == 0 and trial.should_prune():
                    raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise
        except Exception as e:
            if verbose:
                print(f"  Seed {seed} failed: {e}")
            scores.append(0.0)
            all_results.append({})

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score, scores, all_results


# =============================================================================
# SANITY CHECK TARGETS (from sanity_check_ewc.py with λ=1000)
# =============================================================================
# These are the performance targets for a SINGLE expert on 2 tasks.
# Each MoB expert should aim to match these when handling its assigned tasks.
SANITY_CHECK_TARGETS = {
    'task_accuracy': 0.88,      # T1 after T2 = 0.8807
    'new_task_accuracy': 0.92,  # T2 after T2 = 0.9171
    'max_forgetting': 0.12,     # Forgetting = 0.1179
    'min_avg_accuracy': 0.85,   # Avg = 0.8989, allow some slack
}


def compute_per_expert_score(results: Dict, config: Dict, verbose: bool = False) -> Dict:
    """
    Compute per-expert performance metrics for MoB/Continual MoB.

    This function analyzes how well each expert performs on its assigned tasks,
    comparing against the sanity check targets.

    Returns:
        Dict with score components and detailed metrics
    """
    # Extract results
    task_accuracies = results.get('task_accuracies', [])  # Accuracy right after training each task
    final_accuracies = results.get('final_accuracies', [])  # Accuracy at the end on all tasks
    expert_task_wins = results.get('expert_task_wins', {})  # Which expert handled which task

    num_tasks = len(final_accuracies)
    num_experts = config.get('num_experts', 4)

    # Build expert -> tasks mapping
    expert_tasks = {i: [] for i in range(num_experts)}
    for task_id, expert_id in expert_task_wins.items():
        if isinstance(task_id, str):
            task_id = int(task_id)
        expert_tasks[expert_id].append(task_id)

    # Compute per-expert metrics
    expert_metrics = {}
    for expert_id in range(num_experts):
        tasks = expert_tasks[expert_id]
        if not tasks:
            continue

        # Get accuracies for this expert's tasks
        expert_final_accs = [final_accuracies[t] for t in tasks if t < len(final_accuracies)]

        if not expert_final_accs:
            continue

        # Compute forgetting for this expert's tasks
        forgetting = []
        for t in tasks:
            if t < len(task_accuracies) and t < len(final_accuracies):
                forg = max(0, task_accuracies[t] - final_accuracies[t])
                forgetting.append(forg)

        expert_metrics[expert_id] = {
            'tasks': tasks,
            'num_tasks': len(tasks),
            'final_accuracies': expert_final_accs,
            'mean_accuracy': np.mean(expert_final_accs),
            'min_accuracy': min(expert_final_accs),
            'forgetting': forgetting,
            'mean_forgetting': np.mean(forgetting) if forgetting else 0.0,
            'max_forgetting': max(forgetting) if forgetting else 0.0,
        }

    # Compute score components
    # 1. Per-expert accuracy: each expert should achieve ~88% on its tasks
    expert_acc_scores = []
    for eid, metrics in expert_metrics.items():
        # Score based on how close to target (0.88)
        acc_score = min(1.0, metrics['mean_accuracy'] / SANITY_CHECK_TARGETS['task_accuracy'])
        expert_acc_scores.append(acc_score)

    # 2. Per-expert forgetting: each expert should have <12% forgetting
    expert_forg_scores = []
    for eid, metrics in expert_metrics.items():
        # Penalize forgetting above target
        forg = metrics['mean_forgetting']
        forg_score = max(0, 1.0 - forg / SANITY_CHECK_TARGETS['max_forgetting'])
        expert_forg_scores.append(forg_score)

    # 3. Expert utilization: ideally all experts contribute
    utilization = len(expert_metrics) / num_experts

    # 4. Task coverage: all tasks should be handled
    covered_tasks = set()
    for metrics in expert_metrics.values():
        covered_tasks.update(metrics['tasks'])
    coverage = len(covered_tasks) / num_tasks

    # Compute composite score
    # Prioritize: min expert accuracy > mean expert accuracy > low forgetting > utilization
    if expert_acc_scores:
        min_expert_acc_score = min(expert_acc_scores)
        mean_expert_acc_score = np.mean(expert_acc_scores)
    else:
        min_expert_acc_score = 0.0
        mean_expert_acc_score = 0.0

    if expert_forg_scores:
        mean_forg_score = np.mean(expert_forg_scores)
    else:
        mean_forg_score = 1.0  # No forgetting if no experts

    # Weighted composite:
    # - 40% minimum expert accuracy (ensures ALL experts are good)
    # - 30% mean expert accuracy
    # - 20% forgetting penalty
    # - 10% utilization/coverage
    composite_score = (
        0.40 * min_expert_acc_score +
        0.30 * mean_expert_acc_score +
        0.20 * mean_forg_score +
        0.05 * utilization +
        0.05 * coverage
    )

    result = {
        'composite_score': composite_score,
        'min_expert_acc_score': min_expert_acc_score,
        'mean_expert_acc_score': mean_expert_acc_score,
        'mean_forg_score': mean_forg_score,
        'utilization': utilization,
        'coverage': coverage,
        'expert_metrics': expert_metrics,
        'global_avg_accuracy': np.mean(final_accuracies) if final_accuracies else 0.0,
        'global_forgetting': results.get('forgetting', 0.0),
    }

    if verbose:
        print(f"\n  Per-Expert Analysis:")
        for eid, metrics in expert_metrics.items():
            print(f"    Expert {eid}: Tasks {metrics['tasks']}, "
                  f"Acc={metrics['mean_accuracy']:.4f}, "
                  f"Forget={metrics['mean_forgetting']:.4f}")
        print(f"  Composite Score: {composite_score:.4f}")

    return result


# =============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =============================================================================

def create_objective(
    model_type: str,
    base_config: Dict[str, Any],
    seeds: List[int],
    verbose: bool = False,
    optimize_per_expert: bool = False
):
    """
    Create Optuna objective function for a model type.

    Args:
        optimize_per_expert: If True, optimize for per-expert performance
                            matching sanity check targets (T1≈0.88, T2≈0.92, Forget<0.12)
    """

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
        if optimize_per_expert and model_type in ['mob', 'continual']:
            # Per-expert optimization: get full results for analysis
            mean_score, std_score, all_scores, all_results = run_multi_seed_experiment_detailed(
                model_type=model_type,
                config=config,
                base_config=base_config,
                seeds=seeds,
                trial=trial,
                verbose=verbose
            )

            if verbose:
                print(f"  -> Per-Expert Score: {mean_score:.4f} +/- {std_score:.4f}")
                print(f"  -> All: {[f'{s:.4f}' for s in all_scores]}")

            # Store detailed metrics
            trial.set_user_attr('std_score', std_score)
            trial.set_user_attr('all_scores', all_scores)
            trial.set_user_attr('config', config)
            trial.set_user_attr('optimization_mode', 'per_expert')

            # Store per-expert metrics from first seed for inspection
            if all_results:
                first_result = all_results[0]
                trial.set_user_attr('expert_metrics', first_result.get('expert_metrics', {}))
                trial.set_user_attr('global_avg_accuracy', first_result.get('global_avg_accuracy', 0))

            return mean_score
        else:
            # Standard optimization: just average accuracy
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
            trial.set_user_attr('optimization_mode', 'avg_accuracy')

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
    save_results: bool = True,
    optimize_per_expert: bool = False
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
        optimize_per_expert: If True, optimize for per-expert performance
                            matching sanity check targets (MoB/Continual only)

    Returns:
        Dict with best params, best score, and study results
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available. Install with: pip install optuna")

    if seeds is None:
        seeds = DEFAULT_SEEDS

    opt_mode = "PER-EXPERT (sanity check targets)" if optimize_per_expert else "AVERAGE ACCURACY"

    print(f"\n{'='*70}")
    print(f"OPTUNA HYPERPARAMETER SEARCH: {model_type.upper()}")
    print(f"{'='*70}")
    print(f"Optimization Mode: {opt_mode}")
    print(f"Trials: {n_trials}")
    print(f"Seeds per trial: {len(seeds)} ({seeds})")
    print(f"Sampler: TPE (Tree-structured Parzen Estimator)")
    print(f"Pruner: Percentile ({PRUNE_PERCENTILE}th percentile)")
    if optimize_per_expert:
        print(f"Targets: Task Acc≥{SANITY_CHECK_TARGETS['task_accuracy']:.0%}, "
              f"Forget<{SANITY_CHECK_TARGETS['max_forgetting']:.0%}")
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
    objective = create_objective(
        model_type, base_config, seeds,
        verbose=verbose,
        optimize_per_expert=optimize_per_expert
    )

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

    if optimize_per_expert:
        print(f"Per-Expert Score: {best_trial.value:.4f} +/- {best_trial.user_attrs.get('std_score', 0):.4f}")
        print(f"All Seeds: {best_trial.user_attrs.get('all_scores', [])}")
        print(f"Global Avg Accuracy: {best_trial.user_attrs.get('global_avg_accuracy', 0):.4f}")

        # Print expert breakdown if available
        expert_metrics = best_trial.user_attrs.get('expert_metrics', {})
        if expert_metrics:
            print(f"\nPer-Expert Breakdown:")
            for eid, metrics in expert_metrics.items():
                print(f"  Expert {eid}: Tasks {metrics.get('tasks', [])}, "
                      f"Acc={metrics.get('mean_accuracy', 0):.4f}, "
                      f"Forget={metrics.get('mean_forgetting', 0):.4f}")
    else:
        print(f"Mean Accuracy: {best_trial.value:.4f} +/- {best_trial.user_attrs.get('std_accuracy', 0):.4f}")
        print(f"All Seeds: {best_trial.user_attrs.get('all_accuracies', [])}")

    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Compile results
    results = {
        'model_type': model_type,
        'optimization_mode': 'per_expert' if optimize_per_expert else 'avg_accuracy',
        'best_score': best_trial.value,
        'best_std': best_trial.user_attrs.get('std_score' if optimize_per_expert else 'std_accuracy', 0),
        'best_all_scores': best_trial.user_attrs.get('all_scores' if optimize_per_expert else 'all_accuracies', []),
        'best_params': best_trial.params,
        'n_trials': n_trials,
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'seeds': seeds,
        'sanity_check_targets': SANITY_CHECK_TARGETS if optimize_per_expert else None,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state),
                'std': t.user_attrs.get('std_score' if optimize_per_expert else 'std_accuracy'),
                'all_scores': t.user_attrs.get('all_scores' if optimize_per_expert else 'all_accuracies'),
                'expert_metrics': t.user_attrs.get('expert_metrics') if optimize_per_expert else None,
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
    save_results: bool = True,
    optimize_per_expert: bool = False
) -> Dict[str, Dict]:
    """Run hyperparameter search for multiple model types."""

    all_results = {}

    for model_type in model_types:
        try:
            # Per-expert optimization only applies to MoB and Continual
            use_per_expert = optimize_per_expert and model_type in ['mob', 'continual']

            results = run_optuna_search(
                model_type=model_type,
                base_config=base_config,
                n_trials=n_trials,
                seeds=seeds,
                verbose=verbose,
                save_results=save_results,
                optimize_per_expert=use_per_expert
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
            print(f"  Best Accuracy: {results['best_score']:.4f} +/- {results['best_std']:.4f}")
            print(f"  Trials: {results['n_trials']} ({results['n_pruned']} pruned)")
            print(f"  Best Params: {results['best_params']}")

    return all_results


# =============================================================================
# FALLBACK: GRID SEARCH (if Optuna not available)
# =============================================================================

def get_grid_search_space(model_type: str, quick: bool = False) -> Dict[str, List[Any]]:
    """
    Get grid search space when Optuna is not available.

    Ranges informed by EWC sanity check (sanity_check_ewc.py):
    - Single expert optimal λ_ewc ≈ 1000 (avg accuracy 0.8989)
    - λ_ewc = 500-2500 gives best stability-plasticity tradeoff
    - For <5% forgetting: λ_ewc ≈ 2500
    """

    if model_type == 'mob':
        if quick:
            return {
                # Quick test: sample from optimal range (500-2500)
                'lambda_ewc': [500.0, 1000.0, 2000.0],
                'alpha': [0.5],
                'beta': [0.5],
                'learning_rate': [0.001],
                'forgetting_cost_scale': [1.0],
            }
        return {
            # Full search: centered on sanity check optimal (1000)
            'lambda_ewc': [250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0],
            'alpha': [0.4, 0.5, 0.6],
            'beta': [0.4, 0.5, 0.6],
            'learning_rate': [0.0005, 0.001, 0.002],
            'forgetting_cost_scale': [0.5, 1.0, 2.0],
        }

    elif model_type == 'gated_moe':
        if quick:
            return {
                # Gated MoE uses same experts as MoB, so similar λ range
                'lambda_ewc': [500.0, 1000.0, 2000.0],
                'gater_ewc': [True],
                'learning_rate': [0.001],
                'gater_hidden_size': [256],
            }
        return {
            # Full search: informed by EWC sanity check
            'lambda_ewc': [250.0, 500.0, 1000.0, 2000.0, 3000.0],
            'gater_ewc': [True, False],
            'learning_rate': [0.0005, 0.001, 0.002],
            'gater_hidden_size': [128, 256, 512],
        }

    elif model_type == 'continual':
        if quick:
            return {
                # Continual MoB: may need higher λ due to less frequent consolidation
                'lambda_ewc': [500.0, 1000.0, 2500.0],
                'alpha': [0.5],
                'beta': [0.5],
                'shift_threshold': [2.0],
                'learning_rate': [0.001],
                'forgetting_cost_scale': [1.0],
            }
        return {
            # Full search: higher λ range for continual setting
            'lambda_ewc': [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3500.0],
            'alpha': [0.4, 0.5, 0.6],
            'beta': [0.4, 0.5, 0.6],
            'shift_threshold': [1.5, 2.0, 2.5, 3.0],
            'learning_rate': [0.0005, 0.001, 0.002],
            'forgetting_cost_scale': [0.5, 1.0, 2.0],
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

    # Per-expert optimization (for MoB/Continual)
    parser.add_argument('--per_expert', action='store_true',
                        help='Optimize for per-expert performance matching sanity check targets '
                             '(Task Acc≥88%%, Forget<12%%). Only applies to MoB and Continual.')

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
    if args.per_expert:
        print(f"Optimization: PER-EXPERT (sanity check targets)")
        print(f"  Target Task Accuracy: ≥{SANITY_CHECK_TARGETS['task_accuracy']:.0%}")
        print(f"  Target Max Forgetting: <{SANITY_CHECK_TARGETS['max_forgetting']:.0%}")
    else:
        print(f"Optimization: AVERAGE ACCURACY")
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
            save_results=not args.no_save,
            optimize_per_expert=args.per_expert
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
