"""
Example script for running MoB on Split-CIFAR10 with Phase 2 system.

This demonstrates the high-level MoBSystem interface for easy experimentation.

Usage:
    # Single run with default seed
    python experiments/run_cifar10_example.py

    # Multi-seed for statistical rigor (recommended for publication)
    python experiments/run_cifar10_example.py --seeds 5

    # Specific seed
    python experiments/run_cifar10_example.py --single-seed 42

Requirements:
    pip install avalanche-lib plotly kaleido scipy

Note: This is a simplified example. For full experiments with all baselines,
      use the Phase 1 test_baselines.py pattern with CIFAR10 data.
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mob import (
    MoBSystem,
    create_split_cifar10,
    print_benchmark_summary,
    set_seed,
    plot_learning_curves,
    plot_performance_comparison
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def main(seed=None):
    """Run MoB on Split-CIFAR10 with a specific seed.

    Args:
        seed: Random seed for reproducibility. If None, uses default (42).

    Returns:
        dict: Experiment results including accuracies, forgetting, etc.
    """
    # Set seed for reproducibility
    if seed is None:
        seed = 42
    set_seed(seed)

    # Configuration
    config = {
        'num_experts': 4,
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 3,  # CIFAR10 is RGB
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 5000,
        'learning_rate': 0.001,
        'dropout': 0.5
    }

    print("="*70)
    print(f"MoB on Split-CIFAR10 Example (Seed {seed})")
    print("="*70)

    # Create benchmark
    print("\nCreating Split-CIFAR10 benchmark...")
    try:
        train_tasks, test_tasks = create_split_cifar10(
            num_tasks=5,
            batch_size=32,
            seed=seed
        )
        print(f"✓ Created {len(train_tasks)} tasks")

        # Print benchmark info
        print_benchmark_summary(train_tasks, test_tasks)

    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo run this example, install Avalanche:")
        print("  pip install avalanche-lib")
        return

    # Initialize MoB System
    print("\n" + "="*70)
    print("Initializing MoB System")
    print("="*70)

    system = MoBSystem(
        config=config,
        experiment_name=f"mob_cifar10_seed{seed}",
        save_dir=f"phase2/results/cifar10_seed{seed}",
        seed=seed
    )

    # Print system summary
    print(system.get_summary())

    # Training
    print("\n" + "="*70)
    print("Training")
    print("="*70)

    metrics = system.train(
        train_tasks=train_tasks,
        num_epochs_per_task=1,  # Use 2-3 for better results
        fisher_samples=200,
        verbose=True
    )

    # Evaluation
    print("\n" + "="*70)
    print("Evaluation")
    print("="*70)

    results = system.evaluate(
        test_tasks=test_tasks,
        verbose=True
    )

    # Print final summary
    print("\n" + system.get_summary())

    # Save checkpoint
    system.save_checkpoint("final_model.pt")

    print("\n" + "="*70)
    print("Single Run Complete!")
    print("="*70)
    print(f"\nResults saved to: {system.save_dir}")
    print(f"Average Accuracy: {results['avg_accuracy']:.4f}")
    print(f"Average Forgetting: {results['avg_forgetting']:.4f}")

    return results


def compute_statistics(all_results):
    """
    Compute statistics across multiple seeds.

    Args:
        all_results: List of result dictionaries from multiple runs

    Returns:
        dict: Statistics including mean, std, min, max for each metric
    """
    statistics = {}

    # Extract metrics
    avg_accs = [r['avg_accuracy'] for r in all_results]
    avg_forgettings = [r['avg_forgetting'] for r in all_results]
    final_accs_per_task = [r['final_accuracies'] for r in all_results]

    # Compute statistics for average accuracy
    statistics['avg_accuracy'] = {
        'mean': np.mean(avg_accs),
        'std': np.std(avg_accs, ddof=1),  # Sample standard deviation
        'min': np.min(avg_accs),
        'max': np.max(avg_accs),
        'values': avg_accs
    }

    # Compute statistics for average forgetting
    statistics['avg_forgetting'] = {
        'mean': np.mean(avg_forgettings),
        'std': np.std(avg_forgettings, ddof=1),
        'min': np.min(avg_forgettings),
        'max': np.max(avg_forgettings),
        'values': avg_forgettings
    }

    # Compute per-task statistics
    num_tasks = len(final_accs_per_task[0])
    statistics['per_task_accuracy'] = {}

    for task_id in range(num_tasks):
        task_accs = [run_accs[task_id] for run_accs in final_accs_per_task]
        statistics['per_task_accuracy'][f'task_{task_id}'] = {
            'mean': np.mean(task_accs),
            'std': np.std(task_accs, ddof=1),
            'min': np.min(task_accs),
            'max': np.max(task_accs),
            'values': task_accs
        }

    return statistics


def run_multi_seed_experiments(num_seeds=5):
    """
    Run MoB experiments with multiple random seeds for statistical validity.

    Neural network training is stochastic due to:
    - Random weight initialization
    - Random data shuffling
    - Dropout randomness

    Running with multiple seeds allows us to:
    - Estimate true performance (mean across seeds)
    - Measure stability/variance (std across seeds)
    - Report confidence in results

    Recommended: 5-10 seeds for validation, 20-30 for publication.

    Args:
        num_seeds: Number of different random seeds to run

    Returns:
        tuple: (all_results, statistics)
    """
    print("\n" + "="*70)
    print(f"Multi-Seed Experiment: Running {num_seeds} seeds")
    print("="*70)
    print("\nThis provides statistical validity by:")
    print("  - Estimating true performance (mean ± std)")
    print("  - Measuring stability across random initializations")
    print("  - Enabling confidence in reported results")
    print("="*70)

    seeds = [42 + i for i in range(num_seeds)]
    all_results = []

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*70}")
        print(f"Run {seed_idx}/{num_seeds} - Seed {seed}")
        print(f"{'='*70}")

        results = main(seed=seed)
        all_results.append(results)

        print(f"\nSeed {seed} Results:")
        print(f"  Average Accuracy: {results['avg_accuracy']:.4f}")
        print(f"  Average Forgetting: {results['avg_forgetting']:.4f}")

    # Compute statistics across all seeds
    statistics = compute_statistics(all_results)

    # Print summary statistics
    print("\n" + "="*70)
    print("Multi-Seed Experiment Summary")
    print("="*70)
    print(f"\nRuns completed: {num_seeds}")
    print(f"\nAverage Accuracy:")
    print(f"  Mean: {statistics['avg_accuracy']['mean']:.4f} ± {statistics['avg_accuracy']['std']:.4f}")
    print(f"  Min:  {statistics['avg_accuracy']['min']:.4f}")
    print(f"  Max:  {statistics['avg_accuracy']['max']:.4f}")

    print(f"\nAverage Forgetting:")
    print(f"  Mean: {statistics['avg_forgetting']['mean']:.4f} ± {statistics['avg_forgetting']['std']:.4f}")
    print(f"  Min:  {statistics['avg_forgetting']['min']:.4f}")
    print(f"  Max:  {statistics['avg_forgetting']['max']:.4f}")

    print(f"\nPer-Task Accuracy:")
    for task_id in range(len(statistics['per_task_accuracy'])):
        task_stats = statistics['per_task_accuracy'][f'task_{task_id}']
        print(f"  Task {task_id}: {task_stats['mean']:.4f} ± {task_stats['std']:.4f}")

    # Save results
    results_dir = Path("phase2/results/multi_seed")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    raw_results_path = results_dir / "multi_seed_raw_results.json"
    with open(raw_results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for r in all_results:
            serializable_r = {k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in r.items()}
            serializable_results.append(serializable_r)
        json.dump(serializable_results, f, indent=2)
    print(f"\n✓ Raw results saved to: {raw_results_path}")

    # Save statistics
    stats_path = results_dir / "multi_seed_statistics.json"
    with open(stats_path, 'w') as f:
        # Convert numpy types to native Python types for JSON
        serializable_stats = {}
        for key, value in statistics.items():
            if isinstance(value, dict):
                if 'values' in value:
                    # Per-metric statistics
                    serializable_stats[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.floating) else v
                        for k, v in value.items()
                    }
                else:
                    # Per-task statistics
                    serializable_stats[key] = {
                        task_key: {k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, np.floating) else v
                                  for k, v in task_value.items()}
                        for task_key, task_value in value.items()
                    }
        json.dump(serializable_stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_path}")

    return all_results, statistics


def plot_multi_seed_results(statistics, save_path=None):
    """
    Create visualization of multi-seed results with error bars.

    Args:
        statistics: Statistics dictionary from compute_statistics()
        save_path: Path to save the plot (HTML format)
    """
    if not PLOTLY_AVAILABLE:
        print("⚠  Plotly not available. Skipping visualization.")
        print("   Install with: pip install plotly kaleido")
        return

    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Accuracy', 'Per-Task Accuracy'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Average accuracy with error bar
    fig.add_trace(
        go.Bar(
            x=['MoB'],
            y=[statistics['avg_accuracy']['mean']],
            error_y=dict(
                type='data',
                array=[statistics['avg_accuracy']['std']],
                visible=True
            ),
            marker_color='#2ecc71',
            name='Average Accuracy',
            showlegend=False
        ),
        row=1, col=1
    )

    # Per-task accuracies with error bars
    num_tasks = len(statistics['per_task_accuracy'])
    task_names = [f'Task {i}' for i in range(num_tasks)]
    task_means = [statistics['per_task_accuracy'][f'task_{i}']['mean'] for i in range(num_tasks)]
    task_stds = [statistics['per_task_accuracy'][f'task_{i}']['std'] for i in range(num_tasks)]

    fig.add_trace(
        go.Bar(
            x=task_names,
            y=task_means,
            error_y=dict(
                type='data',
                array=task_stds,
                visible=True
            ),
            marker_color='#3498db',
            name='Task Accuracy',
            showlegend=False
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="MoB Multi-Seed Experiment Results",
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)

    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        print(f"\n✓ Visualization saved to: {save_path}")
    else:
        fig.show()

    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MoB on Split-CIFAR10 with multi-seed support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run with default seed
  python experiments/run_cifar10_example.py

  # Multi-seed for statistical rigor (recommended: 5-10)
  python experiments/run_cifar10_example.py --seeds 5

  # Specific seed
  python experiments/run_cifar10_example.py --single-seed 42
        """
    )

    parser.add_argument(
        '--seeds',
        type=int,
        default=1,
        help='Number of random seeds to run (default: 1, recommended: 5-10 for validation, 20-30 for publication)'
    )

    parser.add_argument(
        '--single-seed',
        type=int,
        default=None,
        help='Run with a specific seed (overrides --seeds)'
    )

    args = parser.parse_args()

    # Single seed mode
    if args.single_seed is not None:
        print(f"Running with specific seed: {args.single_seed}")
        results = main(seed=args.single_seed)

    # Multi-seed mode
    elif args.seeds > 1:
        all_results, statistics = run_multi_seed_experiments(num_seeds=args.seeds)

        # Create visualization
        try:
            plot_path = Path("phase2/results/multi_seed/multi_seed_comparison.html")
            plot_multi_seed_results(statistics, save_path=plot_path)
        except Exception as e:
            print(f"\n⚠  Could not create visualization: {e}")

        print("\n" + "="*70)
        print("Multi-Seed Experiment Complete!")
        print("="*70)
        print("\nFor publication-ready results, use --seeds 20 or higher")
        print("Error bars show ± 1 standard deviation across seeds")

    # Single default run
    else:
        results = main()
        print("\n💡 Tip: Use --seeds 5 for statistically rigorous results")
