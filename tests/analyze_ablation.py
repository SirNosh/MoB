"""
Ablation Study Analysis from Hyperparameter Search Results

This script analyzes the saved hyperparameter search results to perform
ablation studies on α (alpha), β (beta), λ_ewc (lambda_ewc), and other
hyperparameters.

Usage:
    python tests/analyze_ablation.py results/optuna_search_mob_20260208_*.json
    python tests/analyze_ablation.py results/optuna_search_*.json --plot
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


def load_results(pattern: str) -> Dict[str, pd.DataFrame]:
    """
    Load all result files matching the pattern.

    Returns:
        Dict mapping model_type to DataFrame of trials
    """
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching: {pattern}")
        return {}

    print(f"Found {len(files)} result file(s)")

    all_data = {}

    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)

        model_type = data['model_type']
        trials = data['all_trials']

        # Convert to DataFrame
        df_trials = []
        for trial in trials:
            if trial['state'] != 'COMPLETE':
                continue  # Skip pruned/failed trials

            row = {
                'trial_number': trial['number'],
                'mean_accuracy': trial['value'],
                'std_accuracy': trial['std'],
                **trial['params']  # Unpack all hyperparameters
            }

            # Add individual seed accuracies if available
            if trial.get('all_accs'):
                for i, acc in enumerate(trial['all_accs']):
                    row[f'seed_{i}_accuracy'] = acc

            df_trials.append(row)

        df = pd.DataFrame(df_trials)

        if model_type in all_data:
            # Concatenate if multiple files for same model
            all_data[model_type] = pd.concat([all_data[model_type], df], ignore_index=True)
        else:
            all_data[model_type] = df

        print(f"  {filepath}: {model_type} ({len(df)} completed trials)")

    return all_data


def ablation_analysis(df: pd.DataFrame, param_name: str, bins: int = 5):
    """
    Analyze effect of a single hyperparameter.

    Args:
        df: DataFrame of trials
        param_name: Name of hyperparameter to analyze
        bins: Number of bins for continuous parameters

    Returns:
        Summary statistics grouped by parameter value
    """
    if param_name not in df.columns:
        print(f"Parameter '{param_name}' not found in data")
        return None

    # Check if parameter is categorical or continuous
    unique_values = df[param_name].nunique()

    if unique_values <= 10:
        # Categorical or few unique values - group directly
        grouped = df.groupby(param_name)['mean_accuracy'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
    else:
        # Continuous - bin the values
        df['param_bin'] = pd.cut(df[param_name], bins=bins)
        grouped = df.groupby('param_bin')['mean_accuracy'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)

    return grouped


def interaction_analysis(df: pd.DataFrame, param1: str, param2: str):
    """
    Analyze interaction between two hyperparameters.

    Args:
        df: DataFrame of trials
        param1: First parameter name
        param2: Second parameter name

    Returns:
        Pivot table showing mean accuracy for each combination
    """
    if param1 not in df.columns or param2 not in df.columns:
        print(f"Parameters '{param1}' and/or '{param2}' not found")
        return None

    # Create pivot table
    pivot = df.pivot_table(
        values='mean_accuracy',
        index=param1,
        columns=param2,
        aggfunc='mean'
    ).round(4)

    return pivot


def main():
    parser = argparse.ArgumentParser(description='Ablation study analysis')
    parser.add_argument('pattern', type=str,
                        help='File pattern (e.g., "results/optuna_search_mob_*.json")')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter by model type')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots (requires matplotlib)')
    parser.add_argument('--output', type=str, default='ablation_results.txt',
                        help='Output file for results')

    args = parser.parse_args()

    # Load results
    all_data = load_results(args.pattern)

    if not all_data:
        return

    # Filter by model if specified
    if args.model:
        if args.model in all_data:
            all_data = {args.model: all_data[args.model]}
        else:
            print(f"Model '{args.model}' not found. Available: {list(all_data.keys())}")
            return

    # Open output file
    with open(args.output, 'w') as out:

        # Analyze each model
        for model_type, df in all_data.items():
            print(f"\n{'='*70}")
            print(f"ABLATION ANALYSIS: {model_type.upper()}")
            print(f"{'='*70}")
            print(f"Total completed trials: {len(df)}")

            out.write(f"\n{'='*70}\n")
            out.write(f"ABLATION ANALYSIS: {model_type.upper()}\n")
            out.write(f"{'='*70}\n")
            out.write(f"Total completed trials: {len(df)}\n\n")

            # Get hyperparameter columns (exclude metadata)
            exclude_cols = ['trial_number', 'mean_accuracy', 'std_accuracy', 'param_bin']
            exclude_cols += [c for c in df.columns if c.startswith('seed_')]
            param_cols = [c for c in df.columns if c not in exclude_cols]

            print(f"\nHyperparameters found: {param_cols}")

            # Single parameter ablation
            for param in param_cols:
                print(f"\n--- Effect of {param} ---")
                result = ablation_analysis(df, param)
                if result is not None:
                    print(result)
                    out.write(f"\n--- Effect of {param} ---\n")
                    out.write(result.to_string())
                    out.write("\n")

            # Two-way interactions (only for key params to avoid explosion)
            if model_type in ['mob', 'continual']:
                key_params = ['alpha', 'beta', 'lambda_ewc']
                key_params = [p for p in key_params if p in param_cols]

                print(f"\n--- Two-way Interactions ---")
                out.write(f"\n--- Two-way Interactions ---\n")

                for i, param1 in enumerate(key_params):
                    for param2 in key_params[i+1:]:
                        print(f"\n{param1} × {param2}:")
                        result = interaction_analysis(df, param1, param2)
                        if result is not None:
                            print(result)
                            out.write(f"\n{param1} × {param2}:\n")
                            out.write(result.to_string())
                            out.write("\n")

            # Best configurations
            print(f"\n--- Top 5 Configurations ---")
            out.write(f"\n--- Top 5 Configurations ---\n")

            top5 = df.nlargest(5, 'mean_accuracy')[param_cols + ['mean_accuracy', 'std_accuracy']]
            print(top5.to_string(index=False))
            out.write(top5.to_string(index=False))
            out.write("\n")

    print(f"\nResults saved to: {args.output}")

    # Generate plots if requested
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")

            for model_type, df in all_data.items():
                # Get param columns
                exclude_cols = ['trial_number', 'mean_accuracy', 'std_accuracy', 'param_bin']
                exclude_cols += [c for c in df.columns if c.startswith('seed_')]
                param_cols = [c for c in df.columns if c not in exclude_cols]

                n_params = len(param_cols)
                n_cols = min(3, n_params)
                n_rows = (n_params + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_params == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for i, param in enumerate(param_cols):
                    ax = axes[i]

                    # Scatter plot
                    ax.scatter(df[param], df['mean_accuracy'], alpha=0.5)
                    ax.set_xlabel(param)
                    ax.set_ylabel('Mean Accuracy')
                    ax.set_title(f'Effect of {param}')
                    ax.grid(True, alpha=0.3)

                # Hide unused subplots
                for i in range(n_params, len(axes)):
                    axes[i].axis('off')

                plt.suptitle(f'Ablation Study: {model_type.upper()}', fontsize=16, y=1.0)
                plt.tight_layout()

                plot_file = f"ablation_{model_type}.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {plot_file}")
                plt.close()

        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib seaborn")


if __name__ == '__main__':
    main()
