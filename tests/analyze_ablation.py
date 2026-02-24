"""
Ablation Study Analysis from Hyperparameter Search Results

This script analyzes the saved hyperparameter search results to perform
ablation studies on α (alpha), β (beta), λ_ewc (lambda_ewc), and other
hyperparameters.

Usage:
    python tests/analyze_ablation.py results/optuna_search_mob_20260208_*.json
    python tests/analyze_ablation.py results/optuna_search_*.json --plot
    python tests/analyze_ablation.py results/optuna_search_mob_*.json --plot --focus alpha beta lambda_ewc
"""

import json
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from itertools import combinations


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
            # Handle multiple state formats:
            # Optuna saves state as str(TrialState.X) which can be "1" (COMPLETE),
            # "2" (PRUNED), or "COMPLETE"/"PRUNED" depending on Optuna version
            state = str(trial.get('state', ''))
            is_complete = state in ('1', 'COMPLETE', 'TrialState.COMPLETE')
            if not is_complete:
                continue  # Skip pruned/failed trials

            # Skip trials with no value
            if trial.get('value') is None:
                continue

            row = {
                'trial_number': trial['number'],
                'mean_accuracy': trial['value'],
                'std_accuracy': trial.get('std') if trial.get('std') is not None else 0.0,
                **trial['params']  # Unpack all hyperparameters
            }

            # Add individual seed accuracies if available
            all_accs = trial.get('all_accs')
            if all_accs is not None and isinstance(all_accs, list):
                for i, acc in enumerate(all_accs):
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


# ---------------------------------------------------------------------------
# Focused hyperparameter analysis functions (publication-quality plots)
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    'alpha': r'$\alpha$ (Execution Cost Weight)',
    'beta': r'$\beta$ (Forgetting Cost Weight)',
    'lambda_ewc': r'$\lambda_{EWC}$ (Regularization Strength)',
}

LOG_SCALE_PARAMS = {'lambda_ewc'}


def _pub_style():
    """Return matplotlib rcParams overrides for publication-quality plots."""
    return {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 150,
    }


def marginal_effect_plot(df: pd.DataFrame, param: str, output_dir: Path,
                         model_type: str):
    """Scatter + LOWESS + Spearman correlation for one hyperparameter."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    if param not in df.columns:
        print(f"  [marginal] Skipping '{param}' — not in data")
        return

    with plt.rc_context(_pub_style()):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        x = df[param].values
        y = df['mean_accuracy'].values
        yerr = df['std_accuracy'].values

        # Error-bar scatter
        ax.errorbar(x, y, yerr=yerr, fmt='o', alpha=0.45, markersize=4,
                     elinewidth=0.8, capsize=2, color='#4C72B0',
                     label='Trials')

        # LOWESS trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sort_idx = np.argsort(x)
            smoothed = lowess(y[sort_idx], x[sort_idx], frac=0.4,
                              return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color='#C44E52',
                    linewidth=2.5, label='LOWESS trend')
        except ImportError:
            print("  [marginal] statsmodels not installed — skipping LOWESS")

        # Spearman correlation
        rho, pval = spearmanr(x, y)
        ax.annotate(f'Spearman $\\rho$={rho:.3f}  (p={pval:.2e})',
                    xy=(0.03, 0.97), xycoords='axes fraction',
                    fontsize=10, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat',
                              alpha=0.7))

        label = PARAM_LABELS.get(param, param)
        ax.set_xlabel(label)
        ax.set_ylabel('Mean Accuracy')
        ax.set_title(f'Marginal Effect of {label}')
        if param in LOG_SCALE_PARAMS:
            ax.set_xscale('log')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = output_dir / f'{model_type}_marginal_{param}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def distribution_plot(df: pd.DataFrame, param: str, output_dir: Path,
                      model_type: str):
    """Violin plot of accuracy distribution in quantile-based bins."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if param not in df.columns:
        print(f"  [distribution] Skipping '{param}' — not in data")
        return

    with plt.rc_context(_pub_style()):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        tmp = df[[param, 'mean_accuracy']].dropna()
        try:
            tmp['bin'] = pd.qcut(tmp[param], q=3, labels=['Low', 'Medium', 'High'],
                                 duplicates='drop')
        except ValueError:
            # Fallback if too few unique values for 3 quantiles
            tmp['bin'] = pd.cut(tmp[param], bins=3, labels=['Low', 'Medium', 'High'])

        order = ['Low', 'Medium', 'High']
        present = [o for o in order if o in tmp['bin'].cat.categories]

        sns.violinplot(data=tmp, x='bin', y='mean_accuracy', order=present,
                       inner='box', palette='Set2', ax=ax)

        # Annotate bin ranges
        for i, lbl in enumerate(present):
            subset = tmp[tmp['bin'] == lbl][param]
            range_str = f'[{subset.min():.3g}, {subset.max():.3g}]'
            ax.text(i, ax.get_ylim()[0], range_str, ha='center', va='top',
                    fontsize=8, color='grey')

        label = PARAM_LABELS.get(param, param)
        ax.set_xlabel(f'{label} (quantile bin)')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title(f'Accuracy Distribution by {label}')
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        out_path = output_dir / f'{model_type}_distribution_{param}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def heatmap_interaction(df: pd.DataFrame, param1: str, param2: str,
                        output_dir: Path, model_type: str, n_bins: int = 5):
    """2-D heatmap of mean accuracy for binned param1 x param2."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if param1 not in df.columns or param2 not in df.columns:
        print(f"  [heatmap] Skipping '{param1}' x '{param2}' — missing column")
        return

    with plt.rc_context(_pub_style()):
        tmp = df[[param1, param2, 'mean_accuracy']].dropna()

        for p in [param1, param2]:
            try:
                tmp[f'{p}_bin'] = pd.qcut(tmp[p], q=n_bins, duplicates='drop')
            except ValueError:
                tmp[f'{p}_bin'] = pd.cut(tmp[p], bins=min(n_bins, tmp[p].nunique()))

        pivot = tmp.pivot_table(values='mean_accuracy',
                                index=f'{param1}_bin',
                                columns=f'{param2}_bin',
                                aggfunc='mean')

        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, cbar_kws={'label': 'Mean Accuracy'})

        l1 = PARAM_LABELS.get(param1, param1)
        l2 = PARAM_LABELS.get(param2, param2)
        ax.set_ylabel(l1)
        ax.set_xlabel(l2)
        ax.set_title(f'Interaction: {l1} vs {l2}')

        fig.tight_layout()
        out_path = output_dir / f'{model_type}_heatmap_{param1}_vs_{param2}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def sensitivity_summary(df: pd.DataFrame, params: List[str],
                        output_dir: Path, model_type: str):
    """Bar chart of |Spearman rho| for each param — shows relative importance."""
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    with plt.rc_context(_pub_style()):
        results = []
        for p in params:
            if p not in df.columns:
                continue
            tmp = df[[p, 'mean_accuracy']].dropna()
            if len(tmp) < 5:
                continue
            rho, pval = spearmanr(tmp[p], tmp['mean_accuracy'])
            results.append({'param': p, 'abs_rho': abs(rho), 'rho': rho,
                            'pval': pval})

        if not results:
            print("  [sensitivity] No valid params for summary")
            return

        res_df = pd.DataFrame(results).sort_values('abs_rho', ascending=True)

        fig, ax = plt.subplots(figsize=(6, max(3, 0.8 * len(res_df) + 1)))
        labels = [PARAM_LABELS.get(p, p) for p in res_df['param']]
        colors = ['#4C72B0' if r >= 0 else '#C44E52' for r in res_df['rho']]
        bars = ax.barh(labels, res_df['abs_rho'], color=colors, edgecolor='white')

        # Annotate with rho and p-value
        for bar, row in zip(bars, res_df.itertuples()):
            sig = '*' if row.pval < 0.05 else ''
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'$\\rho$={row.rho:+.3f}{sig}',
                    va='center', fontsize=10)

        ax.set_xlabel('|Spearman $\\rho$|')
        ax.set_title(f'Hyperparameter Sensitivity ({model_type.upper()})')
        ax.set_xlim(0, min(1.0, res_df['abs_rho'].max() + 0.15))
        ax.grid(True, alpha=0.3, axis='x')

        fig.tight_layout()
        out_path = output_dir / f'{model_type}_sensitivity.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


def run_focused_analysis(df: pd.DataFrame, model_type: str,
                         focus_params: List[str], output_dir: Path):
    """Run all focused analysis plots for the given params."""
    # Filter to params that actually exist in the data
    available = [p for p in focus_params if p in df.columns]
    if not available:
        print(f"  No focused params found in data for {model_type}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Focused Analysis ({model_type.upper()}) ---")
    print(f"  Params: {available}")
    print(f"  Output: {output_dir}")

    # Per-param plots
    for param in available:
        marginal_effect_plot(df, param, output_dir, model_type)
        distribution_plot(df, param, output_dir, model_type)

    # Pairwise heatmaps
    for p1, p2 in combinations(available, 2):
        heatmap_interaction(df, p1, p2, output_dir, model_type)

    # Sensitivity summary
    sensitivity_summary(df, available, output_dir, model_type)


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
    parser.add_argument('--focus', nargs='*', default=None,
                        help='Hyperparameters for detailed analysis '
                             '(default: alpha beta lambda_ewc for mob/continual)')
    parser.add_argument('--outdir', type=str, default='results/ablation_plots',
                        help='Directory for individual plot PNGs')

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

    # Run focused hyperparameter analysis
    if args.focus is not None or args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import spearmanr

            output_dir = Path(args.outdir)

            for model_type, df in all_data.items():
                # Determine which params to focus on
                if args.focus is not None:
                    focus_params = args.focus if args.focus else ['alpha', 'beta', 'lambda_ewc']
                elif model_type in ['mob', 'continual']:
                    focus_params = ['alpha', 'beta', 'lambda_ewc']
                else:
                    continue

                run_focused_analysis(df, model_type, focus_params, output_dir)

        except ImportError as e:
            print(f"Focused analysis requires matplotlib, seaborn, scipy: {e}")
            print("Install with: pip install matplotlib seaborn scipy statsmodels")


if __name__ == '__main__':
    main()
