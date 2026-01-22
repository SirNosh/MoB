"""
Visualization utilities for MoB experiments using Plotly.

This module provides interactive visualizations for:
- Accuracy and forgetting over tasks
- Expert specialization heatmaps
- Bid dynamics
- Performance comparisons across methods

Note: Requires plotly to be installed:
    pip install plotly kaleido
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn(
        "Plotly not available. Install with: pip install plotly kaleido"
    )


def plot_accuracy_matrix(
    methods_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[go.Figure]:
    """
    Create accuracy matrix heatmap comparing methods across tasks.

    Parameters:
    -----------
    methods_results : dict
        Dictionary mapping method names to their results dicts.
        Each results dict must have 'final_accuracies' key.
    save_path : str, optional
        Path to save figure as HTML.
    show : bool
        Whether to display figure (default: True).

    Returns:
    --------
    fig : plotly.graph_objects.Figure or None
        Plotly figure object if plotly available, None otherwise.

    Example:
    --------
    >>> results = {
    ...     'MoB': {'final_accuracies': [0.95, 0.93, 0.92, 0.90, 0.89]},
    ...     'Naive': {'final_accuracies': [0.10, 0.08, 0.12, 0.09, 0.85]}
    ... }
    >>> plot_accuracy_matrix(results, save_path='results/accuracy_matrix.html')
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Skipping visualization.")
        return None

    # Prepare data
    methods = list(methods_results.keys())
    num_tasks = len(methods_results[methods[0]]['final_accuracies'])

    # Create matrix: rows=methods, cols=tasks
    matrix = []
    for method in methods:
        accuracies = methods_results[method]['final_accuracies']
        matrix.append(accuracies)

    matrix = np.array(matrix)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f'Task {i+1}' for i in range(num_tasks)],
        y=methods,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=np.round(matrix, 3),
        texttemplate='%{text:.3f}',
        textfont={"size": 12},
        colorbar=dict(title="Accuracy")
    ))

    fig.update_layout(
        title='Final Task Accuracies by Method',
        xaxis_title='Task',
        yaxis_title='Method',
        width=800,
        height=400,
        font=dict(size=12)
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Accuracy matrix saved to {save_path}")

    if show:
        fig.show()

    return fig


def plot_forgetting_analysis(
    methods_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[go.Figure]:
    """
    Create forgetting analysis plot with task-specific forgetting.

    Parameters:
    -----------
    methods_results : dict
        Dictionary mapping method names to results.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display figure.

    Returns:
    --------
    fig : plotly.graph_objects.Figure or None
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Skipping visualization.")
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Forgetting', 'Forgetting by Task'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )

    methods = list(methods_results.keys())
    colors = px.colors.qualitative.Set2[:len(methods)]

    # Average forgetting (left panel)
    avg_forgetting = [methods_results[m].get('avg_forgetting', 0) for m in methods]

    fig.add_trace(
        go.Bar(
            x=methods,
            y=avg_forgetting,
            marker_color=colors,
            name='Avg Forgetting',
            showlegend=False
        ),
        row=1, col=1
    )

    # Forgetting by task (right panel)
    for idx, method in enumerate(methods):
        if 'forgetting_per_task' in methods_results[method]:
            forgetting = methods_results[method]['forgetting_per_task']
            tasks = list(range(1, len(forgetting) + 1))

            fig.add_trace(
                go.Scatter(
                    x=tasks,
                    y=forgetting,
                    mode='lines+markers',
                    name=method,
                    line=dict(color=colors[idx], width=2),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )

    fig.update_xaxes(title_text="Method", row=1, col=1)
    fig.update_yaxes(title_text="Forgetting", row=1, col=1)
    fig.update_xaxes(title_text="Task", row=1, col=2)
    fig.update_yaxes(title_text="Forgetting", row=1, col=2)

    fig.update_layout(
        title='Catastrophic Forgetting Analysis',
        width=1200,
        height=500,
        font=dict(size=12),
        showlegend=True
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Forgetting analysis saved to {save_path}")

    if show:
        fig.show()

    return fig


def plot_performance_comparison(
    methods_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[go.Figure]:
    """
    Create comprehensive performance comparison with accuracy and forgetting.

    Parameters:
    -----------
    methods_results : dict
        Dictionary mapping method names to results.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display figure.

    Returns:
    --------
    fig : plotly.graph_objects.Figure or None
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Skipping visualization.")
        return None

    methods = list(methods_results.keys())
    avg_accuracies = [methods_results[m].get('avg_accuracy', 0) for m in methods]
    avg_forgetting = [methods_results[m].get('avg_forgetting', 0) for m in methods]

    # Sort by average accuracy (descending)
    sorted_indices = np.argsort(avg_accuracies)[::-1]
    methods_sorted = [methods[i] for i in sorted_indices]
    accuracies_sorted = [avg_accuracies[i] for i in sorted_indices]
    forgetting_sorted = [avg_forgetting[i] for i in sorted_indices]

    fig = go.Figure()

    # Accuracy bars
    fig.add_trace(go.Bar(
        name='Average Accuracy',
        x=methods_sorted,
        y=accuracies_sorted,
        marker_color='rgb(55, 83, 109)',
        yaxis='y',
        offsetgroup=1
    ))

    # Forgetting bars
    fig.add_trace(go.Bar(
        name='Average Forgetting',
        x=methods_sorted,
        y=forgetting_sorted,
        marker_color='rgb(219, 64, 82)',
        yaxis='y2',
        offsetgroup=2
    ))

    fig.update_layout(
        title='Method Performance Comparison',
        xaxis=dict(title='Method'),
        yaxis=dict(
            title='Average Accuracy',
            titlefont=dict(color='rgb(55, 83, 109)'),
            tickfont=dict(color='rgb(55, 83, 109)'),
            range=[0, 1]
        ),
        yaxis2=dict(
            title='Average Forgetting',
            titlefont=dict(color='rgb(219, 64, 82)'),
            tickfont=dict(color='rgb(219, 64, 82)'),
            overlaying='y',
            side='right',
            range=[0, max(forgetting_sorted) * 1.2]
        ),
        barmode='group',
        width=900,
        height=500,
        font=dict(size=12),
        legend=dict(x=0.02, y=0.98)
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Performance comparison saved to {save_path}")

    if show:
        fig.show()

    return fig


def plot_expert_specialization(
    win_history: List[int],
    num_experts: int,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[go.Figure]:
    """
    Visualize expert specialization over time.

    Parameters:
    -----------
    win_history : list[int]
        List of winner expert IDs over batches.
    num_experts : int
        Total number of experts.
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display figure.

    Returns:
    --------
    fig : plotly.graph_objects.Figure or None
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Skipping visualization.")
        return None

    # Compute rolling win rates
    window = min(100, len(win_history) // 10)
    if window < 10:
        window = 10

    batches = []
    win_rates = {i: [] for i in range(num_experts)}

    for i in range(window, len(win_history), window // 2):
        window_wins = win_history[i-window:i]
        batches.append(i)

        for expert_id in range(num_experts):
            rate = window_wins.count(expert_id) / len(window_wins)
            win_rates[expert_id].append(rate)

    # Create plot
    fig = go.Figure()

    colors = px.colors.qualitative.Set1[:num_experts]

    for expert_id in range(num_experts):
        fig.add_trace(go.Scatter(
            x=batches,
            y=win_rates[expert_id],
            mode='lines',
            name=f'Expert {expert_id}',
            line=dict(color=colors[expert_id], width=2),
            stackgroup='one'  # Create stacked area chart
        ))

    fig.update_layout(
        title='Expert Specialization Over Time',
        xaxis_title='Batch',
        yaxis_title='Win Rate',
        yaxis=dict(range=[0, 1]),
        width=1000,
        height=500,
        font=dict(size=12),
        hovermode='x unified'
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Expert specialization plot saved to {save_path}")

    if show:
        fig.show()

    return fig


def plot_learning_curves(
    methods_results: Dict[str, Dict],
    metric: str = 'final_accuracies',
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[go.Figure]:
    """
    Plot learning curves showing performance progression across tasks.

    Parameters:
    -----------
    methods_results : dict
        Dictionary mapping method names to results.
    metric : str
        Metric to plot ('final_accuracies' or 'task_accuracies').
    save_path : str, optional
        Path to save figure.
    show : bool
        Whether to display figure.

    Returns:
    --------
    fig : plotly.graph_objects.Figure or None
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Skipping visualization.")
        return None

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (method, results) in enumerate(methods_results.items()):
        if metric in results:
            values = results[metric]
            tasks = list(range(1, len(values) + 1))

            fig.add_trace(go.Scatter(
                x=tasks,
                y=values,
                mode='lines+markers',
                name=method,
                line=dict(color=colors[idx % len(colors)], width=3),
                marker=dict(size=10)
            ))

    title = 'Learning Curves' if metric == 'final_accuracies' else 'Task Performance During Training'

    fig.update_layout(
        title=title,
        xaxis_title='Task',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        width=900,
        height=500,
        font=dict(size=12),
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Learning curves saved to {save_path}")

    if show:
        fig.show()

    return fig


def create_experiment_dashboard(
    methods_results: Dict[str, Dict],
    win_history: Optional[List[int]] = None,
    num_experts: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[go.Figure]:
    """
    Create comprehensive experiment dashboard with all key metrics.

    Parameters:
    -----------
    methods_results : dict
        Dictionary mapping method names to results.
    win_history : list[int], optional
        Expert win history for specialization plot.
    num_experts : int, optional
        Number of experts (required if win_history provided).
    save_path : str, optional
        Path to save dashboard.
    show : bool
        Whether to display dashboard.

    Returns:
    --------
    fig : plotly.graph_objects.Figure or None
    """
    if not PLOTLY_AVAILABLE:
        warnings.warn("Plotly not available. Skipping visualization.")
        return None

    # Create subplots
    rows = 2
    cols = 2
    subplot_titles = [
        'Final Accuracy by Task',
        'Performance Comparison',
        'Forgetting Analysis',
        'Learning Curves'
    ]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    methods = list(methods_results.keys())
    colors = px.colors.qualitative.Set2[:len(methods)]
    num_tasks = len(methods_results[methods[0]]['final_accuracies'])

    # Panel 1: Final accuracy by task (grouped bar chart)
    for idx, method in enumerate(methods):
        accuracies = methods_results[method]['final_accuracies']
        fig.add_trace(
            go.Bar(
                name=method,
                x=[f'T{i+1}' for i in range(num_tasks)],
                y=accuracies,
                marker_color=colors[idx],
                showlegend=True
            ),
            row=1, col=1
        )

    # Panel 2: Average performance (accuracy vs forgetting)
    avg_acc = [methods_results[m].get('avg_accuracy', 0) for m in methods]
    avg_forg = [methods_results[m].get('avg_forgetting', 0) for m in methods]

    fig.add_trace(
        go.Bar(
            name='Avg Accuracy',
            x=methods,
            y=avg_acc,
            marker_color='steelblue',
            showlegend=False
        ),
        row=1, col=2
    )

    # Panel 3: Forgetting by task
    for idx, method in enumerate(methods):
        if 'forgetting_per_task' in methods_results[method]:
            forgetting = methods_results[method]['forgetting_per_task']
            fig.add_trace(
                go.Scatter(
                    name=method,
                    x=list(range(1, len(forgetting) + 1)),
                    y=forgetting,
                    mode='lines+markers',
                    line=dict(color=colors[idx]),
                    showlegend=False
                ),
                row=2, col=1
            )

    # Panel 4: Learning curves
    for idx, method in enumerate(methods):
        accuracies = methods_results[method]['final_accuracies']
        fig.add_trace(
            go.Scatter(
                name=method,
                x=list(range(1, len(accuracies) + 1)),
                y=accuracies,
                mode='lines+markers',
                line=dict(color=colors[idx]),
                showlegend=False
            ),
            row=2, col=2
        )

    # Update axes
    fig.update_xaxes(title_text="Task", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_xaxes(title_text="Method", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="Task", row=2, col=1)
    fig.update_yaxes(title_text="Forgetting", row=2, col=1)
    fig.update_xaxes(title_text="Task", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)

    fig.update_layout(
        title_text="MoB Experiment Dashboard",
        width=1400,
        height=900,
        font=dict(size=10),
        showlegend=True,
        legend=dict(x=1.02, y=1)
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Dashboard saved to {save_path}")

    if show:
        fig.show()

    return fig
