"""
MoB: Mixture of Bidders - A Continual Learning Framework with VCG Auctions

This package implements a novel continual learning system where expert neural networks
compete in truthful auctions to process data batches, enabling emergent specialization
and mitigating catastrophic forgetting.
"""

from .auction import PerBatchVCGAuction, SealedBidProtocol, create_commitment
from .bidding import ExecutionCostEstimator, EWCForgettingEstimator
from .expert import MoBExpert
from .pool import ExpertPool
from .models import SimpleCNN, LeNet5, MLP, create_model
from .baselines import NaiveFineTuning, RandomAssignment, MonolithicEWC, GatedMoE
from .bid_diagnostics import BidLogger
from .utils import (
    set_seed,
    setup_logging,
    count_parameters,
    print_model_summary,
    format_time,
    get_device,
    save_config,
    load_config,
    print_section_header,
    print_metrics_table
)

# Phase 2 additions
from .system import MoBSystem
from .avalanche_wrapper import (
    create_split_cifar10,
    create_split_cifar100,
    create_split_mnist_avalanche,
    get_benchmark_info,
    print_benchmark_summary
)
from .visualization import (
    plot_accuracy_matrix,
    plot_forgetting_analysis,
    plot_performance_comparison,
    plot_expert_specialization,
    plot_learning_curves,
    create_experiment_dashboard
)

__version__ = "0.1.0"
__author__ = "MoB Development Team"

__all__ = [
    # Auction mechanisms
    'PerBatchVCGAuction',
    'SealedBidProtocol',
    'create_commitment',

    # Bidding components
    'ExecutionCostEstimator',
    'EWCForgettingEstimator',
    'BidLogger',

    # Expert management
    'MoBExpert',
    'ExpertPool',

    # Models
    'SimpleCNN',
    'LeNet5',
    'MLP',
    'create_model',

    # Baselines
    'NaiveFineTuning',
    'RandomAssignment',
    'MonolithicEWC',
    'GatedMoE',

    # Utilities
    'set_seed',
    'setup_logging',
    'count_parameters',
    'print_model_summary',
    'format_time',
    'get_device',
    'save_config',
    'load_config',
    'print_section_header',
    'print_metrics_table',

    # Phase 2: System
    'MoBSystem',

    # Phase 2: Avalanche integration
    'create_split_cifar10',
    'create_split_cifar100',
    'create_split_mnist_avalanche',
    'get_benchmark_info',
    'print_benchmark_summary',

    # Phase 2: Visualization
    'plot_accuracy_matrix',
    'plot_forgetting_analysis',
    'plot_performance_comparison',
    'plot_expert_specialization',
    'plot_learning_curves',
    'create_experiment_dashboard',
]
