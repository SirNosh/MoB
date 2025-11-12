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
]
