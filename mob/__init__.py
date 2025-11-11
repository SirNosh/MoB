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
]
