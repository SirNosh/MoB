"""
Utility functions for MoB: Mixture of Bidders.

This module provides common utilities for logging, reproducibility, and debugging.
"""

import torch
import numpy as np
import random
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Parameters:
    -----------
    seed : int
        Random seed value (default: 42).

    Example:
    --------
    >>> set_seed(42)
    >>> # All random operations are now reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU

    # Make cudnn deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[SEED] Random seed set to {seed} for reproducibility")


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging to console and optionally to file.

    Parameters:
    -----------
    log_file : str, optional
        Path to log file. If None, logs only to console.
    level : int
        Logging level (default: logging.INFO).
    format_str : str, optional
        Custom log format string. If None, uses default format.

    Returns:
    --------
    logger : logging.Logger
        Configured logger instance.

    Example:
    --------
    >>> logger = setup_logging('results/experiment.log')
    >>> logger.info('Experiment started')
    """
    if format_str is None:
        format_str = '[%(asctime)s] [%(levelname)s] %(message)s'

    # Create logger
    logger = logging.getLogger('MoB')
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        print(f"[LOGGING] Logging to file: {log_file}")

    return logger


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.

    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model.
    trainable_only : bool
        If True, count only trainable parameters (default: False).

    Returns:
    --------
    num_params : int
        Number of parameters.

    Example:
    --------
    >>> from mob import create_model
    >>> model = create_model('simple_cnn', num_classes=10)
    >>> print(f"Parameters: {count_parameters(model):,}")
    Parameters: 18,816
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: torch.nn.Module, name: str = "Model"):
    """
    Print a summary of model architecture and parameters.

    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model.
    name : str
        Model name for display (default: "Model").

    Example:
    --------
    >>> from mob import create_model
    >>> model = create_model('simple_cnn', num_classes=10, width_multiplier=2)
    >>> print_model_summary(model, name="SimpleCNN (2x)")
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print(f"{'='*60}\n")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Parameters:
    -----------
    seconds : float
        Time in seconds.

    Returns:
    --------
    formatted : str
        Formatted time string.

    Example:
    --------
    >>> format_time(3725.5)
    '1h 2m 5.5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def get_device(device_name: Optional[str] = None, verbose: bool = True) -> torch.device:
    """
    Get PyTorch device with automatic GPU detection.

    Parameters:
    -----------
    device_name : str, optional
        Device name ('cuda', 'cpu', 'mps'). If None, auto-detects best available.
    verbose : bool
        Print device information (default: True).

    Returns:
    --------
    device : torch.device
        PyTorch device object.

    Example:
    --------
    >>> device = get_device()
    [DEVICE] Using device: cuda (NVIDIA GeForce RTX 3090)
    """
    if device_name is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_name)

    if verbose:
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[DEVICE] Using device: {device} ({gpu_name}, {gpu_memory:.1f}GB)")
        else:
            print(f"[DEVICE] Using device: {device}")

    return device


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration dictionary to JSON file.

    Parameters:
    -----------
    config : dict
        Configuration dictionary.
    save_path : str
        Path to save JSON file.

    Example:
    --------
    >>> config = {'num_experts': 4, 'alpha': 0.5, 'beta': 0.5}
    >>> save_config(config, 'results/config.json')
    """
    import json

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable types
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, torch.device):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value

    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)

    print(f"[CONFIG] Configuration saved to {save_path}")


def load_config(load_path: str) -> Dict[str, Any]:
    """
    Load configuration dictionary from JSON file.

    Parameters:
    -----------
    load_path : str
        Path to JSON file.

    Returns:
    --------
    config : dict
        Configuration dictionary.

    Example:
    --------
    >>> config = load_config('results/config.json')
    >>> print(config['num_experts'])
    4
    """
    import json

    with open(load_path, 'r') as f:
        config = json.load(f)

    print(f"[CONFIG] Configuration loaded from {load_path}")
    return config


def print_section_header(title: str, width: int = 60, char: str = '='):
    """
    Print a formatted section header.

    Parameters:
    -----------
    title : str
        Section title.
    width : int
        Total width of header (default: 60).
    char : str
        Character to use for border (default: '=').

    Example:
    --------
    >>> print_section_header('Experiment Results')
    ============================================================
                        Experiment Results
    ============================================================
    """
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted table.

    Parameters:
    -----------
    metrics : dict
        Dictionary of metric names and values.
    title : str
        Table title (default: "Metrics").

    Example:
    --------
    >>> metrics = {'Accuracy': 0.9234, 'Forgetting': 0.0456}
    >>> print_metrics_table(metrics, title="Final Results")
    """
    print(f"\n{title}:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name:.<30} {value:.4f}")
        else:
            print(f"  {metric_name:.<30} {value}")
    print("-" * 40)
