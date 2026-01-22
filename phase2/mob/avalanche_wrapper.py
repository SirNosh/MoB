"""
Avalanche integration for MoB.

This module provides wrappers and utilities for using MoB with the Avalanche
continual learning library, enabling access to standard benchmarks like
Split-CIFAR10, Split-CIFAR100, and CORe50.

Note: Requires avalanche-lib to be installed:
    pip install avalanche-lib
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
import warnings

try:
    from avalanche.benchmarks.classic import (
        SplitCIFAR10,
        SplitCIFAR100,
        SplitMNIST
    )
    from avalanche.benchmarks import nc_benchmark
    AVALANCHE_AVAILABLE = True
except ImportError:
    AVALANCHE_AVAILABLE = False
    warnings.warn(
        "Avalanche library not available. Install with: pip install avalanche-lib"
    )


def create_split_cifar10(
    num_tasks: int = 5,
    return_task_id: bool = False,
    fixed_class_order: Optional[List[int]] = None,
    seed: Optional[int] = None,
    batch_size: int = 32
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create Split-CIFAR10 benchmark using Avalanche.

    Splits CIFAR-10 (10 classes) into multiple tasks.

    Parameters:
    -----------
    num_tasks : int
        Number of tasks (default: 5, i.e., 2 classes per task).
    return_task_id : bool
        If True, include task ID in dataset (default: False).
    fixed_class_order : list, optional
        Fixed order of classes. If None, uses default [0,1,2,...,9].
    seed : int, optional
        Random seed for reproducibility.
    batch_size : int
        Batch size for DataLoaders (default: 32).

    Returns:
    --------
    train_tasks : list[DataLoader]
        List of training DataLoaders, one per task.
    test_tasks : list[DataLoader]
        List of test DataLoaders, one per task.

    Example:
    --------
    >>> train_tasks, test_tasks = create_split_cifar10(num_tasks=5)
    >>> print(f"Number of tasks: {len(train_tasks)}")
    Number of tasks: 5
    """
    if not AVALANCHE_AVAILABLE:
        raise ImportError("Avalanche library required. Install with: pip install avalanche-lib")

    # Create benchmark
    benchmark = SplitCIFAR10(
        n_experiences=num_tasks,
        return_task_id=return_task_id,
        fixed_class_order=fixed_class_order,
        seed=seed
    )

    # Create DataLoaders for each task
    train_tasks = []
    test_tasks = []

    for experience in benchmark.train_stream:
        train_loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        train_tasks.append(train_loader)

    for experience in benchmark.test_stream:
        test_loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        test_tasks.append(test_loader)

    return train_tasks, test_tasks


def create_split_cifar100(
    num_tasks: int = 10,
    return_task_id: bool = False,
    fixed_class_order: Optional[List[int]] = None,
    seed: Optional[int] = None,
    batch_size: int = 32
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create Split-CIFAR100 benchmark using Avalanche.

    Splits CIFAR-100 (100 classes) into multiple tasks.

    Parameters:
    -----------
    num_tasks : int
        Number of tasks (default: 10, i.e., 10 classes per task).
    return_task_id : bool
        If True, include task ID in dataset (default: False).
    fixed_class_order : list, optional
        Fixed order of classes. If None, uses default [0,1,2,...,99].
    seed : int, optional
        Random seed for reproducibility.
    batch_size : int
        Batch size for DataLoaders (default: 32).

    Returns:
    --------
    train_tasks : list[DataLoader]
        List of training DataLoaders, one per task.
    test_tasks : list[DataLoader]
        List of test DataLoaders, one per task.

    Example:
    --------
    >>> train_tasks, test_tasks = create_split_cifar100(num_tasks=10)
    >>> print(f"Number of tasks: {len(train_tasks)}")
    Number of tasks: 10
    """
    if not AVALANCHE_AVAILABLE:
        raise ImportError("Avalanche library required. Install with: pip install avalanche-lib")

    # Create benchmark
    benchmark = SplitCIFAR100(
        n_experiences=num_tasks,
        return_task_id=return_task_id,
        fixed_class_order=fixed_class_order,
        seed=seed
    )

    # Create DataLoaders for each task
    train_tasks = []
    test_tasks = []

    for experience in benchmark.train_stream:
        train_loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        train_tasks.append(train_loader)

    for experience in benchmark.test_stream:
        test_loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        test_tasks.append(test_loader)

    return train_tasks, test_tasks


def create_split_mnist_avalanche(
    num_tasks: int = 5,
    return_task_id: bool = False,
    fixed_class_order: Optional[List[int]] = None,
    seed: Optional[int] = None,
    batch_size: int = 32
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create Split-MNIST benchmark using Avalanche.

    Alternative to the custom Split-MNIST implementation, using Avalanche's
    standard benchmark. Does not include replay mechanism.

    Parameters:
    -----------
    num_tasks : int
        Number of tasks (default: 5, i.e., 2 digits per task).
    return_task_id : bool
        If True, include task ID in dataset (default: False).
    fixed_class_order : list, optional
        Fixed order of classes. If None, uses default [0,1,2,...,9].
    seed : int, optional
        Random seed for reproducibility.
    batch_size : int
        Batch size for DataLoaders (default: 32).

    Returns:
    --------
    train_tasks : list[DataLoader]
        List of training DataLoaders, one per task.
    test_tasks : list[DataLoader]
        List of test DataLoaders, one per task.

    Note:
    -----
    For experiments requiring replay, use the custom create_split_mnist()
    from test_baselines.py instead.
    """
    if not AVALANCHE_AVAILABLE:
        raise ImportError("Avalanche library required. Install with: pip install avalanche-lib")

    # Create benchmark
    benchmark = SplitMNIST(
        n_experiences=num_tasks,
        return_task_id=return_task_id,
        fixed_class_order=fixed_class_order,
        seed=seed
    )

    # Create DataLoaders for each task
    train_tasks = []
    test_tasks = []

    for experience in benchmark.train_stream:
        train_loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        train_tasks.append(train_loader)

    for experience in benchmark.test_stream:
        test_loader = DataLoader(
            experience.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        test_tasks.append(test_loader)

    return train_tasks, test_tasks


def get_benchmark_info(benchmark_name: str) -> dict:
    """
    Get information about a standard continual learning benchmark.

    Parameters:
    -----------
    benchmark_name : str
        Name of benchmark ('split_mnist', 'split_cifar10', 'split_cifar100').

    Returns:
    --------
    info : dict
        Dictionary with benchmark information.

    Example:
    --------
    >>> info = get_benchmark_info('split_cifar10')
    >>> print(info['num_classes'])
    10
    """
    benchmarks = {
        'split_mnist': {
            'name': 'Split-MNIST',
            'num_classes': 10,
            'input_channels': 1,
            'input_size': (28, 28),
            'typical_tasks': 5,
            'classes_per_task': 2,
            'dataset_size': {'train': 60000, 'test': 10000}
        },
        'split_cifar10': {
            'name': 'Split-CIFAR10',
            'num_classes': 10,
            'input_channels': 3,
            'input_size': (32, 32),
            'typical_tasks': 5,
            'classes_per_task': 2,
            'dataset_size': {'train': 50000, 'test': 10000}
        },
        'split_cifar100': {
            'name': 'Split-CIFAR100',
            'num_classes': 100,
            'input_channels': 3,
            'input_size': (32, 32),
            'typical_tasks': [10, 20],  # Common: 10 or 20 tasks
            'classes_per_task': [10, 5],
            'dataset_size': {'train': 50000, 'test': 10000}
        }
    }

    benchmark_name = benchmark_name.lower()
    if benchmark_name not in benchmarks:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Choose from {list(benchmarks.keys())}")

    return benchmarks[benchmark_name]


def print_benchmark_summary(train_tasks: List[DataLoader], test_tasks: List[DataLoader]):
    """
    Print a summary of a continual learning benchmark.

    Parameters:
    -----------
    train_tasks : list[DataLoader]
        List of training DataLoaders.
    test_tasks : list[DataLoader]
        List of test DataLoaders.

    Example:
    --------
    >>> train_tasks, test_tasks = create_split_cifar10()
    >>> print_benchmark_summary(train_tasks, test_tasks)
    """
    print("="*60)
    print("Benchmark Summary")
    print("="*60)
    print(f"Number of tasks: {len(train_tasks)}")

    total_train_samples = 0
    total_test_samples = 0

    for task_id, (train_loader, test_loader) in enumerate(zip(train_tasks, test_tasks)):
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        total_train_samples += train_size
        total_test_samples += test_size

        print(f"\nTask {task_id + 1}:")
        print(f"  Train samples: {train_size}")
        print(f"  Test samples: {test_size}")

        # Get sample to determine shape
        sample_x, sample_y = next(iter(train_loader))
        print(f"  Input shape: {tuple(sample_x.shape[1:])}")

        # Get unique labels
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.tolist())
        unique_labels = sorted(set(all_labels))
        print(f"  Classes: {unique_labels}")

    print(f"\nTotal train samples: {total_train_samples}")
    print(f"Total test samples: {total_test_samples}")
    print("="*60)
