"""
MoBSystem: High-level interface for MoB experiments.

This module provides a unified, easy-to-use interface for running MoB experiments
with automatic metric tracking, logging, and visualization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time

from .pool import ExpertPool
from .auction import PerBatchVCGAuction
from .utils import (
    setup_logging,
    set_seed,
    get_device,
    save_config,
    format_time,
    print_section_header
)


class MoBSystem:
    """
    High-level interface for MoB continual learning experiments.

    This class wraps ExpertPool and PerBatchVCGAuction with automatic:
    - Metric tracking (accuracy, forgetting, specialization)
    - Logging and checkpointing
    - Progress reporting
    - Experiment reproducibility

    Example:
    --------
    >>> config = {
    ...     'num_experts': 4,
    ...     'architecture': 'simple_cnn',
    ...     'num_classes': 10,
    ...     'alpha': 0.5,
    ...     'beta': 0.5,
    ...     'lambda_ewc': 5000,
    ...     'learning_rate': 0.001
    ... }
    >>> system = MoBSystem(config)
    >>> system.train(train_tasks)
    >>> results = system.evaluate(test_tasks)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: Optional[str] = None,
        save_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize MoB system.

        Parameters:
        -----------
        config : dict
            Configuration dictionary with experiment parameters.
        experiment_name : str, optional
            Name for this experiment (for logging/saving).
        save_dir : str, optional
            Directory to save results and checkpoints.
        device : torch.device, optional
            Device to run on. If None, auto-detects.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.config = config
        self.experiment_name = experiment_name or "mob_experiment"
        self.save_dir = Path(save_dir) if save_dir else Path("results") / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set seed if provided
        if seed is not None:
            set_seed(seed)
            self.config['seed'] = seed

        # Setup device
        self.device = device if device is not None else get_device(verbose=True)
        self.config['device'] = str(self.device)

        # Setup logging
        log_file = self.save_dir / "experiment.log"
        self.logger = setup_logging(str(log_file))

        # Log configuration
        self.logger.info(f"Initializing MoB System: {self.experiment_name}")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

        # Save configuration
        save_config(config, str(self.save_dir / "config.json"))

        # Initialize expert pool and auction
        expert_config = {
            'architecture': config.get('architecture', 'simple_cnn'),
            'num_classes': config.get('num_classes', 10),
            'input_channels': config.get('input_channels', 1),
            'alpha': config.get('alpha', 0.5),
            'beta': config.get('beta', 0.5),
            'lambda_ewc': config.get('lambda_ewc', 5000),
            'dropout': config.get('dropout', 0.5)
        }

        num_experts = config.get('num_experts', 4)
        self.pool = ExpertPool(num_experts, expert_config, device=self.device)
        self.auction = PerBatchVCGAuction(num_experts)

        # Create optimizers
        learning_rate = config.get('learning_rate', 0.001)
        self.optimizers = [
            torch.optim.Adam(expert.model.parameters(), lr=learning_rate)
            for expert in self.pool.experts
        ]

        # Metrics tracking
        self.metrics = {
            'task_accuracies': [],  # Accuracy on each task after training it
            'final_accuracies': [],  # Final accuracy on all tasks
            'forgetting': [],  # Forgetting per task
            'training_time': [],  # Time per task
            'specialization_history': [],  # Expert specialization over time
            'bid_history': [],  # Bid statistics
            'auction_history': []  # Auction outcomes
        }

        self.current_task = 0
        self.total_batches_seen = 0

    def train(
        self,
        train_tasks: List,
        num_epochs_per_task: int = 1,
        fisher_samples: int = 200,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train MoB system on a sequence of tasks.

        Parameters:
        -----------
        train_tasks : list
            List of DataLoaders, one per task.
        num_epochs_per_task : int
            Number of epochs to train on each task (default: 1).
        fisher_samples : int
            Number of samples for Fisher matrix computation (default: 200).
        verbose : bool
            Print progress information (default: True).

        Returns:
        --------
        metrics : dict
            Training metrics including task accuracies and timing.
        """
        print_section_header(f"Training MoB: {self.experiment_name}")

        total_start_time = time.time()

        for task_id, task_loader in enumerate(train_tasks):
            task_start_time = time.time()
            self.current_task = task_id

            if verbose:
                print(f"\n{'='*60}")
                print(f"Task {task_id + 1}/{len(train_tasks)}")
                print(f"{'='*60}")

            self.logger.info(f"Starting Task {task_id + 1}/{len(train_tasks)}")

            # Training loop
            epoch_losses = []
            for epoch in range(num_epochs_per_task):
                epoch_loss = 0.0
                batch_count = 0

                for x, y in task_loader:
                    # Collect bids
                    bids, bid_components = self.pool.collect_bids(x, y)

                    # Run auction
                    winner_id, payment, auction_details = self.auction.run_auction(bids)

                    # Train winner
                    metrics = self.pool.train_winner(winner_id, x, y, self.optimizers)

                    # Track metrics
                    epoch_loss += metrics.get('total_loss', 0.0)
                    batch_count += 1
                    self.total_batches_seen += 1

                    # Store bid and auction history (sample every 10 batches)
                    if self.total_batches_seen % 10 == 0:
                        self.metrics['bid_history'].append({
                            'task': task_id,
                            'batch': self.total_batches_seen,
                            'bids': bids.tolist(),
                            'winner': winner_id
                        })

                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                epoch_losses.append(avg_epoch_loss)

                if verbose and num_epochs_per_task > 1:
                    print(f"  Epoch {epoch + 1}/{num_epochs_per_task}: Loss = {avg_epoch_loss:.4f}")

            # Update Fisher matrix after task
            if verbose:
                print(f"  Updating Fisher matrices...")
            self.pool.update_after_task(task_loader, num_samples=fisher_samples)

            # Track training time
            task_time = time.time() - task_start_time
            self.metrics['training_time'].append(task_time)

            # Get expert statistics
            expert_stats = self.pool.get_statistics()
            self.metrics['specialization_history'].append({
                'task': task_id,
                'stats': expert_stats
            })

            if verbose:
                print(f"  Training time: {format_time(task_time)}")
                print(f"  Average loss: {np.mean(epoch_losses):.4f}")
                print(f"  Expert win rates: {expert_stats.get('win_rates', {})}")

            self.logger.info(f"Task {task_id + 1} completed in {format_time(task_time)}")

        total_time = time.time() - total_start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training completed in {format_time(total_time)}")
            print(f"{'='*60}\n")

        self.logger.info(f"Training completed in {format_time(total_time)}")

        return self.metrics

    def evaluate(
        self,
        test_tasks: List,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate MoB system on all tasks.

        Parameters:
        -----------
        test_tasks : list
            List of test DataLoaders, one per task.
        verbose : bool
            Print evaluation results (default: True).

        Returns:
        --------
        results : dict
            Evaluation metrics including accuracies and forgetting.
        """
        if verbose:
            print_section_header("Evaluation")

        self.logger.info("Starting evaluation on all tasks")

        final_accuracies = []

        for task_id, test_loader in enumerate(test_tasks):
            results = self.pool.evaluate_all(test_loader)
            acc = results['ensemble_accuracy']
            final_accuracies.append(acc)

            if verbose:
                print(f"  Task {task_id + 1}: {acc:.4f}")

            self.logger.info(f"Task {task_id + 1} accuracy: {acc:.4f}")

        # Compute aggregate metrics
        avg_accuracy = np.mean(final_accuracies)

        # Compute forgetting (if we have task accuracies from training)
        if self.metrics['task_accuracies']:
            forgetting_per_task = []
            for i in range(len(final_accuracies) - 1):
                task_acc = self.metrics['task_accuracies'][i] if i < len(self.metrics['task_accuracies']) else 0
                final_acc = final_accuracies[i]
                forget = max(0, task_acc - final_acc)
                forgetting_per_task.append(forget)
            avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0
        else:
            avg_forgetting = 0.0
            forgetting_per_task = []

        # Update metrics
        self.metrics['final_accuracies'] = final_accuracies
        self.metrics['avg_accuracy'] = avg_accuracy
        self.metrics['avg_forgetting'] = avg_forgetting
        self.metrics['forgetting_per_task'] = forgetting_per_task

        if verbose:
            print(f"\n  Average Accuracy: {avg_accuracy:.4f}")
            print(f"  Average Forgetting: {avg_forgetting:.4f}")

        self.logger.info(f"Average accuracy: {avg_accuracy:.4f}")
        self.logger.info(f"Average forgetting: {avg_forgetting:.4f}")

        # Save metrics
        self.save_metrics()

        return self.metrics

    def save_metrics(self, filename: Optional[str] = None):
        """
        Save metrics to JSON file.

        Parameters:
        -----------
        filename : str, optional
            Filename to save metrics. If None, uses default.
        """
        if filename is None:
            filename = "metrics.json"

        save_path = self.save_dir / filename

        # Convert to JSON-serializable format
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, list):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = value

        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        self.logger.info(f"Metrics saved to {save_path}")

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pt"):
        """
        Save model checkpoint.

        Parameters:
        -----------
        checkpoint_name : str
            Name of checkpoint file (default: "checkpoint.pt").
        """
        checkpoint_path = self.save_dir / checkpoint_name

        checkpoint = {
            'config': self.config,
            'expert_states': [expert.model.state_dict() for expert in self.pool.experts],
            'optimizer_states': [opt.state_dict() for opt in self.optimizers],
            'metrics': self.metrics,
            'current_task': self.current_task
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Parameters:
        -----------
        checkpoint_path : str
            Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load expert models
        for expert, state_dict in zip(self.pool.experts, checkpoint['expert_states']):
            expert.model.load_state_dict(state_dict)

        # Load optimizers
        for opt, state_dict in zip(self.optimizers, checkpoint['optimizer_states']):
            opt.load_state_dict(state_dict)

        # Load metrics
        self.metrics = checkpoint['metrics']
        self.current_task = checkpoint['current_task']

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def get_summary(self) -> str:
        """
        Get a formatted summary of the system and results.

        Returns:
        --------
        summary : str
            Formatted summary string.
        """
        lines = []
        lines.append("="*60)
        lines.append(f"MoB System Summary: {self.experiment_name}")
        lines.append("="*60)
        lines.append(f"\nConfiguration:")
        lines.append(f"  Num Experts: {self.config.get('num_experts', 'N/A')}")
        lines.append(f"  Architecture: {self.config.get('architecture', 'N/A')}")
        lines.append(f"  Alpha: {self.config.get('alpha', 'N/A')}")
        lines.append(f"  Beta: {self.config.get('beta', 'N/A')}")
        lines.append(f"  Lambda EWC: {self.config.get('lambda_ewc', 'N/A')}")

        if self.metrics['final_accuracies']:
            lines.append(f"\nResults:")
            lines.append(f"  Average Accuracy: {self.metrics.get('avg_accuracy', 0):.4f}")
            lines.append(f"  Average Forgetting: {self.metrics.get('avg_forgetting', 0):.4f}")
            lines.append(f"\n  Task Accuracies:")
            for i, acc in enumerate(self.metrics['final_accuracies']):
                lines.append(f"    Task {i+1}: {acc:.4f}")

        if self.metrics['training_time']:
            total_time = sum(self.metrics['training_time'])
            lines.append(f"\n  Total Training Time: {format_time(total_time)}")

        lines.append("="*60)

        return "\n".join(lines)
