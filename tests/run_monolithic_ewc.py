"""
Monolithic EWC Baseline - Single Large Model for MoB Comparison

This script implements a fair monolithic baseline with the SAME total parameter count
as the 4-expert MoB system. Uses a wide SimpleCNN with width_multiplier=4.

Key design decisions:
1. Single CNN with 4× the width of each MoB expert
2. Same total parameter count as MoB (4 experts × params_per_expert)
3. EWC regularization to prevent catastrophic forgetting
4. Same training hyperparameters as MoB

Expected behavior:
- Should show some forgetting despite EWC (capacity bottleneck)
- Tests whether multi-expert architecture provides value
- Strong baseline: all parameters can be applied to any task

Usage:
    python tests/run_monolithic_ewc.py                    # Default config
    python tests/run_monolithic_ewc.py --lambda_ewc 10    # Match MoB lambda
    python tests/run_monolithic_ewc.py --epochs 4         # Match MoB epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob.models import create_model
from mob.bidding import EWCForgettingEstimator
from mob.utils import set_seed

# Import dataset creation
from tests.test_baselines import create_split_mnist


class MonolithicEWC:
    """
    Monolithic model with EWC regularization for continual learning.
    
    This is the strongest single-model baseline:
    - Uses all capacity for every task (no routing overhead)
    - EWC prevents catastrophic forgetting
    - Same total parameters as 4-expert MoB system
    
    The key question this baseline answers:
    "Is the multi-expert architecture actually beneficial, or would a single
    large model with EWC perform just as well?"
    """
    
    def __init__(
        self,
        model_config: Dict,
        lambda_ewc: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the monolithic model.
        
        Parameters:
        -----------
        model_config : Dict
            Configuration for creating the model (architecture, num_classes, etc.)
        lambda_ewc : float
            EWC regularization strength
        device : torch.device
            Device for computation
        """
        self.device = device if device is not None else torch.device('cpu')
        self.lambda_ewc = lambda_ewc
        
        # Create wide model (4× width to match 4-expert total params)
        self.model = create_model(
            architecture=model_config['architecture'],
            num_classes=model_config['num_classes'],
            input_channels=model_config.get('input_channels', 1),
            dropout=model_config.get('dropout', 0.5),
            width_multiplier=model_config.get('width_multiplier', 4)
        )
        self.model.to(self.device)
        
        # EWC estimator
        self.ewc_estimator = EWCForgettingEstimator(
            self.model,
            lambda_ewc=lambda_ewc,
            device=self.device
        )
        
        # Statistics
        self.total_batches_trained = 0
        self.tasks_trained = []
    
    def train_on_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_idx: int = 0
    ) -> Dict:
        """
        Train on a single batch with EWC regularization.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input batch
        y : torch.Tensor
            Target labels
        optimizer : torch.optim.Optimizer
            Optimizer for the model
        batch_idx : int
            Batch index for logging
            
        Returns:
        --------
        Dict with loss components
        """
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)
        
        optimizer.zero_grad()
        logits = self.model(x)
        
        # Task loss
        task_loss = F.cross_entropy(logits, y)
        
        # EWC penalty
        ewc_penalty = self.ewc_estimator.penalty()
        
        # Total loss
        total_loss = task_loss + ewc_penalty
        
        # Log first few batches
        if batch_idx < 3:
            print(f"[Monolithic EWC] Batch {batch_idx+1}: "
                  f"task_loss={task_loss.item():.4f}, "
                  f"ewc_penalty={ewc_penalty.item():.4f}, "
                  f"total_loss={total_loss.item():.4f}")
        
        total_loss.backward()
        optimizer.step()
        
        self.total_batches_trained += 1
        
        return {
            'task_loss': task_loss.item(),
            'ewc_penalty': ewc_penalty.item(),
            'total_loss': total_loss.item()
        }
    
    def train_on_task(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        task_id: int,
        epochs: int = 1
    ) -> Dict:
        """
        Train on a complete task for multiple epochs.
        
        Parameters:
        -----------
        dataloader : DataLoader
            Task data
        optimizer : torch.optim.Optimizer
            Model optimizer
        task_id : int
            Task identifier
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        Dict with training statistics
        """
        total_batches = 0
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (x, y) in enumerate(pbar):
                metrics = self.train_on_batch(
                    x, y, optimizer,
                    batch_idx if epoch == 0 else 999  # Only log first epoch
                )
                epoch_loss += metrics['total_loss']
                batch_count += 1
                total_batches += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} avg_loss: {avg_loss:.4f}")
        
        self.tasks_trained.append(task_id)
        
        return {
            'total_batches': total_batches,
            'epoch_losses': epoch_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else 0
        }
    
    def update_fisher_after_task(self, task_loader, num_samples: int = 200):
        """
        Update Fisher information after task completion.
        Same approach as MoB for fair comparison.
        """
        print(f"  Updating Fisher for monolithic model")
        self.ewc_estimator.update_fisher(task_loader, num_samples)
    
    def evaluate(self, dataloader) -> Dict:
        """
        Evaluate model accuracy on a dataset.
        
        Parameters:
        -----------
        dataloader : DataLoader
            Evaluation data
            
        Returns:
        --------
        Dict with accuracy and loss metrics
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(x)
                loss = F.cross_entropy(logits, y, reduction='sum')
                
                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                total_loss += loss.item()
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': total_samples
        }
    
    def count_parameters(self) -> Dict:
        """Count total parameters for comparison with MoB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params
        }


def run_experiment(train_tasks, test_tasks, config):
    """Run Monolithic EWC experiment."""
    
    print("\n" + "="*70)
    print("Monolithic EWC Experiment (Fair Baseline for MoB)")
    print("="*70)
    
    device = torch.device(config['device'])
    
    # Model configuration with 4× width to match MoB total params
    model_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'dropout': 0.5,
        'width_multiplier': config.get('width_multiplier', 4)
    }
    
    # Create model
    monolith = MonolithicEWC(
        model_config=model_config,
        lambda_ewc=config['lambda_ewc'],
        device=device
    )
    
    # Print parameter counts for comparison with MoB
    param_counts = monolith.count_parameters()
    print(f"\nParameter Counts:")
    print(f"  Total: {param_counts['total_params']:,}")
    print(f"  Trainable: {param_counts['trainable_params']:,}")
    print(f"  (MoB has {config.get('num_experts', 4)} experts, same total params)")
    
    # Optimizer (same LR as MoB)
    optimizer = torch.optim.Adam(monolith.model.parameters(), lr=config['learning_rate'])
    
    # Metrics
    task_accuracies = []
    final_accuracies = []
    
    epochs_per_task = config.get('epochs_per_task', 4)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n{'='*70}")
        print(f"TASK {task_id + 1}/{len(train_tasks)} (Digits {task_id*2}, {task_id*2+1})")
        print(f"{'='*70}")
        
        metrics = monolith.train_on_task(
            task_loader,
            optimizer,
            task_id,
            epochs=epochs_per_task
        )
        
        print(f"\n  Task {task_id+1} completed: {metrics['total_batches']} batches")
        
        # Update Fisher after task (same as MoB)
        print(f"\n  Updating EWC Fisher information...")
        monolith.update_fisher_after_task(task_loader, num_samples=200)
        
        # Evaluate on current task
        results = monolith.evaluate(test_tasks[task_id])
        task_accuracies.append(results['accuracy'])
        print(f"  Task {task_id+1} accuracy: {results['accuracy']:.4f}")
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    for task_id, test_loader in enumerate(test_tasks):
        results = monolith.evaluate(test_loader)
        acc = results['accuracy']
        final_accuracies.append(acc)
        
        status = "✓" if acc > 0.5 else "✗"
        print(f"  Task {task_id+1} (digits {task_id*2},{task_id*2+1}): {acc:.4f} {status}")
    
    # Metrics
    avg_accuracy = np.mean(final_accuracies)
    forgetting_per_task = [
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]
    avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    print(f"  Average Forgetting: {avg_forgetting:.4f}")
    print(f"  Tasks retained (>50%): {sum(1 for a in final_accuracies if a > 0.5)}/{len(final_accuracies)}")
    
    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': avg_forgetting,
        'param_counts': param_counts
    }


def main():
    parser = argparse.ArgumentParser(description='Run Monolithic EWC experiment')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width_multiplier', type=int, default=4,
                        help='Width multiplier for CNN (4 = same params as 4 experts)')
    parser.add_argument('--lambda_ewc', type=float, default=10.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_results', action='store_true')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = {
        'num_tasks': 5,
        'num_experts': 4,  # For reference (how many experts MoB uses)
        'width_multiplier': args.width_multiplier,
        'lambda_ewc': args.lambda_ewc,
        'learning_rate': args.learning_rate,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs
    }
    
    print("="*70)
    print(f"Monolithic EWC Experiment (Seed {args.seed})")
    print("="*70)
    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print("\nCreating Split-MNIST datasets...")
    train_tasks = create_split_mnist(config['num_tasks'], train=True, batch_size=config['batch_size'])
    test_tasks = create_split_mnist(config['num_tasks'], train=False, batch_size=config['batch_size'])
    
    results = run_experiment(train_tasks, test_tasks, config)
    
    # Save
    if args.save_results:
        os.makedirs('results', exist_ok=True)
        summary = {
            'seed': args.seed,
            'config': config,
            'task_accuracies': results['task_accuracies'],
            'final_accuracies': results['final_accuracies'],
            'avg_accuracy': results['avg_accuracy'],
            'forgetting': results['forgetting'],
            'param_counts': results['param_counts']
        }
        filename = f"results/monolithic_ewc_seed_{args.seed}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Results saved to: {filename}")


if __name__ == '__main__':
    main()
