"""
Experience Replay (ER) Baseline for Continual Learning

This script implements Experience Replay as a baseline for comparison with MoB.
ER is a simple yet effective approach that maintains a replay buffer of past
experiences and jointly trains on current and replay data.

References:
- Rolnick et al. (2019) "Experience Replay for Continual Learning" (NeurIPS)
  https://arxiv.org/abs/1811.11682
- Chaudhry et al. (2019) "Continual Learning with Tiny Episodic Memories"
  https://arxiv.org/abs/1902.10486
- Buzzega et al. (2020) "Rethinking Experience Replay: a Bag of Tricks"
  https://arxiv.org/abs/2010.05595

Key Algorithm:
1. Maintain a fixed-size replay buffer using reservoir sampling
2. For each batch: train on BOTH current data AND a batch from replay buffer
3. The loss is the sum of current loss and replay loss (optionally weighted)
4. After each batch, update the buffer with reservoir sampling

Key hyperparameters:
- memory_size: Total number of samples to store
- replay_batch_size: Number of samples to replay per batch
- replay_weight: Weight for replay loss (default 1.0 = equal weight)

Usage:
    python tests/run_er_baseline.py                     # Default config
    python tests/run_er_baseline.py --memory_size 512   # Larger memory
    python tests/run_er_baseline.py --epochs 4          # Match MoB epochs
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
from typing import Dict, List, Optional, Tuple
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob.models import create_model
from mob.utils import set_seed

# Import dataset creation
from tests.test_baselines import create_split_mnist


class ReplayBuffer:
    """
    Experience Replay buffer with reservoir sampling.

    Implements Vitter's reservoir sampling (1985) to maintain a uniformly
    random sample of all experiences seen so far, regardless of when they
    were encountered.

    This is the standard approach used in:
    - Rolnick et al. (2019) "Experience Replay for Continual Learning"
    - Chaudhry et al. (2019) "Continual Learning with Tiny Episodic Memories"
    """

    def __init__(self, memory_size: int = 256, device: Optional[torch.device] = None):
        """
        Initialize replay buffer.

        Parameters:
        -----------
        memory_size : int
            Maximum number of samples to store
        device : torch.device
            Device for tensors
        """
        self.memory_size = memory_size
        self.device = device if device is not None else torch.device('cpu')

        # Storage as list of (x, y) tuples
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_samples = 0

    def add_samples(self, x: torch.Tensor, y: torch.Tensor):
        """
        Add samples to buffer using reservoir sampling.

        Reservoir sampling ensures each sample has equal probability of being
        in the buffer, regardless of when it was seen.

        Parameters:
        -----------
        x : torch.Tensor
            Input batch
        y : torch.Tensor
            Target labels
        """
        batch_size = x.size(0)

        for i in range(batch_size):
            self.seen_samples += 1

            if len(self.data) < self.memory_size:
                # Buffer not full, just add
                self.data.append((x[i].cpu().clone(), y[i].cpu().clone()))
            else:
                # Reservoir sampling: replace with probability memory_size / seen_samples
                j = random.randint(0, self.seen_samples - 1)
                if j < self.memory_size:
                    self.data[j] = (x[i].cpu().clone(), y[i].cpu().clone())

    def sample_batch(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample a batch from buffer.

        Parameters:
        -----------
        batch_size : int
            Number of samples to retrieve

        Returns:
        --------
        x, y : Tuple[torch.Tensor, torch.Tensor] or (None, None)
            Sampled inputs and labels, or None if buffer is empty
        """
        if len(self.data) == 0:
            return None, None

        # Sample with replacement if batch_size > buffer size
        actual_size = min(batch_size, len(self.data))
        indices = random.sample(range(len(self.data)), actual_size)

        x_batch = torch.stack([self.data[i][0] for i in indices])
        y_batch = torch.stack([self.data[i][1] for i in indices])

        return x_batch.to(self.device), y_batch.to(self.device)

    def __len__(self) -> int:
        return len(self.data)

    def get_class_distribution(self) -> Dict[int, int]:
        """Get count of samples per class in buffer."""
        dist = {}
        for _, y in self.data:
            label = y.item()
            dist[label] = dist.get(label, 0) + 1
        return dist


class ExperienceReplay:
    """
    Experience Replay (ER) for continual learning.

    This implements the simple yet effective ER baseline:
    1. Maintain a replay buffer using reservoir sampling
    2. For each training batch, also train on a batch from replay buffer
    3. Combine losses from current and replay data

    This is considered one of the strongest simple baselines for continual
    learning, often competitive with more complex methods.
    """

    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 256,
        replay_batch_size: int = 32,
        replay_weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Experience Replay.

        Parameters:
        -----------
        model : nn.Module
            Neural network model
        memory_size : int
            Size of replay buffer
        replay_batch_size : int
            Batch size for replay samples
        replay_weight : float
            Weight for replay loss (1.0 = equal to current loss)
        device : torch.device
            Device for computation
        """
        self.device = device if device is not None else torch.device('cpu')
        self.model = model
        self.model.to(self.device)

        self.memory_size = memory_size
        self.replay_batch_size = replay_batch_size
        self.replay_weight = replay_weight

        self.buffer = ReplayBuffer(memory_size, device=self.device)

        # Statistics
        self.total_batches = 0
        self.replay_batches = 0
        self.tasks_trained = []

    def train_on_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_idx: int = 0
    ) -> Dict:
        """
        Train on a single batch with experience replay.

        Algorithm:
        1. Compute loss on current batch
        2. If buffer is not empty, compute loss on replay batch
        3. Total loss = current_loss + replay_weight * replay_loss
        4. Update parameters
        5. Add current batch to buffer (reservoir sampling)

        Parameters:
        -----------
        x : torch.Tensor
            Input batch
        y : torch.Tensor
            Target labels
        optimizer : torch.optim.Optimizer
            Optimizer
        batch_idx : int
            Batch index for logging

        Returns:
        --------
        Dict with training metrics
        """
        self.model.train()
        self.total_batches += 1

        x = x.to(self.device)
        y = y.to(self.device)

        optimizer.zero_grad()

        # =================================================================
        # STEP 1: Compute loss on current batch
        # =================================================================
        logits = self.model(x)
        current_loss = F.cross_entropy(logits, y)

        # =================================================================
        # STEP 2: Compute loss on replay batch (if buffer not empty)
        # =================================================================
        replay_loss = torch.tensor(0.0, device=self.device)
        used_replay = False

        if len(self.buffer) > 0:
            x_replay, y_replay = self.buffer.sample_batch(self.replay_batch_size)
            if x_replay is not None:
                logits_replay = self.model(x_replay)
                replay_loss = F.cross_entropy(logits_replay, y_replay)
                used_replay = True
                self.replay_batches += 1

        # =================================================================
        # STEP 3: Total loss = current + weighted replay
        # =================================================================
        total_loss = current_loss + self.replay_weight * replay_loss

        # =================================================================
        # STEP 4: Backward and update
        # =================================================================
        total_loss.backward()
        optimizer.step()

        # =================================================================
        # STEP 5: Add current batch to buffer
        # =================================================================
        self.buffer.add_samples(x, y)

        # Log first few batches
        if batch_idx < 3:
            replay_str = f", replay_loss={replay_loss.item():.4f}" if used_replay else ""
            print(f"[ER] Batch {batch_idx+1}: current_loss={current_loss.item():.4f}{replay_str}, "
                  f"total_loss={total_loss.item():.4f}, buffer={len(self.buffer)}")

        return {
            'current_loss': current_loss.item(),
            'replay_loss': replay_loss.item() if used_replay else 0.0,
            'total_loss': total_loss.item(),
            'used_replay': used_replay,
            'buffer_size': len(self.buffer)
        }

    def train_on_task(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        task_id: int,
        epochs: int = 1
    ) -> Dict:
        """
        Train on a complete task.

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
        replay_count = 0

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

                if metrics['used_replay']:
                    replay_count += 1

            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} avg_loss: {avg_loss:.4f}, buffer_size: {len(self.buffer)}")

        self.tasks_trained.append(task_id)

        # Log buffer distribution after task
        dist = self.buffer.get_class_distribution()
        print(f"  Buffer distribution: {dist}")

        return {
            'total_batches': total_batches,
            'replay_batches': replay_count,
            'replay_rate': replay_count / total_batches if total_batches > 0 else 0,
            'epoch_losses': epoch_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'buffer_distribution': dist
        }

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
        """Count total parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params
        }


def run_experiment(train_tasks, test_tasks, config):
    """Run Experience Replay experiment."""

    print("\n" + "="*70)
    print("Experience Replay (ER) Experiment")
    print("="*70)

    device = torch.device(config['device'])

    # Create wide model (width_multiplier=2 to match MoB's ~1.7M params)
    model = create_model(
        architecture='simple_cnn',
        num_classes=10,
        input_channels=1,
        dropout=0.5,
        width_multiplier=config.get('width_multiplier', 2)
    )

    # Create ER
    er = ExperienceReplay(
        model=model,
        memory_size=config['memory_size'],
        replay_batch_size=config['replay_batch_size'],
        replay_weight=config.get('replay_weight', 1.0),
        device=device
    )

    # Force lazy initialization of FC layers
    print("  Initializing lazy layers...")
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    model(dummy_input)

    # Print parameter counts
    param_counts = er.count_parameters()
    print(f"\nParameter Counts:")
    print(f"  Total: {param_counts['total_params']:,}")
    print(f"  (MoB has 4 experts with same total params)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

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

        if len(er.buffer) > 0:
            print(f"  Replay buffer: {len(er.buffer)} samples from previous tasks")

        metrics = er.train_on_task(
            task_loader,
            optimizer,
            task_id,
            epochs=epochs_per_task
        )

        print(f"\n  Task {task_id+1} completed: {metrics['total_batches']} batches")
        print(f"  Replay batches: {metrics['replay_batches']} ({metrics['replay_rate']*100:.1f}%)")

        # Evaluate on current task
        results = er.evaluate(test_tasks[task_id])
        task_accuracies.append(results['accuracy'])
        print(f"  Task {task_id+1} accuracy: {results['accuracy']:.4f}")

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    for task_id, test_loader in enumerate(test_tasks):
        results = er.evaluate(test_loader)
        acc = results['accuracy']
        final_accuracies.append(acc)

        status = "OK" if acc > 0.5 else "FAIL"
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
    print(f"  Total Replay Batches: {er.replay_batches}")
    print(f"  Tasks retained (>50%): {sum(1 for a in final_accuracies if a > 0.5)}/{len(final_accuracies)}")
    print(f"  Final buffer size: {len(er.buffer)} samples")
    print(f"  Buffer distribution: {er.buffer.get_class_distribution()}")

    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': avg_forgetting,
        'replay_batches': er.replay_batches,
        'param_counts': param_counts,
        'buffer_distribution': er.buffer.get_class_distribution()
    }


def main():
    parser = argparse.ArgumentParser(description='Run Experience Replay experiment')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width_multiplier', type=int, default=2,
                        help='Width multiplier for CNN (2 = same params as MoB ~1.7M)')
    parser.add_argument('--memory_size', type=int, default=256,
                        help='Total replay buffer size')
    parser.add_argument('--replay_batch_size', type=int, default=32,
                        help='Batch size for replay samples')
    parser.add_argument('--replay_weight', type=float, default=1.0,
                        help='Weight for replay loss (1.0 = equal to current)')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_results', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)

    config = {
        'num_tasks': 5,
        'width_multiplier': args.width_multiplier,
        'memory_size': args.memory_size,
        'replay_batch_size': args.replay_batch_size,
        'replay_weight': args.replay_weight,
        'learning_rate': args.learning_rate,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs
    }

    print("="*70)
    print(f"Experience Replay Experiment (Seed {args.seed})")
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
            'replay_batches': results['replay_batches'],
            'param_counts': results['param_counts'],
            'buffer_distribution': {str(k): v for k, v in results['buffer_distribution'].items()}
        }
        filename = f"results/er_seed_{args.seed}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n Results saved to: {filename}")


if __name__ == '__main__':
    main()
