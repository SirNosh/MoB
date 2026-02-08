"""
A-GEM (Averaged Gradient Episodic Memory) Baseline for Continual Learning

This script implements A-GEM (Chaudhry et al., ICLR 2019) as a baseline for
comparison with MoB. A-GEM prevents catastrophic forgetting by:
1. Maintaining an episodic memory of samples from previous tasks
2. Projecting gradients to avoid increasing loss on past tasks

Paper: "Efficient Lifelong Learning with A-GEM"
       https://arxiv.org/abs/1812.00420

Key properties:
- Memory efficient (small episodic buffer vs storing all past data)
- Computationally efficient (simple gradient projection vs QP in GEM)
- No task oracle needed at inference
- Same parameter count as MoB for fair comparison

Usage:
    python tests/run_agem_baseline.py                    # Default config
    python tests/run_agem_baseline.py --epochs 4         # Match MoB epochs
    python tests/run_agem_baseline.py --memory_size 512  # Larger memory
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


class EpisodicMemory:
    """
    Episodic memory buffer for A-GEM with reservoir sampling.
    
    Stores a fixed number of samples from past tasks using reservoir sampling,
    which ensures each sample has equal probability of being in the buffer.
    
    Reference: Vitter, "Random Sampling with a Reservoir" (1985)
    """
    
    def __init__(self, memory_size: int = 256, device: Optional[torch.device] = None):
        """
        Initialize episodic memory.
        
        Parameters:
        -----------
        memory_size : int
            Maximum number of samples to store
        device : torch.device
            Device for tensors
        """
        self.memory_size = memory_size
        self.device = device if device is not None else torch.device('cpu')
        
        # Storage
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.seen_samples = 0
    
    def add_samples(self, x: torch.Tensor, y: torch.Tensor):
        """
        Add samples to memory using reservoir sampling.
        
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
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a batch from memory.
        
        Parameters:
        -----------
        batch_size : int
            Number of samples to retrieve
            
        Returns:
        --------
        x, y : Tuple[torch.Tensor, torch.Tensor]
            Sampled inputs and labels
        """
        if len(self.data) == 0:
            return None, None
        
        # Sample with replacement if batch_size > memory size
        indices = random.choices(range(len(self.data)), k=min(batch_size, len(self.data)))
        
        x_batch = torch.stack([self.data[i][0] for i in indices])
        y_batch = torch.stack([self.data[i][1] for i in indices])
        
        return x_batch.to(self.device), y_batch.to(self.device)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_task_distribution(self) -> Dict[int, int]:
        """Get count of samples per class in memory."""
        dist = {}
        for _, y in self.data:
            label = y.item()
            dist[label] = dist.get(label, 0) + 1
        return dist


class AGEM:
    """
    Averaged Gradient Episodic Memory (A-GEM) for continual learning.
    
    A-GEM modifies standard SGD by projecting the gradient to prevent
    increasing loss on previous tasks. Uses episodic memory to store
    samples from past tasks.
    
    Key algorithm:
    1. Compute gradient g on current batch
    2. Sample from episodic memory, compute reference gradient g_ref
    3. If g · g_ref < 0: project g to not increase memory loss
       g_projected = g - (g · g_ref / ||g_ref||²) × g_ref
    4. Update parameters with projected gradient
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 256,
        memory_batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        """
        Initialize A-GEM.
        
        Parameters:
        -----------
        model : nn.Module
            Neural network model
        memory_size : int
            Size of episodic memory buffer
        memory_batch_size : int
            Batch size for computing reference gradient
        device : torch.device
            Device for computation
        """
        self.device = device if device is not None else torch.device('cpu')
        self.model = model
        self.model.to(self.device)
        
        self.memory_size = memory_size
        self.memory_batch_size = memory_batch_size
        self.memory = EpisodicMemory(memory_size, device=self.device)
        
        # Statistics
        self.total_batches = 0
        self.projection_count = 0
        self.tasks_trained = []
    
    def _flatten_gradients(self) -> torch.Tensor:
        """Flatten all gradients into a single vector."""
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads) if grads else None
    
    def _set_gradients(self, flat_grad: torch.Tensor):
        """Set model gradients from a flattened vector."""
        offset = 0
        for param in self.model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grad[offset:offset + numel].view_as(param.grad))
                offset += numel
    
    def _compute_reference_gradient(self) -> Optional[torch.Tensor]:
        """
        Compute reference gradient from episodic memory.
        
        Returns:
        --------
        g_ref : torch.Tensor or None
            Flattened reference gradient, or None if memory is empty
        """
        if len(self.memory) == 0:
            return None
        
        x_mem, y_mem = self.memory.sample_batch(self.memory_batch_size)
        if x_mem is None:
            return None
        
        self.model.zero_grad()
        logits = self.model(x_mem)
        loss = F.cross_entropy(logits, y_mem)
        loss.backward()
        
        g_ref = self._flatten_gradients()
        return g_ref
    
    def _project_gradient(self, g: torch.Tensor, g_ref: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Project gradient g to not increase loss on reference gradient direction.
        
        A-GEM projection:
        If g · g_ref < 0:
            g_projected = g - (g · g_ref / ||g_ref||²) × g_ref
        Else:
            g_projected = g (no projection needed)
        
        Parameters:
        -----------
        g : torch.Tensor
            Current task gradient (flattened)
        g_ref : torch.Tensor
            Reference gradient from memory (flattened)
            
        Returns:
        --------
        g_projected : torch.Tensor
            Projected gradient
        was_projected : bool
            Whether projection was applied
        """
        dot_product = torch.dot(g, g_ref)
        
        if dot_product < 0:
            # Project: g - (g·g_ref / ||g_ref||²) × g_ref
            ref_norm_sq = torch.dot(g_ref, g_ref)
            if ref_norm_sq > 1e-10:  # Avoid division by zero
                projection = (dot_product / ref_norm_sq) * g_ref
                g_projected = g - projection
                return g_projected, True
        
        return g, False
    
    def train_on_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_idx: int = 0
    ) -> Dict:
        """
        Train on a single batch with A-GEM gradient projection.
        
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
        
        # Step 1: Compute gradient on current batch
        optimizer.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        
        g_current = self._flatten_gradients()
        
        # Step 2: If memory is not empty, compute reference gradient and project
        was_projected = False
        if len(self.memory) > 0 and g_current is not None:
            # Save current gradient
            g_current_clone = g_current.clone()
            
            # Compute reference gradient from memory
            g_ref = self._compute_reference_gradient()
            
            if g_ref is not None:
                # Step 3: Project if needed
                g_projected, was_projected = self._project_gradient(g_current_clone, g_ref)
                
                if was_projected:
                    self.projection_count += 1
                    # Restore projected gradient
                    self._set_gradients(g_projected)
                else:
                    # Restore original gradient (may have been overwritten)
                    self._set_gradients(g_current_clone)
        
        # Step 4: Update parameters
        optimizer.step()
        
        # Log first few batches
        if batch_idx < 3:
            proj_str = " [PROJECTED]" if was_projected else ""
            print(f"[A-GEM] Batch {batch_idx+1}: loss={loss.item():.4f}{proj_str}")
        
        return {
            'loss': loss.item(),
            'projected': was_projected
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
        projections = 0
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (x, y) in enumerate(pbar):
                metrics = self.train_on_batch(
                    x, y, optimizer,
                    batch_idx if epoch == 0 else 999
                )
                epoch_loss += metrics['loss']
                batch_count += 1
                total_batches += 1
                
                if metrics['projected']:
                    projections += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} avg_loss: {avg_loss:.4f}")
        
        self.tasks_trained.append(task_id)
        
        return {
            'total_batches': total_batches,
            'projections': projections,
            'projection_rate': projections / total_batches if total_batches > 0 else 0,
            'epoch_losses': epoch_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else 0
        }
    
    def update_memory_after_task(self, dataloader, samples_per_task: int = None):
        """
        Add samples from completed task to episodic memory.
        
        Parameters:
        -----------
        dataloader : DataLoader
            Task data to sample from
        samples_per_task : int
            Number of samples to add (None = use natural reservoir sampling)
        """
        samples_added = 0
        for x, y in dataloader:
            self.memory.add_samples(x, y)
            samples_added += x.size(0)
            
            if samples_per_task and samples_added >= samples_per_task:
                break
        
        print(f"  Memory updated: {len(self.memory)} samples total")
        print(f"  Class distribution: {self.memory.get_task_distribution()}")
    
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
    """Run A-GEM experiment."""
    
    print("\n" + "="*70)
    print("A-GEM (Averaged Gradient Episodic Memory) Experiment")
    print("="*70)
    
    device = torch.device(config['device'])
    
    # Create wide model (width_multiplier=4 to match MoB's 4-expert total params)
    model = create_model(
        architecture='simple_cnn',
        num_classes=10,
        input_channels=1,
        dropout=0.5,
        width_multiplier=config.get('width_multiplier', 2) # 2 matches MoB (1.7M params)
    )
    
    # Create A-GEM
    agem = AGEM(
        model=model,
        memory_size=config['memory_size'],
        memory_batch_size=config['memory_batch_size'],
        device=device
    )
    
    # Force lazy initialization of FC layers
    print("  Initializing lazy layers...")
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    model(dummy_input)
    
    # Print parameter counts
    param_counts = agem.count_parameters()
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
        
        if len(agem.memory) > 0:
            print(f"  Episodic memory: {len(agem.memory)} samples from previous tasks")
        
        metrics = agem.train_on_task(
            task_loader,
            optimizer,
            task_id,
            epochs=epochs_per_task
        )
        
        print(f"\n  Task {task_id+1} completed: {metrics['total_batches']} batches")
        print(f"  Gradient projections: {metrics['projections']} ({metrics['projection_rate']*100:.1f}%)")
        
        # Update episodic memory with samples from this task
        print(f"\n  Adding task samples to episodic memory...")
        agem.update_memory_after_task(task_loader)
        
        # Evaluate on current task
        results = agem.evaluate(test_tasks[task_id])
        task_accuracies.append(results['accuracy'])
        print(f"  Task {task_id+1} accuracy: {results['accuracy']:.4f}")
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    for task_id, test_loader in enumerate(test_tasks):
        results = agem.evaluate(test_loader)
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
    print(f"  Total Gradient Projections: {agem.projection_count}")
    print(f"  Tasks retained (>50%): {sum(1 for a in final_accuracies if a > 0.5)}/{len(final_accuracies)}")
    print(f"  Memory size: {len(agem.memory)} samples")
    print(f"  Memory distribution: {agem.memory.get_task_distribution()}")
    
    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy': avg_accuracy,
        'forgetting': avg_forgetting,
        'projection_count': agem.projection_count,
        'param_counts': param_counts,
        'memory_distribution': agem.memory.get_task_distribution()
    }


def main():
    parser = argparse.ArgumentParser(description='Run A-GEM experiment')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width_multiplier', type=int, default=2,
                        help='Width multiplier for CNN (2 = same params as MoB ~1.7M)')
    parser.add_argument('--memory_size', type=int, default=256,
                        help='Total episodic memory size')
    parser.add_argument('--memory_batch_size', type=int, default=32,
                        help='Batch size for computing reference gradient')
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
        'memory_batch_size': args.memory_batch_size,
        'learning_rate': args.learning_rate,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs
    }
    
    print("="*70)
    print(f"A-GEM Experiment (Seed {args.seed})")
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
            'projection_count': results['projection_count'],
            'param_counts': results['param_counts'],
            'memory_distribution': {str(k): v for k, v in results['memory_distribution'].items()}
        }
        filename = f"results/agem_seed_{args.seed}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Results saved to: {filename}")


if __name__ == '__main__':
    main()
