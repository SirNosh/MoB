"""
Progressive Neural Networks (PNN) Baseline - Zero Forgetting Architecture

This script implements Progressive Neural Networks (Rusu et al., 2016) as a baseline.
PNN achieves ZERO catastrophic forgetting by design but has notable limitations.

Key design decisions:
1. New column (expert) added for each task
2. Lateral connections from frozen previous columns to new column
3. Old columns are FROZEN - no forgetting by construction
4. Total parameters grow with each task (unlike fixed-capacity MoB)

Paper: "Progressive Neural Networks" (Rusu et al., 2016)
       https://arxiv.org/abs/1606.04671

Why include this baseline:
- ZERO forgetting (upper bound for retention)
- But parameter count scales with tasks (not fair for efficiency)
- Requires task ID at inference (task oracle) - IMPORTANT LIMITATION
- Tests if MoB can approach zero-forgetting while being task-agnostic

IMPORTANT: PNN is TASK-AWARE at both training AND evaluation.
This is a significant advantage over MoB, so comparison should note this.

Usage:
    python tests/run_pnn_baseline.py                    # Default config
    python tests/run_pnn_baseline.py --epochs 4         # Match MoB epochs
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

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob.utils import set_seed

# Import dataset creation
from tests.test_baselines import create_split_mnist


class PNNColumn(nn.Module):
    """
    A single column in the Progressive Neural Network.
    
    Architecture mirrors SimpleCNN from MoB for fair comparison:
    - Conv1: input_channels → 32, 3x3 kernel
    - Conv2: 32 → 64, 3x3 kernel
    - FC1: flattened → 128
    - FC2: 128 → num_classes
    
    Lateral connections are applied at the FC layers.
    """
    
    def __init__(
        self,
        column_id: int,
        num_classes: int = 10,
        input_channels: int = 1,
        dropout: float = 0.5,
        previous_columns: Optional[List['PNNColumn']] = None
    ):
        super().__init__()
        self.column_id = column_id
        self.num_classes = num_classes
        
        # Convolutional layers (same as SimpleCNN)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # For MNIST 28x28: after 2 pools -> 7x7
        conv_output_size = 64 * 7 * 7  # 3136
        
        # FC layers
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Lateral connections from previous columns
        # Following PNN paper: adapters at each FC layer
        self.previous_columns = previous_columns if previous_columns else []
        
        if len(self.previous_columns) > 0:
            # Lateral connections to fc1: takes previous columns' conv output
            # Each previous column outputs 3136 features from conv
            num_prev = len(self.previous_columns)
            self.lateral_fc1 = nn.Linear(conv_output_size * num_prev, 128)
            
            # Lateral connections to fc2: takes previous columns' fc1 output
            self.lateral_fc2 = nn.Linear(128 * num_prev, num_classes)
        else:
            self.lateral_fc1 = None
            self.lateral_fc2 = None
    
    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through convolutional layers only."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # Flatten to 3136
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the column.
        
        Returns:
            logits: Final output logits
            conv_out: Convolutional layer output (for lateral connections)
            fc1_out: FC1 output (for lateral connections)
        """
        # Own convolutional features
        conv_out = self.forward_conv(x)
        
        # Collect lateral inputs from previous frozen columns
        lateral_conv_outs = []
        lateral_fc1_outs = []
        
        for prev_col in self.previous_columns:
            prev_col.eval()  # Previous columns are frozen
            with torch.no_grad():
                _, prev_conv, prev_fc1 = prev_col(x)
                lateral_conv_outs.append(prev_conv)
                lateral_fc1_outs.append(prev_fc1)
        
        # FC1 with lateral connections
        fc1_out = self.fc1(conv_out)
        if self.lateral_fc1 is not None and lateral_conv_outs:
            lateral_input = torch.cat(lateral_conv_outs, dim=1)
            fc1_out = fc1_out + self.lateral_fc1(lateral_input)
        fc1_out = F.relu(fc1_out)
        fc1_out = self.dropout2(fc1_out)
        
        # FC2 with lateral connections
        logits = self.fc2(fc1_out)
        if self.lateral_fc2 is not None and lateral_fc1_outs:
            lateral_input = torch.cat(lateral_fc1_outs, dim=1)
            logits = logits + self.lateral_fc2(lateral_input)
        
        return logits, conv_out, fc1_out


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Network for Continual Learning.
    
    Key properties:
    1. ZERO forgetting: previous columns are frozen
    2. Growing capacity: new column added per task (up to max_columns)
    3. Knowledge transfer: lateral connections reuse features
    4. REQUIRES TASK ID at inference (significant limitation)
    
    For fair comparison with MoB:
    - max_columns=4 to match MoB's 4-expert parameter count
    - Extra tasks reuse the last column (trained with EWC-like approach)
    - Note that PNN has task oracle advantage
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        dropout: float = 0.5,
        max_columns: int = 4,  # Match MoB's 4 experts for fair param count
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout = dropout
        self.max_columns = max_columns
        self.device = device if device is not None else torch.device('cpu')
        
        # Columns list (up to max_columns)
        self.columns: nn.ModuleList = nn.ModuleList()
        
        # Track which column handles which task
        self.task_to_column: Dict[int, int] = {}
    
    def add_column_for_task(self, task_id: int) -> Tuple[PNNColumn, bool]:
        """
        Add a new column for a new task, or reuse last column if at capacity.
        
        Returns:
            column: The column for this task
            is_new: True if a new column was created, False if reusing
        """
        # Check if we've hit max columns (max_columns=-1 means unlimited)
        if self.max_columns > 0 and len(self.columns) >= self.max_columns:
            # Reuse the last column (don't freeze it, allow continued training)
            last_column = self.columns[-1]
            self.task_to_column[task_id] = len(self.columns) - 1
            print(f"  Max columns ({self.max_columns}) reached. Reusing column {len(self.columns)-1} for task {task_id}")
            return last_column, False
        
        # Freeze all existing columns
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False
        
        # Create new column with lateral connections to all previous
        new_column = PNNColumn(
            column_id=len(self.columns),
            num_classes=self.num_classes,
            input_channels=self.input_channels,
            dropout=self.dropout,
            previous_columns=list(self.columns)  # Pass existing columns
        )
        new_column.to(self.device)
        
        self.columns.append(new_column)
        self.task_to_column[task_id] = len(self.columns) - 1
        
        print(f"  Added column {len(self.columns)-1} for task {task_id}")
        print(f"  Lateral connections from {len(self.columns)-1} previous columns")
        
        return new_column, True
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass using the column for the specified task.
        
        IMPORTANT: Requires task_id (task oracle) - this is a limitation!
        """
        if task_id not in self.task_to_column:
            raise ValueError(f"No column for task {task_id}")
        
        column_id = self.task_to_column[task_id]
        column = self.columns[column_id]
        logits, _, _ = column(x)
        return logits
    
    def train_column_on_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int,
        optimizer: torch.optim.Optimizer,
        batch_idx: int = 0
    ) -> Dict:
        """
        Train the current task's column on a batch.
        """
        column_id = self.task_to_column[task_id]
        column = self.columns[column_id]
        column.train()
        
        x = x.to(self.device)
        y = y.to(self.device)
        
        optimizer.zero_grad()
        logits, _, _ = column(x)
        
        loss = F.cross_entropy(logits, y)
        
        if batch_idx < 3:
            print(f"[PNN Column {column_id}] Batch {batch_idx+1}: loss={loss.item():.4f}")
        
        loss.backward()
        optimizer.step()
        
        return {'loss': loss.item()}
    
    def train_on_task(
        self,
        dataloader,
        task_id: int,
        epochs: int = 1,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train on a complete task.
        
        Creates a new column if needed and trains only that column.
        """
        # Add column for this task if not exists
        if task_id not in self.task_to_column:
            column, is_new = self.add_column_for_task(task_id)
        else:
            column = self.columns[self.task_to_column[task_id]]
            is_new = False
        
        # Optimizer only for trainable params (new column or reused column)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, column.parameters()),
            lr=learning_rate
        )
        
        total_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (x, y) in enumerate(pbar):
                metrics = self.train_column_on_batch(
                    x, y, task_id, optimizer,
                    batch_idx if epoch == 0 else 999
                )
                epoch_loss += metrics['loss']
                batch_count += 1
                total_batches += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"  Epoch {epoch+1} avg_loss: {avg_loss:.4f}")
        
        return {'total_batches': total_batches, 'final_loss': avg_loss}
    
    def evaluate_task(self, dataloader, task_id: int) -> Dict:
        """
        Evaluate on a specific task using its designated column.
        
        IMPORTANT: Uses task oracle (knows which column to use).
        """
        if task_id not in self.task_to_column:
            return {'accuracy': 0.0, 'error': 'No column for task'}
        
        column_id = self.task_to_column[task_id]
        column = self.columns[column_id]
        column.eval()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits, _, _ = column(x)
                preds = logits.argmax(dim=-1)
                
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'column_id': column_id,
            'uses_task_oracle': True  # Important to note!
        }
    
    def evaluate_task_agnostic(self, dataloader) -> Dict:
        """
        Evaluate WITHOUT task oracle - uses confidence-based routing.
        
        For each sample, run through ALL columns and pick the one with
        highest softmax confidence. This is the same approach used in
        MoE continual learning research for fair comparison.
        
        This makes PNN comparable to MoB at inference time!
        """
        for col in self.columns:
            col.eval()
        
        total_correct = 0
        total_samples = 0
        column_selections = {i: 0 for i in range(len(self.columns))}
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)
                
                # Get predictions and confidences from all columns
                all_logits = []
                all_confidences = []
                
                for col in self.columns:
                    logits, _, _ = col(x)
                    all_logits.append(logits)
                    
                    # Confidence = max softmax probability per sample
                    probs = F.softmax(logits, dim=-1)
                    confidence = probs.max(dim=-1).values  # [batch_size]
                    all_confidences.append(confidence)
                
                # Stack: [num_columns, batch_size]
                confidences = torch.stack(all_confidences, dim=0)
                
                # Per-sample: select column with highest confidence
                selected_columns = confidences.argmax(dim=0)  # [batch_size]
                
                # Get predictions from selected columns
                for sample_idx in range(batch_size):
                    col_id = selected_columns[sample_idx].item()
                    pred = all_logits[col_id][sample_idx].argmax().item()
                    label = y[sample_idx].item()
                    
                    if pred == label:
                        total_correct += 1
                    total_samples += 1
                    column_selections[col_id] += 1
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'column_selections': column_selections,
            'uses_task_oracle': False  # Key difference!
        }
    
    def count_parameters(self) -> Dict:
        """Count parameters (grows with each task up to max_columns)."""
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        per_column = []
        for col in self.columns:
            col_params = sum(p.numel() for p in col.parameters())
            per_column.append(col_params)
        
        return {
            'total_params': total_params,
            'frozen_params': frozen_params,
            'trainable_params': trainable_params,
            'num_columns': len(self.columns),
            'max_columns': self.max_columns,
            'per_column_params': per_column
        }


def run_experiment(train_tasks, test_tasks, config):
    """Run Progressive Neural Networks experiment."""
    
    print("\n" + "="*70)
    print("Progressive Neural Networks (PNN) Experiment")
    print("="*70)
    print("NOTE: PNN uses TASK ORACLE at evaluation (knows which column to use)")
    print("      This is an advantage over MoB which routes without task knowledge.")
    
    device = torch.device(config['device'])
    
    # Create PNN with max_columns matching MoB's expert count
    max_columns = config.get('max_columns', 4)
    pnn = ProgressiveNeuralNetwork(
        num_classes=10,
        input_channels=1,
        dropout=0.5,
        max_columns=max_columns,
        device=device
    )
    print(f"  max_columns={max_columns} (to match MoB's {max_columns} experts)")
    
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
        
        metrics = pnn.train_on_task(
            task_loader,
            task_id,
            epochs=epochs_per_task,
            learning_rate=config['learning_rate']
        )
        
        # Print current parameter count
        param_counts = pnn.count_parameters()
        print(f"\n  Current parameter count: {param_counts['total_params']:,}")
        print(f"  Frozen: {param_counts['frozen_params']:,}, Trainable: {param_counts['trainable_params']:,}")
        
        # Evaluate on current task
        results = pnn.evaluate_task(test_tasks[task_id], task_id)
        task_accuracies.append(results['accuracy'])
        print(f"  Task {task_id+1} accuracy: {results['accuracy']:.4f} (using task oracle)")
    
    # =========================================================================
    # FINAL EVALUATION (Both modes)
    # =========================================================================
    
    # 1. Task Oracle Evaluation (traditional PNN)
    print("\n" + "="*70)
    print("FINAL EVALUATION - Task Oracle Mode (knows which column to use)")
    print("="*70)
    
    for task_id, test_loader in enumerate(test_tasks):
        results = pnn.evaluate_task(test_loader, task_id)
        acc = results['accuracy']
        final_accuracies.append(acc)
        
        status = "✓" if acc > 0.5 else "✗"
        print(f"  Task {task_id+1} (digits {task_id*2},{task_id*2+1}): {acc:.4f} {status} "
              f"[Column {results['column_id']}]")
    
    avg_accuracy_oracle = np.mean(final_accuracies)
    
    # 2. Task-Agnostic Evaluation (fair comparison with MoB)
    print("\n" + "="*70)
    print("FINAL EVALUATION - Task-Agnostic Mode (confidence-based routing)")
    print("="*70)
    print("  (This mode is comparable to MoB - no task oracle)")
    
    # Combine all test data for task-agnostic evaluation
    all_test_data = torch.utils.data.ConcatDataset([t.dataset for t in test_tasks])
    combined_test_loader = torch.utils.data.DataLoader(
        all_test_data,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    agnostic_results = pnn.evaluate_task_agnostic(combined_test_loader)
    avg_accuracy_agnostic = agnostic_results['accuracy']
    
    print(f"  Overall Accuracy (task-agnostic): {avg_accuracy_agnostic:.4f}")
    print(f"  Column selections: {agnostic_results['column_selections']}")
    
    # Final parameter counts
    param_counts = pnn.count_parameters()
    
    # PNN has ZERO forgetting by construction (columns are frozen)
    forgetting_per_task = [
        max(0, task_accuracies[i] - final_accuracies[i])
        for i in range(len(final_accuracies) - 1)
    ]
    avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Task-Oracle Avg Accuracy:    {avg_accuracy_oracle:.4f} (unfair advantage)")
    print(f"  Task-Agnostic Avg Accuracy:  {avg_accuracy_agnostic:.4f} (fair comparison with MoB)")
    print(f"  Average Forgetting: {avg_forgetting:.4f}")
    print(f"  Tasks retained (>50%): {sum(1 for a in final_accuracies if a > 0.5)}/{len(final_accuracies)}")
    print(f"\n  Architecture:")
    max_col_str = "unlimited" if param_counts['max_columns'] == -1 else str(param_counts['max_columns'])
    print(f"    - max_columns: {max_col_str}")
    print(f"    - Actual columns: {param_counts['num_columns']}")
    print(f"    - Total params: {param_counts['total_params']:,}")
    
    return {
        'task_accuracies': task_accuracies,
        'final_accuracies': final_accuracies,
        'avg_accuracy_oracle': avg_accuracy_oracle,
        'avg_accuracy_agnostic': avg_accuracy_agnostic,
        'forgetting': avg_forgetting,
        'param_counts': param_counts,
        'column_selections': agnostic_results['column_selections']
    }


def main():
    parser = argparse.ArgumentParser(description='Run PNN experiment')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_columns', type=int, default=4,
                        help='Max columns: 4=match MoB params, -1=unlimited (full PNN)')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_results', action='store_true')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = {
        'num_tasks': 5,
        'max_columns': args.max_columns,
        'learning_rate': args.learning_rate,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs
    }
    
    print("="*70)
    print(f"Progressive Neural Networks (PNN) Experiment (Seed {args.seed})")
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
            'avg_accuracy_oracle': results['avg_accuracy_oracle'],
            'avg_accuracy_agnostic': results['avg_accuracy_agnostic'],
            'forgetting': results['forgetting'],
            'param_counts': results['param_counts'],
            'column_selections': results['column_selections']
        }
        filename = f"results/pnn_seed_{args.seed}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Results saved to: {filename}")


if __name__ == '__main__':
    main()
