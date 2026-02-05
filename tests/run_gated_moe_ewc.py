"""
Gated MoE with EWC Regularization - Fair Baseline for MoB Comparison

This script implements a standard Gated MoE with EWC applied to experts.
The gater uses industry-standard end-to-end training through task loss.

Key design decisions (industry-standard MoE):
1. Gater is a simple MLP (like in Mixtral, Switch Transformer)
2. Gater learns routing through gradients from task loss (no separate gater loss)
3. Each expert has EWC regularization (same as MoB)
4. Optional load balancing loss

Usage:
    python tests/run_gated_moe_ewc.py                     # Default config
    python tests/run_gated_moe_ewc.py --lambda_ewc 10     # Match MoB lambda
    python tests/run_gated_moe_ewc.py --epochs 4          # Match MoB epochs
    python tests/run_gated_moe_ewc.py --gater_ewc         # EWC on gater too
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

from mob.models import create_model
from mob.bidding import EWCForgettingEstimator
from mob.utils import set_seed

# Import dataset creation
from tests.test_baselines import create_split_mnist


class StandardGater(nn.Module):
    """
    Industry-standard MoE gater: simple MLP.
    
    In LLMs like Mixtral/Switch Transformer, gaters are often just a single
    linear layer. For vision tasks with flattened inputs, we use a small MLP
    for slightly more capacity, but still very simple.
    """
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_experts)
        )
    
    def forward(self, x):
        return self.net(x)


class GatedMoEwithEWC(nn.Module):
    """
    Standard Gated Mixture of Experts with EWC Regularization.
    
    This is the industry-standard MoE architecture:
    1. Gater produces routing probabilities
    2. Top-1 expert is selected per sample (or per batch)
    3. Expert output is weighted by gating probability
    4. Task loss backpropagates through everything (end-to-end)
    
    EWC is applied to experts (and optionally gater) to prevent forgetting.
    """
    
    def __init__(
        self,
        num_experts: int,
        expert_config: Dict,
        lambda_ewc: float = 10.0,
        gater_ewc: bool = False,
        gater_hidden_size: int = 256,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.device = device if device is not None else torch.device('cpu')
        self.lambda_ewc = lambda_ewc
        self.gater_ewc = gater_ewc
        
        # Create expert models (same architecture as MoB)
        self.expert_models = nn.ModuleList([
            create_model(
                architecture=expert_config['architecture'],
                num_classes=expert_config['num_classes'],
                input_channels=expert_config.get('input_channels', 1),
                dropout=expert_config.get('dropout', 0.5)
            )
            for _ in range(self.num_experts)
        ])
        self.to(self.device)
        
        # Create EWC estimators for each expert (same as MoB)
        self.expert_ewc: List[EWCForgettingEstimator] = [
            EWCForgettingEstimator(
                model, 
                lambda_ewc=lambda_ewc,
                device=self.device
            )
            for model in self.expert_models
        ]
        
        # Standard MLP gater (industry standard)
        input_size = expert_config.get('input_size', 784)  # 28*28 for MNIST
        self.gater = StandardGater(input_size, num_experts, gater_hidden_size)
        self.gater.to(self.device)
        
        # Optional EWC for gater
        if gater_ewc:
            self.gater_ewc_estimator = EWCForgettingEstimator(
                self.gater,
                lambda_ewc=lambda_ewc,
                device=self.device
            )
        else:
            self.gater_ewc_estimator = None
        
        # Track which experts trained on which tasks
        self.expert_task_history: Dict[int, List[int]] = {i: [] for i in range(num_experts)}
        self.training_stats = {'expert_usage': np.zeros(num_experts)}
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass through gated MoE.
        
        Returns:
            output: Combined expert output
            gating_probs: Softmax probabilities from gater
            winner_id: Selected expert index
        """
        # Get gating probabilities
        gating_logits = self.gater(x)
        gating_probs = F.softmax(gating_logits, dim=-1)
        
        # Top-1 routing (standard in MoE)
        # For batch-level routing, use the most common expert
        winner_ids = gating_probs.argmax(dim=-1)
        winner_id = winner_ids.mode().values.item()
        
        # Get output from winning expert
        output = self.expert_models[winner_id](x)
        
        return output, gating_probs, winner_id
    
    def forward_all_experts(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get outputs from all experts (for evaluation)."""
        outputs = []
        for expert in self.expert_models:
            expert.eval()
            with torch.no_grad():
                outputs.append(expert(x))
        return outputs
    
    def train_on_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        expert_optimizers: List[torch.optim.Optimizer],
        gater_optimizer: torch.optim.Optimizer,
        task_id: int,
        batch_idx: int = 0
    ) -> Dict:
        """
        Train on a single batch using end-to-end learning.
        
        This is the standard MoE training approach:
        1. Gater selects expert based on input
        2. Selected expert produces output
        3. Task loss computed on expert output
        4. Gradients flow through both expert AND gater
        """
        x = x.to(self.device)
        y = y.to(self.device)
        
        # =====================================================================
        # STEP 1: Forward pass through gater and selected expert
        # =====================================================================
        self.gater.train()
        gating_logits = self.gater(x)
        gating_probs = F.softmax(gating_logits, dim=-1)
        
        # Batch-level top-1 routing
        winner_ids = gating_probs.argmax(dim=-1)
        winner_id = winner_ids.mode().values.item()
        
        # Get selected expert
        expert = self.expert_models[winner_id]
        expert.train()
        
        # Forward through expert
        expert_output = expert(x)
        
        # =====================================================================
        # STEP 2: Compute losses
        # =====================================================================
        # Task loss (cross-entropy)
        task_loss = F.cross_entropy(expert_output, y)
        
        # EWC penalty for the winning expert
        ewc_penalty = self.expert_ewc[winner_id].penalty()
        
        # Optional: EWC penalty for gater
        gater_ewc_penalty = torch.tensor(0.0, device=self.device)
        if self.gater_ewc_estimator is not None:
            gater_ewc_penalty = self.gater_ewc_estimator.penalty()
        
        # Total loss
        total_loss = task_loss + ewc_penalty + gater_ewc_penalty
        
        # =====================================================================
        # STEP 3: Backprop through BOTH gater and expert (end-to-end)
        # =====================================================================
        # Zero all gradients
        gater_optimizer.zero_grad()
        expert_optimizers[winner_id].zero_grad()
        
        # Backward pass - gradients flow through expert AND gater
        total_loss.backward()
        
        # Update both
        gater_optimizer.step()
        expert_optimizers[winner_id].step()
        
        # Track usage
        self.training_stats['expert_usage'][winner_id] += 1
        
        # Log first few batches
        if batch_idx < 3:
            print(f"[GatedMoE+EWC] Batch {batch_idx+1}: "
                  f"winner={winner_id}, "
                  f"task_loss={task_loss.item():.4f}, "
                  f"ewc={ewc_penalty.item():.4f}, "
                  f"total={total_loss.item():.4f}")
        
        return {
            'task_loss': task_loss.item(),
            'ewc_penalty': ewc_penalty.item(),
            'gater_ewc_penalty': gater_ewc_penalty.item() if isinstance(gater_ewc_penalty, torch.Tensor) else gater_ewc_penalty,
            'total_loss': total_loss.item(),
            'winner_id': winner_id
        }
    
    def train_on_task(
        self,
        dataloader,
        expert_optimizers: List[torch.optim.Optimizer],
        gater_optimizer: torch.optim.Optimizer,
        task_id: int,
        epochs: int = 1
    ) -> Dict:
        """Train on a complete task for multiple epochs."""
        
        expert_usage = np.zeros(self.num_experts)
        total_batches = 0
        
        for epoch in range(epochs):
            epoch_usage = np.zeros(self.num_experts)
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (x, y) in enumerate(pbar):
                metrics = self.train_on_batch(
                    x, y, expert_optimizers, gater_optimizer,
                    task_id, batch_idx if epoch == 0 else 999  # Only log first epoch
                )
                epoch_usage[metrics['winner_id']] += 1
                total_batches += 1
            
            expert_usage += epoch_usage
            print(f"  Epoch {epoch+1} winners: {dict(enumerate(epoch_usage.astype(int)))}")
        
        # Determine primary expert for this task
        primary_expert = int(np.argmax(expert_usage))
        if task_id not in self.expert_task_history[primary_expert]:
            self.expert_task_history[primary_expert].append(task_id)
        
        return {
            'expert_usage': expert_usage / expert_usage.sum() if expert_usage.sum() > 0 else expert_usage,
            'primary_expert': primary_expert,
            'total_batches': total_batches
        }
    
    def update_fisher_after_task(self, task_loader, num_samples: int = 200):
        """
        Update Fisher information for experts that trained on this task.
        Same approach as run_mob_only.py for fair comparison.
        """
        # Find which experts were used during this task
        usage = self.training_stats['expert_usage']
        trained_experts = [i for i in range(self.num_experts) if usage[i] > 0]
        
        for expert_id in trained_experts:
            print(f"  Updating Fisher for Expert {expert_id}")
            self.expert_ewc[expert_id].update_fisher(task_loader, num_samples)
        
        # Update gater Fisher if enabled (custom method for gater)
        if self.gater_ewc_estimator is not None:
            print(f"  Updating Fisher for Gater")
            self._update_gater_fisher(task_loader, num_samples)
        
        # Reset usage stats for next task
        self.training_stats['expert_usage'] = np.zeros(self.num_experts)
    
    def _update_gater_fisher(self, dataloader, num_samples: int = 200):
        """
        Custom Fisher update for gater using routing targets (not class labels).
        The gater's target is which expert to route to, not the class label.
        """
        self.gater.train()
        
        # Compute Fisher for gater using its current routing decisions
        current_fisher = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.gater.named_parameters() if p.requires_grad
        }
        
        samples_seen = 0
        for x, y in dataloader:
            if samples_seen >= num_samples:
                break
            
            x = x.to(self.device)
            batch_size = x.size(0)
            
            self.gater.zero_grad()
            gating_logits = self.gater(x)
            
            # Use gater's own predictions as targets (self-supervised Fisher)
            targets = gating_logits.argmax(dim=-1).detach()
            log_probs = F.log_softmax(gating_logits, dim=-1)
            loss = F.nll_loss(log_probs, targets)
            loss.backward()
            
            for n, p in self.gater.named_parameters():
                if p.requires_grad and p.grad is not None:
                    current_fisher[n] += p.grad.data.pow(2) * batch_size
            
            samples_seen += batch_size
        
        # Normalize and update with EMA
        if samples_seen > 0:
            gamma = 0.9  # Same decay as expert EWC
            for n in current_fisher:
                current_fisher[n] /= samples_seen
                if n in self.gater_ewc_estimator.fisher:
                    self.gater_ewc_estimator.fisher[n] = (
                        gamma * self.gater_ewc_estimator.fisher[n] + 
                        (1 - gamma) * current_fisher[n]
                    )
                else:
                    self.gater_ewc_estimator.fisher[n] = current_fisher[n].clone()
            
            # Update optimal params
            for n, p in self.gater.named_parameters():
                if p.requires_grad:
                    p_new = p.clone().detach()
                    if n in self.gater_ewc_estimator.optimal_params:
                        self.gater_ewc_estimator.optimal_params[n] = (
                            gamma * self.gater_ewc_estimator.optimal_params[n] + 
                            (1 - gamma) * p_new
                        )
                    else:
                        self.gater_ewc_estimator.optimal_params[n] = p_new
            
            self.gater_ewc_estimator.num_tasks_consolidated += 1
            self.gater_ewc_estimator._normalize_fisher()
            print(f"    Gater Fisher updated (samples={samples_seen})")
    
    def count_parameters(self) -> Dict:
        """Count total parameters for comparison with MoB."""
        expert_params = sum(p.numel() for e in self.expert_models for p in e.parameters())
        gater_params = sum(p.numel() for p in self.gater.parameters())
        return {
            'expert_params': expert_params,
            'gater_params': gater_params,
            'total_params': expert_params + gater_params,
            'per_expert_params': expert_params // self.num_experts
        }
    
    def evaluate_all(self, dataloader) -> Dict:
        """
        Evaluate using gater-based routing (standard MoE evaluation).
        """
        all_labels = []
        gated_preds = []
        expert_selections = {i: 0 for i in range(self.num_experts)}
        
        self.gater.eval()
        for expert in self.expert_models:
            expert.eval()
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                all_labels.append(y.cpu())
                
                # Get gater routing
                gating_logits = self.gater(x)
                gating_probs = F.softmax(gating_logits, dim=-1)
                
                # Per-sample routing
                routes = gating_probs.argmax(dim=-1)
                
                # Get all expert outputs
                expert_outputs = [expert(x) for expert in self.expert_models]
                
                # Select predictions based on routing
                batch_preds = []
                for sample_idx in range(x.size(0)):
                    expert_id = routes[sample_idx].item()
                    pred = expert_outputs[expert_id][sample_idx].argmax().item()
                    batch_preds.append(pred)
                    expert_selections[expert_id] += 1
                
                gated_preds.extend(batch_preds)
        
        all_labels = torch.cat(all_labels)
        gated_preds = torch.tensor(gated_preds)
        
        total_samples = sum(expert_selections.values())
        
        return {
            'gated_accuracy': (gated_preds == all_labels).float().mean().item(),
            'expert_selections': expert_selections,
            'primary_expert': max(expert_selections, key=expert_selections.get),
            'routing_distribution': {k: v/total_samples for k, v in expert_selections.items()}
        }


def run_experiment(train_tasks, test_tasks, config):
    """Run Gated MoE + EWC experiment."""
    
    print("\n" + "="*70)
    print("Gated MoE + EWC Experiment (Fair Baseline for MoB)")
    print("="*70)
    
    device = torch.device(config['device'])
    
    # Expert configuration (same as MoB)
    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'dropout': 0.5
    }
    
    # Create model
    model = GatedMoEwithEWC(
        num_experts=config['num_experts'],
        expert_config=expert_config,
        lambda_ewc=config['lambda_ewc'],
        gater_ewc=config.get('gater_ewc', False),
        gater_hidden_size=config.get('gater_hidden_size', 256),
        device=device
    )
    
    # Print parameter counts for comparison with MoB
    param_counts = model.count_parameters()
    print(f"\nParameter Counts:")
    print(f"  Experts: {param_counts['expert_params']:,} ({param_counts['per_expert_params']:,} per expert)")
    print(f"  Gater: {param_counts['gater_params']:,}")
    print(f"  Total: {param_counts['total_params']:,}")
    print(f"  (MoB has same experts, no gater overhead)")
    
    # Optimizers (same LR as MoB)
    expert_optimizers = [
        torch.optim.Adam(expert.parameters(), lr=config['learning_rate'])
        for expert in model.expert_models
    ]
    gater_optimizer = torch.optim.Adam(model.gater.parameters(), lr=config['learning_rate'])
    
    # Metrics
    task_accuracies = []
    final_accuracies = []
    expert_task_wins = {}
    
    epochs_per_task = config.get('epochs_per_task', 4)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    for task_id, task_loader in enumerate(train_tasks):
        print(f"\n{'='*70}")
        print(f"TASK {task_id + 1}/{len(train_tasks)} (Digits {task_id*2}, {task_id*2+1})")
        print(f"{'='*70}")
        
        metrics = model.train_on_task(
            task_loader,
            expert_optimizers,
            gater_optimizer,
            task_id,
            epochs=epochs_per_task
        )
        
        expert_task_wins[task_id] = metrics['primary_expert']
        print(f"\n  Task {task_id+1} summary: usage={metrics['expert_usage'].round(2)}, "
              f"primary=Expert {metrics['primary_expert']}")
        
        # Update Fisher after task (same as MoB)
        print(f"\n  Updating EWC Fisher information...")
        model.update_fisher_after_task(task_loader, num_samples=200)
        
        # Evaluate on current task
        results = model.evaluate_all(test_tasks[task_id])
        task_accuracies.append(results['gated_accuracy'])
        print(f"  Task {task_id+1} accuracy: {results['gated_accuracy']:.4f}")
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    for task_id, test_loader in enumerate(test_tasks):
        results = model.evaluate_all(test_loader)
        acc = results['gated_accuracy']
        final_accuracies.append(acc)
        
        trained_expert = expert_task_wins.get(task_id, "?")
        eval_expert = results['primary_expert']
        match_str = "" if trained_expert == eval_expert else f" ** ROUTED TO Expert {eval_expert}!"
        status = "✓" if acc > 0.5 else "✗"
        
        print(f"  Task {task_id+1} (digits {task_id*2},{task_id*2+1}): "
              f"{acc:.4f} {status} [Trained: Expert {trained_expert}]{match_str}")
        print(f"    Routing: {results['routing_distribution']}")
    
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
        'expert_task_wins': expert_task_wins
    }


def main():
    parser = argparse.ArgumentParser(description='Run Gated MoE + EWC experiment')
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--lambda_ewc', type=float, default=50.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gater_hidden_size', type=int, default=256,
                        help='Hidden layer size for gater MLP')
    parser.add_argument('--disable_gater_ewc', action='store_true',
                        help='Disable EWC to gater network')
    parser.add_argument('--save_results', action='store_true')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    config = {
        'num_experts': args.num_experts,
        'num_tasks': 5,
        'lambda_ewc': args.lambda_ewc,
        'learning_rate': args.learning_rate,
        'gater_ewc': not args.disable_gater_ewc,
        'gater_hidden_size': args.gater_hidden_size,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'epochs_per_task': args.epochs
    }
    
    print("="*70)
    print(f"Gated MoE + EWC Experiment (Seed {args.seed})")
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
            'expert_task_wins': {str(k): v for k, v in results['expert_task_wins'].items()}
        }
        filename = f"results/gated_moe_ewc_seed_{args.seed}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Results saved to: {filename}")


if __name__ == '__main__':
    main()
