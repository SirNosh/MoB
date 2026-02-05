"""
Bid diagnostics and logging for MoB: Mixture of Bidders.

This module provides comprehensive logging and diagnostic tools to identify
potential issues with the bidding mechanism, including:

- Alpha (PredictedLoss) signal being ignored
- Beta (ForgettingCost) too high preventing learning
- Bids exploding or vanishing
- Detailed component-level tracking

Usage:
    logger = BidLogger()
    logger.log_batch(batch_idx, bids, components, winner_id)
    logger.print_diagnostics()
    logger.save_logs("bid_logs.json")
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional
from pathlib import Path


class BidLogger:
    """
    Comprehensive logging for bid components and diagnostics.

    Tracks execution costs, forgetting costs, final bids, and winner IDs
    across all batches to diagnose potential bidding issues.
    """

    def __init__(self, num_experts: int, alpha: float = 0.5, beta: float = 0.5, log_file: Optional[str] = None):
        """
        Initialize the bid logger.

        Parameters:
        -----------
        num_experts : int
            Number of experts in the system.
        alpha : float
            Weight for execution cost (stored once, not per-batch).
        beta : float
            Weight for forgetting cost (stored once, not per-batch).
        log_file : str, optional
            Path to save logs automatically after each batch.
        """
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        self.log_file = log_file

        # Storage for all batch logs (compact format)
        self.batch_logs: List[Dict] = []

        # Aggregate statistics
        self.stats = {
            'num_batches': 0,
            'expert_wins': [0] * num_experts,
            'exec_cost_history': [[] for _ in range(num_experts)],
            'forget_cost_history': [[] for _ in range(num_experts)],
            'bid_history': [[] for _ in range(num_experts)],
        }

    def log_batch(
        self,
        batch_idx: int,
        bids: np.ndarray,
        components: List[Dict],
        winner_id: int,
        task_id: Optional[int] = None
    ):
        """
        Log all bid information for a single batch (compact format).

        Parameters:
        -----------
        batch_idx : int
            Index of the current batch.
        bids : np.ndarray
            Array of final bids from all experts.
        components : list of dict
            Bid component breakdowns from each expert.
        winner_id : int
            ID of the winning expert.
        task_id : int, optional
            Current task ID for multi-task tracking.
        """
        # Compact format: [exec_cost, forget_cost, bid] per expert
        experts_data = []
        for expert_id in range(self.num_experts):
            comp = components[expert_id]
            experts_data.append([
                round(float(comp['exec_cost']), 6),
                round(float(comp['forget_cost']), 6),
                round(float(comp['bid']), 6)
            ])
            
            # Update aggregate statistics
            self.stats['exec_cost_history'][expert_id].append(float(comp['exec_cost']))
            self.stats['forget_cost_history'][expert_id].append(float(comp['forget_cost']))
            self.stats['bid_history'][expert_id].append(float(comp['bid']))

        # Compact batch log: [batch_idx, task_id, winner_id, [[e0], [e1], ...]]
        batch_log = {
            'b': batch_idx,
            't': task_id,
            'w': winner_id,
            'e': experts_data  # e[i] = [exec, forget, bid]
        }

        # Update statistics
        self.stats['num_batches'] += 1
        self.stats['expert_wins'][winner_id] += 1

        self.batch_logs.append(batch_log)

        # Auto-save if configured
        if self.log_file and batch_idx % 100 == 0:  # Save every 100 batches
            self.save_logs(self.log_file)

    def print_diagnostics(self, last_n_batches: Optional[int] = None):
        """
        Print comprehensive diagnostics to identify bidding issues.

        Parameters:
        -----------
        last_n_batches : int, optional
            Only analyze the last N batches. If None, analyzes all batches.
        """
        if self.stats['num_batches'] == 0:
            print("⚠  No batches logged yet.")
            return

        # Determine which batches to analyze
        if last_n_batches is not None:
            start_idx = max(0, self.stats['num_batches'] - last_n_batches)
        else:
            start_idx = 0
            last_n_batches = self.stats['num_batches']

        print("\n" + "="*80)
        print(f"BID DIAGNOSTICS (Last {last_n_batches} batches)")
        print("="*80)

        # Issue 1: Is alpha (PredictedLoss) signal being ignored?
        print("\n[1] ALPHA SIGNAL CHECK (Execution Cost)")
        print("-" * 80)

        for expert_id in range(self.num_experts):
            exec_costs = self.stats['exec_cost_history'][expert_id][start_idx:]

            if len(exec_costs) > 0:
                mean_exec = np.mean(exec_costs)
                std_exec = np.std(exec_costs)
                min_exec = np.min(exec_costs)
                max_exec = np.max(exec_costs)

                print(f"  Expert {expert_id}:")
                print(f"    Mean: {mean_exec:.6f} ± {std_exec:.6f}")
                print(f"    Range: [{min_exec:.6f}, {max_exec:.6f}]")

                # Check if exec cost is near zero (alpha signal ignored)
                if mean_exec < 1e-6:
                    print(f"    ⚠  WARNING: Execution cost near zero! Alpha signal may be ignored.")
                elif std_exec < 1e-8:
                    print(f"    ⚠  WARNING: No variance in execution cost! All batches look the same.")

        # Issue 2: Is beta (ForgettingCost) too high preventing learning?
        print("\n[2] BETA SIGNAL CHECK (Forgetting Cost)")
        print("-" * 80)

        for expert_id in range(self.num_experts):
            forget_costs = self.stats['forget_cost_history'][expert_id][start_idx:]
            exec_costs = self.stats['exec_cost_history'][expert_id][start_idx:]

            if len(forget_costs) > 0 and len(exec_costs) > 0:
                mean_forget = np.mean(forget_costs)
                std_forget = np.std(forget_costs)
                min_forget = np.min(forget_costs)
                max_forget = np.max(forget_costs)

                mean_exec = np.mean(exec_costs)
                ratio = mean_forget / mean_exec if mean_exec > 0 else float('inf')

                print(f"  Expert {expert_id}:")
                print(f"    Mean: {mean_forget:.6f} ± {std_forget:.6f}")
                print(f"    Range: [{min_forget:.6f}, {max_forget:.6f}]")
                print(f"    Forget/Exec Ratio: {ratio:.2f}x")

                # Check if forgetting cost dominates
                if ratio > 100:
                    print(f"    🔴 CRITICAL: Forgetting cost is {ratio:.0f}x execution cost!")
                    print(f"       This prevents experts from learning new tasks.")
                    print(f"       Consider: reducing β, reducing λ_EWC, or increasing α")
                elif ratio > 10:
                    print(f"    ⚠  WARNING: Forgetting cost is {ratio:.0f}x execution cost.")
                    print(f"       Learning may be significantly hindered.")
                elif ratio < 0.01:
                    print(f"    ⚠  WARNING: Forgetting cost is negligible ({ratio:.4f}x).")
                    print(f"       EWC may not be preventing forgetting effectively.")

        # Issue 3: Are bids exploding or vanishing?
        print("\n[3] BID MAGNITUDE CHECK")
        print("-" * 80)

        for expert_id in range(self.num_experts):
            bids = self.stats['bid_history'][expert_id][start_idx:]

            if len(bids) > 0:
                mean_bid = np.mean(bids)
                std_bid = np.std(bids)
                min_bid = np.min(bids)
                max_bid = np.max(bids)

                print(f"  Expert {expert_id}:")
                print(f"    Mean: {mean_bid:.6f} ± {std_bid:.6f}")
                print(f"    Range: [{min_bid:.6f}, {max_bid:.6f}]")

                # Check for exploding bids
                if max_bid > 1e6:
                    print(f"    🔴 CRITICAL: Bids are exploding! Max bid = {max_bid:.2e}")
                    print(f"       This indicates numerical instability.")
                elif max_bid > 1e3:
                    print(f"    ⚠  WARNING: Bids are very large (max = {max_bid:.2f})")

                # Check for vanishing bids
                if mean_bid < 1e-6:
                    print(f"    ⚠  WARNING: Bids are vanishing! Mean bid = {mean_bid:.2e}")
                    print(f"       Check if both α and β are too small.")

                # Check for NaN or inf
                if np.any(np.isnan(bids)) or np.any(np.isinf(bids)):
                    print(f"    🔴 CRITICAL: Bids contain NaN or Inf values!")

        # Issue 4: Expert specialization
        print("\n[4] EXPERT WIN DISTRIBUTION")
        print("-" * 80)

        total_wins = sum(self.stats['expert_wins'])
        for expert_id in range(self.num_experts):
            wins = self.stats['expert_wins'][expert_id]
            win_rate = wins / total_wins if total_wins > 0 else 0

            bar_length = int(win_rate * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)

            print(f"  Expert {expert_id}: {bar} {win_rate*100:5.1f}% ({wins}/{total_wins})")

        # Check for monopoly
        max_wins = max(self.stats['expert_wins'])
        max_win_rate = max_wins / total_wins if total_wins > 0 else 0

        if max_win_rate > 0.8:
            print(f"\n  ⚠  WARNING: One expert dominates ({max_win_rate*100:.1f}% of batches)")
            print(f"     This may indicate poor auction dynamics.")

        # Check for equal distribution (no specialization)
        expected_rate = 1.0 / self.num_experts
        actual_rates = [w / total_wins if total_wins > 0 else 0 for w in self.stats['expert_wins']]
        variance = np.var(actual_rates)

        if variance < 0.001:
            print(f"\n  ⚠  WARNING: Nearly uniform win distribution (variance={variance:.6f})")
            print(f"     Experts may not be specializing effectively.")

        print("\n" + "="*80)

    def get_batch_details(self, batch_idx: int) -> Optional[Dict]:
        """
        Get detailed information for a specific batch.

        Parameters:
        -----------
        batch_idx : int
            Index of the batch to retrieve.

        Returns:
        --------
        batch_log : dict or None
            Detailed log for the specified batch, or None if not found.
        """
        for log in self.batch_logs:
            if log['batch_idx'] == batch_idx:
                return log
        return None

    def print_batch_details(self, batch_idx: int):
        """
        Print detailed breakdown for a specific batch.

        Parameters:
        -----------
        batch_idx : int
            Index of the batch to analyze.
        """
        log = self.get_batch_details(batch_idx)

        if log is None:
            print(f"⚠  Batch {batch_idx} not found in logs.")
            return

        print(f"\n" + "="*80)
        print(f"BATCH {batch_idx} DETAILED BREAKDOWN")
        if log['task_id'] is not None:
            print(f"Task ID: {log['task_id']}")
        print("="*80)

        print(f"\n{'Expert':>10} {'Exec Cost':>12} {'Forget Cost':>12} {'α':>6} {'β':>6} {'Bid':>12} {'Winner':>8}")
        print("-" * 80)

        for expert_data in log['experts']:
            winner_mark = "✓" if expert_data['is_winner'] else ""

            print(f"{expert_data['expert_id']:>10} "
                  f"{expert_data['exec_cost']:>12.6f} "
                  f"{expert_data['forget_cost']:>12.6f} "
                  f"{expert_data['alpha']:>6.2f} "
                  f"{expert_data['beta']:>6.2f} "
                  f"{expert_data['bid']:>12.6f} "
                  f"{winner_mark:>8}")

        print("="*80 + "\n")

    def save_logs(self, filepath: str):
        """
        Save logs to a compact, LLM-friendly JSON file.

        Parameters:
        -----------
        filepath : str
            Path to save the log file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Compute summary statistics for quick parsing
        summary = {
            'per_expert': {}
        }
        for i in range(self.num_experts):
            exec_costs = self.stats['exec_cost_history'][i]
            forget_costs = self.stats['forget_cost_history'][i]
            bids = self.stats['bid_history'][i]
            
            if len(exec_costs) > 0:
                summary['per_expert'][f'expert_{i}'] = {
                    'wins': self.stats['expert_wins'][i],
                    'win_rate': round(self.stats['expert_wins'][i] / self.stats['num_batches'], 4),
                    'exec_cost': {'mean': round(np.mean(exec_costs), 6), 'std': round(np.std(exec_costs), 6)},
                    'forget_cost': {'mean': round(np.mean(forget_costs), 6), 'std': round(np.std(forget_costs), 6)},
                    'bid': {'mean': round(np.mean(bids), 6), 'std': round(np.std(bids), 6)}
                }

        # Compact LLM-friendly format
        data = {
            '_format': {
                'description': 'MoB bid diagnostics log',
                'batch_format': 'b=batch_idx, t=task_id, w=winner_id, e=experts_data',
                'expert_format': '[exec_cost, forget_cost, final_bid] per expert (index = expert_id)'
            },
            'config': {
                'num_experts': self.num_experts,
                'alpha': self.alpha,
                'beta': self.beta,
                'bid_formula': 'bid = alpha * exec_cost + beta * forget_cost'
            },
            'summary': {
                'total_batches': self.stats['num_batches'],
                'expert_wins': self.stats['expert_wins'],
                **summary
            },
            'batches': self.batch_logs
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Bid logs saved to: {filepath}")

    def load_logs(self, filepath: str):
        """
        Load logs from a JSON file.

        Parameters:
        -----------
        filepath : str
            Path to the log file.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.num_experts = data['num_experts']
        self.batch_logs = data['batch_logs']

        # Reconstruct statistics
        self.stats['num_batches'] = data['num_batches']
        self.stats['expert_wins'] = data['expert_wins']

        # Rebuild history arrays
        self.stats['exec_cost_history'] = [[] for _ in range(self.num_experts)]
        self.stats['forget_cost_history'] = [[] for _ in range(self.num_experts)]
        self.stats['bid_history'] = [[] for _ in range(self.num_experts)]

        for batch_log in self.batch_logs:
            for expert_data in batch_log['experts']:
                expert_id = expert_data['expert_id']
                self.stats['exec_cost_history'][expert_id].append(expert_data['exec_cost'])
                self.stats['forget_cost_history'][expert_id].append(expert_data['forget_cost'])
                self.stats['bid_history'][expert_id].append(expert_data['bid'])

        print(f"✓ Bid logs loaded from: {filepath}")

    def plot_bid_components(self, save_path: Optional[str] = None):
        """
        Create visualization of bid components over time.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot (PNG or HTML).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠  Matplotlib not available. Install with: pip install matplotlib")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Plot execution costs
        for expert_id in range(self.num_experts):
            axes[0].plot(
                self.stats['exec_cost_history'][expert_id],
                label=f'Expert {expert_id}',
                alpha=0.7
            )
        axes[0].set_ylabel('Execution Cost')
        axes[0].set_title('Execution Cost (α signal) Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot forgetting costs
        for expert_id in range(self.num_experts):
            axes[1].plot(
                self.stats['forget_cost_history'][expert_id],
                label=f'Expert {expert_id}',
                alpha=0.7
            )
        axes[1].set_ylabel('Forgetting Cost')
        axes[1].set_title('Forgetting Cost (β signal) Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot final bids
        for expert_id in range(self.num_experts):
            axes[2].plot(
                self.stats['bid_history'][expert_id],
                label=f'Expert {expert_id}',
                alpha=0.7
            )
        axes[2].set_xlabel('Batch')
        axes[2].set_ylabel('Final Bid')
        axes[2].set_title('Final Bids Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Bid component plot saved to: {save_path}")
        else:
            plt.show()
