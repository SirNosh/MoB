"""
Bid diagnostics and logging for MoB: Mixture of Bidders.

Tracks training and evaluation routing for both pseudo-label and prototype strategies.
"""

import json
import math
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class BidLogger:
    """
    Comprehensive logging for bid components at both training and evaluation time.

    Training:  log_batch() — exec+forget bids, winner, task
    Eval:      log_eval_batch() — prototype distances or exec costs, per-digit routing
    Prototype: log_prototype_finalize() — snapshot of centroid state at consolidation
    """

    def __init__(
        self,
        num_experts: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        routing_strategy: str = 'pseudo_label',
        log_file: Optional[str] = None
    ):
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta
        self.routing_strategy = routing_strategy
        self.log_file = log_file

        # --- Training logs ---
        self.batch_logs: List[Dict] = []
        self.stats = {
            'num_batches': 0,
            'expert_wins': [0] * num_experts,
            'exec_cost_history': [[] for _ in range(num_experts)],
            'forget_cost_history': [[] for _ in range(num_experts)],
            'bid_history': [[] for _ in range(num_experts)],
        }

        # --- Evaluation logs (prototype or pseudo-label) ---
        self.eval_logs: List[Dict] = []
        self.eval_stats = {
            'num_eval_batches': 0,
            'eval_winner_history': [],
            # per-expert: primary cost signal (distance or exec_cost)
            'primary_cost_history': [[] for _ in range(num_experts)],
            'eval_forget_cost_history': [[] for _ in range(num_experts)],
            'eval_bid_history': [[] for _ in range(num_experts)],
            # per-digit routing: digit -> {expert_id -> count}
            'digit_routing': {d: {i: 0 for i in range(num_experts)} for d in range(10)},
            'digit_correct': {d: 0 for d in range(10)},
            'digit_total': {d: 0 for d in range(10)},
        }

        # --- Prototype state snapshots ---
        self.prototype_state_log: List[Dict] = []

    # =========================================================================
    # TRAINING LOGGING
    # =========================================================================

    def log_batch(
        self,
        batch_idx: int,
        bids: np.ndarray,
        components: List[Dict],
        winner_id: int,
        task_id: Optional[int] = None
    ):
        """Log training bid components for one batch."""
        experts_data = []
        for expert_id in range(self.num_experts):
            comp = components[expert_id]
            experts_data.append([
                round(float(comp['exec_cost']), 6),
                round(float(comp['forget_cost']), 6),
                round(float(comp['bid']), 6)
            ])
            self.stats['exec_cost_history'][expert_id].append(float(comp['exec_cost']))
            self.stats['forget_cost_history'][expert_id].append(float(comp['forget_cost']))
            self.stats['bid_history'][expert_id].append(float(comp['bid']))

        self.batch_logs.append({'b': batch_idx, 't': task_id, 'w': winner_id, 'e': experts_data})
        self.stats['num_batches'] += 1
        self.stats['expert_wins'][winner_id] += 1

        if self.log_file and batch_idx % 100 == 0:
            self.save_logs(self.log_file)

    # =========================================================================
    # EVALUATION LOGGING
    # =========================================================================

    def log_eval_batch(
        self,
        batch_idx: int,
        per_expert_data: List[Dict],
        winner_id: int,
        digit_labels: Optional[List[int]] = None,
        winner_preds: Optional[List[int]] = None
    ):
        """
        Log evaluation routing decisions.

        per_expert_data: list of dicts per expert, each with:
            - 'primary_cost': distance_score (prototype) or exec_cost (pseudo-label)
            - 'forget_cost': raw forget cost
            - 'norm_primary': normalized primary cost
            - 'norm_forget': normalized forget cost
            - 'bid': final bid
        digit_labels: ground-truth digit labels for this batch (for per-digit routing stats)
        winner_preds: predictions from winning expert (for per-digit accuracy)
        """
        entry = {
            'b': batch_idx,
            'w': winner_id,
            'e': []
        }
        for i, d in enumerate(per_expert_data):
            entry['e'].append({
                'pc': round(float(d['primary_cost']), 6),
                'fc': round(float(d['forget_cost']), 6),
                'np': round(float(d['norm_primary']), 6),
                'nf': round(float(d['norm_forget']), 6),
                'bid': round(float(d['bid']), 6)
            })
            self.eval_stats['primary_cost_history'][i].append(float(d['primary_cost']))
            self.eval_stats['eval_forget_cost_history'][i].append(float(d['forget_cost']))
            self.eval_stats['eval_bid_history'][i].append(float(d['bid']))

        self.eval_stats['eval_winner_history'].append(winner_id)
        self.eval_stats['num_eval_batches'] += 1

        # Per-digit routing
        if digit_labels is not None:
            for i, digit in enumerate(digit_labels):
                d = int(digit)
                self.eval_stats['digit_routing'][d][winner_id] += 1
                self.eval_stats['digit_total'][d] += 1
                if winner_preds is not None and int(winner_preds[i]) == d:
                    self.eval_stats['digit_correct'][d] += 1

        self.eval_logs.append(entry)

    def log_prototype_finalize(
        self,
        expert_id: int,
        event: str,
        batch_idx: int,
        classes_seen: List[int],
        sample_counts: Dict[int, int],
        has_mahalanobis: bool,
        feature_dim: int
    ):
        """
        Log prototype store state at consolidation.

        event: 'finalize' or 'update'
        """
        self.prototype_state_log.append({
            'expert_id': expert_id,
            'event': event,
            'batch_idx': batch_idx,
            'classes_seen': sorted(classes_seen),
            'sample_counts': {str(k): v for k, v in sample_counts.items()},
            'total_samples': sum(sample_counts.values()),
            'has_mahalanobis': has_mahalanobis,
            'feature_dim': feature_dim
        })

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def print_diagnostics(self, last_n_batches: Optional[int] = None):
        """Print comprehensive diagnostics for training and evaluation."""
        print("\n" + "="*80)
        print(f"BID DIAGNOSTICS  [routing_strategy={self.routing_strategy}]")
        print("="*80)

        self._print_training_diagnostics(last_n_batches)

        if self.eval_stats['num_eval_batches'] > 0:
            self._print_eval_diagnostics()

        if self.stats['num_batches'] > 0 or self.eval_stats['num_eval_batches'] > 0:
            self._print_load_balancing_diagnostics()

        if self.prototype_state_log:
            self._print_prototype_state_diagnostics()

    def _print_training_diagnostics(self, last_n_batches):
        if self.stats['num_batches'] == 0:
            print("[WARN] No training batches logged.")
            return

        start_idx = 0 if last_n_batches is None else max(0, self.stats['num_batches'] - last_n_batches)
        n = self.stats['num_batches'] - start_idx

        print(f"\n[TRAINING] {n} batches")
        print("-"*80)

        print(f"\n  {'Expert':<10} {'ExecCost(mean)':>16} {'ForgetCost(mean)':>18} {'Bid(mean)':>12} {'Wins':>8} {'WinRate':>9}")
        print("  " + "-"*75)
        total_wins = sum(self.stats['expert_wins'])
        for i in range(self.num_experts):
            ec = self.stats['exec_cost_history'][i][start_idx:]
            fc = self.stats['forget_cost_history'][i][start_idx:]
            bids = self.stats['bid_history'][i][start_idx:]
            wins = self.stats['expert_wins'][i]
            wr = wins / total_wins if total_wins > 0 else 0
            print(f"  Expert {i:<4} {np.mean(ec):>16.4f} {np.mean(fc):>18.4f} {np.mean(bids):>12.4f} {wins:>8} {wr*100:>8.1f}%")

        # Warn if exec_cost is near-zero (confidently wrong problem)
        print()
        for i in range(self.num_experts):
            ec = self.stats['exec_cost_history'][i][start_idx:]
            if len(ec) > 0 and np.mean(ec) < 0.05:
                print(f"  [WARN] Expert {i}: mean exec_cost={np.mean(ec):.5f} — near-zero, "
                      f"'confidently wrong' problem likely")

        print(f"\n  Win distribution:")
        for i in range(self.num_experts):
            w = self.stats['expert_wins'][i]
            r = w / total_wins if total_wins > 0 else 0
            bar = "#" * int(r * 40) + "-" * (40 - int(r * 40))
            print(f"  Expert {i}: [{bar}] {r*100:5.1f}%")

    def _print_eval_diagnostics(self):
        n = self.eval_stats['num_eval_batches']
        strategy = self.routing_strategy
        primary_label = "Distance" if strategy == 'prototype' else "ExecCost"

        print(f"\n[EVALUATION] {n} batches  (strategy={strategy})")
        print("-"*80)

        # Per-expert eval bid breakdown
        print(f"\n  {'Expert':<10} {primary_label+' (mean)':>18} {'ForgetCost(mean)':>18} {'Bid(mean)':>12} {'EvalWins':>10}")
        print("  " + "-"*65)

        eval_winners = self.eval_stats['eval_winner_history']
        total_eval = len(eval_winners)
        eval_win_counts = {i: eval_winners.count(i) for i in range(self.num_experts)}

        for i in range(self.num_experts):
            pc = self.eval_stats['primary_cost_history'][i]
            fc = self.eval_stats['eval_forget_cost_history'][i]
            bids = self.eval_stats['eval_bid_history'][i]
            wins = eval_win_counts.get(i, 0)
            print(f"  Expert {i:<4} {np.mean(pc):>18.4f} {np.mean(fc):>18.4f} {np.mean(bids):>12.4f} {wins:>8} ({wins/total_eval*100:.1f}%)")

        # Prototype-specific: distance separation analysis
        if strategy == 'prototype' and len(self.eval_stats['primary_cost_history'][0]) > 0:
            print(f"\n  [PROTOTYPE DISTANCE SEPARATION]")
            print(f"  Ideal: winner should have LOWER distance than losers.")
            all_winner_distances = []
            all_loser_distances = []
            for log in self.eval_logs:
                w = log['w']
                for i, e in enumerate(log['e']):
                    if i == w:
                        all_winner_distances.append(e['pc'])
                    else:
                        all_loser_distances.append(e['pc'])
            if all_winner_distances and all_loser_distances:
                wm = np.mean(all_winner_distances)
                lm = np.mean(all_loser_distances)
                separation = (lm - wm) / (lm + 1e-8) * 100
                print(f"  Mean winner distance:  {wm:.4f}")
                print(f"  Mean loser distance:   {lm:.4f}")
                print(f"  Separation (loser-winner)/loser: {separation:.1f}%")
                if separation < 10:
                    print(f"  [WARN] Low separation — prototype routing may be near-random")
                elif separation > 30:
                    print(f"  [OK] Good separation — prototype routing is discriminative")

        # Per-digit routing
        print(f"\n  [PER-DIGIT ROUTING]")
        print(f"  {'Digit':<8} {'Total':>8} {'Accuracy':>10}  Routing distribution")
        print("  " + "-"*65)
        for d in range(10):
            total = self.eval_stats['digit_total'][d]
            correct = self.eval_stats['digit_correct'][d]
            if total == 0:
                continue
            acc = correct / total * 100
            routing = self.eval_stats['digit_routing'][d]
            # Primary expert for this digit
            primary = max(routing, key=routing.get)
            routing_str = " ".join([f"E{i}:{routing[i]}" for i in range(self.num_experts) if routing[i] > 0])
            acc_flag = "" if acc >= 80 else " [WARN low]"
            print(f"  Digit {d:<4} {total:>8} {acc:>9.1f}%  [{routing_str}] -> primary=E{primary}{acc_flag}")

        # Check for routing collapse (all samples go to one expert)
        dominant = max(eval_win_counts.values()) / total_eval if total_eval > 0 else 0
        if dominant > 0.9:
            print(f"\n  [WARN] Eval routing collapse: one expert gets {dominant*100:.0f}% of batches")
        # Check for uniform routing (no specialization at eval)
        eval_rates = [eval_win_counts.get(i, 0) / total_eval for i in range(self.num_experts)]
        if np.var(eval_rates) < 0.002 and self.num_experts > 1:
            print(f"\n  [WARN] Eval routing near-uniform — routing not discriminating between experts")

    # =========================================================================
    # LOAD BALANCING METRICS (Experiment 4)
    # =========================================================================

    def compute_utilization_metrics(self, win_counts: Dict[int, int]) -> Dict:
        """Compute load-balancing metrics from expert win counts."""
        total = sum(win_counts.values())
        if total == 0:
            return {'entropy': 0.0, 'max_entropy': 0.0, 'normalized_entropy': 0.0, 'gini': 0.0}

        n = self.num_experts
        probs = np.array([win_counts.get(i, 0) / total for i in range(n)])

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(n) if n > 1 else 0.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        sorted_probs = np.sort(probs)
        cumulative = np.cumsum(sorted_probs)
        gini = 1.0 - 2.0 * np.sum(cumulative) / (n * np.sum(sorted_probs)) if np.sum(sorted_probs) > 0 else 0.0

        return {
            'entropy': round(entropy, 4),
            'max_entropy': round(max_entropy, 4),
            'normalized_entropy': round(normalized_entropy, 4),
            'gini': round(gini, 4)
        }

    def _print_load_balancing_diagnostics(self):
        """Print load balancing analysis for training and eval."""
        print(f"\n[LOAD BALANCING ANALYSIS]")
        print("-"*80)

        total_train = sum(self.stats['expert_wins'])
        if total_train > 0:
            train_counts = {i: self.stats['expert_wins'][i] for i in range(self.num_experts)}
            train_metrics = self.compute_utilization_metrics(train_counts)
            print(f"\n  Training ({total_train} batches):")
            print(f"    Entropy:     {train_metrics['entropy']:.4f} / {train_metrics['max_entropy']:.4f} "
                  f"(normalized: {train_metrics['normalized_entropy']:.4f})")
            print(f"    Gini coeff:  {train_metrics['gini']:.4f}")
            if train_metrics['normalized_entropy'] > 0.7:
                print(f"    [OK] Good load balance (normalized entropy > 0.7)")
            elif train_metrics['normalized_entropy'] < 0.4:
                print(f"    [WARN] Poor load balance (normalized entropy < 0.4)")

        eval_winners = self.eval_stats['eval_winner_history']
        if eval_winners:
            eval_counts = {i: eval_winners.count(i) for i in range(self.num_experts)}
            eval_metrics = self.compute_utilization_metrics(eval_counts)
            print(f"\n  Evaluation ({len(eval_winners)} batches):")
            print(f"    Entropy:     {eval_metrics['entropy']:.4f} / {eval_metrics['max_entropy']:.4f} "
                  f"(normalized: {eval_metrics['normalized_entropy']:.4f})")
            print(f"    Gini coeff:  {eval_metrics['gini']:.4f}")

        if self.stats['num_batches'] > 0:
            print(f"\n  [FORGETTING COST AS LOAD BALANCER]")
            for i in range(self.num_experts):
                fc = self.stats['forget_cost_history'][i]
                if len(fc) > 0:
                    wins = self.stats['expert_wins'][i]
                    wr = wins / total_train * 100 if total_train > 0 else 0
                    print(f"    Expert {i}: mean_forget_cost={np.mean(fc):.4f}, "
                          f"final_forget_cost={fc[-1]:.4f}, win_rate={wr:.1f}%")

    def _print_prototype_state_diagnostics(self):
        print(f"\n[PROTOTYPE STATE SNAPSHOTS] {len(self.prototype_state_log)} events")
        print("-"*80)
        for snap in self.prototype_state_log:
            eid = snap['expert_id']
            classes = snap['classes_seen']
            counts = snap['sample_counts']
            total = snap['total_samples']
            mah = "Mahalanobis" if snap['has_mahalanobis'] else "Euclidean (fallback)"
            counts_str = ", ".join([f"cls{c}:{counts.get(str(c), 0)}" for c in classes])
            print(f"  Expert {eid} @ batch {snap['batch_idx']} [{snap['event']}]: "
                  f"classes={classes}, total={total}, dist={mah}")
            print(f"    Counts: {counts_str}")

        print()

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save_logs(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Summary stats
        per_expert_train = {}
        for i in range(self.num_experts):
            ec = self.stats['exec_cost_history'][i]
            fc = self.stats['forget_cost_history'][i]
            bids = self.stats['bid_history'][i]
            if ec:
                per_expert_train[f'expert_{i}'] = {
                    'wins': self.stats['expert_wins'][i],
                    'win_rate': round(self.stats['expert_wins'][i] / max(self.stats['num_batches'], 1), 4),
                    'exec_cost': {'mean': round(np.mean(ec), 6), 'std': round(np.std(ec), 6)},
                    'forget_cost': {'mean': round(np.mean(fc), 6), 'std': round(np.std(fc), 6)},
                    'bid': {'mean': round(np.mean(bids), 6), 'std': round(np.std(bids), 6)}
                }

        per_expert_eval = {}
        if self.eval_stats['num_eval_batches'] > 0:
            eval_winners = self.eval_stats['eval_winner_history']
            total_eval = len(eval_winners)
            primary_label = 'distance_score' if self.routing_strategy == 'prototype' else 'exec_cost'
            for i in range(self.num_experts):
                pc = self.eval_stats['primary_cost_history'][i]
                fc = self.eval_stats['eval_forget_cost_history'][i]
                bids = self.eval_stats['eval_bid_history'][i]
                wins = eval_winners.count(i)
                if pc:
                    per_expert_eval[f'expert_{i}'] = {
                        'eval_wins': wins,
                        'eval_win_rate': round(wins / total_eval, 4),
                        primary_label: {'mean': round(np.mean(pc), 6), 'std': round(np.std(pc), 6)},
                        'forget_cost': {'mean': round(np.mean(fc), 6), 'std': round(np.std(fc), 6)},
                        'bid': {'mean': round(np.mean(bids), 6), 'std': round(np.std(bids), 6)}
                    }

        # Load balancing metrics
        train_balance = {}
        total_train = sum(self.stats['expert_wins'])
        if total_train > 0:
            train_counts = {i: self.stats['expert_wins'][i] for i in range(self.num_experts)}
            train_balance = self.compute_utilization_metrics(train_counts)

        eval_balance = {}
        if self.eval_stats['num_eval_batches'] > 0:
            eval_winners = self.eval_stats['eval_winner_history']
            eval_counts = {i: eval_winners.count(i) for i in range(self.num_experts)}
            eval_balance = self.compute_utilization_metrics(eval_counts)

        data = {
            '_format': {
                'description': 'MoB bid diagnostics log',
                'routing_strategy': self.routing_strategy,
                'train_batch_format': 'b=batch_idx, t=task_id, w=winner_id, e=[[exec,forget,bid], ...]',
                'eval_batch_format': 'b=batch_idx, w=winner_id, e=[{pc,fc,np,nf,bid}, ...]',
                'pc_meaning': 'distance_score (prototype) or exec_cost (pseudo-label)',
            },
            'config': {
                'num_experts': self.num_experts,
                'alpha': self.alpha,
                'beta': self.beta,
                'routing_strategy': self.routing_strategy,
            },
            'training_summary': {
                'total_batches': self.stats['num_batches'],
                'expert_wins': self.stats['expert_wins'],
                'per_expert': per_expert_train,
                'load_balance': train_balance
            },
            'eval_summary': {
                'total_eval_batches': self.eval_stats['num_eval_batches'],
                'per_expert': per_expert_eval,
                'load_balance': eval_balance,
                'per_digit_routing': {
                    str(d): {
                        'total': self.eval_stats['digit_total'][d],
                        'correct': self.eval_stats['digit_correct'][d],
                        'accuracy': round(
                            self.eval_stats['digit_correct'][d] / self.eval_stats['digit_total'][d], 4
                        ) if self.eval_stats['digit_total'][d] > 0 else 0.0,
                        'routing': {str(k): v for k, v in self.eval_stats['digit_routing'][d].items()}
                    }
                    for d in range(10) if self.eval_stats['digit_total'][d] > 0
                }
            },
            'prototype_state_log': self.prototype_state_log,
            'training_batches': self.batch_logs,
            'eval_batches': self.eval_logs
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Bid logs saved to: {filepath}")
        print(f"     Training: {self.stats['num_batches']} batches | "
              f"Eval: {self.eval_stats['num_eval_batches']} batches | "
              f"Prototype snapshots: {len(self.prototype_state_log)}")
