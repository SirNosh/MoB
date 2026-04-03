"""
Expert Pool management for MoB: Mixture of Bidders.

This module provides the ExpertPool class, which manages a collection of
independent MoBExpert agents.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  # Ensure F is imported
from typing import Dict, List, Optional, Tuple

from .expert import MoBExpert
from .models import create_model
from .prototype_store import PrototypeStore


class ShiftDetector:
    """
    Detects distribution shifts using Exponential Moving Average (EMA) of execution cost.
    """
    def __init__(self, alpha: float = 0.99, threshold_multiplier: float = 50.0):
        self.alpha = alpha  # Smoothing factor (high = slow adaptation)
        self.threshold_multiplier = threshold_multiplier
        self.ema_cost = None
        self.shift_cooldown = 0

    def update(self, cost: float) -> bool:
        """
        Update the tracker and check for shift.
        Returns True if a significant upward spike is detected.
        """
        # If cooldown is active, decrement and ignore (allow new task to stabilize)
        if self.shift_cooldown > 0:
            self.shift_cooldown -= 1
            # Still update EMA quickly during cooldown to adapting to new normal
            if self.ema_cost is None:
                self.ema_cost = cost
            else:
                self.ema_cost = 0.5 * self.ema_cost + 0.5 * cost
            return False

        if self.ema_cost is None:
            self.ema_cost = cost
            return False

        # Check for spike with a minimum floor to avoid noise at very low losses
        # If ema_cost is tiny (e.g. 0.001), a small jump to 0.05 is 50x but meaningless.
        # We enforce that the cost must also be significantly larger than a baseline (e.g. 0.5)
        # effectively: cost > max(ema_cost, 0.5) * threshold
        baseline = max(self.ema_cost, 0.5) 
        is_shift = cost > (baseline * self.threshold_multiplier)

        # Update EMA
        self.ema_cost = self.alpha * self.ema_cost + (1 - self.alpha) * cost
        
        if is_shift:
            # Reset/Cooldwon to absorb the new distribution
            self.shift_cooldown = 50 
            self.ema_cost = cost # Jump to new level
            
        return is_shift


class ExpertPool:
    """
    A collection of independent MoBExpert agents.

    This class manages the experts but contains NO centralized gater.
    The auction mechanism itself serves as the dynamic routing layer.
    """

    def __init__(
        self,
        num_experts: int,
        expert_config: Dict,
        device: Optional[torch.device] = None,
        use_shift_detection: bool = False,
        reset_optimizer: bool = False,
        idle_threshold: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Initialize the expert pool.
        """
        self.num_experts = num_experts
        self.expert_config = expert_config
        self.device = device if device is not None else torch.device('cpu')
        self.experts: List[MoBExpert] = []

        # Shift Detection
        self.shift_detector = ShiftDetector() if use_shift_detection else None

        # =====================================================================
        # Optimizer Reset Configuration (for idle-based reset in continual learning)
        # =====================================================================
        self.reset_optimizer = reset_optimizer
        self.idle_threshold = idle_threshold
        self.learning_rate = learning_rate
        self.global_batch_idx = 0  # Track global batch index
        # =====================================================================
        self.load_bias = np.zeros(num_experts)
        self.expert_win_counts = np.zeros(num_experts)
        self.total_auctions = 0
        self.use_conscience = expert_config.get('use_conscience', False)
        self.conscience_rate = expert_config.get('conscience_rate', 0.005)
        self.conscience_decay = expert_config.get('conscience_decay', 0.999)
        self.conscience_max_bias = expert_config.get('conscience_max_bias', 0.1)
        self.seed_idle_prototypes_enabled = expert_config.get('seed_idle_prototypes', False)
        self.routing_temperature = expert_config.get('routing_temperature', 0.0)
        self.temp_min = expert_config.get('temp_min', 0.01)
        self.temp_decay = expert_config.get('temp_decay', 0.995)
        self.current_temperature = self.routing_temperature
        self._prototypes_seeded = False
        self._latest_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # Create experts
        for i in range(num_experts):
            model = self._create_expert_model(expert_config)
            expert = MoBExpert(
                expert_id=i,
                model=model,
                alpha=expert_config.get('alpha', 0.5),
                beta=expert_config.get('beta', 0.5),
                lambda_ewc=expert_config.get('lambda_ewc', 5000),
                forgetting_cost_scale=expert_config.get('forgetting_cost_scale', 1.0),
                device=self.device
            )
            self.experts.append(expert)

    def _expert_has_prototypes(self, expert: MoBExpert) -> bool:
        return (
            expert.prototype_store is not None
            and len(expert.prototype_store.centroids) > 0
        )

    def seed_idle_prototypes(
        self,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None
    ) -> int:
        """
        Seed prototype stores for experts that do not have prototypes yet.
        Uses pseudo-labels from each expert's own logits on the provided batch.
        """
        if x is None or y is None:
            if self._latest_batch is None:
                return 0
            x, y = self._latest_batch

        x_device = x.to(self.device)
        seeded = 0

        for expert in self.experts:
            if self._expert_has_prototypes(expert):
                continue
            if not hasattr(expert.model, 'forward_features'):
                continue

            was_training = expert.model.training
            expert.model.eval()
            with torch.no_grad():
                features, logits = expert.model.forward_features(x_device)
                pseudo_labels = logits.argmax(dim=-1).detach()
            if was_training:
                expert.model.train()

            if expert.prototype_store is None:
                expert.prototype_store = PrototypeStore(
                    feature_dim=features.shape[1],
                    device=self.device
                )

            expert.prototype_store.update(features.detach(), pseudo_labels)
            seeded += 1

        return seeded

    def _create_expert_model(self, config: Dict) -> nn.Module:
        """
        Factory method for creating expert neural networks.
        """
        return create_model(
            architecture=config['architecture'],
            num_classes=config['num_classes'],
            input_channels=config.get('input_channels', 1),
            dropout=config.get('dropout', 0.5),
            input_size=config.get('input_size', 784),
            hidden_sizes=config.get('hidden_sizes', [256, 128])
        )

    def collect_bids(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_routing: str = 'label'
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Gathers bids from all experts for a given batch.

        Args:
            x: Input batch.
            y: Labels (used for label-based routing and training loss).
            train_routing: 'label' (default, uses exec_cost with true y) or
                           'prototype' (uses prototype distance instead of exec_cost).

        Each expert uses its own Z-Score normalization based on running statistics,
        which preserves bid independence (each bid depends only on the expert's own history).
        """
        import math

        if train_routing == 'prototype':
            # Experiment 3: Prototype-distance-based training routing
            if self.seed_idle_prototypes_enabled and not self._prototypes_seeded:
                idle_exists = any(not self._expert_has_prototypes(expert) for expert in self.experts)
                if idle_exists:
                    seeded_count = self.seed_idle_prototypes()
                    if seeded_count == 0:
                        seeded_count = self.seed_idle_prototypes(x, y)
                    if seeded_count > 0:
                        print(f"[Prototype Seeding] Seeded {seeded_count} idle experts")
                    self._prototypes_seeded = True

            bids = np.zeros(self.num_experts)
            components = []
            for i, expert in enumerate(self.experts):
                expert.total_batches_seen += 1

                # Check if prototype routing is available for this expert
                has_prototypes = self._expert_has_prototypes(expert) and hasattr(expert.model, 'forward_features')

                # Get features and logits
                expert.model.eval()
                with torch.no_grad():
                    if hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x.to(self.device))
                    else:
                        logits = expert.model(x.to(self.device))
                        features = None
                expert.model.train()

                if has_prototypes and features is not None:
                    distance_score = expert.prototype_store.compute_routing_score(features)
                    routing_tag = 'prototype'
                else:
                    distance_score = 100.0
                    routing_tag = 'prototype_default'

                pseudo_labels = logits.argmax(dim=-1).detach()
                raw_forget = expert.forget_estimator.compute_forgetting_cost(
                    x.to(self.device), pseudo_labels
                )

                norm_distance = distance_score / 10.0
                norm_forget = math.log1p(raw_forget) / 10.0
                bid = expert.alpha * norm_distance + expert.beta * norm_forget

                if self.use_conscience:
                    adjusted_bid = bid + self.load_bias[i]
                    comp_bid = adjusted_bid
                else:
                    comp_bid = bid

                comp = {
                    'exec_cost': distance_score,
                    'forget_cost': raw_forget,
                    'norm_exec_cost': norm_distance,
                    'norm_forget_cost': norm_forget,
                    'bid': comp_bid,
                    'alpha': expert.alpha,
                    'beta': expert.beta,
                    'routing': routing_tag
                }

                bids[i] = comp_bid
                components.append(comp)
            return bids, components
        else:
            # Default: label-based routing
            bids = np.zeros(self.num_experts)
            components = []
            for i, expert in enumerate(self.experts):
                bid, comp = expert.compute_bid(x, y)
                if self.use_conscience:
                    adjusted_bid = bid + self.load_bias[i]
                    comp['bid'] = adjusted_bid
                    bids[i] = adjusted_bid
                else:
                    bids[i] = bid
                components.append(comp)
            return bids, components

    def update_conscience_bias(self, winner_id: int):
        """Update per-expert load-balancing bias after each auction."""
        if not self.use_conscience:
            return

        self.total_auctions += 1
        self.expert_win_counts[winner_id] += 1

        target_freq = 1.0 / self.num_experts
        for i in range(self.num_experts):
            actual_freq = self.expert_win_counts[i] / self.total_auctions
            self.load_bias[i] += self.conscience_rate * (actual_freq - target_freq)

        self.load_bias *= self.conscience_decay
        self.load_bias = np.clip(
            self.load_bias,
            -self.conscience_max_bias,
            self.conscience_max_bias
        )

    def select_winner(
        self,
        bids: np.ndarray,
        train_routing: str = 'label',
        is_training: bool = True
    ) -> int:
        """
        Select winning expert from bids.

        Temperature-annealed stochastic routing applies ONLY during training
        when prototype routing is active. All other cases use deterministic argmin.
        """
        if (
            is_training
            and train_routing == 'prototype'
            and self.routing_temperature > 0.0
            and self.current_temperature > self.temp_min
        ):
            neg_bids = -np.asarray(bids, dtype=np.float64) / self.current_temperature
            neg_bids = neg_bids - np.max(neg_bids)  # numerical stability
            probs = np.exp(neg_bids)
            probs_sum = probs.sum()
            if probs_sum <= 0.0:
                winner_id = int(np.argmin(bids))
            else:
                probs = probs / probs_sum
                winner_id = int(np.random.choice(self.num_experts, p=probs))

            self.current_temperature = max(
                self.temp_min,
                self.current_temperature * self.temp_decay
            )
            return winner_id

        return int(np.argmin(bids))

    def _should_reset_optimizer_on_shift(self, expert, shift_detected: bool) -> bool:
        """
        Check if expert's optimizer should be reset based on shift detection.

        For continual learning, we reset when:
        1. reset_optimizer is enabled AND
        2. A distribution shift was detected AND
        3. Expert has trained before (has Fisher - nothing to reset otherwise)

        This provides a natural "task boundary" signal in task-free learning.
        """
        if not self.reset_optimizer:
            return False

        if not shift_detected:
            return False

        # Never reset if expert hasn't trained yet
        if not expert.forget_estimator.has_fisher():
            return False

        return True

    def train_winner(
        self,
        winner_id: int,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizers: List[torch.optim.Optimizer]
    ) -> Dict:
        """
        Train the winning expert on the batch.
        """
        winner = self.experts[winner_id]
        self._latest_batch = (x, y)

        # Check for distribution shift BEFORE training
        shift_detected = False
        if self.shift_detector:
            # We use the current loss as the signal
            # Note: winner.exec_estimator.compute_predicted_loss(x, y) was likely called during bidding
            # so we could optimization this, but recalculating is safer for clean code.
            with torch.no_grad():
                current_loss = winner.exec_estimator.compute_predicted_loss(x, y)
            shift_detected = self.shift_detector.update(current_loss)

        # =====================================================================
        # Optimizer Reset on Shift Detection
        # Key insight: Reset optimizer when distribution shift is detected.
        # This provides a natural "task boundary" signal in task-free learning.
        # =====================================================================
        optimizer_reset = False
        if self._should_reset_optimizer_on_shift(winner, shift_detected):
            # Reset optimizer for this expert
            optimizers[winner_id] = torch.optim.Adam(
                winner.model.parameters(),
                lr=self.learning_rate
            )
            print(f"  [Optimizer Reset] Expert {winner_id} optimizer reset (shift detected)")
            optimizer_reset = True
        # =====================================================================

        metrics = winner.train_on_batch(x, y, optimizers[winner_id])
        metrics['shift_detected'] = shift_detected
        metrics['optimizer_reset'] = optimizer_reset

        # Update tracking for optimizer reset logic
        winner.last_won_global_batch = self.global_batch_idx
        self.global_batch_idx += 1
        self.update_conscience_bias(winner_id)

        return metrics

    def consolidate(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 200,
        expert_ids: Optional[List[int]] = None
    ):
        """
        Consolidates knowledge for specific experts by updating their EWC parameters.
        Triggered when a distribution shift is detected.
        
        Args:
            expert_ids: List of expert IDs to update. If None, updates ALL experts.
        """
        targets = expert_ids if expert_ids is not None else range(len(self.experts))
        
        for i in targets:
            self.experts[i].consolidate(dataloader, num_samples=num_samples)

    def evaluate_all(
        self,
        dataloader: torch.utils.data.DataLoader,
        routing_strategy: str = 'pseudo_label'
    ) -> Dict:
        """
        Evaluates the MoB system using auction-based routing.

        Args:
            dataloader: Test data loader.
            routing_strategy: 'pseudo_label' (default) or 'prototype' (feature-distance).

        IMPORTANT: Neither strategy uses ground truth labels for routing.
        """
        import math
        results = {}
        all_labels = []
        winner_preds = []

        use_prototype = (routing_strategy == 'prototype')

        # 1. Calculate individual expert accuracies (for diagnostics)
        for i, expert in enumerate(self.experts):
            expert.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in dataloader:
                    x_device = x.to(self.device)
                    y_device = y.to(self.device)
                    logits = expert.model(x_device)
                    preds = logits.argmax(dim=-1)
                    correct += (preds == y_device).sum().item()
                    total += len(y_device)
            accuracy = correct / total if total > 0 else 0.0
            results[f'expert_{i}_accuracy'] = accuracy

        # 2. Calculate MoB accuracy using auction routing (no ground truth labels!)
        for x, y in dataloader:
            x_device = x.to(self.device)
            y_device = y.to(self.device)
            all_labels.append(y_device.cpu())

            batch_bids = np.zeros(self.num_experts)
            batch_logits = []

            for i, expert in enumerate(self.experts):
                expert.model.eval()
                with torch.no_grad():
                    if use_prototype and hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x_device)
                    else:
                        logits = expert.model(x_device)
                        features = None
                    batch_logits.append(logits)

                if use_prototype and expert.prototype_store is not None and features is not None:
                    # Prototype distance replaces exec_cost
                    distance_score = expert.prototype_store.compute_routing_score(features)

                    # Keep forget cost (still useful signal)
                    pseudo_labels = logits.argmax(dim=-1).detach()
                    forget_cost = expert.forget_estimator.compute_forgetting_cost(x_device, pseudo_labels)

                    # Bid: distance replaces exec_cost
                    # distance_score typically ranges 0-20; normalize to ~0-1
                    norm_distance = distance_score / 10.0
                    norm_forget = math.log1p(forget_cost) / 10.0
                    batch_bids[i] = expert.alpha * norm_distance + expert.beta * norm_forget
                else:
                    # Pseudo-label routing (original)
                    pseudo_labels = logits.argmax(dim=-1).detach()
                    raw_exec = F.cross_entropy(logits, pseudo_labels).item()
                    forget_cost = expert.forget_estimator.compute_forgetting_cost(x_device, pseudo_labels)

                    norm_exec = raw_exec / 2.5
                    norm_forget = math.log1p(forget_cost) / 10.0
                    batch_bids[i] = expert.alpha * norm_exec + expert.beta * norm_forget

            # Select Winner: Lowest Bid = WINS the auction
            auction_winner_id = np.argmin(batch_bids)

            # DEBUG-LOG: Print evaluation decision occasionally
            if len(all_labels) == 1:
                strategy_str = "prototype" if use_prototype else "pseudo-label"
                print(f"[EVAL DEBUG ({strategy_str})] Batch 0: Bids={np.round(batch_bids, 4)}")
                print(f"             Winner={auction_winner_id} (Min Bid)")

            # Get the winning expert's predictions for this batch
            winning_logits = batch_logits[auction_winner_id]
            winning_preds_batch = winning_logits.argmax(dim=-1).cpu()
            winner_preds.append(winning_preds_batch)

        # Concatenate all predictions and labels
        if all_labels:
            all_labels = torch.cat(all_labels)
            winner_preds = torch.cat(winner_preds)

            ensemble_accuracy = (winner_preds == all_labels).float().mean().item()
            results['ensemble_accuracy'] = ensemble_accuracy
        else:
            results['ensemble_accuracy'] = 0.0

        return results

    def evaluate_all_per_sample(
        self,
        dataloader: torch.utils.data.DataLoader,
        top_k: int = 1,
        temperature: float = 1.0,
        eval_bid_mode: str = 'full',
        routing_strategy: str = 'prototype'
    ) -> Dict:
        """
        Per-sample top-k evaluation using prototype distance routing.

        Each sample is independently routed to its top-k experts by lowest
        prototype distance, and outputs are combined via softmax-weighted sum.

        Args:
            dataloader: Test data loader.
            top_k: Number of experts to combine per sample (1=winner-take-all, 2=Mixtral-style).
            temperature: Softmax temperature for combining expert outputs (low=sharp, high=uniform).
            eval_bid_mode: 'full' (distance + forget) or 'distance_only' (skip EWC component).
            routing_strategy: 'prototype' or 'pseudo_label'.

        Returns:
            Dict with accuracy, per-expert selection counts, routing stats.
        """
        import math

        use_prototype = (routing_strategy == 'prototype')

        all_labels = []
        all_preds = []
        expert_sample_wins = {i: 0 for i in range(self.num_experts)}
        # Track how often top-2 experts agree on prediction
        top2_agreement_count = 0
        top2_total = 0

        for x, y in dataloader:
            x_device = x.to(self.device)
            y_device = y.to(self.device)
            batch_size = x_device.shape[0]
            all_labels.append(y_device.cpu())

            # Collect per-expert features, logits, and per-sample distances
            expert_logits = []  # list of (B, num_classes) tensors
            expert_distances = []  # list of (B,) tensors — per-sample distances

            for i, expert in enumerate(self.experts):
                expert.model.eval()
                with torch.no_grad():
                    if use_prototype and hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x_device)
                    else:
                        logits = expert.model(x_device)
                        features = None
                    expert_logits.append(logits)

                if use_prototype and expert.prototype_store is not None and features is not None:
                    per_sample_dist = expert.prototype_store.compute_per_sample_distances(features)
                else:
                    # No prototypes: high default distance (same scale)
                    per_sample_dist = torch.full((batch_size,), 100.0, device=x_device.device)

                if eval_bid_mode == 'full':
                    pseudo_labels = logits.argmax(dim=-1).detach()
                    forget_cost = expert.forget_estimator.compute_forgetting_cost(
                        x_device, pseudo_labels
                    )
                    norm_dist = per_sample_dist / 10.0
                    norm_forget = math.log1p(forget_cost) / 10.0
                    per_sample_bid = expert.alpha * norm_dist + expert.beta * norm_forget
                else:
                    per_sample_bid = per_sample_dist

                expert_distances.append(per_sample_bid)

            # Stack: (num_experts, B) -> (B, num_experts)
            distance_matrix = torch.stack(expert_distances, dim=0).T  # (B, N)
            logits_stack = torch.stack(expert_logits, dim=0)  # (N, B, C)

            # Per-sample top-k selection
            # Lower distance = better → select k experts with smallest distance
            _, top_k_indices = torch.topk(distance_matrix, k=min(top_k, self.num_experts),
                                          dim=1, largest=False)  # (B, k)
            top_k_distances = torch.gather(distance_matrix, 1, top_k_indices)  # (B, k)

            if top_k == 1:
                # Winner-take-all: just use the single best expert per sample
                winners = top_k_indices.squeeze(1)  # (B,)
                batch_preds = torch.zeros(batch_size, dtype=torch.long, device=x_device.device)
                for b in range(batch_size):
                    w = winners[b].item()
                    batch_preds[b] = logits_stack[w, b].argmax()
                    expert_sample_wins[w] += 1
            else:
                # Top-k combination: weighted sum of logits
                # weights = softmax(-distance / temperature) over top-k experts
                weights = F.softmax(-top_k_distances / temperature, dim=1)  # (B, k)

                # Gather logits for top-k experts per sample
                # logits_stack: (N, B, C) -> need (B, k, C)
                num_classes = logits_stack.shape[2]
                combined_logits = torch.zeros(batch_size, num_classes, device=x_device.device)

                for b in range(batch_size):
                    for j in range(min(top_k, self.num_experts)):
                        expert_idx = top_k_indices[b, j].item()
                        combined_logits[b] += weights[b, j] * logits_stack[expert_idx, b]
                    # Count primary (closest) expert as winner
                    expert_sample_wins[top_k_indices[b, 0].item()] += 1

                batch_preds = combined_logits.argmax(dim=1)

                # Track top-2 agreement
                if top_k >= 2:
                    for b in range(batch_size):
                        e1 = top_k_indices[b, 0].item()
                        e2 = top_k_indices[b, 1].item()
                        pred1 = logits_stack[e1, b].argmax().item()
                        pred2 = logits_stack[e2, b].argmax().item()
                        if pred1 == pred2:
                            top2_agreement_count += 1
                        top2_total += 1

            all_preds.append(batch_preds.cpu())

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
        accuracy = (all_preds == all_labels).float().mean().item()

        total_samples = sum(expert_sample_wins.values())
        results = {
            'ensemble_accuracy': accuracy,
            'expert_sample_wins': expert_sample_wins,
            'expert_sample_rates': {k: v / total_samples for k, v in expert_sample_wins.items()},
            'top_k': top_k,
            'eval_bid_mode': eval_bid_mode,
            'temperature': temperature,
        }

        if top2_total > 0:
            results['top2_agreement'] = top2_agreement_count / top2_total

        return results

    def get_expert_statistics(self) -> List[Dict]:
        """Get statistics for all experts."""
        return [expert.get_statistics() for expert in self.experts]

    def reset_statistics(self):
        """Reset statistics for all experts."""
        for expert in self.experts:
            expert.reset_statistics()

    def save_all(self, directory: str):
        """Save all experts to a directory."""
        import os
        os.makedirs(directory, exist_ok=True)
        for expert in self.experts:
            path = os.path.join(directory, f'expert_{expert.expert_id}.pt')
            expert.save(path)

    def load_all(self, directory: str):
        """Load all experts from a directory."""
        import os
        for expert in self.experts:
            path = os.path.join(directory, f'expert_{expert.expert_id}.pt')
            if os.path.exists(path):
                expert.load(path)

    def __len__(self) -> int:
        return self.num_experts

    def __getitem__(self, idx: int) -> MoBExpert:
        return self.experts[idx]

    def __repr__(self) -> str:
        return f"ExpertPool(num_experts={self.num_experts}, device={self.device})"