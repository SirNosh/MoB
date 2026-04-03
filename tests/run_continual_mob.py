"""
Continual MoB Experiment Runner (Task-Free / Digit-Granular).

Features:
1. Continuous Data Stream (ConcatDataset).
2. Auction-Based Routing for both Training and Evaluation.
3. Per-Digit Diagnostics (Online Accuracy & Routing).
4. Shift Detection & Selective Consolidation.
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
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORT FROM contibualmob (The new environment)
from contibualmob.pool import ExpertPool
from contibualmob.auction import PerBatchAuction
from contibualmob.bid_diagnostics import BidLogger
from contibualmob.utils import set_seed

# Import dataset creation (safe to reuse)
from tests.test_baselines import create_split_mnist


def run_continual_experiment(train_tasks, test_tasks, config):
    print("\n" + "="*70)
    print("CONTINUAL MoB Experiment (Task-Free / Digit-Granular)")
    print("="*70)

    device = torch.device(config['device'])

    # Expert configuration
    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': config['alpha'],
        'beta': config['beta'],
        'lambda_ewc': config['lambda_ewc'],
        'forgetting_cost_scale': config.get('forgetting_cost_scale', 1.0),
        'use_conscience': config.get('use_conscience', False),
        'conscience_rate': config.get('conscience_rate', 0.01),
        'conscience_decay': config.get('conscience_decay', 0.999),
        'seed_idle_prototypes': config.get('seed_idle_prototypes', False),
        'dropout': 0.5
    }

    # Create Pool with Shift Detection ENABLED and optional Optimizer Reset
    pool = ExpertPool(
        config['num_experts'],
        expert_config,
        device=device,
        use_shift_detection=True,
        reset_optimizer=config.get('reset_optimizer', False),
        idle_threshold=config.get('idle_threshold', 100),
        learning_rate=config['learning_rate']
    )
    
    if pool.shift_detector:
        pool.shift_detector.threshold_multiplier = config['shift_threshold']

    auction = PerBatchAuction(config['num_experts'])

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=config['learning_rate'])
        for expert in pool.experts
    ]

    bid_logger = BidLogger(
        num_experts=config['num_experts'],
        alpha=config['alpha'],
        beta=config['beta'],
        routing_strategy=config.get('routing_strategy', 'pseudo_label'),
        log_file=None
    )

    # --- DIGIT GRANULAR STATS ---
    # 1. Routing Dist: Which expert gets which digit?
    expert_digit_dist = {i: {d: 0 for d in range(10)} for i in range(config['num_experts'])}
    
    # 2. Online Training Accuracy: How well does Expert X learn Digit D?
    # stats[expert_id][digit] = {'correct': 0, 'total': 0}
    online_digit_stats = {i: {d: {'correct': 0, 'total': 0} for d in range(10)} for i in range(config['num_experts'])}

    detected_shifts = []

    # =========================================================================
    # CREATE CONTINUOUS DATA STREAM
    # =========================================================================
    epochs_per_task = config.get('epochs_per_task', 2)
    stream_datasets = []
    
    print(f"\nConstructing Data Stream ({len(train_tasks)} Latent Contexts)...")
    for task_loader in train_tasks:
        ds = task_loader.dataset
        for _ in range(epochs_per_task):
            stream_datasets.append(ds)

    full_stream_dataset = torch.utils.data.ConcatDataset(stream_datasets)
    stream_loader = torch.utils.data.DataLoader(
        full_stream_dataset,
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=0
    )
    
    # Buffer for Fisher updates
    replay_buffer = [] 
    MAX_BUFFER_SIZE = 500

    print(f"Stream started: {len(full_stream_dataset)} samples.")
    pbar = tqdm(stream_loader, desc="Learning from Stream")
    
    train_routing = config.get('train_routing', 'label')
    train_routing_warmup = config.get('train_routing_warmup', 1000)
    prototype_routing_active = False

    for batch_idx, (x, y) in enumerate(pbar):
        # 1. Auction Phase
        # Experiment 3: Switch to prototype routing after warmup
        current_routing = 'label'
        if train_routing == 'prototype' and batch_idx >= train_routing_warmup:
            current_routing = 'prototype'
            if not prototype_routing_active:
                tqdm.write(f"\n>>> Switching to PROTOTYPE training routing at batch {batch_idx}")
                prototype_routing_active = True

        bids, components = pool.collect_bids(x, y, train_routing=current_routing)
        winner_id, payment, _ = auction.run_auction(bids)

        # 2. Logging & Buffer
        replay_buffer.append((x, y, winner_id))
        if len(replay_buffer) > MAX_BUFFER_SIZE:
            replay_buffer.pop(0)

        bid_logger.log_batch(batch_idx, bids, components, winner_id, -1)

        # 3. Track Stats (Online Accuracy Check)
        # We perform a quick inference to check if the winner actually knows these digits
        # (This is extra overhead but requested for diagnostics)
        pool.experts[winner_id].model.eval()
        with torch.no_grad():
            logits = pool.experts[winner_id].model(x.to(device))
            preds = logits.argmax(dim=1).cpu()
        
        for i, label in enumerate(y):
            digit = label.item()
            is_correct = (preds[i] == label).item()
            
            # Update Routing Count
            expert_digit_dist[winner_id][digit] += 1
            
            # Update Online Accuracy
            online_digit_stats[winner_id][digit]['total'] += 1
            if is_correct:
                online_digit_stats[winner_id][digit]['correct'] += 1

        # 4. Training Phase
        metrics = pool.train_winner(winner_id, x, y, optimizers)

        # 5. Shift Detection & Consolidation
        if metrics.get('shift_detected', False):
            tqdm.write(f"\n>>> SHIFT DETECTED at Batch {batch_idx} (Winner: Expert {winner_id})")
            detected_shifts.append(batch_idx)

            tqdm.write("    Consolidating knowledge (Fisher Update)...")
            if replay_buffer:
                 buffer_loader = [(bx, by) for bx, by, _ in replay_buffer]
                 buffer_winners = [w for _, _, w in replay_buffer]
                 
                 # Filter: Only consolidate experts with significant presence (>1% of buffer)
                 # This prevents locking in "random" weights for experts that just got lucky on 1-2 batches
                 from collections import Counter
                 counts = Counter(buffer_winners)
                 threshold = len(replay_buffer) * 0.01
                 active_experts = [eid for eid, count in counts.items() if count > threshold]
                 
                 tqdm.write(f"    Active experts in buffer: {list(set(buffer_winners))} -> Filtered: {active_experts}")
                 
                 if active_experts:
                    pool.consolidate(buffer_loader, num_samples=200, expert_ids=active_experts)

                    # Finalize prototypes for experts that didn't get consolidated
                    for eid in range(config['num_experts']):
                        expert = pool.experts[eid]
                        if expert.prototype_store is not None and eid not in active_experts:
                            expert.prototype_store.finalize()

                    # Log prototype state for all experts with stores
                    for eid in range(config['num_experts']):
                        expert = pool.experts[eid]
                        if expert.prototype_store is not None:
                            ps = expert.prototype_store
                            bid_logger.log_prototype_finalize(
                                expert_id=eid,
                                event='consolidate',
                                batch_idx=batch_idx,
                                classes_seen=list(ps.class_count.keys()),
                                sample_counts=dict(ps.class_count),
                                has_mahalanobis=(ps.inv_cov is not None),
                                feature_dim=ps.feature_dim
                            )

            replay_buffer = []

    # =========================================================================
    # FINAL EVALUATION (DIGIT-BOUND)
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL EVALUATION (Per Digit)")
    print("="*70)
    print(f"Detected Shifts: {detected_shifts}")

    # Create One Test Set (All Digits)
    all_test_data = torch.utils.data.ConcatDataset([t.dataset for t in test_tasks])
    test_loader = torch.utils.data.DataLoader(all_test_data, batch_size=config['batch_size'], shuffle=False)
    
    digit_eval_stats = {d: {'correct': 0, 'total': 0, 'routed_to': defaultdict(int)} for d in range(10)}
    
    pool.device = device # Ensure device mapping
    
    import math as _math

    routing_strategy = config.get('routing_strategy', 'pseudo_label')
    use_prototype = (routing_strategy == 'prototype')

    # Finalize any prototypes that haven't been finalized yet (end-of-stream)
    if use_prototype:
        for expert in pool.experts:
            if expert.prototype_store is not None:
                expert.prototype_store.finalize()

    per_sample_mode = config.get('per_sample', False)
    top_k = config.get('top_k', 1)
    eval_bid_mode = config.get('eval_bid_mode', 'full')
    temperature = config.get('temperature', 1.0)

    if per_sample_mode or top_k > 1:
        # Per-sample top-k routing (Experiment 1 + 2)
        print(f"Running Per-Sample Evaluation (top_k={top_k}, bid_mode={eval_bid_mode}, "
              f"temp={temperature}, routing={routing_strategy})...")

        ps_results = pool.evaluate_all_per_sample(
            test_loader,
            top_k=top_k,
            temperature=temperature,
            eval_bid_mode=eval_bid_mode,
            routing_strategy=routing_strategy
        )

        # Map per-sample results into digit_eval_stats
        # Need a second pass for per-digit stats
        for x, y in test_loader:
            x_device, y_device = x.to(device), y.to(device)
            # We already have the global accuracy; fill in per-digit stats from this pass
            # using the same routing as evaluate_all_per_sample
            batch_size = x_device.shape[0]

            expert_logits_list = []
            expert_distances_list = []
            for i, expert in enumerate(pool.experts):
                expert.model.eval()
                with torch.no_grad():
                    if use_prototype and hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x_device)
                    else:
                        logits = expert.model(x_device)
                        features = None
                    expert_logits_list.append(logits)

                if use_prototype and expert.prototype_store is not None and features is not None:
                    per_sample_dist = expert.prototype_store.compute_per_sample_distances(features)
                    if eval_bid_mode == 'distance_only':
                        expert_distances_list.append(per_sample_dist)
                    else:
                        pseudo_labels = logits.argmax(dim=-1).detach()
                        forget_cost = expert.forget_estimator.compute_forgetting_cost(x_device, pseudo_labels)
                        norm_dist = per_sample_dist / 10.0
                        norm_forget = _math.log1p(forget_cost) / 10.0
                        expert_distances_list.append(expert.alpha * norm_dist + expert.beta * norm_forget)
                else:
                    pseudo_labels = logits.argmax(dim=-1).detach()
                    per_sample_ce = F.cross_entropy(logits, pseudo_labels, reduction='none')
                    expert_distances_list.append(per_sample_ce)

            import torch as _torch
            distance_matrix = _torch.stack(expert_distances_list, dim=0).T  # (B, N)
            logits_stack = _torch.stack(expert_logits_list, dim=0)  # (N, B, C)
            _, top_k_idx = _torch.topk(distance_matrix, k=min(top_k, config['num_experts']),
                                        dim=1, largest=False)

            if top_k == 1:
                winners = top_k_idx.squeeze(1)
                for b in range(batch_size):
                    w = winners[b].item()
                    pred = logits_stack[w, b].argmax().item()
                    digit = y_device[b].item()
                    digit_eval_stats[digit]['total'] += 1
                    if pred == digit:
                        digit_eval_stats[digit]['correct'] += 1
                    digit_eval_stats[digit]['routed_to'][w] += 1
            else:
                top_k_distances = _torch.gather(distance_matrix, 1, top_k_idx)
                weights = F.softmax(-top_k_distances / temperature, dim=1)
                num_classes = logits_stack.shape[2]
                for b in range(batch_size):
                    combined = _torch.zeros(num_classes, device=x_device.device)
                    for j in range(min(top_k, config['num_experts'])):
                        eidx = top_k_idx[b, j].item()
                        combined += weights[b, j] * logits_stack[eidx, b]
                    pred = combined.argmax().item()
                    digit = y_device[b].item()
                    w = top_k_idx[b, 0].item()
                    digit_eval_stats[digit]['total'] += 1
                    if pred == digit:
                        digit_eval_stats[digit]['correct'] += 1
                    digit_eval_stats[digit]['routed_to'][w] += 1

        print(f"\n  Per-sample ensemble accuracy: {ps_results['ensemble_accuracy']:.4f}")
        print(f"  Expert selection distribution: {ps_results['expert_sample_rates']}")
        if 'top2_agreement' in ps_results:
            print(f"  Top-2 agreement rate: {ps_results['top2_agreement']:.4f}")

    else:
        # Original per-batch evaluation
        print(f"Running Auction-Based Evaluation on Test Set (routing: {routing_strategy})...")
        eval_batch_idx = 0
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)

            batch_bids = np.zeros(config['num_experts'])
            batch_logits = []
            per_expert_log_data = []

            for i, expert in enumerate(pool.experts):
                expert.model.eval()
                with torch.no_grad():
                    if use_prototype and hasattr(expert.model, 'forward_features'):
                        features, logits = expert.model.forward_features(x)
                    else:
                        logits = expert.model(x)
                        features = None
                    batch_logits.append(logits)

                if use_prototype and expert.prototype_store is not None and features is not None:
                    distance_score = expert.prototype_store.compute_routing_score(features)
                    pseudo_labels = logits.argmax(dim=-1).detach()
                    forget_cost = expert.forget_estimator.compute_forgetting_cost(x, pseudo_labels)

                    if eval_bid_mode == 'distance_only':
                        batch_bids[i] = distance_score
                        per_expert_log_data.append({
                            'primary_cost': distance_score,
                            'forget_cost': forget_cost,
                            'norm_primary': distance_score / 10.0,
                            'norm_forget': _math.log1p(forget_cost) / 10.0,
                            'bid': distance_score
                        })
                    else:
                        norm_distance = distance_score / 10.0
                        norm_forget = _math.log1p(forget_cost) / 10.0
                        batch_bids[i] = expert.alpha * norm_distance + expert.beta * norm_forget
                        per_expert_log_data.append({
                            'primary_cost': distance_score,
                            'forget_cost': forget_cost,
                            'norm_primary': norm_distance,
                            'norm_forget': norm_forget,
                            'bid': float(batch_bids[i])
                        })
                else:
                    pseudo_labels = logits.argmax(dim=-1).detach()
                    raw_exec = F.cross_entropy(logits, pseudo_labels).item()
                    forget_cost = expert.forget_estimator.compute_forgetting_cost(x, pseudo_labels)
                    norm_exec = raw_exec / 2.5
                    norm_forget = _math.log1p(forget_cost) / 10.0
                    batch_bids[i] = expert.alpha * norm_exec + expert.beta * norm_forget
                    per_expert_log_data.append({
                        'primary_cost': raw_exec,
                        'forget_cost': forget_cost,
                        'norm_primary': norm_exec,
                        'norm_forget': norm_forget,
                        'bid': float(batch_bids[i])
                    })

            winner_id = np.argmin(batch_bids)

            # Prediction from winning expert
            preds = batch_logits[winner_id].argmax(dim=1)

            # Log eval batch
            bid_logger.log_eval_batch(
                batch_idx=eval_batch_idx,
                per_expert_data=per_expert_log_data,
                winner_id=int(winner_id),
                digit_labels=y.cpu().tolist(),
                winner_preds=preds.cpu().tolist()
            )
            eval_batch_idx += 1

            # Record Results Per Sample
            for i, label in enumerate(y):
                digit = label.item()
                correct = (preds[i] == label).item()

                digit_eval_stats[digit]['total'] += 1
                if correct:
                    digit_eval_stats[digit]['correct'] += 1
                digit_eval_stats[digit]['routed_to'][winner_id] += 1

    # --- REPORTING ---
    
    # 1. Online Training Accuracy (Did they learn it?)
    print("\n" + "="*30)
    print("ONLINE TRAINING ACCURACY")
    print("="*30)
    for expert_id in range(config['num_experts']):
        print(f"\nExpert {expert_id}:")
        has_data = False
        for digit in range(10):
            stats = online_digit_stats[expert_id][digit]
            if stats['total'] > 0:
                acc = (stats['correct'] / stats['total']) * 100
                print(f"  Digit {digit}: {acc:6.2f}% ({stats['total']} samples)")
                has_data = True
        if not has_data:
            print("  (No training data routed)")

    # 2. Test Accuracy & Routing (Do they remember it?)
    print("\n" + "="*30)
    print("TEST ACCURACY & ROUTING")
    print("="*30)
    
    overall_correct = 0
    overall_total = 0
    
    for digit in range(10):
        s = digit_eval_stats[digit]
        acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0.0
        overall_correct += s['correct']
        overall_total += s['total']
        
        # Routing distribution string
        routing_str = ", ".join([f"E{eid}:{count}" for eid, count in s['routed_to'].items()])
        
        print(f"Digit {digit}: {acc:6.2f}% | Routed: [{routing_str}]")

    final_avg = (overall_correct / overall_total) * 100 if overall_total > 0 else 0.0
    print(f"\nOverall Average Accuracy: {final_avg:.2f}%")
    
    # Save Summary
    exp_name = config.get('experiment_name')
    if exp_name:
        summary_path = f"results/{exp_name}_summary.txt"
    else:
        summary_path = f"results/continual_mob_summary_{config['seed']}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Avg Accuracy: {final_avg:.4f}\n")
        f.write(f"Detected Shifts: {detected_shifts}\n")
        
    results = {
        'avg_accuracy': final_avg,
        'detected_shifts': detected_shifts,
        'digit_eval_stats': digit_eval_stats,
        'online_digit_stats': online_digit_stats,
        'bid_logger': bid_logger
    }
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lambda_ewc', type=float, default=40.0) 
    parser.add_argument('--shift_threshold', type=float, default=2.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--save_bids', action='store_true', help='Save bid logs to JSON')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='Reset optimizer when expert is idle too long')
    parser.add_argument('--idle_threshold', type=int, default=100,
                        help='Batches of inactivity before optimizer reset (default: 100)')
    parser.add_argument('--forgetting_cost_scale', type=float, default=1.0,
                        help='Scale for forgetting cost in bid computation')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Unique name for this experiment run (used in output filenames)')
    parser.add_argument('--routing_strategy', type=str, default='pseudo_label',
                        choices=['pseudo_label', 'prototype'],
                        help='Evaluation routing strategy: pseudo_label (default) or prototype (feature-distance)')
    # Experiment 1: Per-sample top-k routing
    parser.add_argument('--top_k', type=int, default=1,
                        help='Number of experts to combine per sample (1=winner-take-all, 2=Mixtral-style)')
    parser.add_argument('--per_sample', action='store_true',
                        help='Enable per-sample routing (each sample routed independently)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for top-k expert combination')
    # Experiment 2: Distance-only bidding
    parser.add_argument('--eval_bid_mode', type=str, default='full',
                        choices=['full', 'distance_only'],
                        help='Eval bid mode: full (distance+forget) or distance_only')
    # Experiment 3: Training-time prototype routing
    parser.add_argument('--train_routing', type=str, default='label',
                        choices=['label', 'prototype'],
                        help='Training routing: label (default) or prototype (after warmup)')
    parser.add_argument('--train_routing_warmup', type=int, default=1000,
                        help='Batches of label-based warmup before switching to prototype routing')
    parser.add_argument('--use_conscience', action='store_true',
                        help='Enable conscience bias load-balancing in bid computation')
    parser.add_argument('--conscience_rate', type=float, default=0.01,
                        help='Conscience bias update rate (default: 0.01)')
    parser.add_argument('--conscience_decay', type=float, default=0.999,
                        help='Conscience bias decay factor (default: 0.999)')
    parser.add_argument('--seed_idle_prototypes', action='store_true',
                        help='Seed prototype stores for idle experts at prototype-routing switch')

    args = parser.parse_args()
    set_seed(args.seed)

    config = {
        'seed': args.seed,
        'num_experts': args.num_experts, 
        'num_tasks': 5,
        'alpha': args.alpha,
        'beta': args.beta,
        'lambda_ewc': args.lambda_ewc,
        'learning_rate': args.learning_rate,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'epochs_per_task': args.epochs,
        'shift_threshold': args.shift_threshold,
        'reset_optimizer': args.reset_optimizer,
        'idle_threshold': args.idle_threshold,
        'forgetting_cost_scale': args.forgetting_cost_scale,
        'routing_strategy': args.routing_strategy,
        'experiment_name': args.experiment_name,
        'top_k': args.top_k,
        'per_sample': args.per_sample,
        'temperature': args.temperature,
        'eval_bid_mode': args.eval_bid_mode,
        'train_routing': args.train_routing,
        'train_routing_warmup': args.train_routing_warmup,
        'use_conscience': args.use_conscience,
        'conscience_rate': args.conscience_rate,
        'conscience_decay': args.conscience_decay,
        'seed_idle_prototypes': args.seed_idle_prototypes
    }

    print("Creating datasets...")
    train_tasks = create_split_mnist(config['num_tasks'], train=True, batch_size=32)
    test_tasks = create_split_mnist(config['num_tasks'], train=False, batch_size=32)

    results = run_continual_experiment(train_tasks, test_tasks, config)

    os.makedirs('results', exist_ok=True)

    if args.save_bids and 'bid_logger' in results:
        exp_name = args.experiment_name
        if exp_name:
            bids_path = f"results/{exp_name}_bids.json"
        else:
            seed = config['seed']
            strategy = config['routing_strategy']
            bids_path = f"results/continual_mob_bids_{strategy}_seed_{seed}.json"
        results['bid_logger'].save_logs(bids_path)
        print("\n" + "="*70)
        print("BID DIAGNOSTICS")
        print("="*70)
        results['bid_logger'].print_diagnostics()

if __name__ == '__main__':
    main()
