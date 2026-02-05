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
from contibualmob.auction import PerBatchVCGAuction
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
        'dropout': 0.5
    }

    # Create Pool with Shift Detection ENABLED
    pool = ExpertPool(
        config['num_experts'], 
        expert_config, 
        device=device,
        use_shift_detection=True
    )
    
    if pool.shift_detector:
        pool.shift_detector.threshold_multiplier = config['shift_threshold']

    auction = PerBatchVCGAuction(config['num_experts'])

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=config['learning_rate'])
        for expert in pool.experts
    ]

    bid_logger = BidLogger(
        num_experts=config['num_experts'],
        alpha=config['alpha'],
        beta=config['beta'],
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
    
    for batch_idx, (x, y) in enumerate(pbar):
        # 1. Auction Phase
        bids, components = pool.collect_bids(x, y)
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
    
    print("Running Auction-Based Evaluation on Test Set...")
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x, y = x.to(device), y.to(device)
        
        # 1. Auction Routing
        bids, _ = pool.collect_bids(x, y.cpu()) # collect_bids expects cpu targets if used, but here uses x primarily
        winner_id = np.argmin(bids) # AUCTION LOGIC
        
        # 2. Prediction
        winner_model = pool.experts[winner_id].model
        winner_model.eval()
        with torch.no_grad():
            logits = winner_model(x)
            preds = logits.argmax(dim=1)
            
        # 3. Record Results Per Sample
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
    with open(f"results/continual_mob_summary_{config['seed']}.txt", "w") as f:
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
        'shift_threshold': args.shift_threshold
    }

    print("Creating datasets...")
    train_tasks = create_split_mnist(config['num_tasks'], train=True, batch_size=32)
    test_tasks = create_split_mnist(config['num_tasks'], train=False, batch_size=32)

    run_continual_experiment(train_tasks, test_tasks, config)

if __name__ == '__main__':
    main()
