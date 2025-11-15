"""
Runs a single, dedicated MoB experiment for detailed bid analysis.

This script uses the final, tuned hyperparameters to generate the bid logs
and plots for a single, representative run.
"""

import torch
import os
import sys

# Add parent directory to path to find the 'mob' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary functions from the existing benchmark script
from tests.test_baselines import run_mob_experiment, create_split_mnist, set_seed

def main():
    """
    Runs a single, dedicated MoB experiment for bid analysis.
    """
    SEED = 42  # Use a fixed seed for reproducibility
    set_seed(SEED)
    print(f"============================================================")
    print(f"  Running Single MoB Experiment for Bid Analysis (Seed {SEED})")
    print(f"============================================================")

    # The final, winning hyperparameter configuration from our debugging
    config = {
        'num_experts': 4,
        'num_tasks': 5,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 2500,
        'learning_rate': 0.001,
        'forgetting_cost_scale': 1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 32,
        'epochs_per_task': 4  # Set to 4 epochs as requested for deep analysis
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create datasets
    print("\nCreating Split-MNIST datasets...")
    train_tasks = create_split_mnist(config['num_tasks'], train=True, batch_size=config['batch_size'])
    test_tasks = create_split_mnist(config['num_tasks'], train=False, batch_size=config['batch_size'])

    # Run only the MoB experiment
    results = run_mob_experiment(train_tasks, test_tasks, config)

    # Print and save bid diagnostics
    if 'bid_logger' in results:
        print("\n" + "="*80)
        print("BID DIAGNOSTICS FOR MoB")
        print("="*80)
        results['bid_logger'].print_diagnostics()
        
        # Save logs and plot to a specific analysis directory for clarity
        output_dir = "results/bid_analysis"
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"mob_bids_seed_{SEED}.json")
        plot_path = os.path.join(output_dir, f"mob_bids_seed_{SEED}.png")
        
        results['bid_logger'].save_logs(log_path)
        try:
            results['bid_logger'].plot_bid_components(save_path=plot_path)
            print(f"✓ Bid component plot saved to: {plot_path}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
            
    print("\n" + "="*60)
    print("FINAL MoB RESULTS")
    print("="*60)
    print(f"  Average Accuracy: {results['avg_accuracy']:.4f}")
    print(f"  Average Forgetting: {results['forgetting']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()