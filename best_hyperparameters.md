# Best Hyperparameters

This document records the best hyperparameters found for each model configuration as of February 3, 2026.

## Common Configuration
*   **Number of Experts**: 4
*   **Number of Tasks**: 5
*   **Batch Size**: 32
*   **Epochs per Task**: 4
*   **Device**: cuda

## 1. Gated MoE + EWC Experiment
*Script*: `tests/run_gated_moe_ewc.py`
*   **Seed**: 42
*   **Lambda EWC**: 50.0
*   **Learning Rate**: 0.001
*   **Gater EWC**: True
*   **Gater Hidden Size**: 256

## 2. Task Aware MoB (MoB Standalone)
*Script*: `tests/run_mob_only.py`
*   **Seed**: 42
*   **Alpha**: 0.5
*   **Beta**: 0.5
*   **Lambda EWC**: 10.0
*   **Learning Rate**: 0.001
*   **Forgetting Cost Scale**: 1.0
*   **Use LwF**: False
*   **LwF Temperature**: 2.0
*   **LwF Alpha**: 0.1

## 3. Continual MoB
*Script*: `tests/run_continual_mob.py`
*   **Base Configuration**: Same as Task Aware MoB
*   **Lambda EWC**: 40.0
*   **Shift Threshold**: 2.0
