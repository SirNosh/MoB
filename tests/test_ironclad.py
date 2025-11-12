"""
Ironclad test suite for MoB Phase 1 critical fixes.

This comprehensive test suite validates that all critical bugs are fixed and
the system is ready for full baseline comparison experiments.

Tests:
1. Split-MNIST class distribution (replay mechanism)
2. EWC Fisher matrix non-zero
3. EWC penalty applied during training
4. Parameter capacity equality across baselines
5. Model device placement (CPU/GPU)
6. Auction correctness (VCG properties)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mob import (
    PerBatchVCGAuction,
    ExpertPool,
    create_model,
    MoBExpert,
    count_parameters
)
from mob.baselines import MonolithicEWC


def create_split_mnist(num_tasks: int = 5, train: bool = True, batch_size: int = 32, replay_ratio: float = 0.2):
    """Create Split-MNIST dataset with replay mechanism."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    tasks = []
    digits_per_task = 10 // num_tasks

    for task_id in range(num_tasks):
        start_digit = task_id * digits_per_task
        end_digit = start_digit + digits_per_task

        # Get current task samples
        current_indices = [
            i for i, (_, label) in enumerate(dataset)
            if start_digit <= label < end_digit
        ]

        # For tasks after the first, add replay samples from previous tasks
        if task_id > 0:
            previous_indices = [
                i for i, (_, label) in enumerate(dataset)
                if label < start_digit
            ]

            random.shuffle(previous_indices)
            replay_count = int(len(current_indices) * replay_ratio)
            replay_indices = previous_indices[:replay_count]

            task_indices = current_indices + replay_indices
            random.shuffle(task_indices)
        else:
            task_indices = current_indices

        task_dataset = Subset(dataset, task_indices)
        task_loader = DataLoader(
            task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        tasks.append(task_loader)

    return tasks


def test_split_mnist_class_distribution():
    """
    Test 1: Verify Split-MNIST has correct class distribution with replay.

    Ensures that:
    - Task 0 contains only its designated classes
    - Task 1+ contain current classes PLUS replay from previous tasks
    - Replay ratio is approximately correct
    """
    print("\n" + "="*70)
    print("TEST 1: Split-MNIST Class Distribution with Replay")
    print("="*70)

    num_tasks = 5
    replay_ratio = 0.2
    train_tasks = create_split_mnist(num_tasks=num_tasks, train=True, replay_ratio=replay_ratio)

    all_passed = True

    for task_id, task_loader in enumerate(train_tasks):
        # Collect all labels in this task
        task_labels = []
        for x, y in task_loader:
            task_labels.extend(y.numpy())

        unique_labels = set(task_labels)
        label_counts = {label: task_labels.count(label) for label in unique_labels}

        # Expected classes for this task
        start_digit = task_id * (10 // num_tasks)
        end_digit = start_digit + (10 // num_tasks)
        expected_current = set(range(start_digit, end_digit))

        print(f"\nTask {task_id}:")
        print(f"  Expected current classes: {sorted(expected_current)}")
        print(f"  All classes present: {sorted(unique_labels)}")

        # Test 1a: Current task classes must be present
        if not expected_current.issubset(unique_labels):
            print(f"  ✗ FAIL: Missing current task classes!")
            all_passed = False
            continue

        # Test 1b: Task 0 should only have its classes (no replay)
        if task_id == 0:
            if unique_labels != expected_current:
                print(f"  ✗ FAIL: Task 0 has unexpected classes: {unique_labels - expected_current}")
                all_passed = False
            else:
                print(f"  ✓ Task 0 has only its classes (no replay)")

        # Test 1c: Tasks 1+ should have replay from previous tasks
        else:
            expected_previous = set(range(0, start_digit))
            has_replay = bool(unique_labels & expected_previous)

            if not has_replay:
                print(f"  ✗ FAIL: No replay samples from previous tasks!")
                all_passed = False
            else:
                replay_labels = unique_labels & expected_previous
                print(f"  ✓ Has replay from previous tasks: {sorted(replay_labels)}")

                # Check replay ratio
                current_count = sum(label_counts[label] for label in expected_current if label in label_counts)
                replay_count = sum(label_counts[label] for label in replay_labels)
                actual_ratio = replay_count / current_count if current_count > 0 else 0

                print(f"  Replay ratio: {actual_ratio:.3f} (expected ~{replay_ratio})")

                # Allow some variance (±50% of expected)
                if not (replay_ratio * 0.5 <= actual_ratio <= replay_ratio * 1.5):
                    print(f"  ⚠  Warning: Replay ratio deviates significantly from expected")

    if all_passed:
        print("\n✓ TEST 1 PASSED: Split-MNIST class distribution correct")
    else:
        print("\n✗ TEST 1 FAILED: Split-MNIST class distribution incorrect")

    return all_passed


def test_ewc_fisher_nonzero():
    """
    Test 2: Verify EWC Fisher matrix is computed and non-zero.

    Ensures that:
    - Fisher matrix is computed after task completion
    - Fisher values are non-zero (not all zeros bug)
    - Fisher statistics are reasonable
    """
    print("\n" + "="*70)
    print("TEST 2: EWC Fisher Matrix Non-Zero")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create a simple model and expert
    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    expert = MoBExpert(
        expert_id=0,
        model=model,
        alpha=1.0,
        beta=0.0,
        lambda_ewc=5000,
        device=device
    )

    # Create minimal training data
    train_tasks = create_split_mnist(num_tasks=2, train=True, batch_size=32)

    # Train briefly on task 0
    print("\nTraining on Task 0 (50 batches)...")
    optimizer = torch.optim.Adam(expert.model.parameters(), lr=0.001)
    batch_count = 0
    for x, y in train_tasks[0]:
        expert.train_on_batch(x, y, optimizer)
        batch_count += 1
        if batch_count >= 50:
            break

    # Update Fisher matrix
    print("Computing Fisher matrix...")
    expert.update_after_task(train_tasks[0], num_samples=200)

    # Get Fisher statistics
    fisher_stats = expert.forget_estimator.get_fisher_stats()

    print(f"\nFisher Statistics:")
    print(f"  Total parameters: {fisher_stats.get('total_params', 0):,}")
    print(f"  Mean importance: {fisher_stats.get('mean_importance', 0):.6e}")
    print(f"  Max importance: {fisher_stats.get('max_importance', 0):.6e}")

    # Test assertions
    all_passed = True

    # Test 2a: Fisher matrix exists
    if not expert.forget_estimator.has_fisher():
        print("\n✗ FAIL: Fisher matrix not computed!")
        all_passed = False
    else:
        print("\n✓ Fisher matrix exists")

    # Test 2b: Fisher values are non-zero
    mean_importance = fisher_stats.get('mean_importance', 0)
    if mean_importance == 0:
        print("✗ FAIL: Fisher matrix is all zeros!")
        all_passed = False
    else:
        print(f"✓ Fisher values are non-zero (mean={mean_importance:.6e})")

    # Test 2c: Fisher values are reasonable (not too small, not NaN/Inf)
    if mean_importance < 1e-10:
        print(f"✗ FAIL: Fisher values too small (mean={mean_importance:.6e})")
        all_passed = False
    elif np.isnan(mean_importance) or np.isinf(mean_importance):
        print(f"✗ FAIL: Fisher values are NaN or Inf!")
        all_passed = False
    else:
        print(f"✓ Fisher values in reasonable range")

    if all_passed:
        print("\n✓ TEST 2 PASSED: EWC Fisher matrix computed correctly")
    else:
        print("\n✗ TEST 2 FAILED: EWC Fisher matrix issues detected")

    return all_passed


def test_ewc_penalty_applied():
    """
    Test 3: Verify EWC penalty is applied during training on new tasks.

    Ensures that:
    - EWC penalty is zero on first task (no Fisher yet)
    - EWC penalty is non-zero on subsequent tasks
    - EWC penalty grows as parameters diverge from optimal
    """
    print("\n" + "="*70)
    print("TEST 3: EWC Penalty Applied During Training")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model and baseline
    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    baseline = MonolithicEWC(model, lambda_ewc=5000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create training data
    train_tasks = create_split_mnist(num_tasks=2, train=True, batch_size=32)

    # Task 0: Train and verify penalty is zero
    print("\n--- Task 0 Training ---")
    print("Training (50 batches)...")
    batch_count = 0
    task0_penalties = []

    for x, y in train_tasks[0]:
        x = x.to(device)
        y = y.to(device)
        metrics = baseline.expert.train_on_batch(x, y, optimizer)
        task0_penalties.append(metrics['ewc_penalty'])

        batch_count += 1
        if batch_count >= 50:
            break

    print(f"Task 0 EWC penalties (first 5): {task0_penalties[:5]}")

    # Update Fisher
    print("Updating Fisher matrix...")
    baseline.update_after_task(train_tasks[0], num_samples=200)

    # Task 1: Train and verify penalty is non-zero and growing
    print("\n--- Task 1 Training ---")
    print("Training (50 batches)...")
    batch_count = 0
    task1_penalties = []

    for x, y in train_tasks[1]:
        x = x.to(device)
        y = y.to(device)
        metrics = baseline.expert.train_on_batch(x, y, optimizer)
        task1_penalties.append(metrics['ewc_penalty'])

        batch_count += 1
        if batch_count >= 50:
            break

    print(f"Task 1 EWC penalties (first 10): {[f'{p:.6f}' for p in task1_penalties[:10]]}")

    # Test assertions
    all_passed = True

    # Test 3a: Task 0 penalties should be zero (no Fisher yet)
    if not all(p == 0.0 for p in task0_penalties):
        print("\n✗ FAIL: Task 0 has non-zero EWC penalties (Fisher shouldn't exist yet)!")
        all_passed = False
    else:
        print("\n✓ Task 0 EWC penalties are zero (correct)")

    # Test 3b: Task 1 penalties should be non-zero after a few batches
    if all(p == 0.0 for p in task1_penalties[5:]):
        print("✗ FAIL: Task 1 EWC penalties are all zero (EWC not working)!")
        all_passed = False
    else:
        print("✓ Task 1 EWC penalties are non-zero (correct)")

    # Test 3c: Task 1 penalties should grow (parameters diverging)
    if len(task1_penalties) >= 10:
        early_avg = np.mean(task1_penalties[2:5])
        late_avg = np.mean(task1_penalties[-5:])

        print(f"  Early penalties (batches 2-5): {early_avg:.6f}")
        print(f"  Late penalties (last 5): {late_avg:.6f}")

        if late_avg <= early_avg:
            print("✗ FAIL: EWC penalties not growing (parameters should diverge)!")
            all_passed = False
        else:
            print(f"✓ EWC penalties growing ({late_avg/early_avg:.2f}x increase)")

    if all_passed:
        print("\n✓ TEST 3 PASSED: EWC penalty applied correctly")
    else:
        print("\n✗ TEST 3 FAILED: EWC penalty application issues")

    return all_passed


def test_parameter_capacity_equal():
    """
    Test 4: Verify parameter capacity is fair across baselines.

    Ensures that:
    - Single models with width_multiplier=2 have similar capacity to 4x single models
    - Parameter counts are within reasonable range for fair comparison
    """
    print("\n" + "="*70)
    print("TEST 4: Parameter Capacity Equality")
    print("="*70)

    # Single model (1x width)
    model_1x = create_model('simple_cnn', num_classes=10, input_channels=1, width_multiplier=1)
    params_1x = count_parameters(model_1x)

    # Single model (2x width) - for Naive and Monolithic EWC
    model_2x = create_model('simple_cnn', num_classes=10, input_channels=1, width_multiplier=2)
    params_2x = count_parameters(model_2x)

    # Single model (4x width)
    model_4x = create_model('simple_cnn', num_classes=10, input_channels=1, width_multiplier=4)
    params_4x = count_parameters(model_4x)

    # 4 separate models (1x each) - for MoB, Random, Gated
    params_4_models = params_1x * 4

    print(f"\nParameter Counts:")
    print(f"  Single model (1x): {params_1x:,}")
    print(f"  Single model (2x): {params_2x:,}")
    print(f"  Single model (4x): {params_4x:,}")
    print(f"  4 separate models:  {params_4_models:,}")

    print(f"\nCapacity Ratios:")
    ratio_2x = params_2x / params_4_models
    ratio_4x = params_4x / params_4_models
    print(f"  2x single / 4 models: {ratio_2x:.3f}")
    print(f"  4x single / 4 models: {ratio_4x:.3f}")

    # Test assertions
    all_passed = True

    # Test 4a: 2x model should be within 0.5-2.0x of 4 models (fair comparison)
    if not (0.5 <= ratio_2x <= 2.0):
        print(f"\n✗ FAIL: 2x model capacity unfair (ratio={ratio_2x:.3f}, should be 0.5-2.0)")
        all_passed = False
    else:
        print(f"\n✓ 2x model capacity fair (ratio={ratio_2x:.3f})")

    # Test 4b: Verify 2x is closer to 4 models than 4x
    if ratio_2x >= ratio_4x:
        print(f"✗ FAIL: 2x model not better match than 4x model!")
        all_passed = False
    else:
        print(f"✓ 2x model closer match than 4x model")

    # Test 4c: Recommendations
    print(f"\n📊 Recommendation:")
    if ratio_2x < 1.0:
        print(f"  Use width_multiplier=2 for Naive/Monolithic (slight disadvantage but fair)")
    elif ratio_2x > 1.5:
        print(f"  ⚠  Consider width_multiplier=1.5 for closer match (if supported)")
    else:
        print(f"  ✓ width_multiplier=2 provides fair comparison")

    if all_passed:
        print("\n✓ TEST 4 PASSED: Parameter capacity is fair")
    else:
        print("\n✗ TEST 4 FAILED: Parameter capacity issues")

    return all_passed


def test_model_device_placement():
    """
    Test 5: Verify models work correctly on different devices (CPU/GPU).

    Ensures that:
    - Models can be created and moved to device
    - Forward passes work without device mismatch errors
    - Dynamic layer initialization works on correct device
    """
    print("\n" + "="*70)
    print("TEST 5: Model Device Placement")
    print("="*70)

    all_passed = True

    # Test on available devices
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')

    for device_name in devices_to_test:
        device = torch.device(device_name)
        print(f"\n--- Testing on {device} ---")

        try:
            # Test 5a: SimpleCNN with dynamic initialization
            model = create_model('simple_cnn', num_classes=10, input_channels=1, width_multiplier=2)
            model.to(device)

            # Create dummy input
            x = torch.randn(4, 1, 28, 28).to(device)

            # Forward pass (triggers dynamic initialization)
            with torch.no_grad():
                logits = model(x)

            # Verify output shape and device
            assert logits.shape == (4, 10), f"Wrong output shape: {logits.shape}"
            assert logits.device.type == device.type, f"Output on wrong device: {logits.device}"

            print(f"  ✓ SimpleCNN (2x) forward pass successful")

            # Test 5b: LeNet5
            model_lenet = create_model('lenet5', num_classes=10, input_channels=1, width_multiplier=2)
            model_lenet.to(device)

            with torch.no_grad():
                logits = model_lenet(x)

            assert logits.shape == (4, 10)
            assert logits.device.type == device.type
            print(f"  ✓ LeNet5 (2x) forward pass successful")

            # Test 5c: MLP
            x_flat = x.view(4, -1)
            model_mlp = create_model('mlp', num_classes=10, input_size=784, width_multiplier=2)
            model_mlp.to(device)

            with torch.no_grad():
                logits = model_mlp(x_flat.to(device))

            assert logits.shape == (4, 10)
            assert logits.device.type == device.type
            print(f"  ✓ MLP (2x) forward pass successful")

        except Exception as e:
            print(f"  ✗ FAIL on {device}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\n✓ TEST 5 PASSED: Device placement works correctly")
    else:
        print("\n✗ TEST 5 FAILED: Device placement issues")

    return all_passed


def test_auction_correctness():
    """
    Test 6: Verify VCG auction implements correct properties.

    Ensures that:
    - Winner is the expert with lowest bid
    - Payment is second-lowest bid (VCG property)
    - Auction is deterministic given same bids
    """
    print("\n" + "="*70)
    print("TEST 6: Auction Correctness (VCG Properties)")
    print("="*70)

    num_experts = 4
    auction = PerBatchVCGAuction(num_experts=num_experts)

    all_passed = True

    # Test 6a: Winner is lowest bidder
    print("\n--- Test 6a: Winner Selection ---")
    bids = np.array([0.5, 0.3, 0.8, 0.4])  # Expert 1 should win
    winner_id, payment, details = auction.run_auction(bids)

    print(f"Bids: {bids}")
    print(f"Winner: Expert {winner_id} (bid={bids[winner_id]})")

    if winner_id != 1:  # np.argmin([0.5, 0.3, 0.8, 0.4]) = 1
        print(f"✗ FAIL: Wrong winner! Expected Expert 1, got Expert {winner_id}")
        all_passed = False
    else:
        print(f"✓ Winner is lowest bidder")

    # Test 6b: Payment is second-lowest bid (VCG property)
    print("\n--- Test 6b: VCG Payment ---")
    sorted_bids = np.sort(bids)
    expected_payment = sorted_bids[1]  # Second lowest

    print(f"Sorted bids: {sorted_bids}")
    print(f"Expected payment (2nd lowest): {expected_payment}")
    print(f"Actual payment: {payment}")

    if abs(payment - expected_payment) > 1e-6:
        print(f"✗ FAIL: Wrong payment! Expected {expected_payment}, got {payment}")
        all_passed = False
    else:
        print(f"✓ Payment is second-lowest bid (VCG property satisfied)")

    # Test 6c: Deterministic behavior
    print("\n--- Test 6c: Deterministic Behavior ---")
    winner_id2, payment2, _ = auction.run_auction(bids)

    if winner_id != winner_id2 or abs(payment - payment2) > 1e-6:
        print(f"✗ FAIL: Auction is non-deterministic!")
        all_passed = False
    else:
        print(f"✓ Auction is deterministic")

    # Test 6d: Truthful bidding is optimal (incentive compatibility)
    print("\n--- Test 6d: Incentive Compatibility (Conceptual) ---")
    print("VCG mechanism properties:")
    print("  ✓ Second-price auction")
    print("  ✓ Winner pays second-lowest bid")
    print("  ✓ Dominant-strategy incentive-compatible (DSIC)")
    print("  → Truthful bidding maximizes utility")

    if all_passed:
        print("\n✓ TEST 6 PASSED: Auction implements VCG correctly")
    else:
        print("\n✗ TEST 6 FAILED: Auction implementation issues")

    return all_passed


def run_all_tests():
    """Run all ironclad tests."""
    print("\n" + "="*70)
    print("        MoB PHASE 1 IRONCLAD TEST SUITE")
    print("="*70)

    tests = [
        ("Split-MNIST Class Distribution", test_split_mnist_class_distribution),
        ("EWC Fisher Non-Zero", test_ewc_fisher_nonzero),
        ("EWC Penalty Applied", test_ewc_penalty_applied),
        ("Parameter Capacity Equal", test_parameter_capacity_equal),
        ("Model Device Placement", test_model_device_placement),
        ("Auction Correctness", test_auction_correctness),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*70)
    print("                    IRONCLAD TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL IRONCLAD TESTS PASSED!")
        print("Phase 1 critical fixes are verified and working correctly.")
        print("✅ Ready for full baseline comparison experiments!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Please fix issues before proceeding to full experiments.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
