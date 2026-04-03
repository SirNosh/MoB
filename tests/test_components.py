"""
Unit tests for MoB Phase 1 components.

This script validates that all core components can be imported and instantiated correctly.
"""

import sys
import os
import torch
import numpy as np
import types

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")

    try:
        from mob import (
            PerBatchAuction,
            SealedBidProtocol,
            create_commitment,
            ExecutionCostEstimator,
            EWCForgettingEstimator,
            MoBExpert,
            ExpertPool,
            SimpleCNN,
            LeNet5,
            MLP,
            create_model
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_auction():
    """Test auction mechanism."""
    print("\nTesting Auction...")

    from mob import PerBatchAuction

    num_experts = 4
    auction = PerBatchAuction(num_experts)

    # Test auction with sample bids
    bids = np.array([0.5, 0.3, 0.7, 0.4])
    winner, payment, metrics = auction.run_auction(bids)

    assert winner == 1, f"Expected winner 1 (lowest bid), got {winner}"
    assert payment == 0.4, f"Expected payment 0.4 (second lowest), got {payment}"
    assert metrics['winning_bid'] == 0.3

    print(f"✓ Auction works correctly")
    print(f"  Winner: {winner}, Payment: {payment}")
    print(f"  Bid spread: {metrics['bid_spread']:.3f}")

    return True


def test_sealed_bid():
    """Test sealed bid protocol."""
    print("\nTesting Sealed Bid Protocol...")

    from mob import SealedBidProtocol, create_commitment

    num_experts = 3
    protocol = SealedBidProtocol(num_experts)

    # Expert 0 commits
    bid_0 = 0.5
    commitment_0, nonce_0 = create_commitment(bid_0)
    success = protocol.commit_bid(0, commitment_0)
    assert success, "Commitment should succeed"

    # Expert 1 commits
    bid_1 = 0.3
    commitment_1, nonce_1 = create_commitment(bid_1)
    protocol.commit_bid(1, commitment_1)

    # Reveal bids
    revealed_0 = protocol.reveal_bid(0, bid_0, nonce_0)
    revealed_1 = protocol.reveal_bid(1, bid_1, nonce_1)

    assert revealed_0 and revealed_1, "Reveals should succeed"

    bids = protocol.get_revealed_bids()
    assert bids[0] == bid_0
    assert bids[1] == bid_1
    assert bids[2] == np.inf  # Expert 2 didn't reveal

    print(f"✓ Sealed bid protocol works correctly")
    print(f"  Revealed bids: {bids}")

    return True


def test_models():
    """Test model architectures."""
    print("\nTesting Model Architectures...")

    from mob import SimpleCNN, LeNet5, MLP, create_model

    # Test SimpleCNN
    model = SimpleCNN(num_classes=10, input_channels=1)
    x = torch.randn(2, 1, 28, 28)
    output = model(x)
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
    print(f"✓ SimpleCNN works: input {x.shape} -> output {output.shape}")

    # Test LeNet5
    model = LeNet5(num_classes=10, input_channels=1)
    output = model(x)
    assert output.shape == (2, 10)
    print(f"✓ LeNet5 works: input {x.shape} -> output {output.shape}")

    # Test MLP
    model = MLP(input_size=784, num_classes=10)
    output = model(x)
    assert output.shape == (2, 10)
    print(f"✓ MLP works: input {x.shape} -> output {output.shape}")

    # Test factory
    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    output = model(x)
    assert output.shape == (2, 10)
    print(f"✓ Model factory works")

    return True


def test_bidding():
    """Test bidding components."""
    print("\nTesting Bidding Components...")

    from mob import SimpleCNN, ExecutionCostEstimator, EWCForgettingEstimator

    model = SimpleCNN(num_classes=10, input_channels=1)

    # Test ExecutionCostEstimator
    exec_estimator = ExecutionCostEstimator(model)
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    exec_cost = exec_estimator.compute_predicted_loss(x, y)

    assert isinstance(exec_cost, float), "Execution cost should be float"
    assert exec_cost > 0, "Execution cost should be positive"
    print(f"✓ ExecutionCostEstimator works: cost = {exec_cost:.4f}")

    # Test EWCForgettingEstimator
    ewc_estimator = EWCForgettingEstimator(model, lambda_ewc=1000)

    # Initially no forgetting cost (no Fisher)
    forget_cost = ewc_estimator.compute_forgetting_cost(x, y)
    assert forget_cost == 0.0, "Initial forgetting cost should be 0"
    print(f"✓ EWCForgettingEstimator initialized: initial cost = {forget_cost}")

    # Check penalty (should be 0 initially)
    penalty = ewc_estimator.penalty()
    assert penalty == 0.0, "Initial penalty should be 0"
    print(f"✓ EWC penalty works: initial penalty = {penalty}")

    return True


def test_expert():
    """Test MoBExpert."""
    print("\nTesting MoBExpert...")

    from mob import SimpleCNN, MoBExpert

    model = SimpleCNN(num_classes=10, input_channels=1)
    expert = MoBExpert(
        expert_id=0,
        model=model,
        alpha=0.6,
        beta=0.4,
        lambda_ewc=1000,
        forgetting_cost_scale=1.0
    )

    # Test bidding
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    bid, components = expert.compute_bid(x, y)

    assert isinstance(bid, float), "Bid should be float"
    assert 'exec_cost' in components
    assert 'forget_cost' in components
    assert components['forget_cost'] == 0.0  # No Fisher yet

    print(f"✓ Expert bidding works:")
    print(f"  Bid: {bid:.4f}")
    print(f"  Components: exec={components['exec_cost']:.4f}, forget={components['forget_cost']:.4f}")

    # Test training
    optimizer = torch.optim.Adam(expert.model.parameters(), lr=0.001)
    metrics = expert.train_on_batch(x, y, optimizer)

    assert 'task_loss' in metrics
    assert 'ewc_penalty' in metrics
    assert 'total_loss' in metrics

    print(f"✓ Expert training works:")
    print(f"  Task loss: {metrics['task_loss']:.4f}")
    print(f"  Total loss: {metrics['total_loss']:.4f}")

    # Test statistics
    stats = expert.get_statistics()
    assert stats['expert_id'] == 0
    assert stats['batches_won'] == 1

    print(f"✓ Expert statistics work:")
    print(f"  Batches won: {stats['batches_won']}")
    print(f"  Win rate: {stats['win_rate']:.3f}")

    return True


def test_expert_pool():
    """Test ExpertPool."""
    print("\nTesting ExpertPool...")

    from mob import ExpertPool

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1000
    }

    pool = ExpertPool(num_experts=3, expert_config=expert_config)

    assert len(pool) == 3, "Pool should have 3 experts"
    print(f"✓ ExpertPool created with {len(pool)} experts")

    # Test bid collection
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    bids, components = pool.collect_bids(x, y)

    assert len(bids) == 3, "Should have 3 bids"
    assert len(components) == 3, "Should have 3 component dicts"

    print(f"✓ Bid collection works:")
    print(f"  Bids: {[f'{b:.4f}' for b in bids]}")

    # Test training winner
    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=0.001)
        for expert in pool.experts
    ]

    winner_id = 1
    metrics = pool.train_winner(winner_id, x, y, optimizers)

    assert 'task_loss' in metrics
    print(f"✓ Winner training works")

    return True


def test_integration():
    """Test full integration of components."""
    print("\nTesting Full Integration...")

    from mob import ExpertPool, PerBatchAuction

    # Setup
    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1000
    }

    num_experts = 4
    pool = ExpertPool(num_experts, expert_config)
    auction = PerBatchAuction(num_experts)

    optimizers = [
        torch.optim.Adam(expert.model.parameters(), lr=0.001)
        for expert in pool.experts
    ]

    # Simulate one training step
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    # 1. Collect bids
    bids, components = pool.collect_bids(x, y)

    # 2. Run auction
    winner_id, payment, auction_metrics = auction.run_auction(bids)

    # 3. Train winner
    train_metrics = pool.train_winner(winner_id, x, y, optimizers)

    print(f"✓ Full integration works:")
    print(f"  Winner: Expert {winner_id}")
    print(f"  Winning bid: {auction_metrics['winning_bid']:.4f}")
    print(f"  Payment: {payment:.4f}")
    print(f"  Training loss: {train_metrics['task_loss']:.4f}")

    return True


def test_conscience_bias_reduces_imbalance():
    """Conscience bias should penalize frequent winners and allow others to win."""
    from contibualmob.pool import ExpertPool

    max_bias = 0.05
    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 1.0,
        'beta': 0.0,
        'lambda_ewc': 1.0,
        'use_conscience': True,
        'conscience_rate': 0.5,
        'conscience_decay': 1.0,
        'conscience_max_bias': max_bias
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)

    raw_bids = [0.10, 0.11, 0.12, 0.13]
    for i, expert in enumerate(pool.experts):
        def _fixed_bid(self, x, y, _rb=raw_bids[i]):
            return _rb, {
                'exec_cost': _rb,
                'forget_cost': 0.0,
                'norm_exec_cost': _rb,
                'norm_forget_cost': 0.0,
                'bid': _rb,
                'alpha': 1.0,
                'beta': 0.0
            }
        expert.compute_bid = types.MethodType(_fixed_bid, expert)

    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))

    wins = np.zeros(4, dtype=int)
    for _ in range(100):
        bids, _ = pool.collect_bids(x, y, train_routing='label')
        winner = int(np.argmin(bids))
        wins[winner] += 1
        pool.update_conscience_bias(winner)

    assert wins[0] < 100, "Conscience bias should prevent expert 0 from winning all auctions"
    assert wins[1:].sum() > 0, "Other experts should win at least some auctions with conscience enabled"

    final_bids, _ = pool.collect_bids(x, y, train_routing='label')
    assert final_bids[0] > raw_bids[0], "Conscience should increase expert 0 adjusted bid after frequent wins"
    assert np.all(np.abs(pool.load_bias) <= max_bias + 1e-12), \
        "Conscience load_bias must be clipped to [-conscience_max_bias, conscience_max_bias]"

    # Force an overflow and verify clipping still enforces the cap.
    pool.load_bias = np.array([0.2, -0.2, 0.2, -0.2], dtype=float)
    pool.update_conscience_bias(0)
    assert np.all(np.abs(pool.load_bias) <= max_bias + 1e-12), \
        "Clipping should cap pre-existing oversized biases to max_bias"


def test_conscience_backward_compat():
    """When conscience is disabled, load_bias must not affect bids."""
    from contibualmob.pool import ExpertPool

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 1.0,
        'beta': 0.0,
        'lambda_ewc': 1.0,
        'use_conscience': False
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)
    pool.load_bias = np.array([10.0, -10.0, 5.0, -5.0], dtype=float)

    raw_bids = [0.15, 0.25, 0.35, 0.45]
    for i, expert in enumerate(pool.experts):
        def _fixed_bid(self, x, y, _rb=raw_bids[i]):
            return _rb, {
                'exec_cost': _rb,
                'forget_cost': 0.0,
                'norm_exec_cost': _rb,
                'norm_forget_cost': 0.0,
                'bid': _rb,
                'alpha': 1.0,
                'beta': 0.0
            }
        expert.compute_bid = types.MethodType(_fixed_bid, expert)

    x = torch.randn(2, 1, 28, 28)
    y = torch.randint(0, 10, (2,))

    bids, _ = pool.collect_bids(x, y, train_routing='label')
    assert np.allclose(bids, np.array(raw_bids)), "Bids must remain raw values when conscience is disabled"


def test_conscience_params_configurable():
    """Different conscience rates should produce different bias magnitudes."""
    from contibualmob.pool import ExpertPool

    common = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'use_conscience': True,
        'conscience_decay': 1.0,
        'conscience_max_bias': 10.0
    }
    slow = ExpertPool(num_experts=4, expert_config={**common, 'conscience_rate': 0.01})
    fast = ExpertPool(num_experts=4, expert_config={**common, 'conscience_rate': 0.5})

    for _ in range(20):
        slow.update_conscience_bias(0)
        fast.update_conscience_bias(0)

    assert abs(fast.load_bias[0]) > abs(slow.load_bias[0]), \
        "Higher conscience_rate should produce larger bias magnitude"


def _has_prototypes(expert):
    return expert.prototype_store is not None and len(expert.prototype_store.centroids) > 0


def test_prototype_seeding_creates_prototypes():
    """Seeding should create prototypes for all experts."""
    from contibualmob.pool import ExpertPool

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'seed_idle_prototypes': True
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    optimizers = [torch.optim.Adam(expert.model.parameters(), lr=0.001) for expert in pool.experts]

    # Train one expert so others remain idle.
    pool.train_winner(0, x, y, optimizers)
    assert _has_prototypes(pool.experts[0])
    assert any(not _has_prototypes(expert) for expert in pool.experts[1:])

    pool.seed_idle_prototypes(x, y)

    assert all(_has_prototypes(expert) for expert in pool.experts), \
        "All experts should have prototypes after seeding"


def test_seeding_removes_default_distance():
    """After seeding, prototype routing should not use the 100.0 default distance."""
    from contibualmob.pool import ExpertPool

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'seed_idle_prototypes': True
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    optimizers = [torch.optim.Adam(expert.model.parameters(), lr=0.001) for expert in pool.experts]

    pool.train_winner(0, x, y, optimizers)
    pool.seed_idle_prototypes(x, y)

    _, components = pool.collect_bids(x, y, train_routing='prototype')
    assert all(comp['exec_cost'] != 100.0 for comp in components), \
        "No expert should use the default 100.0 distance after seeding"


def test_seeding_backward_compat():
    """When seed_idle_prototypes is disabled, idle experts should still use distance 100.0."""
    from contibualmob.pool import ExpertPool

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'seed_idle_prototypes': False
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    optimizers = [torch.optim.Adam(expert.model.parameters(), lr=0.001) for expert in pool.experts]

    pool.train_winner(0, x, y, optimizers)
    _, components = pool.collect_bids(x, y, train_routing='prototype')

    assert any(comp['exec_cost'] == 100.0 for comp in components), \
        "Backward compatibility requires idle experts to keep default distance 100.0 when seeding is disabled"


def test_temperature_annealing_schedule():
    """Temperature should decay each training batch until temp_min."""
    from contibualmob.pool import ExpertPool

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'routing_temperature': 2.0,
        'temp_decay': 0.5,
        'temp_min': 0.01
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)
    bids = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    seen_temps = []
    for _ in range(10):
        seen_temps.append(pool.current_temperature)
        pool.select_winner(bids, train_routing='prototype', is_training=True)

    expected = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.01, 0.01]
    assert np.allclose(seen_temps, expected), f"Expected annealing schedule {expected}, got {seen_temps}"
    assert pool.current_temperature == 0.01


def test_temperature_enables_exploration():
    """High temperature should allow non-argmin experts to win some auctions."""
    from contibualmob.pool import ExpertPool

    np.random.seed(0)

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'routing_temperature': 5.0,
        'temp_decay': 1.0,
        'temp_min': 0.01
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)
    bids = np.array([0.0, 3.0, 3.0, 3.0], dtype=float)

    wins = np.zeros(4, dtype=int)
    for _ in range(100):
        winner = pool.select_winner(bids, train_routing='prototype', is_training=True)
        wins[winner] += 1

    non_best_win_rate = wins[1:].sum() / 100.0
    assert non_best_win_rate > 0.10, \
        f"Expected >10% non-argmin wins with high temperature, got {non_best_win_rate:.2%}"


def test_temperature_backward_compat():
    """With routing_temperature=0.0, selection must remain deterministic argmin."""
    from contibualmob.pool import ExpertPool

    np.random.seed(123)

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'routing_temperature': 0.0,
        'temp_decay': 0.5,
        'temp_min': 0.01
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)
    bids = np.array([0.05, 0.2, 0.3, 0.4], dtype=float)

    winners = [
        pool.select_winner(bids, train_routing='prototype', is_training=True)
        for _ in range(100)
    ]
    assert all(w == 0 for w in winners), "Temperature disabled should use deterministic argmin"
    assert pool.current_temperature == 0.0

    # Eval-time routing must always be deterministic, even when temperature is enabled.
    eval_config = dict(expert_config)
    eval_config['routing_temperature'] = 5.0
    eval_pool = ExpertPool(num_experts=4, expert_config=eval_config)
    eval_winners = [
        eval_pool.select_winner(bids, train_routing='prototype', is_training=False)
        for _ in range(50)
    ]
    assert all(w == 0 for w in eval_winners), "Eval routing must remain deterministic argmin"
    assert eval_pool.current_temperature == 5.0, "Eval routing should not decay temperature"


def test_combined_mechanisms():
    """Conscience + seeding + temperature should work together with valid routing behavior."""
    from contibualmob.pool import ExpertPool

    np.random.seed(7)
    torch.manual_seed(7)

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 1.0,
        'use_conscience': True,
        'conscience_rate': 0.1,
        'conscience_decay': 1.0,
        'seed_idle_prototypes': True,
        'routing_temperature': 3.0,
        'temp_decay': 0.98,
        'temp_min': 0.05
    }
    pool = ExpertPool(num_experts=4, expert_config=expert_config)
    optimizers = [torch.optim.Adam(expert.model.parameters(), lr=0.001) for expert in pool.experts]

    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    wins = np.zeros(4, dtype=int)
    for _ in range(50):
        bids, components = pool.collect_bids(x, y, train_routing='prototype')
        assert np.isfinite(bids).all(), "Combined mechanisms should produce finite bids"
        assert all(np.isfinite(comp['bid']) for comp in components), "Component bids should be finite"

        winner = pool.select_winner(bids, train_routing='prototype', is_training=True)
        metrics = pool.train_winner(winner, x, y, optimizers)
        wins[winner] += 1

        assert np.isfinite(metrics['total_loss']), "Training loss should remain finite"

    assert all(_has_prototypes(expert) for expert in pool.experts), \
        "Seeding should ensure all experts have prototypes in combined mode"
    assert pool.total_auctions == 50, "Conscience should update after each training auction"
    assert pool.current_temperature <= 3.0 and pool.current_temperature >= 0.05
    assert wins.sum() == 50
    assert np.count_nonzero(wins) >= 2, "Combined mechanisms should avoid single-expert monopoly"
    assert wins.max() / wins.sum() < 0.95, "Routing distribution should remain reasonably spread"


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("MoB Phase 1 Component Tests")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Auction", test_auction),
        ("Sealed Bid Protocol", test_sealed_bid),
        ("Model Architectures", test_models),
        ("Bidding Components", test_bidding),
        ("MoBExpert", test_expert),
        ("ExpertPool", test_expert_pool),
        ("Full Integration", test_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Phase 1 implementation is complete.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")

    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
