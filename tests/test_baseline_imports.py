"""
Quick test to verify all baseline implementations can be imported and instantiated.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_baseline_imports():
    """Test that all baselines can be imported."""
    print("Testing baseline imports...")

    try:
        from mob.baselines import (
            NaiveFineTuning,
            RandomAssignment,
            MonolithicEWC,
            GatedMoE
        )
        print("✓ All baseline imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_naive_instantiation():
    """Test Naive Fine-tuning instantiation."""
    print("\nTesting Naive Fine-tuning...")

    from mob.baselines import NaiveFineTuning
    from mob import create_model

    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    baseline = NaiveFineTuning(model)

    print(f"✓ Naive Fine-tuning instantiated: {baseline.__class__.__name__}")
    return True


def test_random_instantiation():
    """Test Random Assignment instantiation."""
    print("\nTesting Random Assignment...")

    from mob.baselines import RandomAssignment

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 5000
    }

    baseline = RandomAssignment(num_experts=3, expert_config=expert_config)

    assert len(baseline.experts) == 3
    print(f"✓ Random Assignment instantiated with {len(baseline.experts)} experts")
    return True


def test_monolithic_ewc_instantiation():
    """Test Monolithic EWC instantiation."""
    print("\nTesting Monolithic EWC...")

    from mob.baselines import MonolithicEWC
    from mob import create_model

    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    baseline = MonolithicEWC(model, lambda_ewc=5000)

    assert baseline.expert is not None
    print(f"✓ Monolithic EWC instantiated")
    return True


def test_gated_moe_instantiation():
    """Test Gated MoE instantiation."""
    print("\nTesting Gated MoE...")

    from mob.baselines import GatedMoE

    expert_config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 784
    }

    baseline = GatedMoE(num_experts=3, expert_config=expert_config)

    assert len(baseline.expert_models) == 3
    assert baseline.gater is not None
    print(f"✓ Gated MoE instantiated with {len(baseline.expert_models)} experts and gater")
    return True


def test_forward_passes():
    """Test that all baselines can do forward passes."""
    print("\nTesting forward passes...")

    from mob.baselines import NaiveFineTuning, RandomAssignment, MonolithicEWC, GatedMoE
    from mob import create_model

    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))

    # Naive
    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    naive = NaiveFineTuning(model)
    with torch.no_grad():
        logits = naive.model(x)
    assert logits.shape == (4, 10)
    print("✓ Naive forward pass works")

    # Random Assignment
    config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'alpha': 0.5,
        'beta': 0.5,
        'lambda_ewc': 5000
    }
    random_baseline = RandomAssignment(num_experts=2, expert_config=config)
    for expert in random_baseline.experts:
        with torch.no_grad():
            logits = expert.model(x)
        assert logits.shape == (4, 10)
    print("✓ Random Assignment forward pass works")

    # Monolithic EWC
    model = create_model('simple_cnn', num_classes=10, input_channels=1)
    mono = MonolithicEWC(model, lambda_ewc=1000)
    with torch.no_grad():
        logits = mono.expert.model(x)
    assert logits.shape == (4, 10)
    print("✓ Monolithic EWC forward pass works")

    # Gated MoE
    config = {
        'architecture': 'simple_cnn',
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 784
    }
    gated = GatedMoE(num_experts=2, expert_config=config)
    with torch.no_grad():
        gating_logits = gated.gater(x)
        assert gating_logits.shape == (4, 2)
        for expert in gated.expert_models:
            logits = expert(x)
            assert logits.shape == (4, 10)
    print("✓ Gated MoE forward pass works")

    return True


def run_all_tests():
    """Run all baseline tests."""
    print("="*60)
    print("Baseline Implementation Tests")
    print("="*60)

    tests = [
        ("Imports", test_baseline_imports),
        ("Naive Instantiation", test_naive_instantiation),
        ("Random Assignment Instantiation", test_random_instantiation),
        ("Monolithic EWC Instantiation", test_monolithic_ewc_instantiation),
        ("Gated MoE Instantiation", test_gated_moe_instantiation),
        ("Forward Passes", test_forward_passes),
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
        print("\n🎉 All baseline tests passed! Ready for validation experiments.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")

    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
