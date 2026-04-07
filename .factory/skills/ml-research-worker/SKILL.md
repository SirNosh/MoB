---
name: ml-research-worker
description: Implements features for the MoB research codebase — routing mechanisms, tests, and experiment scripts.
---

# ML Research Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for any feature that involves:
- Modifying Python source files in `contibualmob/` or `tests/`
- Adding new routing mechanisms, CLI flags, or experiment configurations
- Writing unit tests for ML components
- Creating experiment runner scripts

## Required Skills

None — all work is done via file editing and CLI commands.

## Work Procedure

### Step 1: Understand the Feature

1. Read the feature description, preconditions, expectedBehavior, and verificationSteps from features.json.
2. Read `C:\MoB Final\.factory\library\architecture.md` to understand where your changes integrate.
3. Read the specific source files you'll modify. Always read the FULL file before editing.
4. Read `C:\Users\devya\.factory\missions\57a33d41-d638-42d2-bb9a-9eb08c0f8c74\AGENTS.md` for coding conventions and boundaries.

### Step 2: Write Tests First (TDD)

1. Read `tests/test_components.py` to understand existing test patterns.
2. Write failing tests for the new behavior. Each test should:
   - Create minimal instances of the components being tested
   - Exercise the specific behavior defined in expectedBehavior
   - Assert concrete outcomes (not just "doesn't crash")
3. Run `pytest tests/test_components.py -v` to confirm the new tests FAIL (red phase).

### Step 3: Implement

1. Make the minimum changes needed to pass the tests.
2. Follow existing code patterns:
   - Match the style of surrounding code (no type hints, minimal comments)
   - Use the same import patterns
   - Keep new parameters backward-compatible (sensible defaults that preserve existing behavior)
3. Key files you may modify:
   - `contibualmob/pool.py` — ExpertPool bid collection, routing logic
   - `contibualmob/expert.py` — Expert prototype management
   - `contibualmob/prototype_store.py` — Distance computation, centroid management
   - `tests/run_mob_only.py` — CLI flags, training loop integration
   - `tests/run_continual_mob.py` — CLI flags, continual training loop
   - `tests/test_components.py` — Unit tests
4. NEVER modify files in `mob/` package or `results/experiments_v2/`.

### Step 4: Verify Tests Pass

1. Run `pytest tests/test_components.py -v` — all tests must pass (green phase).
2. If any test fails, fix the implementation (not the test, unless the test itself is wrong).

### Step 5: Integration Verification

1. If the feature adds CLI flags, verify they work:
   ```powershell
   python tests/run_mob_only.py --help
   ```
   Confirm new flags appear in help output.

2. If the feature modifies routing behavior, run a quick smoke test:
   ```powershell
   python tests/run_mob_only.py --seed 42 --epochs 1 --train_routing prototype [new_flags]
   ```
   Verify it completes without errors and produces output.

3. If the feature should be backward-compatible, run without new flags:
   ```powershell
   python tests/run_mob_only.py --seed 42 --epochs 1
   ```
   Verify it still works identically to before your changes.

### Step 6: Run Full Verification (for later features)

For features that need full experiment runs (not just unit tests):
```powershell
python tests/run_mob_only.py --seed 42 --train_routing prototype [flags] --experiment_name v3_test
```
Check the result JSON in `results/` for accuracy, load balance, and routing metrics.

**IMPORTANT:** Full experiment runs take ~90-110 seconds. Only run them when verification steps explicitly require it. For most features, unit tests + smoke tests are sufficient.

## Example Handoff

```json
{
  "salientSummary": "Implemented conscience bias mechanism in ExpertPool.collect_bids(). Added --use_conscience, --conscience_rate, --conscience_decay flags to run_mob_only.py. Wrote 3 unit tests (test_conscience_bias_reduces_imbalance, test_conscience_backward_compat, test_conscience_params_configurable). All 11 tests pass. Smoke test with --train_routing prototype --use_conscience runs without error.",
  "whatWasImplemented": "Added load_bias array and update_conscience_bias() method to ExpertPool in contibualmob/pool.py. Conscience bias is applied in collect_bids() when use_conscience=True. Added 3 CLI flags to tests/run_mob_only.py argparser. Added 3 tests to tests/test_components.py.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "pytest tests/test_components.py -v",
        "exitCode": 0,
        "observation": "11 passed, 0 failed. New tests test_conscience_bias_reduces_imbalance, test_conscience_backward_compat, test_conscience_params_configurable all pass."
      },
      {
        "command": "python tests/run_mob_only.py --help",
        "exitCode": 0,
        "observation": "Help output includes --use_conscience, --conscience_rate (default 0.01), --conscience_decay (default 0.999) flags."
      },
      {
        "command": "python tests/run_mob_only.py --seed 42 --epochs 1 --train_routing prototype --use_conscience",
        "exitCode": 0,
        "observation": "Completed in 23s. Expert win distribution: [28%, 22%, 25%, 25%] — no collapse. Accuracy: 45% (1 epoch, expected to be low)."
      }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_components.py",
        "cases": [
          { "name": "test_conscience_bias_reduces_imbalance", "verifies": "Conscience bias penalizes frequent winners and redistributes load" },
          { "name": "test_conscience_backward_compat", "verifies": "Without use_conscience flag, behavior is identical to baseline" },
          { "name": "test_conscience_params_configurable", "verifies": "Different conscience_rate and conscience_decay values produce different bias trajectories" }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- A required source file doesn't exist or has unexpected structure
- Existing tests fail BEFORE your changes (pre-existing failure)
- The feature requires modifying `mob/` package (off-limits)
- Full experiment runs produce NaN/infinity values indicating a mathematical error
- A dependency on another feature's output is not yet available
