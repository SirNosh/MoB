"""
Formula Comparison: Addition vs Subtraction

Runs run_mob_only.py with different lambda and formula configurations.
"""

import subprocess
import json
import sys

lambdas = [10, 100, 500, 1000]
formulas = ['subtraction', 'addition']

print("=" * 90)
print("FORMULA COMPARISON: ADDITION vs SUBTRACTION (with pseudo-label eval routing)")
print("=" * 90)
print(f"\n{'Lambda':<10} {'Formula':<12} {'T1':<8} {'T2':<8} {'T3':<8} {'T4':<8} {'T5':<8} {'Avg':<8}")
print("-" * 90)

results = []

for lam in lambdas:
    for formula in formulas:
        # Run the experiment
        cmd = [
            sys.executable, 'tests/run_mob_only.py',
            '--seed', '42',
            '--lambda_ewc', str(lam),
            '--formula', formula,
            '--epochs', '4',
            '--reset_optimizer'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Read results
        try:
            with open('results/mob_results_seed_42.json', 'r') as f:
                data = json.load(f)

            accs = data['final_accuracies']
            avg = data['avg_accuracy']

            acc_str = "  ".join([f"{a*100:5.1f}%" for a in accs])
            print(f"{lam:<10} {formula:<12} {acc_str}  {avg*100:5.1f}%")
            sys.stdout.flush()

            results.append({
                'lambda': lam,
                'formula': formula,
                'accs': accs,
                'avg': avg
            })
        except Exception as e:
            print(f"{lam:<10} {formula:<12} ERROR: {e}")
            sys.stdout.flush()

print("\n" + "=" * 90)
print("ANALYSIS")
print("=" * 90)

print("\nDifference (SUBTRACTION - ADDITION) in avg accuracy:")
for lam in lambdas:
    sub = [r for r in results if r['lambda'] == lam and r['formula'] == 'subtraction']
    add = [r for r in results if r['lambda'] == lam and r['formula'] == 'addition']
    if sub and add:
        diff = (sub[0]['avg'] - add[0]['avg']) * 100
        winner = "SUBTRACTION" if diff > 1 else "ADDITION" if diff < -1 else "TIE"
        print(f"  Lambda {lam:>4}: {diff:+.1f}% -> {winner}")

if results:
    best = max(results, key=lambda r: r['avg'])
    print(f"\nBest config: lambda={best['lambda']}, formula={best['formula']}, avg={best['avg']*100:.1f}%")
