#!/bin/bash
# ===========================================================================
# MoB Routing Experiments v2 — Comprehensive Suite
# ===========================================================================
#
# 4 experts, 5 tasks. One expert MUST handle 2 tasks — that's the continual
# learning challenge MoB is designed to solve.
#
# Changes from v1:
#   - Default lambda_ewc=1000 (v1 used 5.0, catastrophically weak)
#   - Exp3 (training-time prototype routing) bug fixed: incremental centroids
#   - Multi-seed runs for statistical significance
#   - Forgetting cost scale & epochs ablations
#   - Fully label-free pipeline experiments (the LLM scaling path)
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh 2>&1 | tee experiment_log.txt
#
# ===========================================================================

set -e

RESULTS_DIR="results/experiments_v2"
mkdir -p "$RESULTS_DIR"

EPOCHS=4
COMMON="--epochs $EPOCHS --save_bids --reset_optimizer"

EXP_NUM=0
TOTAL_EXPS=47
START_TIME=$(date +%s)

run_mob() {
    EXP_NUM=$((EXP_NUM + 1))
    local name="$1"
    shift
    echo ""
    echo "=================================================================="
    echo "[$EXP_NUM/$TOTAL_EXPS] $name"
    echo "=================================================================="
    local exp_start=$(date +%s)
    python tests/run_mob_only.py $COMMON --experiment_name "$name" "$@"
    local exp_end=$(date +%s)
    echo "[TIMER] $name completed in $((exp_end - exp_start))s"
    echo ""
    [ -f "results/${name}_bids.json" ]    && mv "results/${name}_bids.json"    "$RESULTS_DIR/"
    [ -f "results/${name}_results.json" ] && mv "results/${name}_results.json" "$RESULTS_DIR/"
}

run_continual() {
    EXP_NUM=$((EXP_NUM + 1))
    local name="$1"
    shift
    echo ""
    echo "=================================================================="
    echo "[$EXP_NUM/$TOTAL_EXPS] $name"
    echo "=================================================================="
    local exp_start=$(date +%s)
    python tests/run_continual_mob.py $COMMON --experiment_name "$name" "$@"
    local exp_end=$(date +%s)
    echo "[TIMER] $name completed in $((exp_end - exp_start))s"
    echo ""
    [ -f "results/${name}_bids.json" ]    && mv "results/${name}_bids.json"    "$RESULTS_DIR/"
    [ -f "results/${name}_summary.txt" ]  && mv "results/${name}_summary.txt"  "$RESULTS_DIR/"
}

echo "=================================================================="
echo "MoB ROUTING EXPERIMENTS v2 — COMPREHENSIVE"
echo "=================================================================="
echo "4 experts, 5 tasks (one expert handles 2 tasks)"
echo "Epochs: $EPOCHS | Output: $RESULTS_DIR/"
echo "Total experiments: $TOTAL_EXPS"
echo "Default lambda_ewc: 1000"
echo "Started at: $(date)"
echo "=================================================================="


# ===================================================================
# PHASE 1: BASELINES (lambda=1000, seed=42)
# ===================================================================
echo ""
echo "#################### PHASE 1: BASELINES ####################"

# 1. Pseudo-label routing (original MoB eval)
run_mob "base_pseudolabel" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy pseudo_label

# 2. Prototype per-batch
run_mob "base_prototype_perbatch" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype

# 3. Per-sample k=1 (per-token analog)
run_mob "base_persample_k1" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1

# 4-5. Per-sample k=2 at two temperatures
run_mob "base_persample_k2_t0.5" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 2 --temperature 0.5

run_mob "base_persample_k2_t1.0" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 2 --temperature 1.0


# ===================================================================
# PHASE 2: DISTANCE-ONLY BIDDING
# Can we drop forget_cost from eval entirely?
# ===================================================================
echo ""
echo "#################### PHASE 2: DISTANCE-ONLY ####################"

# 6. Distance-only per-batch (compare to #2)
run_mob "distonly_perbatch" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --eval_bid_mode distance_only

# 7. Distance-only per-sample k=1 (compare to #3)
run_mob "distonly_persample_k1" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1 --eval_bid_mode distance_only

# 8. Distance-only per-sample k=2 (compare to #4)
run_mob "distonly_persample_k2" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 2 --temperature 0.5 --eval_bid_mode distance_only


# ===================================================================
# PHASE 3: TRAINING-TIME PROTOTYPE ROUTING (BUG FIXED)
# ===================================================================
echo ""
echo "#################### PHASE 3: TRAIN-TIME ROUTING (FIXED) ####################"

# 9. Warmup=0 (pure prototype from batch 1 — strongest fix validation)
run_mob "trainproto_w0" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 0

# 10. Warmup=500
run_mob "trainproto_w500" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 500

# 11. Warmup=1000
run_mob "trainproto_w1000" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000

# 12. Warmup=1500
run_mob "trainproto_w1500" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1500


# ===================================================================
# PHASE 4: FULLY LABEL-FREE PIPELINE
# Train with prototypes + eval with distance-only = zero label dependency
# This is the LLM scaling path
# ===================================================================
echo ""
echo "#################### PHASE 4: FULLY LABEL-FREE ####################"

# 13. Label-free, per-sample k=1
run_mob "labelfree_k1" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000 \
    --per_sample --top_k 1 --eval_bid_mode distance_only

# 14. Label-free, per-batch
run_mob "labelfree_perbatch" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000 \
    --eval_bid_mode distance_only

# 15. Label-free, k=2 (does blending help with noisier prototype routing?)
run_mob "labelfree_k2" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000 \
    --per_sample --top_k 2 --temperature 0.5 --eval_bid_mode distance_only


# ===================================================================
# PHASE 5: ALPHA/BETA ABLATION (lambda=1000, per-sample k=1)
# v1 showed alpha=0.3/beta=0.7 was dramatically better
# ===================================================================
echo ""
echo "#################### PHASE 5: ALPHA/BETA ####################"

# 16. Favor forget cost (load balancing signal)
run_mob "ab_a0.3_b0.7" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1

# 17. Favor distance (specialization signal)
run_mob "ab_a0.7_b0.3" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.7 --beta 0.3 \
    --routing_strategy prototype --per_sample --top_k 1

# 18. Alpha/beta with train-proto (does the ratio matter for label-free training?)
run_mob "ab_a0.3_b0.7_trainproto" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1 \
    --train_routing prototype --train_routing_warmup 1000


# ===================================================================
# PHASE 6: EWC ABLATION (per-sample k=1)
# ===================================================================
echo ""
echo "#################### PHASE 6: EWC ABLATION ####################"

# 19-22. Bracket lambda (base_persample_k1 is lambda=1000)
run_mob "ewc_l100" \
    --seed 42 --lambda_ewc 100.0 --routing_strategy prototype --per_sample --top_k 1

run_mob "ewc_l500" \
    --seed 42 --lambda_ewc 500.0 --routing_strategy prototype --per_sample --top_k 1

run_mob "ewc_l2000" \
    --seed 42 --lambda_ewc 2000.0 --routing_strategy prototype --per_sample --top_k 1

run_mob "ewc_l5000" \
    --seed 42 --lambda_ewc 5000.0 --routing_strategy prototype --per_sample --top_k 1


# ===================================================================
# PHASE 7: FORGETTING COST SCALE ABLATION
# Optuna found forgetting_cost_scale=2.13 for task-aware MoB
# ===================================================================
echo ""
echo "#################### PHASE 7: FORGETTING SCALE ####################"

# 23-25. (base_persample_k1 is scale=1.0)
run_mob "fscale_0.5" \
    --seed 42 --lambda_ewc 1000.0 --forgetting_cost_scale 0.5 \
    --routing_strategy prototype --per_sample --top_k 1

run_mob "fscale_2.0" \
    --seed 42 --lambda_ewc 1000.0 --forgetting_cost_scale 2.0 \
    --routing_strategy prototype --per_sample --top_k 1

run_mob "fscale_3.0" \
    --seed 42 --lambda_ewc 1000.0 --forgetting_cost_scale 3.0 \
    --routing_strategy prototype --per_sample --top_k 1


# ===================================================================
# PHASE 8: EPOCHS ABLATION
# ===================================================================
echo ""
echo "#################### PHASE 8: EPOCHS ABLATION ####################"

# 26. 2 epochs
EXP_NUM=$((EXP_NUM + 1))
echo ""
echo "=================================================================="
echo "[$EXP_NUM/$TOTAL_EXPS] epochs_2"
echo "=================================================================="
python tests/run_mob_only.py --epochs 2 --save_bids --reset_optimizer \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1 \
    --experiment_name "epochs_2"
[ -f "results/epochs_2_bids.json" ]    && mv "results/epochs_2_bids.json"    "$RESULTS_DIR/"
[ -f "results/epochs_2_results.json" ] && mv "results/epochs_2_results.json" "$RESULTS_DIR/"

# 27. 8 epochs
EXP_NUM=$((EXP_NUM + 1))
echo ""
echo "=================================================================="
echo "[$EXP_NUM/$TOTAL_EXPS] epochs_8"
echo "=================================================================="
python tests/run_mob_only.py --epochs 8 --save_bids --reset_optimizer \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1 \
    --experiment_name "epochs_8"
[ -f "results/epochs_8_bids.json" ]    && mv "results/epochs_8_bids.json"    "$RESULTS_DIR/"
[ -f "results/epochs_8_results.json" ] && mv "results/epochs_8_results.json" "$RESULTS_DIR/"


# ===================================================================
# PHASE 9: COMBINED BEST (lambda=1000, alpha=0.3, beta=0.7)
# ===================================================================
echo ""
echo "#################### PHASE 9: COMBINED BEST ####################"

# 28. Combined best, per-sample k=1
run_mob "combined_best" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1

# 29. Combined best + distance-only eval
run_mob "combined_best_distonly" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1 --eval_bid_mode distance_only

# 30. Combined best + fully label-free
run_mob "combined_best_labelfree" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1 \
    --train_routing prototype --train_routing_warmup 1000 \
    --eval_bid_mode distance_only

# 31. Combined best + forgetting scale 2.0
run_mob "combined_best_fscale2" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 --forgetting_cost_scale 2.0 \
    --routing_strategy prototype --per_sample --top_k 1

# 32. Combined best + forgetting scale 2.0 + label-free (full kitchen sink)
run_mob "combined_full" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 --forgetting_cost_scale 2.0 \
    --routing_strategy prototype --per_sample --top_k 1 \
    --train_routing prototype --train_routing_warmup 1000 \
    --eval_bid_mode distance_only


# ===================================================================
# PHASE 10: MULTI-SEED VALIDATION (seeds 123, 456)
# Seed 42 covered in previous phases. 3 seeds for significance.
# ===================================================================
echo ""
echo "#################### PHASE 10: MULTI-SEED ####################"

# --- Per-sample k=1 baseline ---
run_mob "ms_persample_s123" \
    --seed 123 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1

run_mob "ms_persample_s456" \
    --seed 456 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1

# --- Combined best ---
run_mob "ms_combined_s123" \
    --seed 123 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1

run_mob "ms_combined_s456" \
    --seed 456 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1

# --- Fully label-free ---
run_mob "ms_labelfree_s123" \
    --seed 123 --lambda_ewc 1000.0 --routing_strategy prototype \
    --per_sample --top_k 1 --train_routing prototype --train_routing_warmup 1000 \
    --eval_bid_mode distance_only

run_mob "ms_labelfree_s456" \
    --seed 456 --lambda_ewc 1000.0 --routing_strategy prototype \
    --per_sample --top_k 1 --train_routing prototype --train_routing_warmup 1000 \
    --eval_bid_mode distance_only

# --- Train-proto (check if warmup matters across seeds) ---
run_mob "ms_trainproto_s123" \
    --seed 123 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000

run_mob "ms_trainproto_s456" \
    --seed 456 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000


# ===================================================================
# PHASE 11: CONTINUAL MOB (task-free, shift detection)
# ===================================================================
echo ""
echo "#################### PHASE 11: CONTINUAL MOB ####################"

# --- Core comparisons (seed 42) ---
# 41. Pseudo-label baseline
run_continual "cmob_pseudolabel" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy pseudo_label

# 42. Prototype per-batch
run_continual "cmob_prototype" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype

# 43. Per-sample k=1
run_continual "cmob_persample_k1" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1

# 44. Distance-only
run_continual "cmob_distonly" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --per_sample --top_k 1 --eval_bid_mode distance_only

# 45. Train-proto warmup=500
run_continual "cmob_trainproto_w500" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 500

# 46. Train-proto warmup=1000
run_continual "cmob_trainproto_w1000" \
    --seed 42 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000

# 47. Alpha/beta tuning for continual
run_continual "cmob_a0.3_b0.7" \
    --seed 42 --lambda_ewc 1000.0 --alpha 0.3 --beta 0.7 \
    --routing_strategy prototype --per_sample --top_k 1

# --- Multi-seed continual ---
run_continual "cmob_ms_s123" \
    --seed 123 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1

run_continual "cmob_ms_s456" \
    --seed 456 --lambda_ewc 1000.0 --routing_strategy prototype --per_sample --top_k 1

run_continual "cmob_ms_trainproto_s123" \
    --seed 123 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000

run_continual "cmob_ms_trainproto_s456" \
    --seed 456 --lambda_ewc 1000.0 --routing_strategy prototype \
    --train_routing prototype --train_routing_warmup 1000


# ===================================================================
# SUMMARY
# ===================================================================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "=================================================================="
echo "ALL $TOTAL_EXPS EXPERIMENTS COMPLETE"
echo "=================================================================="
echo "Total time: $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
echo "Results directory: $RESULTS_DIR/"
echo ""

# Generate comparison table
echo "=================================================================="
echo "RESULTS TABLE — TASK-AWARE MOB"
echo "=================================================================="
echo ""
printf "%-42s %10s %10s %6s\n" "Experiment" "Accuracy" "Forgetting" "Seed"
printf "%-42s %10s %10s %6s\n" "------------------------------------------" "----------" "----------" "------"

for f in "$RESULTS_DIR"/*_results.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _results.json)
        acc=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['avg_accuracy']:.4f}\")" 2>/dev/null || echo "N/A")
        fgt=$(python -c "import json; d=json.load(open('$f')); print(f\"{d['forgetting']:.4f}\")" 2>/dev/null || echo "N/A")
        seed=$(python -c "import json; d=json.load(open('$f')); print(d.get('seed','?'))" 2>/dev/null || echo "?")
        printf "%-42s %10s %10s %6s\n" "$name" "$acc" "$fgt" "$seed"
    fi
done

echo ""
echo "=================================================================="
echo "RESULTS TABLE — CONTINUAL MOB"
echo "=================================================================="
echo ""
for f in "$RESULTS_DIR"/cmob_*_summary.txt; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _summary.txt)
        content=$(head -1 "$f")
        printf "%-42s %s\n" "$name" "$content"
    fi
done

echo ""
echo "=================================================================="
echo "KEY COMPARISONS"
echo "=================================================================="
echo ""
echo "ROUTING QUALITY:"
echo "  base_persample_k1 vs base_prototype_perbatch vs base_pseudolabel"
echo "  k=2: base_persample_k2_t0.5 vs base_persample_k2_t1.0 vs base_persample_k1"
echo ""
echo "DISTANCE-ONLY (LLM-viable eval):"
echo "  distonly_persample_k1 vs base_persample_k1 (should be close)"
echo "  distonly_perbatch vs base_prototype_perbatch (should be identical)"
echo ""
echo "TRAINING-TIME ROUTING (bug fixed):"
echo "  trainproto_w0 / w500 / w1000 / w1500 (should NOT collapse to 19.5%)"
echo ""
echo "FULLY LABEL-FREE (the LLM story):"
echo "  labelfree_k1, labelfree_perbatch, labelfree_k2"
echo "  combined_best_labelfree, combined_full"
echo ""
echo "HYPERPARAMETERS:"
echo "  ewc:    ewc_l100 / l500 / base(1000) / l2000 / l5000"
echo "  a/b:    ab_a0.3_b0.7 vs base(0.5/0.5) vs ab_a0.7_b0.3"
echo "  fscale: fscale_0.5 vs base(1.0) vs fscale_2.0 vs fscale_3.0"
echo "  epochs: epochs_2 vs base(4) vs epochs_8"
echo ""
echo "ROBUSTNESS (multi-seed, compute mean +/- std):"
echo "  persample:  base_persample_k1 + ms_persample_s123 + ms_persample_s456"
echo "  combined:   combined_best + ms_combined_s123 + ms_combined_s456"
echo "  labelfree:  labelfree_k1 + ms_labelfree_s123 + ms_labelfree_s456"
echo "  trainproto: trainproto_w1000 + ms_trainproto_s123 + ms_trainproto_s456"
echo "  continual:  cmob_persample_k1 + cmob_ms_s123 + cmob_ms_s456"
