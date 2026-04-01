#!/usr/bin/env bash
# sweep_leaky_relu.sh — sweep leaky relu slope + related hyperparameters
#
# Phase 1: run each experiment for 10 minutes, save logs
# Phase 2: pick the best val loss, run it for 20 minutes
#
# Logs land in ./sweep_logs/<experiment_name>/
# Safe to delete this script and ./sweep_logs/ when done.
#
# Usage (single GPU):
#   bash sweep_leaky_relu.sh
#
# Usage (multi-GPU, e.g. 4):
#   NPROC=4 bash sweep_leaky_relu.sh

set -euo pipefail

NPROC="${NPROC:-1}"
LOG_DIR="./sweep_logs"
PHASE1_SECONDS=600    # 10 min per experiment
PHASE2_SECONDS=1200   # 20 min for the winner

# ── single-GPU friendly defaults (matches 1-GPU section of run.sh) ────────────
BASE_ARGS=(
    NUM_LAYERS=9
    MLP_MULT=2
    TRAIN_SEQ_LEN=1024
    TRAIN_BATCH_TOKENS=524288
    WARMDOWN_ITERS=1200
    MUON_MOMENTUM=0.99
    MUON_MOMENTUM_WARMUP_START=0.92
    MUON_MOMENTUM_WARMUP_STEPS=500
    MATRIX_LR=0.025
    SCALAR_LR=0.025
    TIED_EMBED_LR=0.035
    MUON_WEIGHT_DECAY=0.04
    ADAM_WEIGHT_DECAY=0.04
    GRAD_CLIP_NORM=0.3
    INT6_LAYER_START=0
    INT6_LAYER_END=8
    DATA_PATH=./data/datasets/fineweb10B_sp1024/
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
    VOCAB_SIZE=1024
    VAL_LOSS_EVERY=200
)

# ── experiments: name + overrides ─────────────────────────────────────────────
# Each entry: "name|KEY=VAL KEY=VAL ..."
# Vary leaky relu slope (0.0 = pure relu^2 baseline) and a few LR configs
# since activation scale affects gradient magnitude.
EXPERIMENTS=(
    "relu2_baseline|LEAKY_RELU_SLOPE=0.0"
    "leaky_0.01|LEAKY_RELU_SLOPE=0.01"
    "leaky_0.05|LEAKY_RELU_SLOPE=0.05"
    "leaky_0.1|LEAKY_RELU_SLOPE=0.1"
    "leaky_0.25|LEAKY_RELU_SLOPE=0.25"
    "leaky_0.5|LEAKY_RELU_SLOPE=0.5"
    "leaky_0.1_lr_low|LEAKY_RELU_SLOPE=0.1 MATRIX_LR=0.02 SCALAR_LR=0.02"
    "leaky_0.1_lr_high|LEAKY_RELU_SLOPE=0.1 MATRIX_LR=0.03 SCALAR_LR=0.03"
)

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  PHASE 1 — 10-min sweeps (${#EXPERIMENTS[@]} experiments)"
echo "  Logs → $LOG_DIR"
echo "============================================================"

for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%|*}"
    overrides="${entry##*|}"
    exp_dir="$LOG_DIR/$name"
    mkdir -p "$exp_dir"

    echo ""
    echo "── Running: $name ──────────────────────────────────────"
    echo "   Overrides: $overrides"
    echo "   Log: $exp_dir/train.log"

    # Build env: base args + per-experiment overrides
    env_prefix=""
    for kv in "${BASE_ARGS[@]}"; do
        env_prefix="$env_prefix $kv"
    done
    env_prefix="$env_prefix $overrides"
    env_prefix="$env_prefix RUN_ID=$name MAX_WALLCLOCK_SECONDS=$PHASE1_SECONDS"

    eval "env $env_prefix torchrun --standalone --nproc_per_node=$NPROC train_gpt.py" \
        2>&1 | tee "$exp_dir/train.log"

    echo "   Done: $name"
done

echo ""
echo "============================================================"
echo "  PHASE 1 complete — extracting best val loss"
echo "============================================================"

# Parse the last reported val loss from each log.
# train_gpt.py prints lines like: step X | val_loss Y.YYYY | ...
best_name=""
best_loss=999999

for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%|*}"
    log="$LOG_DIR/$name/train.log"
    if [[ ! -f "$log" ]]; then
        echo "  WARNING: log not found for $name, skipping"
        continue
    fi

    # grab last val_loss value from the log
    last_val=$(grep -oP 'val_loss\s+\K[0-9]+\.[0-9]+' "$log" | tail -1)
    if [[ -z "$last_val" ]]; then
        echo "  WARNING: no val_loss found in $name log, skipping"
        continue
    fi

    echo "  $name → val_loss = $last_val"

    # compare with awk (bash can't do float comparison natively)
    is_better=$(awk -v cur="$last_val" -v best="$best_loss" 'BEGIN { print (cur < best) ? "yes" : "no" }')
    if [[ "$is_better" == "yes" ]]; then
        best_loss="$last_val"
        best_name="$name"
    fi
done

if [[ -z "$best_name" ]]; then
    echo "ERROR: could not determine a winner. Check logs in $LOG_DIR."
    exit 1
fi

echo ""
echo "  Winner: $best_name  (val_loss = $best_loss)"

# Retrieve the winner's overrides
winner_overrides=""
for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%|*}"
    if [[ "$name" == "$best_name" ]]; then
        winner_overrides="${entry##*|}"
        break
    fi
done

echo ""
echo "============================================================"
echo "  PHASE 2 — 20-min run with winner: $best_name"
echo "  Overrides: $winner_overrides"
echo "============================================================"

winner_dir="$LOG_DIR/${best_name}_phase2"
mkdir -p "$winner_dir"

env_prefix=""
for kv in "${BASE_ARGS[@]}"; do
    env_prefix="$env_prefix $kv"
done
env_prefix="$env_prefix $winner_overrides"
env_prefix="$env_prefix RUN_ID=${best_name}_phase2 MAX_WALLCLOCK_SECONDS=$PHASE2_SECONDS"

eval "env $env_prefix torchrun --standalone --nproc_per_node=$NPROC train_gpt.py" \
    2>&1 | tee "$winner_dir/train.log"

echo ""
echo "============================================================"
echo "  All done."
echo "  Phase 1 logs : $LOG_DIR/<experiment_name>/train.log"
echo "  Phase 2 log  : $winner_dir/train.log"
echo "  Winner config: $winner_overrides"
echo "============================================================"
