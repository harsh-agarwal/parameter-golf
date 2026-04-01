#!/usr/bin/env bash
# sweep_hparams.sh — architecture sweep for the 16MB budget constraint
#
# CONTEXT: The 11L MLP3x model is ~19MB compressed, must be <16MB.
# This sweep uses the best hyperparameters from the previous ablation and
# searches for the best architecture that fits in budget.
#
# ABLATION WINNERS (from previous sweep, incorporated into BASE):
#   WARMDOWN_ITERS=2000      (+0.016 over 2700 — clear winner)
#   MATRIX_LR=0.030          (+0.005 over 0.025)
#   TIED_EMBED_LR=0.025      (+0.004 over 0.035)
#   XSA_LAST_N=0             (+0.002 — XSA hurts slightly)
#   LATE_QAT_THRESHOLD=0.25  (+0.001, marginal)
#
# SIZE ESTIMATES (compressed, assuming 1.38x compression):
#   11L MLP2x + bigram2048:  ~15.0MB  (0.98MB headroom)
#   9L  MLP3x + bigram2048:  ~15.7MB  (0.26MB headroom — risky!)
#   9L  MLP3x + no bigram:   ~15.4MB  (0.62MB headroom)
#   10L MLP2x + bigram2048:  ~13.7MB  (uses 85% of budget — ok)
#
# NOTE: 9L MLP3x is VERY tight. The size estimate has ±0.3MB uncertainty.
# Run the 9L MLP3x experiments and verify file size before submitting.
#
# Usage:
#   NPROC=4 bash sweep_hparams.sh              # recommended
#   NPROC=4 bash sweep_hparams.sh phase2       # skip to phase 2
#   NPROC=4 PHASE1_SECONDS=600 bash sweep_hparams.sh

set -uo pipefail

NPROC="${NPROC:-4}"
LOG_DIR="./sweep_arch_logs"
PHASE1_SECONDS="${PHASE1_SECONDS:-600}"
PHASE2_SECONDS="${PHASE2_SECONDS:-1200}"
SKIP_SLIDING_WINDOW="${SKIP_SLIDING_WINDOW:-0}"
PROGRESS_FILE="$LOG_DIR/progress.tsv"
WINNER_FILE="$LOG_DIR/winner.txt"

# ── Base config: ablation winners + 11L MLP2x as default arch ─────────────────
# INT6_LAYER_END must match num_layers-1; each experiment overrides it if
# changing NUM_LAYERS (e.g. 9L needs INT6_LAYER_END=8, 10L needs INT6_LAYER_END=9).
BASE=(
    NUM_LAYERS=11
    MLP_MULT=2
    LEAKY_RELU_SLOPE=0.5
    LN_SCALE=1
    EMA_ENABLED=1
    EMA_DECAY=0.997
    SWA_ENABLED=1
    SWA_EVERY=50
    XSA_LAST_N=0
    ROPE_DIMS=16
    BIGRAM_VOCAB_SIZE=2048
    TRAIN_SEQ_LEN=2048
    TRAIN_BATCH_TOKENS=786432
    WARMDOWN_ITERS=2000
    MUON_MOMENTUM=0.99
    MUON_MOMENTUM_WARMUP_START=0.92
    MUON_MOMENTUM_WARMUP_STEPS=1500
    MATRIX_LR=0.030
    SCALAR_LR=0.025
    TIED_EMBED_LR=0.025
    MUON_WEIGHT_DECAY=0.04
    ADAM_WEIGHT_DECAY=0.04
    GRAD_CLIP_NORM=0.3
    BETA1=0.9
    BETA2=0.95
    LATE_QAT_THRESHOLD=0.25
    INT6_LAYER_START=0
    INT6_LAYER_END=10
    DATA_PATH=./data/datasets/fineweb10B_sp1024/
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
    VOCAB_SIZE=1024
    VAL_LOSS_EVERY=200
)

# ── Experiments ───────────────────────────────────────────────────────────────
# TIER 1: Architecture candidates (all must fit <16MB)
#
#   11L_mlp2  ~15.0MB — deepest model that comfortably fits; baseline for this sweep
#   9L_mlp3   ~15.7MB — same depth as record but MLP3x; RISKY (only 0.26MB headroom)
#   9L_mlp3_nb ~15.4MB — 9L MLP3x with bigram disabled; safer size
#   10L_mlp2  ~13.7MB — 10 layers; leaves 2.3MB unused but could beat 11L MLP2x in BPB
#
# TIER 2: LR search for winning architecture
#   Optimal MATRIX_LR may shift with new architecture (different param counts).
#   0.030 won the ablation but test 0.025 and 0.035 to confirm.
#
# TIER 3: Warmdown for winning architecture
#   WARMDOWN_ITERS=2000 won ablation but that was for 11L MLP3x at ~212ms/step.
#   9L/10L models are faster (~160-180ms/step), giving more total steps.
#   WARMDOWN_ITERS=2000 still gives ~24-28% warmdown — test 1500 and 2500 too.

EXPERIMENTS=(
    # ── Tier 1: Architecture ──────────────────────────────────────────────────
    "11L_mlp2|"
    "9L_mlp3|NUM_LAYERS=9 MLP_MULT=3 INT6_LAYER_END=8"
    "9L_mlp3_nb|NUM_LAYERS=9 MLP_MULT=3 INT6_LAYER_END=8 BIGRAM_VOCAB_SIZE=0"
    "10L_mlp2|NUM_LAYERS=10 MLP_MULT=2 INT6_LAYER_END=9"

    # ── Tier 2: LR search (on 11L MLP2x base) ────────────────────────────────
    "11L_mlp2_mlr025|MATRIX_LR=0.025"
    "11L_mlp2_mlr035|MATRIX_LR=0.035"
    "11L_mlp2_mlr040|MATRIX_LR=0.040"

    # ── Tier 3: Warmdown variants (on 11L MLP2x base) ────────────────────────
    # With MLP2x the model is faster (~160ms/step), giving ~7500 steps in 1200s.
    # 2000 iters = 26.7% warmdown; 1500 = 20%; 2500 = 33%
    "11L_mlp2_wd1500|WARMDOWN_ITERS=1500"
    "11L_mlp2_wd2500|WARMDOWN_ITERS=2500"

    # ── Tier 4: XSA re-test with MLP2x ───────────────────────────────────────
    # XSA_LAST_N=0 won the ablation (11L MLP3x). Re-test with MLP2x since
    # smaller MLP might benefit more from XSA attention enhancement.
    "11L_mlp2_xsa4|XSA_LAST_N=4"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
build_env() {
    local overrides="$1"
    local prefix=""
    for kv in "${BASE[@]}"; do prefix="$prefix $kv"; done
    [[ -n "$overrides" ]] && prefix="$prefix $overrides"
    echo "$prefix"
}

log_progress() {
    local name="$1" status="$2" val_loss="$3" overrides="$4"
    local ts; ts=$(date '+%Y-%m-%d %H:%M:%S')
    printf '%s\t%s\t%s\t%s\t%s\n' "$ts" "$name" "$status" "$val_loss" "$overrides" \
        >> "$PROGRESS_FILE"
}

extract_val_loss() {
    local log="$1"
    local val
    val=$(grep -oP 'final_sliding_window val_loss:\K[0-9]+\.[0-9]+' "$log" 2>/dev/null | tail -1 || true)
    if [[ -z "$val" ]]; then
        val=$(grep -oP 'final_quant_zlib_roundtrip val_loss:\K[0-9]+\.[0-9]+' "$log" 2>/dev/null | tail -1 || true)
    fi
    echo "$val"
}

# ── Init ──────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
[[ ! -f "$PROGRESS_FILE" ]] && printf 'timestamp\tname\tstatus\tval_loss\toverrides\n' > "$PROGRESS_FILE"

# ── PHASE 1 ───────────────────────────────────────────────────────────────────
if [[ "${1:-}" != "phase2" ]]; then

    echo "============================================================"
    echo "  PHASE 1 — ${PHASE1_SECONDS}s sweeps (${#EXPERIMENTS[@]} experiments)"
    echo "  NPROC=$NPROC  |  Progress: $PROGRESS_FILE"
    echo "  IMPORTANT: After phase 1, check file sizes for 9L MLP3x candidates:"
    echo "    ls -lh sweep_arch_logs/9L_mlp3*/final_model.pt 2>/dev/null"
    echo "  They must be <16MB or cannot be submitted."
    echo "============================================================"

    for entry in "${EXPERIMENTS[@]}"; do
        name="${entry%%|*}"
        overrides="${entry##*|}"
        exp_dir="$LOG_DIR/$name"
        done_marker="$exp_dir/.done"

        if [[ -f "$done_marker" ]]; then
            val_loss=$(extract_val_loss "$exp_dir/train.log")
            echo "  SKIP $name (already done, val_loss=${val_loss:-unknown})"
            continue
        fi

        mkdir -p "$exp_dir"
        echo ""
        echo "── $name ────────────────────────────────────────────────"
        [[ -n "$overrides" ]] && echo "   Overrides : $overrides" || echo "   (baseline: 11L MLP2x + ablation winners)"
        echo "   Log       : $exp_dir/train.log"
        echo "   Started   : $(date '+%Y-%m-%d %H:%M:%S')"

        log_progress "$name" "started" "" "$overrides"

        local_eval_stride=""
        [[ "$SKIP_SLIDING_WINDOW" == "1" ]] && local_eval_stride="EVAL_STRIDE=0"
        env_str="$(build_env "$overrides") $local_eval_stride RUN_ID=$name MAX_WALLCLOCK_SECONDS=$PHASE1_SECONDS"

        set +e
        eval "env $env_str stdbuf -oL torchrun --standalone --nproc_per_node=$NPROC train_gpt.py" \
            2>&1 | stdbuf -oL tee -a "$exp_dir/train.log"
        exit_code=${PIPESTATUS[0]}
        set -e

        val_loss=$(extract_val_loss "$exp_dir/train.log")
        finished_at=$(date '+%Y-%m-%d %H:%M:%S')

        if [[ $exit_code -eq 0 ]]; then
            touch "$done_marker"
            echo "   Finished  : $finished_at  |  val_loss = ${val_loss:-not found}"
            log_progress "$name" "done" "${val_loss:-}" "$overrides"
        else
            echo "   FAILED (exit $exit_code) at $finished_at"
            log_progress "$name" "failed(exit=$exit_code)" "${val_loss:-partial}" "$overrides"
        fi
    done

    # ── Pick winner ───────────────────────────────────────────────────────────
    echo ""
    echo "============================================================"
    echo "  PHASE 1 complete — val loss summary (ascending = better)"
    echo "============================================================"

    best_name=""
    best_loss=999999

    for entry in "${EXPERIMENTS[@]}"; do
        name="${entry%%|*}"
        log="$LOG_DIR/$name/train.log"
        [[ ! -f "$log" ]] && echo "  MISSING  $name" && continue
        val_loss=$(extract_val_loss "$log")
        [[ -z "$val_loss" ]] && echo "  NO_VAL   $name" && continue
        echo "  $val_loss  $name"
        is_better=$(awk -v c="$val_loss" -v b="$best_loss" 'BEGIN { print (c < b) ? "yes" : "no" }')
        if [[ "$is_better" == "yes" ]]; then
            best_loss="$val_loss"
            best_name="$name"
        fi
    done

    if [[ -z "$best_name" ]]; then
        echo "ERROR: no winner found. Check logs in $LOG_DIR."
        exit 1
    fi

    winner_overrides=""
    for entry in "${EXPERIMENTS[@]}"; do
        [[ "${entry%%|*}" == "$best_name" ]] && winner_overrides="${entry##*|}" && break
    done

    echo ""
    echo "  Winner : $best_name  (val_loss = $best_loss)"
    [[ -n "$winner_overrides" ]] && echo "  Config : $winner_overrides" || echo "  Config : (11L MLP2x baseline)"
    echo ""
    echo "  REMINDER: If winner uses 9L MLP3x, verify final model fits <16MB"
    echo "    before running phase 2 and certainly before submitting."

    printf '%s\n%s\n%s\n' "$best_name" "$best_loss" "$winner_overrides" > "$WINNER_FILE"
    echo "  Saved  : $WINNER_FILE"
fi

# ── PHASE 2 ───────────────────────────────────────────────────────────────────
[[ ! -f "$WINNER_FILE" ]] && echo "ERROR: $WINNER_FILE not found. Run phase 1 first." && exit 1

best_name=$(sed -n '1p' "$WINNER_FILE")
best_loss=$(sed -n '2p' "$WINNER_FILE")
winner_overrides=$(sed -n '3p' "$WINNER_FILE")

phase2_dir="$LOG_DIR/${best_name}_phase2"
phase2_done="$phase2_dir/.done"

echo ""
echo "============================================================"
echo "  PHASE 2 — ${PHASE2_SECONDS}s  |  winner: $best_name  (phase1 val_loss = $best_loss)"
[[ -n "$winner_overrides" ]] && echo "  Config : $winner_overrides" || echo "  Config : (11L MLP2x baseline)"
echo "============================================================"

if [[ -f "$phase2_done" ]]; then
    val_loss=$(extract_val_loss "$phase2_dir/train.log")
    echo "  Phase 2 already done — val_loss = ${val_loss:-unknown}"
    echo "  Delete $phase2_done to re-run."
else
    mkdir -p "$phase2_dir"
    echo "  Log: $phase2_dir/train.log"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    log_progress "${best_name}_phase2" "started" "" "$winner_overrides"

    env_str="$(build_env "$winner_overrides") RUN_ID=${best_name}_phase2 MAX_WALLCLOCK_SECONDS=$PHASE2_SECONDS"

    set +e
    eval "env $env_str stdbuf -oL torchrun --standalone --nproc_per_node=$NPROC train_gpt.py" \
        2>&1 | stdbuf -oL tee -a "$phase2_dir/train.log"
    exit_code=${PIPESTATUS[0]}
    set -e

    val_loss=$(extract_val_loss "$phase2_dir/train.log")

    if [[ $exit_code -eq 0 ]]; then
        touch "$phase2_done"
        echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')  |  val_loss = ${val_loss:-not found}"
        log_progress "${best_name}_phase2" "done" "${val_loss:-}" "$winner_overrides"
        # Show final model size for budget check
        model_file="final_model.pt"
        [[ -f "$model_file" ]] && echo "  Model size: $(du -sh $model_file | cut -f1) (must be <16MB)"
    else
        echo "  FAILED (exit $exit_code) — re-run: NPROC=$NPROC bash sweep_hparams.sh phase2"
        log_progress "${best_name}_phase2" "failed(exit=$exit_code)" "${val_loss:-partial}" "$winner_overrides"
    fi
fi

echo ""
echo "============================================================"
echo "  All done."
echo "  Progress  : $PROGRESS_FILE"
echo "  Phase 2   : $phase2_dir/train.log"
echo "============================================================"
