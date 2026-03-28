#!/usr/bin/env bash
# run.sh — training commands for the Parameter Golf competition.
#
# Changes active by default in train_gpt.py:
#   - FP16 tied embedding export            (saves ~0.5MB artifact space)
#   - Int6 quantization on all block layers (saves ~25% vs int8)
#   - Sliding window eval, stride=64        (free ~0.03 BPB improvement)
#   - Late QAT at lr_scale<0.15             (closes quant gap from ~2 BPB to ~0.03 BPB)
#   - Decoupled weight decay 0.04           (Muon + AdamW)
#
# ─────────────────────────────────────────────────────────────────────────────
# SINGLE GPU — for local testing and iteration
#
# WHY different settings: 11L + MLP3x + seq_len=2048 runs at ~1200ms/step on 1
# GPU. In 600s that's only ~500 steps — not enough to converge or reach warmdown.
# These smaller settings match the baseline speed (~400ms/step, ~1500 steps),
# so you can verify training is healthy and QAT is working before scaling up.
# ─────────────────────────────────────────────────────────────────────────────

# RUN_ID=test_1gpu \
# NUM_LAYERS=9 \
# MLP_MULT=2 \
# TRAIN_SEQ_LEN=1024 \
# TRAIN_BATCH_TOKENS=524288 \
# WARMDOWN_ITERS=1200 \
# MUON_MOMENTUM=0.99 \
# MUON_MOMENTUM_WARMUP_START=0.92 \
# MUON_MOMENTUM_WARMUP_STEPS=500 \
# MATRIX_LR=0.025 \
# SCALAR_LR=0.025 \
# TIED_EMBED_LR=0.035 \
# MUON_WEIGHT_DECAY=0.04 \
# ADAM_WEIGHT_DECAY=0.04 \
# GRAD_CLIP_NORM=0.3 \
# INT6_LAYER_START=0 \
# INT6_LAYER_END=8 \
# DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
# TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
# VOCAB_SIZE=1024 \
# torchrun --standalone --nproc_per_node=1 train_gpt.py

# ─────────────────────────────────────────────────────────────────────────────
# 8x H100 — leaderboard submission
#
# At ~83ms/step on 8xH100 you get ~7200 steps in 600s. WARMDOWN_ITERS=3500
# means warmdown starts at ~3700 steps, QAT activates at ~15% of peak LR
# (deep into warmdown), giving the model ~500 steps of QAT adaptation before
# export. INT6_LAYER_END=10 covers all 11 layers (0-indexed).
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID=submission_8gpu \
NUM_LAYERS=11 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
WARMDOWN_ITERS=3500 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_WEIGHT_DECAY=0.04 \
ADAM_WEIGHT_DECAY=0.04 \
GRAD_CLIP_NORM=0.3 \
INT6_LAYER_START=0 \
INT6_LAYER_END=10 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
