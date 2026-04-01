#!/usr/bin/env bash
# winner.sh — 9L MLP3x leaderboard submission (8x H100, 600s)
#
# Architecture: 9 layers, MLP 3x width (~15.7MB compressed, under 16MB limit)
# Hyperparameters: all ablation winners incorporated (warmdown=2000, matrix_lr=0.030, etc.)
#
# Verify model size after run: ls -lh final_model.pt  (must be <16MB)

RUN_ID=submission_9L_mlp3 \
NUM_LAYERS=9 \
MLP_MULT=3 \
LEAKY_RELU_SLOPE=0.5 \
LN_SCALE=1 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
XSA_LAST_N=0 \
ROPE_DIMS=16 \
BIGRAM_VOCAB_SIZE=2048 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=393216 \
WARMDOWN_ITERS=3500 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MATRIX_LR=0.030 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.025 \
MUON_WEIGHT_DECAY=0.04 \
ADAM_WEIGHT_DECAY=0.04 \
GRAD_CLIP_NORM=0.3 \
LATE_QAT_THRESHOLD=0.25 \
INT6_LAYER_START=0 \
INT6_LAYER_END=8 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=1200 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
