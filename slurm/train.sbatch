#!/bin/bash
#SBATCH -J prisma-train
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
# Compute nodes may lack internet: log W&B offline, `wandb sync` later from a login node.
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Override with: sbatch --export=ALL,TOTAL_TOKENS=...,SWEEP=...,CKPT_DIR=... slurm/train.sbatch
# SWEEP set but empty (SWEEP=) trains one default run (k=64, 16x, lr 1e-3).
TOTAL_TOKENS=${TOTAL_TOKENS:-128000000}
SWEEP=${SWEEP-sweeps/first_sweep.py}
CKPT_DIR=${CKPT_DIR:-$RUNS_DIR}
SWEEP_ARGS=()
[ -n "$SWEEP" ] && SWEEP_ARGS=(--sweep "$SWEEP")
uv run python train_prisma.py --route shards \
  --train-shards $SHARDS_ROOT/$TRAIN_HASH --val-shards $SHARDS_ROOT/$VAL_HASH --saev-layer 48 --model-local-path $MODEL_LOCAL \
  --ckpt-dir $CKPT_DIR --total-tokens $TOTAL_TOKENS "${SWEEP_ARGS[@]}"
