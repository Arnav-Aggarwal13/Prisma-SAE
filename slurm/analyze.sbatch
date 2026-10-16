#!/bin/bash
#SBATCH -J prisma-analyze
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
# Default: the k=128 tied-init run (best nmse 0.219). Override: sbatch --export=ALL,CKPT=...,OUT=...,FEATURES=12+345 slurm/analyze.sbatch
CKPT=${CKPT:-$RUNS_DIR/full_tied/5dc3a416-bigg-prisma-sae/n_images_496128.pt}
OUT=${OUT:-$RUNS_DIR/analysis/k128}
EXTRA=()
[ -n "${FEATURES:-}" ] && EXTRA+=(--features "${FEATURES//+/,}")  # + instead of , because sbatch --export splits on commas
[ -n "${SKIP_ENCODE:-}" ] && EXTRA+=(--skip-encode)
uv run python analyze_sae.py --ckpt $CKPT --val-shards $SHARDS_ROOT/$VAL_HASH --images $VAL_IMAGES --saev-layer 48 --out $OUT "${EXTRA[@]}"
