#!/bin/bash
#SBATCH -J prisma-valcos
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
# Default: the four SAEs of the full sweep (k=64/128, 16x/32x). One pass over the val shards for all of them, then upload to each SAE's W&B run.
# Override: sbatch --export=ALL,CKPTS="path1 path2" slurm/val_cosine.sbatch
T=$RUNS_DIR/full_tied
CKPTS=${CKPTS:-"$T/af59271f-bigg-prisma-sae/n_images_496128.pt $T/f9c689a1-bigg-prisma-sae/n_images_496128.pt $T/5dc3a416-bigg-prisma-sae/n_images_496128.pt $T/acf07f5c-bigg-prisma-sae/n_images_496128.pt"}
export WANDB_MODE=online
uv run python eval_val_cosine.py --ckpt $CKPTS --val-shards $SHARDS_ROOT/$VAL_HASH --saev-layer 48 --out-dir $RUNS_DIR/analysis/val_cosine
