#!/bin/bash
#SBATCH -J prisma-check
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
uv run python check_tokens_prisma.py --images $TRAIN_IMAGES --shards $SHARDS_ROOT/$TRAIN_HASH --saev-layer 48
