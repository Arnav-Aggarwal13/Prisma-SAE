#!/bin/bash
#SBATCH -J prisma-apply
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
# Default SAE: k=128, 32x, tied init (best val nmse 0.211). Override: sbatch --export=ALL,CKPT=...,COND=sae_pre slurm/apply_sae.sbatch
# Oracle control (SAE applied to the true image embeddings): sbatch --export=ALL,ORACLE=1 slurm/apply_sae.sbatch
CKPT=${CKPT:-$RUNS_DIR/full_tied/acf07f5c-bigg-prisma-sae/n_images_496128.pt}
EXTRA=()
[ -n "${ORACLE:-}" ] && EXTRA+=(--oracle)
[ -n "${COND:-}" ] && EXTRA+=(--cond "$COND")
[ -n "${OVERWRITE:-}" ] && EXTRA+=(--overwrite)
uv run python apply_sae.py --ckpt $CKPT "${EXTRA[@]}"
