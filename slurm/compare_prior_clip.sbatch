#!/bin/bash
#SBATCH -J prisma-cmp
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
M=$HOME/orcd/scratch/mindeye/outputs
uv run python compare_prior_clip.py --prior-sae $M/sae_pre2gp9q3hp --clip-sae $M/sae_pre_oracle2gp9q3hp --out $RUNS_DIR/analysis/prior_vs_clip.json
