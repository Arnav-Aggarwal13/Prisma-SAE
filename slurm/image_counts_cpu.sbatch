#!/bin/bash
#SBATCH -J prisma-imgcounts
#SBATCH -p mit_normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
# CPU only, reads each analysis dir's features.npz. Override: sbatch --export=ALL,DIRS="dir1 dir2" slurm/image_counts_cpu.sbatch
DIRS=${DIRS:-"$RUNS_DIR/analysis/k128 $RUNS_DIR/analysis/k64_16x $RUNS_DIR/analysis/k64_32x"}
uv run python image_count_estimate.py $DIRS
