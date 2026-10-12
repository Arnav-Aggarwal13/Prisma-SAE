# Paths for Prisma-SAE. Values copied from SAETraining/contrib/bigg_openimages/env.sh (not sourced, on purpose).
# Usage: source env.sh   (from the repo root)
export POOL=$HOME/orcd/pool
export SCRATCH=$HOME/orcd/scratch
export OI_DIR=$POOL/openimages
export TRAIN_IMAGES=$OI_DIR/train
export VAL_IMAGES=$OI_DIR/val
export SHARDS_ROOT=$SCRATCH/saev/shards
export TRAIN_HASH=99a257f8
export VAL_HASH=c3eca3fc
export HF_HOME=$POOL/hf_cache
export RUNS_DIR=$HOME/prisma_runs
export WANDB_DIR=$RUNS_DIR/wandb
mkdir -p "$RUNS_DIR" "$WANDB_DIR"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON=3.12
