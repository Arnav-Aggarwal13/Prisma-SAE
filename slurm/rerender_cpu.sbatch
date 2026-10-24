#!/bin/bash
#SBATCH -J prisma-regrid
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/Prisma-SAE && source env.sh
export OMP_NUM_THREADS=8
# Re-render the grids of an existing analysis (stage 2 only, on CPU, no GPU needed). OUT must hold features.npz from analyze_sae.py.
# Required: CKPT (the SAE the analysis was made with), OUT. Optional: COPY_FROM (analysis dir whose features.npz is copied into OUT first, for tests),
#   FEATURES=12+345 (extra latents), N_PER_GROUP (default 20), HEAT_FLOOR (default 0.1).
: "${CKPT:?set CKPT}" "${OUT:?set OUT}"
if [ -n "${COPY_FROM:-}" ]; then mkdir -p "$OUT" && cp "$COPY_FROM/features.npz" "$OUT/"; fi
EXTRA=()
[ -n "${FEATURES:-}" ] && EXTRA+=(--features "${FEATURES//+/,}")  # + instead of , because sbatch --export splits on commas
[ -n "${HEAT_FLOOR:-}" ] && EXTRA+=(--heat-floor "$HEAT_FLOOR")
uv run python analyze_sae.py --ckpt $CKPT --val-shards $SHARDS_ROOT/$VAL_HASH --images $VAL_IMAGES --saev-layer 48 --out $OUT --skip-encode --device cpu --n-per-group ${N_PER_GROUP:-20} "${EXTRA[@]}"
