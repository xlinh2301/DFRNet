#!/bin/bash
#SBATCH --job-name=obb_staging
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/slurm/logs/%x_%j.out
#SBATCH --error=/datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

WS=/datastore/cndt_thangcpd/linhtruong/workspace3
REQUIRED_VRAM=14000
source ${WS}/wmr_char_attention/slurm/gpu_setup.sh

BATCH_ID="${BATCH_ID:?must set BATCH_ID=<batch_timestamp>}"
IMAGES_DIR="${IMAGES_DIR:-${WS}/supplement_staging/${BATCH_ID}/images}"
OUT="${OUT:-${WS}/supplement_staging/${BATCH_ID}/predictions_obb_staging_${BATCH_ID}.json}"

if [ ! -d "$IMAGES_DIR" ]; then
    echo "[run_obb_staging] ERROR: images dir not found at $IMAGES_DIR"
    exit 1
fi

$PYTHON "${WS}/wmr_char_attention/slurm/infer_obb_staging.py" \
    --images-dir "$IMAGES_DIR" \
    --out "$OUT"

echo "[run_obb_staging] done -> $OUT"
