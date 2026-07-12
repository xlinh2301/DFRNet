#!/bin/bash
#SBATCH --job-name=ppocr_staging
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/slurm/logs/%x_%j.out
#SBATCH --error=/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

WS=/datastore/cndt_thangcpd/linhtruong/workspace3
REQUIRED_VRAM=8000
PYTHON=${WS}/envs/paddle/bin/python
export LD_LIBRARY_PATH=${WS}/envs/paddle/lib:${LD_LIBRARY_PATH:-}
source ${WS}/water_meter_amr/slurm/gpu_setup.sh

BATCH_ID="${BATCH_ID:?must set BATCH_ID=<batch_timestamp>}"
IMAGES_DIR="${IMAGES_DIR:-${WS}/supplement_staging/${BATCH_ID}/images}"
OBB_JSON="${OBB_JSON:-${WS}/supplement_staging/${BATCH_ID}/predictions_obb_staging_${BATCH_ID}.json}"
OUT="${OUT:-${WS}/supplement_staging/${BATCH_ID}/predictions_ppocrv5_staging_${BATCH_ID}.json}"

if [ ! -f "$OBB_JSON" ]; then
    echo "[run_ppocrv5_staging] ERROR: OBB JSON not found at $OBB_JSON"
    echo "  Run run_obb_staging.slurm first."
    exit 1
fi

$PYTHON "${WS}/water_meter_amr/slurm/infer_ppocrv5_staging.py" \
    --obb-json "$OBB_JSON" \
    --images-dir "$IMAGES_DIR" \
    --out "$OUT"

echo "[run_ppocrv5_staging] done -> $OUT"
