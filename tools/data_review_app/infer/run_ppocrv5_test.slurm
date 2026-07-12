#!/bin/bash
#SBATCH --job-name=ppocrv5_eval_test
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/slurm/logs/%x_%j.out
#SBATCH --error=/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue

# PPOCRv5 inference over the DATA_COCO test set (439 images).
# Prereq: scp instances_test.json to $WS/water_meter_amr/outputs/instances_test_manifest.json first.
#
# NOT auto-submitted. Review then run manually:
#   sbatch run_ppocrv5_test.slurm

WS=/datastore/cndt_thangcpd/linhtruong/workspace3
REQUIRED_VRAM=8000
PYTHON=${WS}/envs/paddle/bin/python
export LD_LIBRARY_PATH=${WS}/envs/paddle/lib:${LD_LIBRARY_PATH:-}
source ${WS}/water_meter_amr/slurm/gpu_setup.sh

MANIFEST="${MANIFEST:-${WS}/water_meter_amr/outputs/instances_test_manifest.json}"
OUT="${OUT:-${WS}/water_meter_amr/outputs/predictions_ppocrv5_test.json}"

if [ ! -f "$MANIFEST" ]; then
    echo "[run_ppocrv5_test] ERROR: manifest not found at $MANIFEST"
    echo "  scp data/DATA_COCO/annotations/instances_test.json to that path first."
    exit 1
fi

$PYTHON "${WS}/water_meter_amr/slurm/infer_ppocrv5_test.py" --instances "$MANIFEST" --out "$OUT"
echo "[run_ppocrv5_test] done"
