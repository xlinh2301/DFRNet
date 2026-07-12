#!/bin/bash
#SBATCH --job-name=obb_infer_100k
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/slurm/logs/%x_%j.out
#SBATCH --error=/datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue

# YOLOv8m-OBB inference over the full 100k image pool (~100,000 images — sized for
# up to 12h; adjust --time if the actual throughput needs more).
# Prereq: scp data/100k/results_v2/e2e/label.json to
#   $WS/wmr_char_attention/outputs/e2e_label_manifest.json first (for id/file_name alignment).
#
# NOT auto-submitted. Review then run manually:
#   sbatch run_obb_100k.slurm

WS=/datastore/cndt_thangcpd/linhtruong/workspace3
REQUIRED_VRAM=14000
source ${WS}/wmr_char_attention/slurm/gpu_setup.sh

MANIFEST="${MANIFEST:-${WS}/wmr_char_attention/outputs/e2e_label_manifest.json}"
OUT="${OUT:-${WS}/wmr_char_attention/outputs/predictions_obb_100k.json}"

if [ ! -f "$MANIFEST" ]; then
    echo "[run_obb_100k] ERROR: manifest not found at $MANIFEST"
    echo "  scp data/100k/results_v2/e2e/label.json to that path first."
    exit 1
fi

$PYTHON "${WS}/wmr_char_attention/slurm/infer_obb_100k.py" --images-manifest "$MANIFEST" --out "$OUT"
echo "[run_obb_100k] done"
