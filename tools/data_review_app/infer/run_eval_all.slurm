#!/bin/bash
#SBATCH --job-name=eval_all_models
#SBATCH --output=/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/outputs/eval_new_testset/eval_%j.out
#SBATCH --error=/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/outputs/eval_new_testset/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

WS=/datastore/cndt_thangcpd/linhtruong/workspace3
AMR=${WS}/water_meter_amr
mkdir -p ${AMR}/outputs/eval_new_testset

# Use anaconda3 (has torch + paddle + ultralytics + mmocr)
PYTHON=${WS}/anaconda3/bin/python

# Paddle needs its libs
export LD_LIBRARY_PATH=${WS}/envs/paddle/lib:${LD_LIBRARY_PATH:-}

REQUIRED_VRAM=16000
source ${WS}/water_meter_amr/slurm/gpu_setup.sh

$PYTHON "${WS}/water_meter_amr/slurm/eval_all_models.py"
echo "[run_eval_all] done"
