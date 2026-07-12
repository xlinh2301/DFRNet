# Batch inference (manual, SLURM)

These scripts are prepared but **not submitted automatically** — running them
consumes shared GPU resources on `cndt_thangcpd@slurm.uit.edu.vn`. Review and
submit yourself, or ask explicitly.

## 1. PPOCRv5 over the test set

```
# from local machine
scp data/DATA_COCO/annotations/instances_test.json \
    cndt_thangcpd@slurm.uit.edu.vn:/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/outputs/instances_test_manifest.json

scp tools/data_review_app/infer/infer_ppocrv5_test.py tools/data_review_app/infer/run_ppocrv5_test.slurm \
    cndt_thangcpd@slurm.uit.edu.vn:/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/slurm/

# on server
ssh cndt_thangcpd@slurm.uit.edu.vn
cd /datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/slurm
sbatch run_ppocrv5_test.slurm
squeue -u cndt_thangcpd   # watch it run
# check slurm/logs/ppocrv5_eval_test_<jobid>.out for progress/errors

# once done, pull results back locally
scp cndt_thangcpd@slurm.uit.edu.vn:/datastore/cndt_thangcpd/linhtruong/workspace3/water_meter_amr/outputs/predictions_ppocrv5_test.json \
    data/predictions_ppocrv5_test.json
```

## 2. YOLO-OBB over the 100k pool

```
# from local machine
scp data/100k/results_v2/e2e/label.json \
    cndt_thangcpd@slurm.uit.edu.vn:/datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/outputs/e2e_label_manifest.json

scp tools/data_review_app/infer/infer_obb_100k.py tools/data_review_app/infer/run_obb_100k.slurm \
    cndt_thangcpd@slurm.uit.edu.vn:/datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/slurm/

# on server
ssh cndt_thangcpd@slurm.uit.edu.vn
cd /datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/slurm
sbatch run_obb_100k.slurm
squeue -u cndt_thangcpd
# check slurm/logs/obb_infer_100k_<jobid>.out for progress/errors
# ~100k images — adjust #SBATCH --time in run_obb_100k.slurm if it needs longer than 12h

# once done, pull results back locally
scp cndt_thangcpd@slurm.uit.edu.vn:/datastore/cndt_thangcpd/linhtruong/workspace3/wmr_char_attention/outputs/predictions_obb_100k.json \
    data/predictions_obb_100k.json
```

## After both are pulled back

Restart the web app (`uvicorn app:app --port 8008`) so it picks up the new
`data/predictions_ppocrv5_test.json` and `data/predictions_obb_100k.json` files.
The Eval view and the OBB-sourced Supplement view will then be populated.
