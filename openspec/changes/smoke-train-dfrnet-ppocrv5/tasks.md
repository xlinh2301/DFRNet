## 1. Data label paths

- [ ] 1.1 Write a small script/one-off to rewrite `data/gt/rec/rec_gt_train.txt`
      image paths to `set1/train/<basename>`, `rec_gt_val.txt` to
      `set2/val/<basename>`, `rec_gt_test.txt` to `set3/test/<basename>`
- [ ] 1.2 Verify every rewritten row resolves to an existing file under
      `data/`, and row counts match the originals (2464/422/439)

## 2. Config fixes

- [ ] 2.1 Fix `configs/dfrnet.yaml` Train/Eval `data_dir`/`label_file` to
      point at `data/` + the rewritten label files, and `pretrained` to
      `data/checkpoints/release/ppocr_v5_paddle/best_accuracy.pdparams`
- [ ] 2.2 Create `configs/dfrnet_smoke.yaml` (copy of the fixed config) with
      a small `epochs` (e.g. 2) and `eval_step`/`save_epoch_step` tuned so a
      run finishes in minutes

## 3. Checkpoint sanity check (no Paddle needed)

- [ ] 3.1 Run `tools/inspect_checkpoint.py --ckpt data/checkpoints/release/ppocr_v5_paddle/best_accuracy.pdparams`
      locally and confirm backbone/ctc_encoder/ctc_head.fc keys are matched

## 4. Container environment

- [ ] 4.1 On `MLR_LinhNX` (via `docker exec`), install a paddlepaddle-gpu
      wheel compatible with the container's CUDA (12.8); fall back to the
      closest available CUDA 12.x build or CPU build if no exact match exists
- [ ] 4.2 Shallow-clone `PaddleOCR` into `/workspace/PaddleOCR` inside the
      container
- [ ] 4.3 Verify `import paddle` and `from ppocr.data import build_dataloader`
      both succeed inside the container

## 5. Smoke run

- [ ] 5.1 Run `python train.py --config configs/dfrnet_smoke.yaml` from
      `/workspace/DRNet` inside the container
- [ ] 5.2 Confirm the checkpoint-loaded log line reports matched/skipped
      counts consistent with task 3.1
- [ ] 5.3 Confirm the run completes without a crash and logged `loss`
      decreases from first to last step
- [ ] 5.4 Record the actual paddlepaddle-gpu version installed and any
      deviations from the design's assumptions in the change's notes
