## ADDED Requirements

### Requirement: Runnable training config
`configs/dfrnet.yaml` and a new `configs/dfrnet_smoke.yaml` SHALL contain
real, resolvable `data_dir`/`label_file` paths for Train and Eval (no
placeholder `path/to/...` values), and a `pretrained` path that points at an
existing PPOCRv5 `.pdparams` file on disk.

#### Scenario: Config paths resolve
- **WHEN** `configs/dfrnet_smoke.yaml` is loaded and its Train/Eval
  `data_dir` + `label_file` and Model `pretrained` paths are checked against
  the filesystem
- **THEN** every referenced path exists

### Requirement: Label files reference real images
`data/gt/rec/rec_gt_train.txt`, `rec_gt_val.txt`, and `rec_gt_test.txt` SHALL
reference image paths that exist under `data/set1/train/`, `data/set2/val/`,
and `data/set3/test/` respectively (rewritten from their original Colab
absolute paths).

#### Scenario: All label rows resolve to existing files
- **WHEN** each row's rewritten image path in `rec_gt_train.txt`,
  `rec_gt_val.txt`, `rec_gt_test.txt` is resolved relative to `data/`
- **THEN** the file exists on disk for every row, and the row count in each
  file is unchanged from before the rewrite

### Requirement: PPOCRv5 checkpoint loads into DFRNet
Given a working Paddle + PaddleOCR environment, `DFRNet(..., pretrained=<ppocrv5 checkpoint path>)` SHALL load backbone, `ctc_encoder`, and
`ctc_fc` weights from the checkpoint (via the existing key remapping in
`dfrnet/model.py::_load_pretrained`), leaving only the OFR/corruption modules
randomly initialized.

#### Scenario: Checkpoint keys match expected modules
- **WHEN** `tools/inspect_checkpoint.py --ckpt data/checkpoints/release/ppocr_v5_paddle/best_accuracy.pdparams` is run
- **THEN** it reports backbone.*, head.ctc_encoder.*, and head.ctc_head.fc.*
  keys as loaded (matched), and only head.gtc_head.*/head.before_gtc.* as
  skipped

### Requirement: Smoke training run completes and loss decreases
Running `python train.py --config configs/dfrnet_smoke.yaml` inside the
`MLR_LinhNX` container (with paddlepaddle-gpu and PaddleOCR installed) SHALL
complete without error for a short number of epochs, and the reported
training `loss` SHALL trend downward across logged steps.

#### Scenario: Smoke run finishes cleanly with decreasing loss
- **WHEN** the smoke config's training loop runs to completion
- **THEN** the process exits without a traceback/crash, and the last logged
  `loss` value is lower than the first logged `loss` value
