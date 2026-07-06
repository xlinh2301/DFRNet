## Why

DFRNet has model code (`dfrnet/`), a training script (`train.py`), and a config
(`configs/dfrnet.yaml`), but it has never been run end-to-end. The config still
has placeholder paths (`path/to/train`), the label files reference Colab
absolute paths that don't exist here, and the target container has neither
`paddle` nor `PaddleOCR` installed. Before investing in a real training run we
need to prove the pipeline actually executes: data loads, the PPOCRv5
checkpoint loads into the backbone/encoder/head, and the loss decreases over a
handful of steps.

## What Changes

- Rewrite `data/gt/rec/rec_gt_{train,val,test}.txt` image paths from the
  original Colab absolute paths to paths relative to `data/set1` (train),
  `data/set2` (val), `data/set3` (test).
- Fix `configs/dfrnet.yaml`: real `data_dir`/`label_file` paths for Train/Eval,
  correct `pretrained` path to the downloaded
  `data/checkpoints/release/ppocr_v5_paddle/best_accuracy.pdparams`, and a
  small `epochs`/`eval_step` smoke-test override (separate smoke config, not
  overwriting the production one).
- Provision the `MLR_LinhNX` container with `paddlepaddle-gpu` and a
  `PaddleOCR` checkout at the sibling path `train.py` expects
  (`/workspace/PaddleOCR`), matching the digit character dict used by the
  PPOCRv5 checkpoint (`ppocr/utils/dict/digits_dict.txt`).
- Run a short smoke train (few epochs, small step count) on the container and
  confirm: checkpoint keys load (backbone/ctc_encoder/ctc_fc matched, OFR/
  corruption randomly initialized), no crashes, and `loss` trends down.

**BREAKING**: none — this only touches local data label files and a new
config; existing `configs/dfrnet.yaml` production paths are also being fixed
since they are currently non-functional placeholders.

## Capabilities

### New Capabilities
- `dfrnet-smoke-train`: ability to run a short, verifiable training pass of
  DFRNet on the water-meter digit dataset using the PPOCRv5 pretrained
  checkpoint, confirming the pipeline (data loading, checkpoint remapping,
  OFR corruption/recovery, loss computation) works end-to-end.

### Modified Capabilities
(none — no other existing spec covers training config/paths today)

## Impact

- `configs/dfrnet.yaml` (paths fixed), new `configs/dfrnet_smoke.yaml`
- `data/gt/rec/rec_gt_train.txt`, `rec_gt_val.txt`, `rec_gt_test.txt` (paths rewritten)
- `MLR_LinhNX` container on `linhnx_gmoe`: new `paddlepaddle-gpu` install,
  new `/workspace/PaddleOCR` checkout (pip/system state change inside the
  container, not the image itself)
- No changes to `dfrnet/*.py` model code expected unless the smoke run
  surfaces a bug in checkpoint key remapping or shapes.
