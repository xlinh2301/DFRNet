## Why

The `smoke-train-dfrnet-ppocrv5` change proved the DFRNet training pipeline
runs correctly and converges (loss_main → ~0, eval acc ~92-93% after 15
epochs). But that alone doesn't answer the actual research question: **does
the OFR (Occlusion-aware Feature Refinement) branch make the model better**,
or would plain fine-tuning of the same PPOCRv5 backbone/encoder/CTC-head do
just as well without it? A naive comparison (zero-shot PPOCRv5 vs. our
fine-tuned DFRNet checkpoint) would be confounded — any improvement could
just be from fine-tuning itself, not from OFR.

## What Changes

- Add a **baseline training path**: fine-tune the same PPOCRv5
  backbone+encoder+CTC-head with plain CTC loss only (no corruption, no OFR
  branch), same data/epochs/optimizer schedule as the DFRNet smoke run, so
  the only difference between the two trained models is the presence of the
  OFR branch during training.
- Add an **evaluation script** that runs both trained checkpoints (baseline,
  DFRNet) — plus the untouched zero-shot PPOCRv5 checkpoint as a reference
  point — on `data/set3/test` (`rec_gt_test.txt`), reporting sequence
  accuracy and normalized edit distance for each.
- Extend the comparison to **synthetic occlusion at test time** (reusing
  `dfrnet/corruption.py`'s span-masking, applied to the *encoder output* at
  inference, OFR module bypassed either way) at 2-3 severity levels, since
  the DFRNet hypothesis specifically predicts an advantage under occlusion,
  not necessarily on clean input.
- Produce a results report (numbers + verdict) answering: does DFRNet
  outperform the plain-fine-tuned baseline, and if so, is the gap larger
  under occlusion (supporting the OFR hypothesis) or uniform (suggesting
  it's just extra regularization/capacity)?

**BREAKING**: none.

## Capabilities

### New Capabilities
- `dfrnet-ablation`: ability to fine-tune a plain-CTC baseline (same
  architecture, no OFR) and evaluate it against the DFRNet-trained model and
  the zero-shot checkpoint on the real test set, both clean and under
  synthetic occlusion, producing a clear contribution verdict.

### Modified Capabilities
(none)

## Impact

- New `train_baseline.py` (or a `--no_ofr` mode in `train.py`) reusing the
  existing data pipeline/optimizer wiring
- New `configs/dfrnet_baseline.yaml` (mirrors `dfrnet_smoke.yaml`'s
  epochs/data, OFR-related keys unused)
- New `tools/eval_ablation.py`: loads a checkpoint, evaluates on
  `data/set3/test`, optional occlusion severity parameter
- Runs on `MLR_LinhNX` container (already has paddle/PaddleOCR from the
  prior change)
- No changes to existing `dfrnet/*.py` model code expected (reuses classes
  already fixed and verified in `smoke-train-dfrnet-ppocrv5`)
