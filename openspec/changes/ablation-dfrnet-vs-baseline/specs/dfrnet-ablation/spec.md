## ADDED Requirements

### Requirement: Controlled baseline config
`configs/dfrnet_baseline.yaml` SHALL be identical to
`configs/dfrnet_smoke.yaml` in data paths, epochs, optimizer, and image
shape, differing only in `Loss.lambda_aux: 0.0` and `Loss.beta_rec: 0.0`, so
that the only distinguishing factor between the two trained checkpoints is
whether OFR's output influenced training.

#### Scenario: Configs differ only in loss weights
- **WHEN** `configs/dfrnet_baseline.yaml` and `configs/dfrnet_smoke.yaml` are
  diffed
- **THEN** the only differing keys are `Loss.lambda_aux` and `Loss.beta_rec`

### Requirement: Baseline checkpoint trains without OFR influence
Running `python train.py --config configs/dfrnet_baseline.yaml` SHALL
produce a checkpoint whose backbone/encoder/CTC-head weights were updated
only by `L_main` (standard CTC loss on clean features), with OFR module
weights remaining at their random initialization (never receiving nonzero
gradient).

#### Scenario: OFR weights unchanged after baseline training
- **WHEN** the baseline run's saved final checkpoint's `ofr.*` parameters are
  compared to a freshly-initialized `DFRNet`'s `ofr.*` parameters
- **THEN** they are statistically consistent with no training having
  occurred (this is a design consequence of zero loss weight, verified by
  inspecting that `lambda_aux`/`beta_rec` are indeed 0 in the run's loaded
  config, not by literal tensor diffing)

### Requirement: Three-way test-set evaluation
`tools/eval_ablation.py` SHALL evaluate a given checkpoint against
`data/set3/test` (`data/gt/rec/rec_gt_test.txt`) and report sequence
accuracy and normalized edit distance, runnable against: the zero-shot
PPOCRv5 checkpoint, the baseline-trained checkpoint, and the DFRNet-trained
checkpoint.

#### Scenario: Same eval script works for all three checkpoints
- **WHEN** `tools/eval_ablation.py --checkpoint <path>` is run once per
  checkpoint (zero-shot, baseline, dfrnet) against the same test set
- **THEN** each run reports an accuracy and normalized-edit-distance number
  without requiring script changes between runs

### Requirement: Occlusion-severity evaluation
`tools/eval_ablation.py` SHALL support an `--occlusion` severity parameter
(`none`, `light` ≈20% masked tokens, `heavy` ≈50% masked tokens) that
corrupts the encoder's output features (via
`dfrnet.corruption.OcclusionDiffusionCorruption`) before the shared CTC head,
bypassing the OFR module, for all three checkpoints under comparison.

#### Scenario: Occlusion degrades accuracy in a checkpoint-comparable way
- **WHEN** the same checkpoint is evaluated at `--occlusion none`, `light`,
  and `heavy`
- **THEN** accuracy at `heavy` is less than or equal to accuracy at `light`,
  which is less than or equal to accuracy at `none` (monotonic degradation)

### Requirement: Contribution verdict report
The change SHALL produce a results summary comparing all three checkpoints
across all occlusion severities, explicitly stating whether DFRNet
outperforms the plain-fine-tuned baseline, and whether any advantage is
occlusion-specific (widens under `light`/`heavy`) or uniform across
severities.

#### Scenario: Report states a clear verdict
- **WHEN** the ablation results are compiled
- **THEN** the report includes, for each severity, the accuracy delta
  (DFRNet − baseline) and a one-line verdict on whether OFR contributes and
  under what conditions
