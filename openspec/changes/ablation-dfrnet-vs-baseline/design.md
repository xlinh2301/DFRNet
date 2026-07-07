## Context

`dfrnet/model.py`'s `DFRNet.forward()` always runs the corruption+OFR branch
when `self.training` is True, and the total loss is
`L_main + λ·L_aux + β·L_rec` (`dfrnet/loss.py`). Critically, this means the
"baseline, no-OFR" model doesn't need a separate architecture or training
script at all: **setting `lambda_aux: 0.0` and `beta_rec: 0.0` in the Loss
config makes OFR's output contribute zero gradient**, while backbone,
`ctc_encoder`, and `ctc_fc` still train on `L_main` exactly like a plain CTC
fine-tune. OFR still runs forward (wasted compute, harmless) but its weights
never move and never influence the shared head's gradient. This is the same
class, same checkpoint format, same eval code path — the cleanest possible
control for isolating OFR's contribution.

At inference, `DFRNet.forward()` (when `self.training` is False) already
skips OFR entirely and returns softmax logits from `encode()` → `ctc_fc()` —
this is the real deployed inference path, identical for baseline and DFRNet
checkpoints.

For occlusion robustness testing (the actual hypothesis DFRNet is meant to
validate), we need to corrupt features at test time and skip OFR recovery
(since OFR is documented as inference-time-unused/zero-latency by design —
testing "what if a real deployment saw occluded input" means testing the
*encoder's own* robustness, not OFR's ability to un-corrupt it). This is done
by calling `model.encode(images)` directly, applying
`dfrnet.corruption.OcclusionDiffusionCorruption` at a fixed severity `t`, then
`model.ctc_fc()` on the corrupted feature — bypassing `model.forward()`
entirely for this eval path.

## Goals / Non-Goals

**Goals:**
- Produce a controlled 3-way comparison on the real test set
  (`data/set3/test`, `rec_gt_test.txt`): (1) zero-shot PPOCRv5 (no
  fine-tuning at all), (2) baseline fine-tune (CTC-only, same epochs/data as
  the DFRNet smoke run), (3) DFRNet (OFR-trained).
- Test all three under clean input and under 2 synthetic occlusion
  severities, to see whether any DFRNet advantage is occlusion-specific
  (supporting the OFR hypothesis) or uniform (just extra capacity/noise
  regularization).
- Reuse the exact `DFRNet` class and existing fixed code from
  `smoke-train-dfrnet-ppocrv5` — no new model code.

**Non-Goals:**
- Statistical significance testing / multiple seeds (single run each,
  consistent with the smoke-test scope so far — flag this limitation in the
  report rather than engineering multi-seed infra).
- Hyperparameter search for either baseline or DFRNet.
- Testing "real" occlusion (physical watermeter occlusion) — only the
  synthetic span-masking already implemented in `dfrnet/corruption.py`.

## Decisions

1. **Baseline = same `DFRNet` class, `configs/dfrnet_baseline.yaml` copy of
   `dfrnet_smoke.yaml` with `lambda_aux: 0.0`, `beta_rec: 0.0`.** Alternative
   considered: a separate `Encoder+CTCHead`-only script mirroring
   `tools/ablation_contribution.py`'s synthetic-task structure — rejected,
   because building a second real-data training path risks its own bugs
   (exactly what happened with `train.py` in the previous change) and the
   zero-weight trick gives an *exact* apples-to-apples comparison (same
   optimizer param groups, same data order determinism where possible, same
   checkpoint format) for free.
2. **New `tools/eval_ablation.py`**, not reusing `train.py`'s `evaluate()`
   directly, because it needs to: (a) load an arbitrary checkpoint path
   (zero-shot / baseline / dfrnet) into a fresh `DFRNet`, (b) optionally
   corrupt features at a given severity before the CTC head, (c) run against
   `data/set3/test` specifically (not `Eval` data_dir which points at
   `set2/val`). Shares the image-loading/decode logic already written in
   `tools/baseline_check.py` (aspect-preserving resize + normalize, matching
   `resize_norm_img`).
3. **Occlusion severities**: reuse `OcclusionDiffusionCorruption` with fixed
   `t` values corresponding to ~20% and ~50% masked tokens (matching
   `tools/ablation_contribution.py`'s `occ-light`/`occ-heavy` naming for
   consistency), rather than inventing a new corruption scheme.
4. **Same epoch count (15) for baseline as DFRNet's smoke run** — the point
   of this change is a controlled comparison at the training budget already
   proven to converge, not a search for the best budget for either.

## Risks / Trade-offs

- [Risk] Single run, no seed variance — a small accuracy delta could be
  noise, not a real effect. → Mitigation: report deltas explicitly with this
  caveat; recommend multi-seed as future work, don't over-claim from one
  run.
- [Risk] `lambda_aux=0, beta_rec=0` still forwards through OFR/corruption
  every step (wasted GPU time, ~unchanged given prior smoke run's speed) —
  acceptable, not worth a code branch to skip it for a one-off ablation run.
- [Trade-off] Reusing `DFRNet` for the baseline means its `state_dict` still
  contains (randomly-initialized, never-trained) OFR weights — fine, since
  eval never touches them for a baseline checkpoint either.

## Migration Plan

Not applicable — additive tooling/config only, no existing behavior changes.

## Open Questions

- If the ablation shows no meaningful DFRNet advantage even under occlusion,
  is the next step tuning OFR hyperparameters (depth, `mask_ratio_max`,
  `beta_rec`) or is that a signal to reconsider the approach? Out of scope
  for this change — decide after seeing the numbers.
