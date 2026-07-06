## Context

`train.py` builds on PaddleOCR's data pipeline (`ppocr.data.build_dataloader`,
`SimpleDataSet`, `CTCLabelEncode`, `RecResizeImg`) and imports `PaddleOCR` from
a hardcoded sibling path (`../PaddleOCR` relative to the DFRNet repo root,
i.e. `dfrnet/model.py` walks up to `../../../PaddleOCR`). The `MLR_LinhNX`
container currently has `torch==2.11.0+cu128` but no `paddle` and no
`PaddleOCR` checkout — this is a from-scratch environment provisioning step,
not just a code fix.

The label files under `data/gt/rec/` were exported from a Colab notebook and
contain absolute paths like
`/content/drive/MyDrive/MASTER 2025/.../crops/train/<file>.jpg\t<label>`.
The actual images live in `data/set1/train/`, `data/set2/val/`,
`data/set3/test/` on both the local machine and the server (already rsynced).
Filenames are unique enough (`24302_..._jpg.rf.<hash>_<n>.jpg`) that path
rewriting is a matter of replacing the directory prefix, keeping the basename.

The PPOCRv5 checkpoint's own `config.yml` confirms the architecture DFRNet's
`configs/dfrnet.yaml` already assumes: `PPLCNetV3` backbone (scale 0.95),
`EncoderWithSVTR` head neck (dims=120, depth=2, hidden_dims=120,
kernel_size=[1,3], use_guide=true), digit charset (11 classes: 0-9 + blank),
image shape `[3, 64, 256]`, `max_text_length: 25`.

## Goals / Non-Goals

**Goals:**
- Get `train.py --config configs/dfrnet_smoke.yaml` to run for a handful of
  epochs/steps on the real dataset without crashing.
- Confirm the checkpoint loader in `dfrnet/model.py::_load_pretrained` matches
  the expected number of backbone/ctc_encoder/ctc_fc keys (use
  `tools/inspect_checkpoint.py` first — it needs no Paddle).
- Confirm loss decreases over the smoke run (sanity, not convergence).

**Non-Goals:**
- Full/production training run (100 epochs) — that's a follow-up change once
  the pipeline is proven.
- Hyperparameter tuning, data augmentation review, or accuracy targets.
- Changing `dfrnet/*.py` model architecture (only touched if the smoke run
  surfaces an actual bug).

## Decisions

1. **Rewrite label paths in place, not the images.** The three
   `rec_gt_*.txt` files get their path column rewritten to
   `set1/train/<basename>`, `set2/val/<basename>`, `set3/test/<basename>`
   (relative to `data/`), matching `train_cfg["data_dir"] = "data"` +
   `label_file`. Alternative considered: symlink a `data/train` dir mirroring
   the old absolute structure — rejected, more moving parts than a one-time
   text rewrite, and the paths are already broken/unusable as-is.

2. **New `configs/dfrnet_smoke.yaml` instead of editing `dfrnet.yaml` for the
   smoke run's epoch count.** `dfrnet.yaml` gets its placeholder paths fixed
   (since it's currently unusable either way) but keeps `epochs: 100` for
   real training later. The smoke config inherits the same paths but overrides
   `epochs` (e.g. 2) and `eval_step`/`save_epoch_step` so a smoke run finishes
   in minutes instead of hours.

3. **Install paddlepaddle-gpu + PaddleOCR directly into the running
   `MLR_LinhNX` container**, not baked into a new image. This is the fastest
   path to a smoke test; if it works out, a Dockerfile change to make it
   reproducible is a separate follow-up (out of scope here per Non-Goals).
   Risk: state is lost if the container is recreated again — acceptable for
   a smoke test.

4. **PaddleOCR checkout location**: clone to `/workspace/PaddleOCR` inside
   the container, since `train.py` resolves `../PaddleOCR` relative to the
   DFRNet repo at `/workspace/DRNet`, i.e. `/workspace/PaddleOCR`. Use a
   shallow clone (`--depth 1`) of the official `PaddlePaddle/PaddleOCR` repo;
   only `ppocr.data`, `ppocr.postprocess`, `ppocr.metrics`,
   `ppocr.modeling.backbones`, `ppocr.modeling.necks.rnn` are needed.

5. **paddlepaddle-gpu version**: pick the newest paddlepaddle-gpu wheel
   compatible with CUDA 12.x available at install time (check
   `https://www.paddlepaddle.org.cn/` install matrix); if no prebuilt wheel
   matches CUDA 12.8 exactly, fall back to the closest CUDA 12.x build since
   paddle wheels are typically forward-compatible within a CUDA major
   version, and document whatever version actually gets installed in the
   task/verify notes.

## Risks / Trade-offs

- [Risk] No paddlepaddle-gpu wheel is compatible with CUDA 12.8 / this GPU
  driver → Mitigation: try CPU paddle build as a last resort just to validate
  pipeline logic (slower, but proves correctness); flag to the user if GPU
  paddle truly isn't installable and a production run needs a different
  container/image.
- [Risk] Label file path rewrite silently drops or mismatches rows if
  filenames collide across sets → Mitigation: rewrite via a small script that
  errors loudly if a referenced basename isn't found in the target set
  directory, and prints counts matched vs. total.
- [Risk] Checkpoint key remapping in `_load_pretrained` doesn't match this
  specific PPOCRv5 checkpoint format (keys/shapes differ) → Mitigation: run
  `tools/inspect_checkpoint.py` against `best_accuracy.pdparams` first, before
  wiring up the full training loop, to see matched/skipped counts.
- [Trade-off] Environment changes (pip installs, git clone) live only inside
  the container's writable layer, not committed anywhere → acceptable for a
  smoke test; documented in tasks.md as a manual step to redo if the
  container is recreated.

## Migration Plan

Not applicable — no deployed system, no rollback needed. If the smoke run
fails, no destructive changes are made outside the container's own
package/checkout state (revertible via `docker exec ... pip uninstall` or
container recreation) and the two edited files (`dfrnet.yaml`,
`rec_gt_*.txt`) are backed up before rewrite.

## Open Questions

- Should the eventual production Dockerfile bake in paddlepaddle-gpu +
  PaddleOCR, or should DFRNet move to the pure-PyTorch path hinted at by
  `tools/smoke_test_torch.py`? Out of scope for this smoke-test change —
  revisit once the Paddle path is proven (or proven infeasible).
