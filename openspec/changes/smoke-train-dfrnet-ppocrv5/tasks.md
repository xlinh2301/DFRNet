## 1. Data label paths

- [x] 1.1 Write a small script/one-off to rewrite `data/gt/rec/rec_gt_train.txt`
      image paths to `set1/train/<basename>`, `rec_gt_val.txt` to
      `set2/val/<basename>`, `rec_gt_test.txt` to `set3/test/<basename>`
      (`tools/rewrite_gt_paths.py`)
- [x] 1.2 Verify every rewritten row resolves to an existing file under
      `data/`, and row counts match the originals (2464/422/439) — confirmed,
      0 missing

## 2. Config fixes

- [x] 2.1 Fix `configs/dfrnet.yaml` Train/Eval `data_dir`/`label_file` to
      point at `data/` + the rewritten label files, and `pretrained` to
      `data/checkpoints/release/ppocr_v5_paddle/best_accuracy.pdparams`
- [x] 2.2 Create `configs/dfrnet_smoke.yaml` (copy of the fixed config) with
      a small `epochs` (raised from 2 to 15 — see notes) so a run finishes
      in minutes

## 3. Checkpoint sanity check (no Paddle needed)

- [x] 3.1 Run `tools/inspect_checkpoint.py --ckpt data/checkpoints/release/ppocr_v5_paddle/best_accuracy.pdparams`
      locally and confirm backbone/ctc_encoder/ctc_head.fc keys are matched
      (884/927 after the remap-prefix fix — see notes)

## 4. Container environment

- [x] 4.1 On `MLR_LinhNX` (via `docker exec`), install a paddlepaddle-gpu
      wheel compatible with the container's CUDA (12.8) — installed
      `paddlepaddle-gpu==3.0.0` from the `cu126` index (no `cu128` wheel
      existed at install time; cu126 build runs fine, paddle logs a CUDA
      12.6-vs-12.2-runtime warning but trains correctly on GPU)
- [x] 4.2 Shallow-clone `PaddleOCR` into `/workspace/PaddleOCR` inside the
      container, plus `pip install -r requirements.txt` (scikit-image,
      shapely, albumentations, etc. — not pre-installed)
- [x] 4.3 Verify `import paddle` and `from ppocr.data import build_dataloader`
      both succeed inside the container

## 5. Smoke run

- [x] 5.1 Run `python train.py --config configs/dfrnet_smoke.yaml` from
      `/workspace/DRNet` inside the container
- [x] 5.2 Confirm the checkpoint-loaded log line reports matched/skipped
      counts consistent with task 3.1 — 884/927 matched, 84 skipped
      (gtc_head/before_gtc, NRTR branch, correctly unused)
- [x] 5.3 Confirm the run completes without a crash and logged `loss`
      decreases from first to last step — confirmed after fixing the
      checkpoint remap bug (see notes); `loss_main` drops from 0.06 to
      ~0.0001-0.12 by epoch 6, eval `acc` stabilizes ~0.92-0.93
- [x] 5.4 Record the actual paddlepaddle-gpu version installed and any
      deviations from the design's assumptions in the change's notes — see
      below

## Notes / deviations from design

**paddlepaddle-gpu version**: `3.0.0`, installed from
`https://www.paddlepaddle.org.cn/packages/stable/cu126/` (no `cu128` wheel
existed; cu126 works against the container's CUDA 12.8 runtime with only a
compatibility warning, not an error).

**Bugs found and fixed while wiring up the pipeline** (all pre-existing —
this code had never been run end-to-end before this change):

1. `train.py::build_optimizer` — DFRNet built with `neck_cfg=`/`ctc_out_channels=`
   kwargs that don't exist on `DFRNet.__init__` (real params: `svtr_cfg`,
   `num_classes`). Fixed the call site.
2. `train.py::build_optimizer` — referenced `model.neck` and `model.ctc_head`,
   which don't exist (`model.ctc_encoder`, `model.ctc_fc`). Fixed.
3. `train.py` — `build_dataloader()` calls used PaddleOCR's old-style flat
   config + lowercase mode (`"train"`/`"eval"`) instead of the installed
   PaddleOCR's expected `{"Train": {...}}`/`{"Eval": {...}}` nesting and
   capitalized mode string, and passed `None` for the required `logger` arg
   (crashes inside `SimpleDataSet.__init__`, which calls `logger.info(...)`
   unconditionally). Restructured both dataloader configs and added a
   `Global` section (`max_text_length`, `character_dict_path`) since
   `CTCLabelEncode` needs it — created `configs/digits_dict.txt` (PaddleOCR's
   own checkout doesn't ship one; the original training machine's path
   doesn't exist here).
4. `dfrnet/corruption.py` — `paddle.randn_like` doesn't exist in this Paddle
   version; replaced with `paddle.randn(F.shape, dtype=F.dtype)`.
5. `dfrnet/ofr_module.py` — `paddle.nn.MultiHeadAttention` returns a single
   tensor (no attention-weights tuple, unlike `torch.nn.MultiheadAttention`);
   fixed the unpacking.
6. `dfrnet/loss.py` — Paddle's `F.ctc_loss` requires `label` as `int32` but
   `input_length`/`label_length` as `int64` (mixed dtypes); was casting
   everything to `int32`.
7. `train.py::evaluate` — passed `post_process`/`metric` args incompatible
   with the installed `CTCLabelDecode`/`RecMetric` (needed `label=` kwarg to
   get back a decoded-label tuple, and `labels` needed `.numpy()` before
   decoding).
8. **Root cause of "loss never decreases" in the first two full smoke runs**:
   `dfrnet/model.py::_load_pretrained` remapped `"head.ctc_encoder."` →
   `"ctc_encoder."`, but the actual checkpoint keys are
   `"head.ctc_encoder.encoder."` (extra `.encoder.` level — the SVTR encoder
   sits inside a wrapper the checkpoint's `head.ctc_encoder` object). This
   silently dropped ~50 SVTR-encoder tensors (conv1-4, svtr_block, conv1x1,
   norm) every time, leaving the actual feature encoder randomly initialized
   even though the loader reported "833/927 loaded" (the *backbone* loaded
   fine, masking the problem). Confirmed via a new `tools/baseline_check.py`
   script that runs raw inference with the checkpoint loaded: before the fix,
   0/20 predictions were even non-empty; after the fix, 20/20 exact matches
   at 0.9-1.0 confidence. Fixed the remap prefix in both
   `dfrnet/model.py` and `tools/inspect_checkpoint.py` (which was validating
   against the same wrong assumption and gave false confidence at task 3.1).

**`epochs` in `configs/dfrnet_smoke.yaml`**: raised from the originally
planned `2` to `15` — 2 epochs (~150 steps) wasn't enough to tell a real
downward loss trend from step-to-step noise; 15 epochs (~1150 steps) gave a
clear, confirmable trend.
