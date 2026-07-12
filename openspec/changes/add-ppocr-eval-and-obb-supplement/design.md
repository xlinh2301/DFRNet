## Context

Server: `ssh cndt_thangcpd@slurm.uit.edu.vn`, workdir `/datastore/cndt_thangcpd/linhtruong/workspace3`. It is a SLURM login node — heavy GPU work must be submitted as a job (`sbatch`), not run directly on the login shell (existing precedent: `DATA/locate_anything_inference/submit_coco.slurm`).

Checkpoints confirmed to exist and be viable:
- PPOCRv5 recognizer: `water_meter_amr/outputs/ppocr_v5_paddle/best_accuracy.pdparams` (Paddle format, classic PPOCR rec model, usable with PaddleOCR's `tools/infer_rec.py` / a small custom inference loop — PaddleOCR source available both locally and on the server).
- YOLO-OBB detector: `wmr_char_attention/outputs/detect_obb/train/weights/best.pt` (Ultralytics YOLOv8m-OBB, task: obb, epoch 100 metrics: precision 0.996, recall 0.986, mAP50 0.992, mAP50-95 0.965).

Existing web app (`add-test-data-review-app`) already implements review/edit/delete on `instances_test.json` and browse/import on 100k candidates, writing directly to `data/DATA_COCO/`.

## Goals / Non-Goals

**Goals:**
- Automatically surface test-set images where PPOCRv5 disagrees with GT text, so the reviewer doesn't have to eyeball all 439 images blind.
- Get better (oriented) candidate boxes for the 100k pool to support picking genuinely hard cases.
- Stop mutating the original `data/DATA_COCO/` — all edits go to a `_v2` working copy from now on.

**Non-Goals:**
- No automatic label correction — PPOCRv5 output is only used to flag mismatches; the human still decides delete vs. edit vs. keep.
- No re-running text recognition on the 100k pool — existing `results_v2/e2e/label.json` text is reused as-is.
- No retraining/fine-tuning of either model — both checkpoints are used purely for inference.
- No automatic SLURM job submission from the web app or from `apply` — inference scripts are prepared and handed to the user to submit/confirm, given shared-cluster resource cost.

## Decisions

- **Batch, not live inference.** Both inference jobs run once via SLURM, write JSON prediction files under the server's workdir, then get `scp`'d/copied to `data/predictions_ppocrv5_test.json` and `data/predictions_obb_100k.json` locally. The web app only ever reads these two JSON files — no server round-trip at request time. Rationale: predictable app latency, no coupling of the review UI to cluster availability, matches user's chosen "batch first, app just reads" option.
- **`instances_test.json` format for predictions**: keep it simple —
  - `predictions_ppocrv5_test.json`: `{"<file_name>": "<predicted_text>", ...}` (one prediction per test image; if an image has multiple annotations, PPOCRv5 runs per-annotation-crop and the key becomes `"<file_name>#<annotation_id>"`).
  - `predictions_obb_100k.json`: COCO-shaped `{"images": [...], "annotations": [...]}` mirroring `results_v2/e2e/label.json`'s image list (same `file_name`/`id`), but `annotations[].bbox`/`segmentation` come from the new OBB run, `attributes` only carries `angle`/`conf` from the OBB model (no `text` — text is joined at read time from `results_v2/e2e/label.json` by `file_name`).
- **v2 working copy, created lazily.** On first mutating request (review edit/delete OR supplement import) in a process lifetime, if `data/DATA_COCO_v2/` doesn't exist yet, the app does a full recursive copy of `data/DATA_COCO/` → `data/DATA_COCO_v2/` (images + annotations, all splits) before applying the write. All reads and writes after that point target `_v2`. This keeps the original golden copy at `data/DATA_COCO/` completely untouched, and reuses the same "session backup" logic from the prior change (still takes a timestamped backup of `instances_test.json` inside `_v2` before its first edit).
- **Eval view mismatch rule**: exact string inequality between `attributes.text` (GT, from `_v2/instances_test.json`) and the PPOCRv5 prediction for that annotation. Case-sensitive, no normalization — matches user's choice ("khác 1 ký tự cũng tính sai").
- **Supplement view box source swap**: candidate rows are built by iterating `predictions_obb_100k.json` images/annotations (this becomes the primary iteration source instead of `results_v2/e2e/label.json`), then looking up `text`/`yolo_conf` from the original `results_v2/e2e/label.json` by matching `file_name`. If no OBB detection exists for a given 100k image, it simply doesn't appear as a candidate (OBB detector is expected to have near-universal recall per its 0.986 recall metric, so this should be rare).
- **SLURM job scripts are deliverables, not auto-run.** `tools/data_review_app/infer/run_ppocrv5_test.slurm` and `tools/data_review_app/infer/run_obb_100k.slurm` (plus the Python entrypoints they call) are written and left for the user to review and submit (`sbatch ...`) themselves, or to explicitly ask the assistant to submit — per the safety rule against unprompted resource-intensive/shared-system actions.

## Risks / Trade-offs

- [PPOCRv5 inference environment on the server may need setup (paddlepaddle-gpu, PaddleOCR checkout) — prior proposal `smoke-train-dfrnet-ppocrv5` noted the training container lacks this; the inference script must be validated against whichever env (`envs/paddle` or similar) actually has classic PaddleOCR rec installed] → mitigated by checking env before submitting the job; script assumes `PaddleOCR` repo + a paddle env exist per the earlier repo scan, but this needs a quick `--help`/dry run check before a real submission.
- [YOLO-OBB run over 100k images is a non-trivial GPU job] → run as a SLURM batch job (not on login node), sized/timed appropriately; user must approve submission.
- [`DATA_COCO_v2` full copy duplicates ~hundreds of MB of images] → acceptable one-time cost for keeping the original golden set immutable; only triggered lazily on first edit, not on every app start.
- [Two prediction JSON files can go stale if `_v2` test set changes after import (new images added via supplement have no PPOCRv5 prediction yet)] → acceptable: Eval view only evaluates annotations that have a matching key in `predictions_ppocrv5_test.json`; newly imported annotations simply won't show up in Eval until a future inference re-run — out of scope to auto-refresh.

## Open Questions

None outstanding for this design — SLURM submission timing/approval is a process step, not a design ambiguity.
