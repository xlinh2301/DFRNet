## Why

The review app built in `add-test-data-review-app` lets a human manually eyeball all 439 test images and the 91.8k `100k` candidates, but manual triage alone misses systematic errors and doesn't prioritize review effort. We now have two trained models available on `ssh cndt_thangcpd@slurm.uit.edu.vn`: a fine-tuned PPOCRv5 recognizer (`water_meter_amr/outputs/ppocr_v5_paddle/best_accuracy.pdparams`) and a fine-tuned YOLOv8m-OBB detector (`wmr_char_attention/outputs/detect_obb/train/weights/best.pt`, mAP50 0.992). Running them lets us (1) surface test images where the model disagrees with the ground-truth text so blurry/mislabeled cases are found automatically instead of by brute-force browsing, and (2) get higher-quality oriented bboxes for the 100k pool so the reviewer can pick genuinely hard cases to diversify the test set.

## What Changes

- Add batch inference scripts (run manually via SSH on `cndt_thangcpd@slurm.uit.edu.vn`, submitted as SLURM jobs) that produce local prediction JSON files:
  - PPOCRv5 recognition over the 439 test images → `predictions_ppocrv5_test.json`.
  - YOLO-OBB detection over the 100k images → `predictions_obb_100k.json`.
- Add an **Eval view** to the web app: loads the test set + `predictions_ppocrv5_test.json`, shows only annotations where predicted text != ground-truth text (exact match), with the existing delete/edit actions.
- **BREAKING (data location)**: all mutating actions (delete/edit/import) now target a working copy `data/DATA_COCO_v2/` instead of `data/DATA_COCO/` directly. The app copies `DATA_COCO` → `DATA_COCO_v2` once (if not already present) on first mutating request; `data/DATA_COCO/` is never written to again by the app.
- Supplement view's candidate bboxes now come from `predictions_obb_100k.json` (oriented boxes) instead of the original `results_v2/e2e/label.json` axis-aligned boxes; recognized `text` is still taken from `results_v2/e2e/label.json` (matched by `file_name`), no OCR re-run needed on the 100k side.
- Import target becomes `data/DATA_COCO_v2/` (images + `instances_test.json` under it), consistent with the review view.

## Capabilities

### New Capabilities
- `test-eval-visualize`: batch PPOCRv5 inference over the test set, plus a web view that surfaces and lets the reviewer act on prediction-vs-GT mismatches.
- `obb-candidate-detection`: batch YOLO-OBB inference over the 100k pool, producing improved oriented-box candidates for the supplement flow.

### Modified Capabilities
- `test-data-review`: mutating actions (delete/edit) now operate on `data/DATA_COCO_v2/instances_test.json`, with a one-time copy-on-first-write from `data/DATA_COCO/`, instead of writing directly to the original test set.
- `test-data-supplement`: candidate boxes are sourced from `predictions_obb_100k.json` instead of `results_v2/e2e/label.json`; import target is `data/DATA_COCO_v2/`.

## Impact

- New scripts under `source_code/DFRNet/tools/data_review_app/infer/` (or similar) to run on the SLURM server and scripts to pull results back locally.
- `source_code/DFRNet/tools/data_review_app/app.py` and frontend updated for the v2 working copy, the new eval view, and OBB-sourced candidates.
- New local data artifacts: `data/DATA_COCO_v2/` (working copy, created by the app), `data/predictions_ppocrv5_test.json`, `data/predictions_obb_100k.json` (pulled from server).
- Original `data/DATA_COCO/` becomes read-only from the app's perspective going forward.
- Running the SLURM inference jobs consumes shared GPU resources — requires explicit user confirmation before submission, not run automatically as part of `apply`.
