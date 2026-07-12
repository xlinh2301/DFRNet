## 1. Working copy (DATA_COCO_v2) plumbing

- [x] 1.1 Add `ensure_v2_copy()` helper in `app.py`: on first mutating request, if `data/DATA_COCO_v2/` missing, recursively copy `data/DATA_COCO/` there
- [x] 1.2 Repoint all read/write paths (`TEST_ANN_PATH`, `TEST_IMAGES_DIR`, backup logic) at `data/DATA_COCO_v2/...`, falling back to reading `data/DATA_COCO/...` for display before any edit has happened (so the review grid works before first write)
- [x] 1.3 Update review/delete/edit/import endpoints to call `ensure_v2_copy()` before mutating

## 2. PPOCRv5 batch inference (test-eval-visualize)

- [x] 2.1 Write `tools/data_review_app/infer/infer_ppocrv5_test.py`: loads `ppocr_v5_paddle/best_accuracy.pdparams`, runs recognition over each test annotation's cropped bbox region, writes `predictions_ppocrv5_test.json` (`file_name` or `file_name#annotation_id` -> predicted text)
- [x] 2.2 Write `tools/data_review_app/infer/run_ppocrv5_test.slurm` SLURM submission script (paths per server workdir, GPU request)
- [x] 2.3 Do NOT submit the job — leave for explicit user confirmation

## 3. YOLO-OBB batch inference (obb-candidate-detection)

- [x] 3.1 Write `tools/data_review_app/infer/infer_obb_100k.py`: loads `detect_obb/train/weights/best.pt`, runs OBB detection over all images in `data/100k/wm_100k/wm_100k/` (server-side path), writes COCO-shaped `predictions_obb_100k.json`
- [x] 3.2 Write `tools/data_review_app/infer/run_obb_100k.slurm` SLURM submission script
- [x] 3.3 Do NOT submit the job — leave for explicit user confirmation
- [x] 3.4 Document the manual pull-back step (scp/copy `predictions_ppocrv5_test.json` and `predictions_obb_100k.json` to local `data/`) in a short README in `infer/`

## 4. Eval view (backend + frontend)

- [x] 4.1 `GET /api/eval/mismatches?page=&page_size=` — loads `data/predictions_ppocrv5_test.json`, compares against `_v2` GT text, returns only mismatched annotations (predicted + GT text)
- [x] 4.2 `static/eval.html` + `static/eval.js`: grid reusing review card layout, showing predicted vs GT text side by side, with existing delete/edit actions wired to the same review endpoints
- [x] 4.3 Add Eval to the nav bar across all three views

## 5. Supplement view box-source swap

- [x] 5.1 Update `/api/supplement/candidates` to iterate `data/predictions_obb_100k.json` images/annotations as primary source, join `text`/`yolo_conf` from `results_v2/e2e/label.json` by `file_name`
- [x] 5.2 Update `/api/supplement/import` to use the OBB-sourced `bbox`/`segmentation` and write into `data/DATA_COCO_v2/`
- [x] 5.3 Handle missing-text-match case (empty `text` field, candidate still shown)

## 6. Verification

- [x] 6.1 Confirm `data/DATA_COCO/` is untouched after a full session of edits/deletes/imports (checksum or diff against a pre-session copy) — verified: edit on annotation id 1 left `data/DATA_COCO/annotations/instances_test.json` text as `"02103"` while `_v2` copy shows `"TESTV2"`
- [ ] 6.2 BLOCKED — needs `data/predictions_ppocrv5_test.json` pulled back after the user submits `run_ppocrv5_test.slurm` (not run automatically, see infer/README.md)
- [ ] 6.3 BLOCKED — needs `data/predictions_obb_100k.json` pulled back after the user submits `run_obb_100k.slurm` (not run automatically, see infer/README.md)
