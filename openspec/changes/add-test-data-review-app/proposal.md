## Why

`data/DATA_COCO/annotations/instances_test.json` (439 images, 440 annotations) has some blurry/mislabeled images that inflate or distort text-recognition eval metrics. There is currently no tool to visually inspect each test annotation, remove bad ones, or fix wrong text labels. Separately, `data/100k` has ~91.8k additional bbox+text candidates (with `yolo_conf`) that could supplement the test set once reviewed, but there is no way to browse and hand-pick them either.

## What Changes

- Add a local web app (FastAPI backend + HTML/JS frontend) under `source_code/DFRNet/tools/data_review_app` for reviewing `instances_test.json`.
- **Review view**: paginated grid of test images with bbox overlay and `attributes.text` label shown per annotation.
- **Edit actions**: delete an annotation (cascades to removing the image entry if it was the image's last annotation) or edit its `attributes.text` value. Writes go only to `instances_test.json` — `yolo_obb/test` and image files on disk are untouched.
- **Supplement view**: separate browsable/sortable list of `data/100k/results_v2/e2e/label.json` candidates (sortable/filterable by `yolo_conf`), with manual multi-select to add chosen images to the test set — converts each into DATA_COCO's COCO annotation schema, appends to `instances_test.json`, and copies the source image file into `data/DATA_COCO/images/test`.
- Automatic timestamped backup of `instances_test.json` before the first write of each server session.
- No authentication — single local user, run on demand on Windows.

## Capabilities

### New Capabilities
- `test-data-review`: local web UI + API for viewing, editing, and deleting DATA_COCO test-set annotations, backed directly by `instances_test.json`.
- `test-data-supplement`: local web UI + API for browsing 100k candidates and importing selected ones into the DATA_COCO test set in COCO format.

### Modified Capabilities
(none — no existing specs cover this data yet)

## Impact

- New code only, under `source_code/DFRNet/tools/data_review_app/` (backend + static frontend).
- Reads/writes: `data/DATA_COCO/annotations/instances_test.json` (writes), `data/DATA_COCO/images/test/` (image copies added on import, no deletes), `data/100k/results_v2/e2e/label.json` (read-only), `data/100k/wm_100k/` images (read-only, source for copies).
- No changes to training/eval code, `yolo_obb`, or existing DFRNet modules.
