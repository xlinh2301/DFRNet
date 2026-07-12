## 1. Backend scaffold

- [x] 1.1 Create `source_code/DFRNet/tools/data_review_app/` with `app.py` (FastAPI app), `requirements.txt` (fastapi, uvicorn), and `static/` folder
- [x] 1.2 Load `data/DATA_COCO/annotations/instances_test.json` into memory on startup (dict keyed by image id / annotation list)
- [x] 1.3 Load `data/100k/results_v2/e2e/label.json` into memory on startup (read-only)
- [x] 1.4 Mount `data/DATA_COCO/images/test` and `data/100k/wm_100k/wm_100k` as static file directories for image serving

## 2. Review API (test-data-review)

- [x] 2.1 `GET /api/review/images?page=&page_size=` — return paginated images with their annotation(s) (bbox, text, image URL)
- [x] 2.2 `DELETE /api/review/annotations/{annotation_id}` — remove annotation; cascade-remove image entry if it was the last one; trigger session backup + write-through
- [x] 2.3 `PATCH /api/review/annotations/{annotation_id}` — update `attributes.text`; trigger session backup + write-through
- [x] 2.4 Implement one-time-per-session backup helper (copy `instances_test.json` to timestamped `.bak.json` before first write)

## 3. Supplement API (test-data-supplement)

- [x] 3.1 `GET /api/supplement/candidates?page=&page_size=&sort=yolo_conf&order=&min_conf=&max_conf=` — paginated/sorted/filtered 100k candidates
- [x] 3.2 `POST /api/supplement/import` — accept list of candidate image ids; for each: compute new image/annotation ids, append converted entries (drop `yolo_conf`/`angle`, keep `text`) to in-memory `instances_test.json`, copy image file to `data/DATA_COCO/images/test/`, skip if `file_name` already present or file already exists
- [x] 3.3 Reuse backup + write-through helper from review API for import writes

## 4. Frontend

- [x] 4.1 `static/index.html` + `static/review.js`: paginated grid, canvas/SVG bbox overlay per image, text label display, delete button, edit-text inline form
- [x] 4.2 `static/supplement.html` + `static/supplement.js`: paginated/sortable candidate grid with confidence filter inputs, multi-select checkboxes, "Import selected" action
- [x] 4.3 Basic nav between review and supplement views

## 5. Verification

- [x] 5.1 Run app locally, confirm review grid renders bbox+text correctly for a sample of test images
- [x] 5.2 Delete an annotation (image with 1 annotation) and confirm both annotation and image entry removed from `instances_test.json`, backup file created
- [x] 5.3 Edit a text label and confirm only `attributes.text` changes on disk
- [x] 5.4 Filter/sort supplement candidates by `yolo_conf` and confirm results match
- [x] 5.5 Import 2-3 candidates, confirm new entries appended with correct incremented ids, `attributes` contains only `text`, and image files copied into `images/test`
- [x] 5.6 Re-run import on an already-imported file_name, confirm no duplicate entry/file
