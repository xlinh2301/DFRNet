## Context

Ground truth files:
- `data/DATA_COCO/annotations/instances_test.json` — COCO schema, `categories: [{id:1, name:"meter_display"}]`, each annotation has `bbox [x,y,w,h]`, `segmentation`, `attributes.text`. 439 images / 440 annotations (near 1:1).
- `data/100k/results_v2/e2e/label.json` — same COCO schema plus `attributes.yolo_conf`, `attributes.angle`. 99,986 images / 91,813 annotations.
- Source images for import: `data/100k/wm_100k/wm_100k/<file_name>.jpg` (100k raw full images, matches `file_name` in `e2e/label.json`).
- Target images dir: `data/DATA_COCO/images/test/<file_name>`.

Single local reviewer, run on demand on Windows, no concurrent multi-user concern.

## Goals / Non-Goals

**Goals:**
- Fast visual triage of all 439 test images (bbox + text overlay) with delete / edit-text actions.
- Browse & filter 100k candidates by `yolo_conf`, multi-select, import into test set.
- Single source of truth = `instances_test.json`, edited safely (backup before first write per session).

**Non-Goals:**
- No `yolo_obb/test` sync, no deleting image files from `images/test` when an annotation is deleted (files may remain unreferenced on disk — acceptable, GT JSON is authoritative for training/eval).
- No multi-user auth/locking.
- No undo/redo history beyond the one session-start backup.
- No editing of `bbox`/`segmentation` geometry (only delete annotation or edit `text`).

## Decisions

- **Backend: FastAPI + Uvicorn, served locally.** Chosen (per user) over Streamlit for full control of the bbox-overlay canvas and multi-select supplement grid. Static frontend = plain HTML/CSS + vanilla JS with `<canvas>` for bbox drawing (no build step, keeps the tool self-contained).
- **Data access: load COCO JSON fully into memory on startup, mutate in-memory model, write back to disk on each mutating request.** Dataset sizes (439 anns for test, 91.8k for 100k index) are small enough (~tens of MB) to hold in memory; avoids a DB dependency. 100k JSON is read-only so it's loaded once and never rewritten.
- **Backup strategy:** on first mutating request per process lifetime, copy `instances_test.json` → `instances_test.<ISO8601-timestamp>.bak.json` in the same annotations folder, then proceed with writes. Subsequent writes in the same run overwrite `instances_test.json` directly (no per-edit backup, keeps it simple as scoped).
- **Delete cascade:** deleting an annotation removes it from `annotations[]`; if no other annotation in the JSON references that `image_id`, also remove the corresponding entry from `images[]`. Image file on disk is left as-is (Non-Goal).
- **ID allocation for imports:** new annotation `id` = `max(existing annotation ids) + 1`, new `image` `id` = `max(existing image ids) + 1`, computed at import time against the current in-memory `instances_test.json` state (so multiple imports in one session keep incrementing correctly).
- **Import conversion:** for a selected 100k image, copy its `image` entry (`file_name`, `width`, `height`) and its annotation(s) (`bbox`, `segmentation`, `category_id: 1`, `attributes.text`) into `instances_test.json` with reassigned ids per above; `yolo_conf`/`angle` attributes are dropped (not part of DATA_COCO schema) — only `text` is kept in `attributes`. Image file is copied (not moved) from `data/100k/wm_100k/wm_100k/` to `data/DATA_COCO/images/test/`; if a file with the same name already exists in the target, skip the copy (assume already imported) but still skip re-adding the JSON entry if `file_name` already present in `instances_test.json`.
- **Pagination:** review grid paginates client-requested (default 24/page) via query params against the in-memory image list; supplement grid same pattern, plus a `sort=yolo_conf&order=desc|asc` query and optional `min_conf`/`max_conf` filters.
- **Images served via a static file mount** (FastAPI `StaticFiles`) rooted at each dataset's image directory, referenced by relative URL from the frontend — no image duplication/base64 needed.

## Risks / Trade-offs

- [In-memory mutation + full-file rewrite on every edit] → fine at this scale (hundreds of test annotations); would not scale to millions, but out of scope here.
- [No per-edit undo] → mitigated by the one-time session backup; reviewer can restore from the `.bak.json` file manually if needed.
- [Concurrent tabs/processes could race on writes] → single local user assumption; not mitigated further (Non-Goal).
- [Orphaned image files after annotation delete] → acceptable per Non-Goals; disk cleanup can be a manual/future step.

## Open Questions

None outstanding — scope confirmed with user (FastAPI+HTML/JS, JSON-only sync, manual supplement selection).
