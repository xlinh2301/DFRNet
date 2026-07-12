"""Local review/eval/supplement web app for the DATA_COCO test set.

Run with: uvicorn app:app --reload --port 8008
(from this directory, after `pip install -r requirements.txt`)

Mutating actions (edit/delete/import) always operate on a lazily-created
working copy at data/DATA_COCO_v2/, never on the original data/DATA_COCO/.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data"

ORIGINAL_COCO_DIR = DATA_ROOT / "DATA_COCO"
V2_COCO_DIR = DATA_ROOT / "DATA_COCO_v2"

CANDIDATES_ANN_PATH = DATA_ROOT / "100k" / "results_v2" / "e2e" / "label.json"
CANDIDATES_IMAGES_DIR = DATA_ROOT / "100k" / "wm_100k" / "wm_100k"

PPOCR_PREDICTIONS_PATH = DATA_ROOT / "predictions_ppocrv5_test.json"
OBB_PREDICTIONS_PATH = DATA_ROOT / "predictions_obb_100k.json"

MEMBER_CATEGORY_ID = 1

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_write_lock = Lock()
_backup_done = False
_using_v2 = False


def _active_coco_dir() -> Path:
    return V2_COCO_DIR if _using_v2 else ORIGINAL_COCO_DIR


def _test_ann_path() -> Path:
    return _active_coco_dir() / "annotations" / "instances_test.json"


def _test_images_dir() -> Path:
    return _active_coco_dir() / "images" / "test"


with open(ORIGINAL_COCO_DIR / "annotations" / "instances_test.json", "r", encoding="utf-8") as f:
    test_coco = json.load(f)

with open(CANDIDATES_ANN_PATH, "r", encoding="utf-8") as f:
    candidates_coco = json.load(f)


def _images_by_id(coco: dict) -> dict[int, dict]:
    return {img["id"]: img for img in coco["images"]}


def _anns_by_image(coco: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        out.setdefault(ann["image_id"], []).append(ann)
    return out


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def _ensure_v2_copy() -> None:
    """Lazily create data/DATA_COCO_v2 as a full copy of data/DATA_COCO on first mutation."""
    global _using_v2
    if _using_v2:
        return
    if not V2_COCO_DIR.exists():
        shutil.copytree(ORIGINAL_COCO_DIR, V2_COCO_DIR)
    _using_v2 = True


def _backup_test_ann_once() -> None:
    global _backup_done
    if _backup_done:
        return
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    ann_path = _test_ann_path()
    backup_path = ann_path.with_name(f"instances_test.{ts}.bak.json")
    shutil.copy2(ann_path, backup_path)
    _backup_done = True


def _save_test_ann() -> None:
    with open(_test_ann_path(), "w", encoding="utf-8") as f:
        json.dump(test_coco, f, ensure_ascii=False, indent=2)


def _persist_test_mutation() -> None:
    _ensure_v2_copy()
    _backup_test_ann_once()
    _save_test_ann()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="DATA_COCO Test Review App")

app.mount("/images/100k", StaticFiles(directory=str(CANDIDATES_IMAGES_DIR)), name="candidate-images")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/images/test/{file_name}")
def get_test_image(file_name: str):
    path = _test_images_dir() / file_name
    if not path.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Review API
# ---------------------------------------------------------------------------
@app.get("/api/review/images")
def list_review_images(page: int = 1, page_size: int = 24):
    if page < 1 or page_size < 1:
        raise HTTPException(400, "page and page_size must be >= 1")

    images = sorted(test_coco["images"], key=lambda im: im["id"])
    anns_by_image = _anns_by_image(test_coco)

    start = (page - 1) * page_size
    end = start + page_size
    page_images = images[start:end]

    items = []
    for img in page_images:
        anns = anns_by_image.get(img["id"], [])
        items.append(
            {
                "image_id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "url": f"/images/test/{img['file_name']}",
                "annotations": [
                    {
                        "annotation_id": a["id"],
                        "bbox": a["bbox"],
                        "text": a.get("attributes", {}).get("text", ""),
                    }
                    for a in anns
                ],
            }
        )

    return {
        "page": page,
        "page_size": page_size,
        "total_images": len(images),
        "items": items,
    }


class EditTextRequest(BaseModel):
    text: str


@app.patch("/api/review/annotations/{annotation_id}")
def edit_annotation_text(annotation_id: int, body: EditTextRequest):
    with _write_lock:
        for ann in test_coco["annotations"]:
            if ann["id"] == annotation_id:
                ann.setdefault("attributes", {})["text"] = body.text
                _persist_test_mutation()
                return {"annotation_id": annotation_id, "text": body.text}
    raise HTTPException(404, f"annotation {annotation_id} not found")


@app.delete("/api/review/annotations/{annotation_id}")
def delete_annotation(annotation_id: int):
    with _write_lock:
        target = next((a for a in test_coco["annotations"] if a["id"] == annotation_id), None)
        if target is None:
            raise HTTPException(404, f"annotation {annotation_id} not found")

        image_id = target["image_id"]
        test_coco["annotations"] = [a for a in test_coco["annotations"] if a["id"] != annotation_id]

        remaining_for_image = [a for a in test_coco["annotations"] if a["image_id"] == image_id]
        image_removed = False
        if not remaining_for_image:
            test_coco["images"] = [im for im in test_coco["images"] if im["id"] != image_id]
            image_removed = True

        _persist_test_mutation()
        return {
            "deleted_annotation_id": annotation_id,
            "image_id": image_id,
            "image_removed": image_removed,
        }


# ---------------------------------------------------------------------------
# Eval API (PPOCRv5 mismatches)
# ---------------------------------------------------------------------------
def _load_ppocr_predictions() -> dict[str, str]:
    if not PPOCR_PREDICTIONS_PATH.exists():
        return {}
    with open(PPOCR_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/eval/mismatches")
def list_eval_mismatches(page: int = 1, page_size: int = 24):
    if page < 1 or page_size < 1:
        raise HTTPException(400, "page and page_size must be >= 1")

    predictions = _load_ppocr_predictions()
    images_by_id = _images_by_id(test_coco)
    anns_per_image = {img_id: len(anns) for img_id, anns in _anns_by_image(test_coco).items()}

    mismatches = []
    for ann in test_coco["annotations"]:
        img = images_by_id.get(ann["image_id"])
        if img is None:
            continue
        file_name = img["file_name"]
        key = file_name if anns_per_image.get(ann["image_id"], 1) == 1 else f"{file_name}#{ann['id']}"
        if key not in predictions:
            continue

        gt_text = ann.get("attributes", {}).get("text", "")
        pred_text = predictions[key]
        if pred_text == gt_text:
            continue

        mismatches.append(
            {
                "image_id": img["id"],
                "annotation_id": ann["id"],
                "file_name": file_name,
                "width": img["width"],
                "height": img["height"],
                "url": f"/images/test/{file_name}",
                "bbox": ann["bbox"],
                "gt_text": gt_text,
                "predicted_text": pred_text,
            }
        )

    start = (page - 1) * page_size
    end = start + page_size

    return {
        "page": page,
        "page_size": page_size,
        "total_mismatches": len(mismatches),
        "items": mismatches[start:end],
    }


# ---------------------------------------------------------------------------
# Supplement API (OBB-sourced candidates)
# ---------------------------------------------------------------------------
def _load_obb_predictions() -> dict:
    if not OBB_PREDICTIONS_PATH.exists():
        return {"images": [], "annotations": []}
    with open(OBB_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _candidate_text_lookup() -> dict[str, dict]:
    """file_name -> {text, yolo_conf} from the original 100k e2e label.json."""
    images_by_id = _images_by_id(candidates_coco)
    lookup: dict[str, dict] = {}
    for ann in candidates_coco["annotations"]:
        img = images_by_id.get(ann["image_id"])
        if img is None:
            continue
        lookup[img["file_name"]] = {
            "text": ann.get("attributes", {}).get("text", ""),
            "yolo_conf": ann.get("attributes", {}).get("yolo_conf"),
        }
    return lookup


@app.get("/api/supplement/candidates")
def list_candidates(
    page: int = 1,
    page_size: int = 24,
    sort: str = "yolo_conf",
    order: str = "desc",
    min_conf: Optional[float] = None,
    max_conf: Optional[float] = None,
):
    if page < 1 or page_size < 1:
        raise HTTPException(400, "page and page_size must be >= 1")
    if order not in ("asc", "desc"):
        raise HTTPException(400, "order must be 'asc' or 'desc'")

    obb = _load_obb_predictions()
    obb_images_by_id = _images_by_id(obb)
    obb_anns_by_image = _anns_by_image(obb)
    text_lookup = _candidate_text_lookup()

    rows = []
    for image_id, anns in obb_anns_by_image.items():
        img = obb_images_by_id.get(image_id)
        if img is None:
            continue
        joined = text_lookup.get(img["file_name"], {"text": "", "yolo_conf": None})
        conf = joined["yolo_conf"]
        if min_conf is not None and (conf is None or conf < min_conf):
            continue
        if max_conf is not None and (conf is None or conf > max_conf):
            continue
        for ann in anns:
            rows.append(
                {
                    "image_id": image_id,
                    "annotation_id": ann["id"],
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                    "url": f"/images/100k/{img['file_name']}",
                    "bbox": ann["bbox"],
                    "segmentation": ann.get("segmentation"),
                    "text": joined["text"],
                    "yolo_conf": conf,
                    "angle": ann.get("attributes", {}).get("angle"),
                }
            )

    if sort == "yolo_conf":
        rows.sort(key=lambda r: (r["yolo_conf"] is None, r["yolo_conf"]), reverse=(order == "desc"))

    start = (page - 1) * page_size
    end = start + page_size

    return {
        "page": page,
        "page_size": page_size,
        "total_candidates": len(rows),
        "items": rows[start:end],
    }


class ImportRequest(BaseModel):
    annotation_ids: list[int]


@app.post("/api/supplement/import")
def import_candidates(body: ImportRequest):
    obb = _load_obb_predictions()
    obb_images_by_id = _images_by_id(obb)
    ann_by_id = {a["id"]: a for a in obb["annotations"]}
    text_lookup = _candidate_text_lookup()

    with _write_lock:
        _ensure_v2_copy()

        existing_file_names = {im["file_name"] for im in test_coco["images"]}
        next_image_id = max((im["id"] for im in test_coco["images"]), default=0) + 1
        next_ann_id = max((a["id"] for a in test_coco["annotations"]), default=0) + 1

        imported, skipped = [], []

        for annotation_id in body.annotation_ids:
            ann = ann_by_id.get(annotation_id)
            if ann is None:
                skipped.append({"annotation_id": annotation_id, "reason": "not_found"})
                continue

            img = obb_images_by_id.get(ann["image_id"])
            if img is None:
                skipped.append({"annotation_id": annotation_id, "reason": "image_not_found"})
                continue

            if img["file_name"] in existing_file_names:
                skipped.append({"annotation_id": annotation_id, "reason": "already_imported"})
                continue

            src_path = CANDIDATES_IMAGES_DIR / img["file_name"]
            dst_path = _test_images_dir() / img["file_name"]
            if not dst_path.exists():
                if not src_path.exists():
                    skipped.append({"annotation_id": annotation_id, "reason": "source_image_missing"})
                    continue
                shutil.copy2(src_path, dst_path)

            new_image_id = next_image_id
            next_image_id += 1
            test_coco["images"].append(
                {
                    "id": new_image_id,
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                }
            )
            existing_file_names.add(img["file_name"])

            new_ann_id = next_ann_id
            next_ann_id += 1
            text = text_lookup.get(img["file_name"], {}).get("text", "")
            test_coco["annotations"].append(
                {
                    "id": new_ann_id,
                    "image_id": new_image_id,
                    "category_id": MEMBER_CATEGORY_ID,
                    "segmentation": ann.get("segmentation"),
                    "bbox": ann["bbox"],
                    "area": ann.get("area"),
                    "iscrowd": ann.get("iscrowd", 0),
                    "attributes": {"text": text},
                }
            )

            imported.append(
                {
                    "annotation_id": annotation_id,
                    "new_image_id": new_image_id,
                    "new_annotation_id": new_ann_id,
                    "file_name": img["file_name"],
                }
            )

        if imported:
            _backup_test_ann_once()
            _save_test_ann()

        return {"imported": imported, "skipped": skipped}
