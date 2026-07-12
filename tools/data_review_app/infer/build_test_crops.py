#!/usr/bin/env python3
"""Build OBB perspective-warp crops for all DATA_COCO_v2 annotations.

Creates: data/ocr_oriented/test_new/<crop>.jpg
Updates: data/mmocr/textrecog_test_new.json
Skips crops that already exist (safe to re-run).
"""
import json, os
import cv2
import numpy as np

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
AMR = f"{WS}/water_meter_amr"
DATA = f"{AMR}/data"
COCO_V2_ANN = f"{AMR}/outputs/instances_test_manifest.json"
TEST_IMAGES_DIR = f"{WS}/water_meter_reading_paper/data/raw/raw/test/images"

OUT_CROPS_DIR = f"{DATA}/ocr_oriented/test_new"
OUT_MMOCR_JSON = f"{DATA}/mmocr/textrecog_test_new.json"
os.makedirs(OUT_CROPS_DIR, exist_ok=True)

def _order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)

def obb_crop(img_bgr, seg):
    pts = _order_points(np.array(seg, dtype=np.float32).reshape(4, 2))
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w == 0 or h == 0:
        return cv2.resize(img_bgr, (128, 32))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w, h))
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(warped, (128, 32))

with open(COCO_V2_ANN, encoding="utf-8") as f:
    coco = json.load(f)

img_map = {im["id"]: im["file_name"] for im in coco["images"]}
data_list = []
missing = 0

for idx, ann in enumerate(coco["annotations"]):
    fn = img_map[ann["image_id"]]
    gt = ann.get("attributes", {}).get("text", "")
    crop_name = f"test_{idx:06d}.jpg"
    crop_path = os.path.join(OUT_CROPS_DIR, crop_name)

    if not os.path.exists(crop_path):
        img_path = os.path.join(TEST_IMAGES_DIR, fn)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [warn] missing: {fn}")
            missing += 1
            continue
        seg = ann.get("segmentation", [[]])[0]
        if len(seg) == 8:
            crop = obb_crop(img_bgr, seg)
        else:
            x, y, w, h = ann["bbox"]
            crop = img_bgr[max(0,int(y)):int(y+h), max(0,int(x)):int(x+w)]
            crop = cv2.resize(crop, (128, 32)) if crop.size > 0 else np.zeros((32,128,3), np.uint8)
        cv2.imwrite(crop_path, crop)

    data_list.append({
        "sample_idx": idx,
        "img_path": f"test_new/{crop_name}",
        "instances": [{"text": gt}]
    })

with open(OUT_MMOCR_JSON, "w", encoding="utf-8") as f:
    json.dump({"metainfo": {"dataset_type": "TextRecogDataset"}, "data_list": data_list}, f, indent=2)

print(f"[build_crops] {len(data_list)} crops ready, {missing} missing images")
print(f"[build_crops] mmocr JSON -> {OUT_MMOCR_JSON}")
