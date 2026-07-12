#!/usr/bin/env python3
"""Run the fine-tuned PPOCRv5 recognizer over every DATA_COCO test annotation crop.

Run on the SLURM server (ssh cndt_thangcpd@slurm.uit.edu.vn) via run_ppocrv5_test.slurm,
using the paddle env: /datastore/cndt_thangcpd/linhtruong/workspace3/envs/paddle/bin/python

Model loading mirrors water_meter_amr/scripts/visualize/viz_wrong_ppocr_v5.py.

Input: a COCO-format instances_test.json (upload data/DATA_COCO/annotations/instances_test.json
       from local, or point --instances at data/DATA_COCO_v2/... after the working copy exists).
Output: predictions_ppocrv5_test.json, {"<file_name>" or "<file_name>#<annotation_id>": "<predicted_text>"}
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
PADDLE_DIR = f"{WS}/PaddleOCR"
sys.path.insert(0, PADDLE_DIR)

import paddle  # noqa: E402
from ppocr.modeling.architectures import build_model  # noqa: E402
from ppocr.postprocess import build_post_process  # noqa: E402

DEFAULT_IMAGES_DIR = f"{WS}/water_meter_reading_paper/data/raw/raw/test/images"
CFG_PATH = f"{PADDLE_DIR}/configs/rec/watermeter/v5_rec.yml"
CKPT_BASE = f"{WS}/water_meter_amr/outputs/ppocr_v5_paddle/best_accuracy"
IN_W, IN_H = 256, 64


def load_model():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    global_cfg = cfg.get("Global", {})
    global_cfg.setdefault("character_dict_path", f"{PADDLE_DIR}/ppocr/utils/dict/digits_dict.txt")
    global_cfg.setdefault("use_space_char", False)

    post_process = build_post_process(cfg["PostProcess"], global_cfg)
    char_num = len(getattr(post_process, "character", [])) or 12

    head_cfg = cfg["Architecture"].get("Head", {})
    if head_cfg.get("name") == "MultiHead":
        head_cfg["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
            "NRTRLabelDecode": char_num + 3,
        }
    else:
        head_cfg["out_channels"] = char_num
    cfg["Architecture"]["Head"] = head_cfg

    model = None
    for ckpt in [f"{CKPT_BASE}.pdparams", f"{CKPT_BASE}/model.pdparams"]:
        if os.path.exists(ckpt):
            model = build_model(cfg["Architecture"])
            model.set_state_dict(paddle.load(ckpt))
            model.eval()
            print(f"[infer_ppocrv5] loaded checkpoint: {ckpt}")
            break
    if model is None:
        raise SystemExit(f"ERROR: no checkpoint found at {CKPT_BASE}[.pdparams|/model.pdparams]")

    return model, post_process


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return pts in (top-left, top-right, bottom-right, bottom-left) order."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],    # top-left: min x+y
        pts[np.argmin(diff)], # top-right: min y-x
        pts[np.argmax(s)],    # bottom-right: max x+y
        pts[np.argmax(diff)], # bottom-left: max y-x
    ], dtype=np.float32)


def _obb_crop(img_bgr: np.ndarray, seg: list) -> np.ndarray:
    """Perspective-warp an oriented bounding box region to a flat rectangle."""
    pts = _order_points(np.array(seg, dtype=np.float32).reshape(4, 2))
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w == 0 or h == 0:
        return img_bgr
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w, h))
    # Ensure landscape orientation (width >= height) expected by recognizer
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def preprocess_crop(img_bgr: np.ndarray, ann: dict) -> "paddle.Tensor":
    seg = ann.get("segmentation", [[]])[0]
    if len(seg) == 8:
        crop = _obb_crop(img_bgr, seg)
    else:
        # Fallback: axis-aligned bbox
        x, y, w, h = ann["bbox"]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1 = min(img_bgr.shape[1], int(x + w))
        y1 = min(img_bgr.shape[0], int(y + h))
        crop = img_bgr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else img_bgr

    if crop.size == 0:
        crop = img_bgr
    crop = cv2.resize(crop, (IN_W, IN_H))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    arr = crop.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return paddle.to_tensor(arr[np.newaxis, :])


def predict_text(model, post_process, tensor):
    with paddle.no_grad():
        output = model(tensor)
    if isinstance(output, dict):
        head_out = output.get("CTCHead", list(output.values())[0])
        if isinstance(head_out, (list, tuple)):
            head_out = head_out[0]
    elif isinstance(output, (list, tuple)):
        head_out = output[0]
    else:
        head_out = output
    result = post_process(head_out.numpy())
    if result and isinstance(result[0], (list, tuple)):
        return str(result[0][0])
    if result:
        return str(result[0])
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="path to instances_test.json (uploaded from local)")
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--out", default=f"{WS}/water_meter_amr/outputs/predictions_ppocrv5_test.json")
    args = ap.parse_args()

    with open(args.instances, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_per_image: dict[int, int] = {}
    for ann in coco["annotations"]:
        anns_per_image[ann["image_id"]] = anns_per_image.get(ann["image_id"], 0) + 1

    model, post_process = load_model()

    predictions: dict[str, str] = {}
    total = len(coco["annotations"])
    for i, ann in enumerate(coco["annotations"]):
        img_meta = images_by_id[ann["image_id"]]
        file_name = img_meta["file_name"]
        img_path = os.path.join(args.images_dir, file_name)
        key = file_name if anns_per_image[ann["image_id"]] == 1 else f"{file_name}#{ann['id']}"
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(img_path)
            tensor = preprocess_crop(img_bgr, ann)
            predictions[key] = predict_text(model, post_process, tensor)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed on {key}: {e}")
            predictions[key] = "?"

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"[infer_ppocrv5] wrote {len(predictions)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
