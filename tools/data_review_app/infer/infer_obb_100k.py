#!/usr/bin/env python3
"""Run the fine-tuned YOLOv8m-OBB detector over the full 100k image pool.

Run on the SLURM server (ssh cndt_thangcpd@slurm.uit.edu.vn) via run_obb_100k.slurm,
using the base anaconda env (has torch+cuda+ultralytics):
  /datastore/cndt_thangcpd/anaconda3/bin/python

Checkpoint: wmr_char_attention/outputs/detect_obb/train/weights/best.pt
Images:     DATA/data_raw/wm_100k/ (100000 files)

Output: predictions_obb_100k.json, COCO-shaped {"images": [...], "annotations": [...]}
mirroring data/100k/results_v2/e2e/label.json's image id/file_name (upload that file
as --images-manifest so ids line up for the app's file_name join), with oriented-box
bbox/segmentation from this OBB run and attributes = {angle, conf}.
"""
import argparse
import json
import os

from ultralytics import YOLO

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
DEFAULT_IMAGES_DIR = f"{WS}/DATA/data_raw/wm_100k"
DEFAULT_MODEL = f"{WS}/wmr_char_attention/outputs/detect_obb/train/weights/best.pt"
CATEGORY_ID = 1


def obb_to_bbox_and_seg(xyxyxyxy):
    xs = xyxyxyxy[0::2]
    ys = xyxyxyxy[1::2]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    bbox = [x0, y0, x1 - x0, y1 - y0]
    segmentation = [list(xyxyxyxy)]
    return bbox, segmentation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--images-manifest", required=True, help="uploaded results_v2/e2e/label.json for id/file_name alignment")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=f"{WS}/wmr_char_attention/outputs/predictions_obb_100k.json")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    with open(args.images_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    images_meta = manifest["images"]
    file_names = [im["file_name"] for im in images_meta]
    id_by_file_name = {im["file_name"]: im["id"] for im in images_meta}

    model = YOLO(args.model)

    out_images = []
    out_annotations = []
    next_ann_id = 1
    total = len(file_names)

    for start in range(0, total, args.batch):
        chunk = file_names[start:start + args.batch]
        paths = [os.path.join(args.images_dir, fn) for fn in chunk]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue

        results = model.predict(paths, conf=args.conf, verbose=False)

        for res in results:
            file_name = os.path.basename(res.path)
            image_id = id_by_file_name.get(file_name)
            if image_id is None:
                continue
            h, w = res.orig_shape
            out_images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})

            if res.obb is None or len(res.obb) == 0:
                continue
            xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
            confs = res.obb.conf.cpu().numpy()
            for poly, conf in zip(xyxyxyxy, confs):
                flat = poly.reshape(-1).tolist()
                bbox, seg = obb_to_bbox_and_seg(flat)
                out_annotations.append(
                    {
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": CATEGORY_ID,
                        "bbox": bbox,
                        "segmentation": seg,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "attributes": {"conf": float(conf)},
                    }
                )
                next_ann_id += 1

        if (start // args.batch + 1) % 20 == 0:
            print(f"  {min(start + args.batch, total)}/{total} images done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"images": out_images, "annotations": out_annotations}, f, ensure_ascii=False, indent=2)
    print(f"[infer_obb_100k] wrote {len(out_images)} images / {len(out_annotations)} annotations -> {args.out}")


if __name__ == "__main__":
    main()
