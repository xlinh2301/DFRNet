#!/usr/bin/env python3
"""Run YOLOv8m-OBB detector over a supplement staging batch.

Run on SLURM server via run_obb_staging.slurm.
Python env: /datastore/cndt_thangcpd/anaconda3/bin/python (has ultralytics)

Input:  a flat directory of images (supplement_staging/<batch_id>/images/)
Output: predictions_obb_staging_<batch_id>.json — COCO-shaped {images, annotations}
        with oriented bbox/segmentation from OBB detection.
"""
import argparse
import json
import os

from ultralytics import YOLO

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
DEFAULT_MODEL = f"{WS}/wmr_char_attention/outputs/detect_obb/train/weights/best.pt"
CATEGORY_ID = 1


def obb_to_bbox_and_seg(xyxyxyxy):
    xs = xyxyxyxy[0::2]
    ys = xyxyxyxy[1::2]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0], [list(xyxyxyxy)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="path to staging batch images/ folder on server")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    file_names = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    total = len(file_names)
    print(f"[infer_obb_staging] {total} images in {args.images_dir}")

    model = YOLO(args.model)

    out_images, out_annotations = [], []
    next_image_id = next_ann_id = 1

    for start in range(0, total, args.batch):
        batch_files = file_names[start:start + args.batch]
        paths = [os.path.join(args.images_dir, f) for f in batch_files]
        results = model.predict(paths, conf=args.conf, verbose=False)

        for res, file_name in zip(results, batch_files):
            h, w = res.orig_shape
            image_id = next_image_id
            next_image_id += 1
            out_images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})

            if res.obb is not None and len(res.obb) > 0:
                for box in res.obb:
                    coords = box.xyxyxyxy[0].tolist()
                    coords_flat = [c for xy in coords for c in xy]
                    bbox, seg = obb_to_bbox_and_seg(coords_flat)
                    conf = float(box.conf[0])
                    out_annotations.append({
                        "id": next_ann_id,
                        "image_id": image_id,
                        "category_id": CATEGORY_ID,
                        "bbox": bbox,
                        "segmentation": seg,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "attributes": {"conf": conf},
                    })
                    next_ann_id += 1

        done = min(start + args.batch, total)
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total} done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"images": out_images, "annotations": out_annotations}, f, ensure_ascii=False, indent=2)
    print(f"[infer_obb_staging] wrote {len(out_images)} images / {len(out_annotations)} annotations -> {args.out}")


if __name__ == "__main__":
    main()
