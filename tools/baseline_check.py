"""
Baseline sanity check: load PPOCRv5 checkpoint into DFRNet (eval mode, OFR
never runs at inference), run raw inference on a handful of val images, and
print predicted vs ground-truth digit strings.

Purpose: isolate whether the "loss doesn't decrease" smoke-train result is a
DFRNet/training-loop bug, or a data-pipeline (character dict / image
preprocessing) mismatch that would also break plain inference with the
pretrained checkpoint.

Usage:
    python tools/baseline_check.py --config configs/dfrnet_smoke.yaml --n 20
"""

import argparse
import os
import sys

import yaml
import numpy as np
import paddle
import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PADDLE_OCR_ROOT = os.path.abspath(os.path.join(_ROOT, "../PaddleOCR"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _PADDLE_OCR_ROOT not in sys.path:
    sys.path.insert(0, _PADDLE_OCR_ROOT)

from ppocr.postprocess import build_post_process

from dfrnet import DFRNet


def load_image(path, image_shape):
    """Mirrors ppocr.data.imaug.rec_img_aug.resize_norm_img exactly
    (aspect-ratio-preserving resize + right-padding), since that's what
    RecResizeImg actually does in the training pipeline."""
    imgC, imgH, imgW = image_shape
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[0], img.shape[1]
    ratio = w / float(h)
    if int(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(imgH * ratio) if int(imgH * ratio) > 0 else 1
    resized = cv2.resize(img, (resized_w, imgH))
    resized = resized.astype("float32").transpose(2, 0, 1) / 255.0
    resized -= 0.5
    resized /= 0.5
    padded = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dfrnet_smoke.yaml")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["Model"]
    eval_cfg = cfg["Eval"]

    model = DFRNet(
        backbone_cfg=model_cfg["backbone"],
        svtr_cfg=model_cfg["svtr"],
        num_classes=model_cfg.get("num_classes", 11),
        ofr_nhead=model_cfg.get("ofr_nhead", 4),
        ofr_depth=model_cfg.get("ofr_depth", 2),
        T=model_cfg.get("T", 1000),
        mask_ratio_max=model_cfg.get("mask_ratio_max", 0.5),
        span_len=model_cfg.get("span_len", 3),
        pretrained=model_cfg.get("pretrained"),
    )
    model.eval()

    post_process = build_post_process(
        {
            "name": "CTCLabelDecode",
            "character_dict_path": "configs/digits_dict.txt",
            "use_space_char": False,
        }
    )

    label_path = os.path.join(_ROOT, eval_cfg["label_file"])
    data_dir = os.path.join(_ROOT, eval_cfg["data_dir"])

    with open(label_path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()][: args.n]

    correct = 0
    for line in lines:
        rel_path, gt_text = line.split("\t", 1)
        img = load_image(os.path.join(data_dir, rel_path), eval_cfg["image_shape"])
        batch = paddle.to_tensor(np.expand_dims(img, 0), dtype="float32")
        with paddle.no_grad():
            logits = model(batch)
        (pred_text, conf) = post_process(logits)[0]
        ok = pred_text == gt_text
        correct += ok
        print(f"gt={gt_text!r:>10}  pred={pred_text!r:>10}  conf={conf:.3f}  {'OK' if ok else 'X'}")

    print(f"\n{correct}/{len(lines)} exact matches")


if __name__ == "__main__":
    main()
