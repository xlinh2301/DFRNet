"""
DFRNet inference — identical output format to base PPOCRv5.

Usage:
    python infer.py --config configs/dfrnet.yaml --checkpoint outputs/dfrnet/epoch_100.pdparams --image path/to/img.jpg
"""

import argparse
import sys
import os

import yaml
import cv2
import numpy as np
import paddle

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PADDLE_OCR_ROOT = os.path.abspath(os.path.join(_ROOT, "../PaddleOCR"))
if _PADDLE_OCR_ROOT not in sys.path:
    sys.path.insert(0, _PADDLE_OCR_ROOT)

from ppocr.postprocess import build_post_process

from dfrnet import DFRNet


def preprocess(img_path: str, image_shape: list) -> paddle.Tensor:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_shape[2], image_shape[1]))
    img = img.astype("float32") / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    return paddle.to_tensor(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dfrnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["Model"]

    model = DFRNet(
        backbone_cfg=model_cfg["backbone"],
        neck_cfg=model_cfg["neck"],
        ctc_out_channels=model_cfg["ctc_out_channels"],
        ofr_nhead=model_cfg.get("ofr_nhead", 4),
        ofr_depth=model_cfg.get("ofr_depth", 2),
        T=model_cfg.get("T", 1000),
    )

    state = paddle.load(args.checkpoint)
    model.set_state_dict(state)
    model.eval()

    image_shape = cfg["Eval"]["image_shape"]
    img_tensor = preprocess(args.image, image_shape)

    with paddle.no_grad():
        logits = model(img_tensor)

    post_process = build_post_process({"name": "CTCLabelDecode"})
    result = post_process(logits)
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
