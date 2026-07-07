"""
DFRNet ablation evaluation: run a given checkpoint against the real test set
(data/set3/test, data/gt/rec/rec_gt_test.txt), optionally corrupting the
encoder's output features at test time to simulate occlusion — bypassing the
OFR module either way (OFR is never used at inference in the real deployment
path; this measures the *encoder's own* robustness).

Usage:
    # zero-shot PPOCRv5 checkpoint (needs key remapping via DFRNet.pretrained=)
    python tools/eval_ablation.py --checkpoint data/checkpoints/release/ppocr_v5_paddle/best_accuracy.pdparams \
        --checkpoint-format zero-shot --occlusion none

    # a checkpoint saved by train.py (paddle.save(model.state_dict())) —
    # already in DFRNet's own key format, no remapping
    python tools/eval_ablation.py --checkpoint outputs/dfrnet_baseline/epoch_15.pdparams \
        --checkpoint-format trained --occlusion light
"""

import argparse
import os
import sys

import yaml
import numpy as np
import paddle

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TOOLS_DIR)
_PADDLE_OCR_ROOT = os.path.abspath(os.path.join(_ROOT, "../PaddleOCR"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
if _PADDLE_OCR_ROOT not in sys.path:
    sys.path.insert(0, _PADDLE_OCR_ROOT)

from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric

from dfrnet import DFRNet
from baseline_check import load_image

# t values chosen so mask_ratio_max=0.5 scales down to ~20%/~50% of tokens
# masked, matching tools/ablation_contribution.py's occ-light/occ-heavy
OCCLUSION_T = {
    "none": None,
    "light": 400,   # (400/1000) * 0.5 ~= 0.20
    "heavy": 1000,  # (1000/1000) * 0.5 = 0.50
}


def build_model(cfg, checkpoint, checkpoint_format):
    model_cfg = cfg["Model"]
    pretrained = checkpoint if checkpoint_format == "zero-shot" else None
    model = DFRNet(
        backbone_cfg=model_cfg["backbone"],
        svtr_cfg=model_cfg["svtr"],
        num_classes=model_cfg.get("num_classes", 11),
        ofr_nhead=model_cfg.get("ofr_nhead", 4),
        ofr_depth=model_cfg.get("ofr_depth", 2),
        T=model_cfg.get("T", 1000),
        mask_ratio_max=model_cfg.get("mask_ratio_max", 0.5),
        span_len=model_cfg.get("span_len", 3),
        pretrained=pretrained,
    )
    if checkpoint_format == "trained":
        state = paddle.load(checkpoint)
        model.set_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dfrnet_smoke.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-format", choices=["zero-shot", "trained"], required=True
    )
    parser.add_argument("--occlusion", choices=["none", "light", "heavy"], default="none")
    parser.add_argument(
        "--test-label-file", default="data/gt/rec/rec_gt_test.txt"
    )
    parser.add_argument("--test-data-dir", default="data")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    image_shape = cfg["Eval"]["image_shape"]
    model = build_model(cfg, args.checkpoint, args.checkpoint_format)

    occ_t = OCCLUSION_T[args.occlusion]
    corruption = model.corruption if occ_t is not None else None

    post_process = build_post_process(
        {
            "name": "CTCLabelDecode",
            "character_dict_path": "configs/digits_dict.txt",
            "use_space_char": False,
        }
    )
    metric = build_metric({"name": "RecMetric", "main_indicator": "acc"})

    label_path = os.path.join(_ROOT, args.test_label_file)
    data_dir = os.path.join(_ROOT, args.test_data_dir)

    with open(label_path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    preds, targets = [], []
    for line in lines:
        rel_path, gt_text = line.split("\t", 1)
        img = load_image(os.path.join(data_dir, rel_path), image_shape)
        batch = paddle.to_tensor(np.expand_dims(img, 0), dtype="float32")

        with paddle.no_grad():
            if corruption is None:
                logits = model(batch)
            else:
                F_clean = model.encode(batch)
                B = F_clean.shape[0]
                t = paddle.full([B], occ_t, dtype="int64")
                F_occ, _ = corruption(F_clean, t)
                logits = paddle.nn.functional.softmax(model.ctc_fc(F_occ), axis=2)

        (pred_text, conf) = post_process(logits)[0]
        preds.append((pred_text, conf))
        targets.append((gt_text, 1.0))

    metric.reset()
    metric((preds, targets))
    result = metric.get_metric()

    print(
        f"checkpoint={args.checkpoint} format={args.checkpoint_format} "
        f"occlusion={args.occlusion} n={len(lines)}"
    )
    print(f"  acc={result['acc']:.4f}  norm_edit_dis={result['norm_edit_dis']:.4f}")


if __name__ == "__main__":
    main()
