#!/usr/bin/env python3
"""Evaluate PPOCRv3/v4/v5/v6 fine-tuned (Paddle) on the new test set.

Reads data/ocr_oriented/test_new/ crops + data/mmocr/textrecog_test_new.json.
Outputs: outputs/eval_new_testset/paddle_results.json
"""
import json, os, sys
import cv2
import numpy as np

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
AMR = f"{WS}/water_meter_amr"
DATA = f"{AMR}/data"
PADDLE_DIR = f"{WS}/PaddleOCR"
OUT_DIR = f"{AMR}/outputs/eval_new_testset"
MMOCR_JSON = f"{DATA}/mmocr/textrecog_test_new.json"

sys.path.insert(0, PADDLE_DIR)
import paddle
import yaml
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process

with open(MMOCR_JSON, encoding="utf-8") as f:
    data_list = json.load(f)["data_list"]
print(f"[eval_paddle] {len(data_list)} test samples")

IN_W, IN_H = 256, 64

def run_paddle_model(ckpt_base, cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    global_cfg = cfg.get("Global", {})
    global_cfg.setdefault("character_dict_path", f"{PADDLE_DIR}/ppocr/utils/dict/digits_dict.txt")
    global_cfg.setdefault("use_space_char", False)
    post_process = build_post_process(cfg["PostProcess"], global_cfg)
    char_num = len(getattr(post_process, "character", [])) or 12
    head_cfg = cfg["Architecture"].get("Head", {})
    if head_cfg.get("name") == "MultiHead":
        head_cfg["out_channels_list"] = {"CTCLabelDecode": char_num, "SARLabelDecode": char_num+2, "NRTRLabelDecode": char_num+3}
    else:
        head_cfg["out_channels"] = char_num
    cfg["Architecture"]["Head"] = head_cfg

    model = None
    for ckpt in [f"{ckpt_base}.pdparams", f"{ckpt_base}/model.pdparams"]:
        if os.path.exists(ckpt):
            model = build_model(cfg["Architecture"])
            model.set_state_dict(paddle.load(ckpt))
            model.eval()
            print(f"  loaded: {ckpt}")
            break
    if model is None:
        raise FileNotFoundError(f"No checkpoint at {ckpt_base}")

    correct = total = 0
    char_correct = char_total = 0
    for d in data_list:
        img_path = os.path.join(DATA, "ocr_oriented", d["img_path"])
        gt = d["instances"][0]["text"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img = cv2.resize(img_bgr, (IN_W, IN_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = ((img - 0.5) / 0.5).transpose(2, 0, 1)
        tensor = paddle.to_tensor(img[np.newaxis, :])
        with paddle.no_grad():
            output = model(tensor)
        if isinstance(output, dict):
            head_out = output.get("CTCHead", list(output.values())[0])
            if isinstance(head_out, (list, tuple)): head_out = head_out[0]
        elif isinstance(output, (list, tuple)):
            head_out = output[0]
        else:
            head_out = output
        res = post_process(head_out.numpy())
        pred = str(res[0][0]) if res and isinstance(res[0], (list, tuple)) else (str(res[0]) if res else "?")
        if pred == gt: correct += 1
        total += 1
        for g, p in zip(gt, pred):
            char_total += 1
            if g == p: char_correct += 1
        char_total += abs(len(gt) - len(pred))

    word_acc = correct / total if total else 0
    char_acc = char_correct / char_total if char_total else 0
    return {"word_acc": word_acc, "char_acc": char_acc, "correct": correct, "total": total}


models = {
    "PPOCRv3 (fine-tuned)": (f"{AMR}/outputs/ppocr_v3_paddle/best_accuracy", f"{PADDLE_DIR}/configs/rec/watermeter/v3_rec.yml"),
    "PPOCRv4 (fine-tuned)": (f"{AMR}/outputs/ppocr_v4_paddle/best_accuracy", f"{PADDLE_DIR}/configs/rec/watermeter/v4_rec.yml"),
    "PPOCRv5 (fine-tuned)": (f"{AMR}/outputs/ppocr_v5_paddle/best_accuracy", f"{PADDLE_DIR}/configs/rec/watermeter/v5_rec.yml"),
    "PPOCRv6 (fine-tuned)": (f"{AMR}/outputs/ppocr_v6_paddle/best_accuracy", f"{PADDLE_DIR}/configs/rec/watermeter/v6_rec.yml"),
}

results = {}
for name, (ckpt, cfg) in models.items():
    print(f"\n[eval_paddle] {name}")
    try:
        r = run_paddle_model(ckpt, cfg)
        results[name] = r
        print(f"  word_acc={r['word_acc']:.4f}  char_acc={r['char_acc']:.4f}  ({r['correct']}/{r['total']})")
    except Exception as e:
        print(f"  FAILED: {e}")
        results[name] = None

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/paddle_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[eval_paddle] saved -> {OUT_DIR}/paddle_results.json")
