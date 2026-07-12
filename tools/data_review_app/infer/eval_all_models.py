#!/usr/bin/env python3
"""Evaluate all fine-tuned SOTA models on the new (expanded) test set.

Steps performed:
  1. Load DATA_COCO_v2 annotations (574 items).
  2. Build OBB perspective-warp crops for each annotation → save to
     data/ocr_oriented/test_new/  (skips existing to avoid re-cropping).
  3. Rebuild data/mmocr/textrecog_test_new.json for mmocr/parseq/satrn format.
  4. Run each model and collect word accuracy.
  5. Print a comparison table and save to outputs/eval_new_testset/results.json.

Run on SLURM server via run_eval_all.slurm  (uses anaconda3 env).
"""
import json
import os
import sys
import importlib
import cv2
import numpy as np
from pathlib import Path

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
AMR = f"{WS}/water_meter_amr"
DATA = f"{AMR}/data"
PADDLE_DIR = f"{WS}/PaddleOCR"
PARSEQ_DIR = f"{WS}/parseq_src"
MMOCR_DIR = f"{WS}/mmocr_src"

TEST_IMAGES_DIR = f"{WS}/water_meter_reading_paper/data/raw/raw/test/images"
COCO_V2_ANN = f"{WS}/water_meter_amr/outputs/instances_test_manifest.json"

OUT_CROPS_DIR = f"{DATA}/ocr_oriented/test_new"
OUT_MMOCR_JSON = f"{DATA}/mmocr/textrecog_test_new.json"
OUT_DIR = f"{AMR}/outputs/eval_new_testset"

os.makedirs(OUT_CROPS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1 — load COCO v2 annotations
# ---------------------------------------------------------------------------
print("[eval] loading COCO v2 annotations...")
with open(COCO_V2_ANN, encoding="utf-8") as f:
    coco = json.load(f)

img_map = {im["id"]: im["file_name"] for im in coco["images"]}
anns_per_img = {}
for a in coco["annotations"]:
    anns_per_img[a["image_id"]] = anns_per_img.get(a["image_id"], 0) + 1

# ---------------------------------------------------------------------------
# Step 2 — build OBB perspective-warp crops
# ---------------------------------------------------------------------------
def _order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)

def obb_crop(img_bgr, seg, target_w=128, target_h=32):
    pts = _order_points(np.array(seg, dtype=np.float32).reshape(4, 2))
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if w == 0 or h == 0:
        return cv2.resize(img_bgr, (target_w, target_h))
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w, h))
    if h > w:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(warped, (target_w, target_h))

print("[eval] building crops...")
data_list = []
for idx, ann in enumerate(coco["annotations"]):
    fn = img_map[ann["image_id"]]
    gt = ann.get("attributes", {}).get("text", "")
    crop_name = f"test_{idx:06d}.jpg"
    crop_path = os.path.join(OUT_CROPS_DIR, crop_name)

    if not os.path.exists(crop_path):
        img_path = os.path.join(TEST_IMAGES_DIR, fn)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [warn] missing image: {fn}")
            continue
        seg = ann.get("segmentation", [[]])[0]
        if len(seg) == 8:
            crop = obb_crop(img_bgr, seg)
        else:
            x, y, w, h = ann["bbox"]
            crop = img_bgr[max(0,int(y)):int(y+h), max(0,int(x)):int(x+w)]
            crop = cv2.resize(crop, (128, 32)) if crop.size > 0 else np.zeros((32,128,3), np.uint8)
        cv2.imwrite(crop_path, crop)

    data_list.append({"sample_idx": idx, "img_path": f"test_new/{crop_name}",
                       "instances": [{"text": gt}]})

with open(OUT_MMOCR_JSON, "w", encoding="utf-8") as f:
    json.dump({"metainfo": {"dataset_type": "TextRecogDataset"}, "data_list": data_list}, f, indent=2)
print(f"[eval] {len(data_list)} crops ready. mmocr JSON: {OUT_MMOCR_JSON}")

# ---------------------------------------------------------------------------
# Helper: load GT
# ---------------------------------------------------------------------------
gt_map = {d["img_path"]: d["instances"][0]["text"] for d in data_list}

# ---------------------------------------------------------------------------
# Step 3 — run each model
# ---------------------------------------------------------------------------
results = {}

# ── PPOCRv5 (fine-tuned, Paddle) ─────────────────────────────────────────
def run_ppocrv5(ckpt_base, label, cfg_path=None):
    print(f"\n[eval] {label}...")
    sys.path.insert(0, PADDLE_DIR)
    import paddle
    import yaml
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    cfg_path = cfg_path or f"{PADDLE_DIR}/configs/rec/watermeter/v5_rec.yml"
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
            break
    if model is None:
        print(f"  [warn] checkpoint not found: {ckpt_base}")
        return None

    correct = total = 0
    IN_W, IN_H = 256, 64
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
        if pred == gt:
            correct += 1
        total += 1

    acc = correct / total if total else 0
    print(f"  {label}: {acc:.4f} ({correct}/{total})")
    return acc


results["PPOCRv3 (fine-tuned)"] = run_ppocrv5(
    f"{AMR}/outputs/ppocr_v3_paddle/best_accuracy",
    "PPOCRv3 (fine-tuned)",
    cfg_path=f"{PADDLE_DIR}/configs/rec/watermeter/v3_rec.yml"
)

results["PPOCRv4 (fine-tuned)"] = run_ppocrv5(
    f"{AMR}/outputs/ppocr_v4_paddle/best_accuracy",
    "PPOCRv4 (fine-tuned)",
    cfg_path=f"{PADDLE_DIR}/configs/rec/watermeter/v4_rec.yml"
)

results["PPOCRv5 (fine-tuned)"] = run_ppocrv5(
    f"{AMR}/outputs/ppocr_v5_paddle/best_accuracy",
    "PPOCRv5 (fine-tuned)"
)

results["PPOCRv6 (fine-tuned)"] = run_ppocrv5(
    f"{AMR}/outputs/ppocr_v6_paddle/best_accuracy",
    "PPOCRv6 (fine-tuned)",
    cfg_path=f"{PADDLE_DIR}/configs/rec/watermeter/v6_rec.yml"
)

# ── PARSeq ────────────────────────────────────────────────────────────────
def run_parseq():
    print("\n[eval] PARSeq...")
    try:
        sys.path.insert(0, PARSEQ_DIR)
        import torch
        import torchvision.transforms as T
        from strhub.models.utils import load_from_checkpoint

        ckpt = f"{AMR}/outputs/parseq_baudm/checkpoints/epoch=64-step=1950-val_accuracy=92.4350-val_NED=97.3168.ckpt"
        model = load_from_checkpoint(ckpt).eval().cuda()
        IMG_W, IMG_H = 128, 32
        transform = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        from PIL import Image as PILImage

        correct = total = 0
        for d in data_list:
            img_path = os.path.join(DATA, "ocr_oriented", d["img_path"])
            gt = d["instances"][0]["text"]
            img = PILImage.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = model(tensor)
            pred = model.tokenizer.decode(logits.softmax(-1))[0]
            if pred == gt:
                correct += 1
            total += 1
        acc = correct / total if total else 0
        print(f"  PARSeq: {acc:.4f} ({correct}/{total})")
        return acc
    except Exception as e:
        print(f"  [warn] PARSeq failed: {e}")
        return None

results["PARSeq"] = run_parseq()

# ── SATRN (mmocr) ────────────────────────────────────────────────────────
def run_mmocr_model(cfg_path, ckpt_path, label):
    print(f"\n[eval] {label}...")
    try:
        from mmocr.apis import TextRecognizer
        recognizer = TextRecognizer(cfg_path, ckpt_path, device="cuda:0")
        correct = total = 0
        for d in data_list:
            img_path = os.path.join(DATA, "ocr_oriented", d["img_path"])
            gt = d["instances"][0]["text"]
            res = recognizer(img_path)
            pred = res[0]["text"] if res else "?"
            if pred == gt:
                correct += 1
            total += 1
        acc = correct / total if total else 0
        print(f"  {label}: {acc:.4f} ({correct}/{total})")
        return acc
    except Exception as e:
        print(f"  [warn] {label} failed: {e}")
        return None

SATRN_CFG = f"{MMOCR_DIR}/configs/textrecog/satrn/satrn_shallow_5e_st_mj.py"
SATRN_CKPT = f"{AMR}/outputs/eval_top5/satrn/epoch_5.pth"
results["SATRN"] = run_mmocr_model(SATRN_CFG, SATRN_CKPT, "SATRN")

ABINET_CFG = f"{MMOCR_DIR}/configs/textrecog/abinet/abinet_20e_st-an_mj.py"
ABINET_CKPT = f"{AMR}/outputs/eval_top5/abinet/epoch_20.pth"
results["ABINet"] = run_mmocr_model(ABINET_CFG, ABINET_CKPT, "ABINet")

# ---------------------------------------------------------------------------
# Step 4 — print comparison table & save
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print(f"  Evaluation on New Test Set ({len(data_list)} images)")
print("=" * 55)
print(f"  {'Model':<30} {'Word Acc':>10}")
print("-" * 55)
sorted_results = sorted([(k, v) for k, v in results.items() if v is not None], key=lambda x: x[1], reverse=True)
for model_name, acc in sorted_results:
    print(f"  {model_name:<30} {acc*100:>9.2f}%")
failed = [k for k, v in results.items() if v is None]
if failed:
    print(f"\n  Failed: {', '.join(failed)}")
print("=" * 55)

with open(f"{OUT_DIR}/results.json", "w", encoding="utf-8") as f:
    json.dump({
        "test_set_size": len(data_list),
        "results": {k: round(v, 6) if v else None for k, v in results.items()},
        "sorted": [(k, round(v, 4)) for k, v in sorted_results],
    }, f, indent=2)
print(f"\n[eval] results saved to {OUT_DIR}/results.json")
