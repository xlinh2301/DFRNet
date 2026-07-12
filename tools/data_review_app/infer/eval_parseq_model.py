#!/usr/bin/env python3
"""Evaluate PARSeq (baudm) on new test set crops.

Env: envs/parseq_baudm
Output: outputs/eval_new_testset/parseq_results.json
"""
import json, os, sys
import torch
import torchvision.transforms as T
from PIL import Image

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
AMR = f"{WS}/water_meter_amr"
DATA = f"{AMR}/data"
OUT_DIR = f"{AMR}/outputs/eval_new_testset"
MMOCR_JSON = f"{DATA}/mmocr/textrecog_test_new.json"

sys.path.insert(0, f"{WS}/parseq_src")
from strhub.models.utils import load_from_checkpoint

CKPT = f"{AMR}/outputs/parseq_baudm/checkpoints/epoch=64-step=1950-val_accuracy=92.4350-val_NED=97.3168.ckpt"

with open(MMOCR_JSON, encoding="utf-8") as f:
    data_list = json.load(f)["data_list"]
print(f"[eval_parseq] {len(data_list)} test samples")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_from_checkpoint(CKPT).eval().to(device)
transform = T.Compose([T.Resize((32, 128)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])

correct = total = char_correct = char_total = 0
for d in data_list:
    img_path = os.path.join(DATA, "ocr_oriented", d["img_path"])
    gt = d["instances"][0]["text"]
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = model.tokenizer.decode(logits.softmax(-1))[0]
    if pred == gt: correct += 1
    total += 1
    for g, p in zip(gt, pred):
        char_total += 1
        if g == p: char_correct += 1
    char_total += abs(len(gt) - len(pred))

word_acc = correct / total if total else 0
char_acc = char_correct / char_total if char_total else 0
print(f"[eval_parseq] word_acc={word_acc:.4f}  char_acc={char_acc:.4f}  ({correct}/{total})")

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/parseq_results.json", "w", encoding="utf-8") as f:
    json.dump({"PARSeq": {"word_acc": word_acc, "char_acc": char_acc, "correct": correct, "total": total}}, f, indent=2)
print(f"[eval_parseq] saved -> {OUT_DIR}/parseq_results.json")
