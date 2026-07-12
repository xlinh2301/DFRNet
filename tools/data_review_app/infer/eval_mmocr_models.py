#!/usr/bin/env python3
"""Evaluate SATRN and ABINet (mmocr) on new test set crops.

Creates patched config files pointing at test_new/ crops + textrecog_test_new.json.
Env: envs/mmocr
Output: outputs/eval_new_testset/mmocr_results.json
"""
import json, os, sys, subprocess, tempfile, re

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
AMR = f"{WS}/water_meter_amr"
DATA = f"{AMR}/data"
OUT_DIR = f"{AMR}/outputs/eval_new_testset"
MMOCR_SRC = f"{WS}/mmocr_src"
PYTHON = f"{WS}/envs/mmocr/bin/python"

MMOCR_JSON_NEW = f"{DATA}/mmocr/textrecog_test_new.json"
CROPS_DIR = f"{DATA}/ocr_oriented"   # test_new/ is a subdir here

os.makedirs(OUT_DIR, exist_ok=True)


def patch_config(orig_cfg_path: str, suffix: str) -> str:
    """Copy config, replace _an and _dr to point at new test data, write to /tmp."""
    with open(orig_cfg_path) as f:
        content = f.read()
    # Override the test dataloader to use textrecog_test_new.json + test_new/ images
    patch = f"""
# === auto-patch for new test set ===
_an_new = '{DATA}/mmocr'
_dr_new = '{DATA}/ocr_oriented'
_TEST_JSON = 'textrecog_test_new.json'
_TEST_PREFIX = 'test_new/'

test_dataloader = dict(
    batch_size=64, num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type='OCRDataset',
        data_root=_dr_new,
        ann_file=_an_new + '/' + _TEST_JSON,
        data_prefix=dict(img_path=''),
        test_mode=True,
        pipeline=test_dataloader['dataset']['pipeline'],
    )
)
test_evaluator = dict(dataset_prefixes=['watermeter'])
"""
    patched_path = f"/tmp/mmocr_patched_{suffix}.py"
    with open(patched_path, "w") as f:
        f.write(content + "\n" + patch)
    return patched_path


def run_mmocr_test(cfg_path, ckpt_path, name, orig_cfg_path):
    print(f"\n[eval_mmocr] {name}")
    try:
        patched_cfg = patch_config(orig_cfg_path, name.replace(" ", "_"))
        result_dir = f"{OUT_DIR}/mmocr_{name.replace(' ', '_')}"
        os.makedirs(result_dir, exist_ok=True)
        cmd = [
            PYTHON, f"{MMOCR_SRC}/tools/test.py",
            patched_cfg, ckpt_path,
            "--work-dir", result_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = proc.stdout + proc.stderr
        # Parse word_acc from mmocr output
        m = re.search(r"word_acc[:\s]+([0-9.]+)", output)
        if m:
            word_acc = float(m.group(1))
            print(f"  {name}: word_acc={word_acc:.4f}")
            return {"word_acc": word_acc, "char_acc": None, "correct": None, "total": None}
        else:
            print(f"  [warn] could not parse word_acc from output")
            print(output[-1000:])
            return None
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


models = [
    ("SATRN",
     f"{AMR}/outputs/eval_top5/satrn/satrn_watermeter.py",
     f"{AMR}/outputs/mmocr_satrn/epoch_100.pth"),
    ("ABINet",
     f"{AMR}/outputs/eval_top5/abinet/abinet_watermeter.py",
     f"{AMR}/outputs/mmocr_abinet/epoch_100.pth"),
]

results = {}
for name, orig_cfg, ckpt in models:
    r = run_mmocr_test(None, ckpt, name, orig_cfg)
    results[name] = r

with open(f"{OUT_DIR}/mmocr_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[eval_mmocr] saved -> {OUT_DIR}/mmocr_results.json")
