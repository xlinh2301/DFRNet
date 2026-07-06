"""
One-off: rewrite rec_gt_{train,val,test}.txt image paths from their original
Colab absolute paths to paths relative to data/set{1,2,3}.

Usage:
    python tools/rewrite_gt_paths.py
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")

MAPPING = [
    ("gt/rec/rec_gt_train.txt", "set1/train"),
    ("gt/rec/rec_gt_val.txt", "set2/val"),
    ("gt/rec/rec_gt_test.txt", "set3/test"),
]


def rewrite(label_rel_path: str, image_prefix: str):
    label_path = os.path.join(DATA_DIR, label_rel_path)
    image_dir = os.path.join(DATA_DIR, image_prefix)

    with open(label_path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    out_lines = []
    missing = []
    for line in lines:
        old_path, label = line.split("\t", 1)
        basename = os.path.basename(old_path)
        new_path = f"{image_prefix}/{basename}"
        if not os.path.isfile(os.path.join(DATA_DIR, new_path)):
            missing.append(basename)
        out_lines.append(f"{new_path}\t{label}")

    if missing:
        raise SystemExit(
            f"{label_rel_path}: {len(missing)} rows reference missing files, "
            f"e.g. {missing[:5]}"
        )

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"{label_rel_path}: rewrote {len(out_lines)} rows -> {image_prefix}/")


if __name__ == "__main__":
    for rel_path, prefix in MAPPING:
        rewrite(rel_path, prefix)
