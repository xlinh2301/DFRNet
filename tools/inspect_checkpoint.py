"""
Inspect PPOCRv5 checkpoint and simulate DFRNet key remapping.
No Paddle required — uses pure pickle.

Usage:
    python tools/inspect_checkpoint.py --ckpt ../../release/ppocr_v5_paddle/best_accuracy.pdparams
"""

import argparse
import pickle
import sys


REMAP_PREFIX = {
    "head.ctc_encoder.": "ctc_encoder.",
    "head.ctc_head.fc.": "ctc_fc.",
}

SKIP_PREFIX = (
    "head.gtc_head.",
    "head.before_gtc.",
    "StructuredToParameterName@@",
)

DFRNET_NEW_MODULES = ("ofr.", "corruption.")


def inspect(path: str):
    with open(path, "rb") as f:
        state = pickle.load(f)

    keys = list(state.keys())
    print(f"Total checkpoint keys: {len(keys)}\n")

    loaded, skipped = [], []
    for k in keys:
        if k.startswith(SKIP_PREFIX):
            skipped.append((k, "skipped — not used in DFRNet"))
            continue

        mk = k
        for src, dst in REMAP_PREFIX.items():
            if k.startswith(src):
                mk = dst + k[len(src):]
                break

        v = state[k]
        shape = getattr(v, "shape", "?")
        loaded.append((k, mk, shape))

    print("=== Keys that WILL be loaded into DFRNet ===")
    for ck, mk, shape in loaded:
        arrow = f"  {ck}" if ck == mk else f"  {ck}  →  {mk}"
        print(f"{arrow}  {shape}")

    print(f"\n=== Keys SKIPPED (not in DFRNet) ===")
    for ck, reason in skipped:
        print(f"  {ck} — {reason}")

    print(f"\n=== Summary ===")
    print(f"  Loaded : {len(loaded)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"\nDFRNet-only modules (randomly initialised):")
    for m in DFRNET_NEW_MODULES:
        print(f"  {m}*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="../../release/ppocr_v5_paddle/best_accuracy.pdparams")
    args = parser.parse_args()
    inspect(args.ckpt)
