#!/usr/bin/env python3
"""Combine all partial eval JSONs into one report table."""
import json, os

WS = "/datastore/cndt_thangcpd/linhtruong/workspace3"
OUT_DIR = f"{WS}/water_meter_amr/outputs/eval_new_testset"

results = {}
for fname in ["paddle_results.json", "parseq_results.json", "mmocr_results.json"]:
    path = os.path.join(OUT_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            results.update(json.load(f))

if not results:
    print("No results found yet.")
    exit(0)

valid = [(k, v) for k, v in results.items() if v and v.get("word_acc") is not None]
valid.sort(key=lambda x: x[1]["word_acc"], reverse=True)

print("\n" + "=" * 62)
print(f"  SOTA Evaluation — New Test Set (574 images)")
print("=" * 62)
print(f"  {'Rank':<5} {'Model':<30} {'Word Acc':>10} {'Char Acc':>10}")
print("-" * 62)
for rank, (name, r) in enumerate(valid, 1):
    char = f"{r['char_acc']*100:.2f}%" if r.get("char_acc") else "  n/a  "
    total = f"({r['correct']}/{r['total']})" if r.get("total") else ""
    print(f"  {rank:<5} {name:<30} {r['word_acc']*100:>9.2f}%  {char}  {total}")

failed = [k for k, v in results.items() if not v or v.get("word_acc") is None]
if failed:
    print(f"\n  Failed/missing: {', '.join(failed)}")
print("=" * 62)

combined = {
    "test_set_size": 574,
    "models": {k: v for k, v in results.items()},
    "ranking": [(k, round(v["word_acc"], 6)) for k, v in valid],
}
with open(f"{OUT_DIR}/combined_results.json", "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nSaved -> {OUT_DIR}/combined_results.json")
