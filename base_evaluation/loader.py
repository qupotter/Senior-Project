from datasets import load_dataset
import json

ds = load_dataset("lighteval/hendrycks_ethics", "deontology")

out_path = "ethics_deontology_raw.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for split in ds.keys():
        for ex in ds[split]:
            row = {
                "split": split,
                "group_id": int(ex["group_id"]),
                "label": int(ex["label"]),
                "scenario": ex["scenario"],
                "excuse": ex["excuse"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Wrote:", out_path)
print("Splits:", list(ds.keys()))
print("Example:", ds[list(ds.keys())[0]][0])
