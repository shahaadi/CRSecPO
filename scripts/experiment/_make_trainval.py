"""Merge the train and validation candidate splits into one training set.

Full-epoch training uses no validation-based selection, so the validation split
is folded into training to use all of it. Every merged row's split is set to
"train" (both splits are 100% training-pool, so this passes the training
provenance guard); the held-out test split is left untouched. Output bytes are
deterministic (sorted keys, fixed order) so pair-collection hashing is stable.

Env in: SPLIT_DIR, WORK.
"""
import json, os
from pathlib import Path

split_dir = Path(os.environ["SPLIT_DIR"])
rows = []
for name in ("train", "validation"):
    for line in (split_dir / f"{name}.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            row["split"] = "train"
            rows.append(row)

out = Path(os.environ["WORK"]) / "trainval_candidates.jsonl"
out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
print(f"wrote {out} ({len(rows)} rows: train+validation merged, split=train)")
