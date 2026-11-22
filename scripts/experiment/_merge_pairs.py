"""Merge already-generated train + validation pair exports into one training set.

Use this when stage 1 was already run and produced both train_pairs.jsonl and
validation_pairs.jsonl: it folds validation into training WITHOUT regenerating a
single answer. Each row's split is relabeled to "train" (validation rows carried
split=validation; both are training-pool, so this passes the training provenance
guard), and the combined rows are written as one standalone export at
PAIRS_DIR/merged_pairs.jsonl, which stage 2 freezes.

No model is loaded and no answers are generated. Env in: PAIRS_DIR.
"""
import json, os
from pathlib import Path

from secpo.crsecpo.collection_exports import read_collection_export, write_standalone_export
from secpo.evaluation.sep_judge import load_jsonl

pairs_dir = Path(os.environ["PAIRS_DIR"])
combined, seen = [], set()
for name in ("train", "validation"):
    source = pairs_dir / f"{name}_pairs.jsonl"
    if not source.exists():
        raise SystemExit(
            f"missing {source}; run stage 1 first (both train and validation pairs "
            "must already exist to merge them)")
    payload = read_collection_export(source, "pairs")   # verifies the export
    for row in load_jsonl(source, payload=payload):
        if row["id"] in seen:
            raise SystemExit(f"duplicate pair id across splits: {row['id']}")
        seen.add(row["id"])
        row["split"] = "train"
        combined.append(row)

out = pairs_dir / "merged_pairs.jsonl"
out.unlink(missing_ok=True)
write_standalone_export(combined, out)
print(f"wrote {out} ({len(combined)} pairs: train+validation merged, split=train)")
