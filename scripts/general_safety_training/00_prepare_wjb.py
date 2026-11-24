"""Convert the WildJailbreak *adversarial* split into CRSecPO candidate splits.

Produces train/validation/test JSONL in $SPLIT_DIR, in the exact candidate schema
`validate_candidate` requires, so every downstream stage (01 generate, 02 freeze,
04 eval) reuses them unchanged.

Mapping (only data_types that carry BOTH a vanilla and an adversarial field, so
source != target is satisfiable):
    adversarial_harmful -> source=vanilla, target=adversarial,
                           selection_group=harmful_benign_framing, target_policy=prohibited  (unsafe)
    adversarial_benign  -> source=vanilla, target=adversarial,
                           selection_group=benign_adversarial_control, target_policy=permissible (safe)

Pure vanilla_* rows have no distinct source and are skipped (noted in the README).

Split is done at the FAMILY level (hash of the vanilla request) so adversarial
variants of the same base request never straddle train/test.

Env: SPLIT_DIR (out), GSAFE_MAX_PER_CLASS ('all' or int), GSAFE_FORCE_PREP, SEED.
Needs: the `datasets` package and HF access to the gated allenai/wildjailbreak
(export HF_TOKEN after accepting the dataset terms).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DATASET = "allenai/wildjailbreak"
SEED = os.environ.get("SEED", "42")
# data_type -> (selection_group, target_policy, class label)
MAPPING = {
    "adversarial_harmful": ("harmful_benign_framing", "prohibited", "unsafe"),
    "adversarial_benign": ("benign_adversarial_control", "permissible", "safe"),
}
# Family-hash buckets 0..9 -> split. Keeps variants of one base request together.
SPLIT_BUCKETS = {**{b: "train" for b in range(8)}, 8: "validation", 9: "test"}


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _bucket(family_id: str) -> int:
    return int(_sha("split", SEED, family_id)[:8], 16) % 10


def _max_per_class() -> float:
    value = os.environ.get("GSAFE_MAX_PER_CLASS", "5000").strip().lower()
    return float("inf") if value in {"all", ""} else int(value)


def _load_rows():
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover
        raise SystemExit("the 'datasets' package is required: pip install datasets") from error
    # WildJailbreak ships as TSV; keep_default_na avoids turning 'null'-like text into NaN.
    ds = load_dataset(DATASET, "train", delimiter="\t", keep_default_na=False)["train"]
    cols = set(ds.column_names)
    for required in ("vanilla", "adversarial", "data_type"):
        if required not in cols:
            raise SystemExit(f"unexpected WildJailbreak columns {sorted(cols)}; missing {required!r}")
    return ds


def build_candidates():
    ds = _load_rows()
    cap = _max_per_class()
    kept, counts, skipped = [], {"unsafe": 0, "safe": 0}, 0
    seen_ids = set()
    for index, row in enumerate(ds):
        mapped = MAPPING.get((row.get("data_type") or "").strip())
        if mapped is None:
            continue  # pure vanilla_* or unknown type
        group, policy, label = mapped
        source = (row.get("vanilla") or "").strip()
        target = (row.get("adversarial") or "").strip()
        if not source or not target or source == target:
            skipped += 1
            continue
        if counts[label] >= cap:
            continue
        family_id = _sha("family", source)
        cand_id = _sha("cand", source, target)
        if cand_id in seen_ids:
            continue
        seen_ids.add(cand_id)
        counts[label] += 1
        kept.append({
            "id": cand_id,
            "family_id": family_id,
            "source": source,
            "target": target,
            "split": SPLIT_BUCKETS[_bucket(family_id)],
            "target_policy": policy,
            "selection_group": group,
            "selection_status": "accepted",
            "provenance": {"dataset": DATASET, "config": "train",
                           "data_type": row.get("data_type"), "row_index": str(index)},
        })
    return kept, counts, skipped


def _validate(row) -> None:
    # Mirror the key invariants of pairs.validate_candidate without importing torch-heavy modules.
    for key in ("id", "source", "target", "family_id", "split", "target_policy",
                "selection_group", "selection_status"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"candidate {key} must be nonempty text")
    if row["source"].strip() == row["target"].strip():
        raise ValueError("source and target must differ")
    if row["target_policy"] not in {"prohibited", "limited", "permissible"}:
        raise ValueError("bad target_policy")
    if (row["selection_group"] == "harmful_benign_framing") != (row["target_policy"] in {"prohibited", "limited"}):
        raise ValueError("selection_group/target_policy inconsistent")


def main():
    out_dir = Path(os.environ["SPLIT_DIR"])
    force = os.environ.get("GSAFE_FORCE_PREP", "0") == "1"
    train_path = out_dir / "train.jsonl"
    if train_path.exists() and train_path.stat().st_size and not force:
        print(f"[00] candidate splits already present in {out_dir}; skipping "
              f"(set GSAFE_FORCE_PREP=1 to rebuild)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, counts, skipped = build_candidates()
    for row in rows:
        _validate(row)
    by_split = {"train": [], "validation": [], "test": []}
    for row in rows:
        by_split[row["split"]].append(row)
    for split, items in by_split.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_unsafe = sum(r["selection_group"] == "harmful_benign_framing" for r in items)
        print(f"[00] {split:<10} {len(items):>6} rows  (unsafe={n_unsafe}, safe={len(items) - n_unsafe})")
    print(f"[00] total kept={len(rows)} (unsafe={counts['unsafe']}, safe={counts['safe']}), "
          f"skipped={skipped}; wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
