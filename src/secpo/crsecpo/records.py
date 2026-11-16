"""Read immutable review exports and prepare family-disjoint experiment inputs.

The filtering service owns its SQLite database. This adapter consumes exported
snapshots only, never edits that database, and never exposes publisher completions
or other reviewers' decisions to answer-generation or evaluation calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata
from urllib.parse import urlparse

from secpo.evaluation.sep_judge import atomic_write_jsonl, load_jsonl


POLICIES = {"prohibited", "limited", "permissible"}
GROUPS = {"harmful_benign_framing", "benign_adversarial_control", "benign_goal_change"}
INPUT_ASSET_MANIFEST = Path(__file__).resolve().parents[3] / "configs/crsecpo_inputs.json"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(text):
    # Used only to group records, never to rewrite model inputs.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def eligible(row):
    if row["selection_status"] != "accepted":
        return False
    if row["target_policy"] not in POLICIES or row["selection_group"] not in GROUPS:
        return False
    harmful = row["target_policy"] in {"prohibited", "limited"}
    return (harmful == (row["selection_group"] == "harmful_benign_framing")
            and bool(row["source"].strip()) and bool(row["target"].strip())
            and row["source"].strip() != row["target"].strip()
            and row["has_rewrite"])


def read_review_export(path, *, pool):
    """Read all rows, including unresolved and rejected records, for split grouping."""
    if pool not in {"training", "evaluation"}:
        raise ValueError("Unknown source pool")
    path = Path(path)
    file_hash = sha256(path)
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, 64 * 1024 * 1024))
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = set(reader.fieldnames or [])
            audit_columns = ("record_id", "consensus", "target_policy", "selection_group",
                             "has_rewrite", "revision", "row_index", "partition")
            prefixes = [k[:-len("record_id")] for k in fields
                        if k.startswith("crsecpo_") and k.endswith("record_id")
                        and all(k[:-len("record_id")] + suffix in fields for suffix in audit_columns)]
            if len(prefixes) != 1:
                raise ValueError("Expected one namespaced review export")
            prefix = prefixes[0]
            for row in reader:
                partition = row[prefix + "partition"]
                if partition != ("train" if pool == "training" else "eval"):
                    raise ValueError("Training/evaluation export partition mismatch")
                source, target = row.get("vanilla") or "", row.get("adversarial") or ""
                yield {"id": row[prefix + "record_id"], "source": source, "target": target,
                       "has_rewrite": row[prefix + "has_rewrite"] in {"1", "True", "true"},
                       "selection_status": row[prefix + "consensus"],
                       "target_policy": row[prefix + "target_policy"],
                       "selection_group": row[prefix + "selection_group"], "pool": pool,
                       "provenance": {"review_export_sha256": file_hash,
                                      "publisher_revision": row[prefix + "revision"],
                                      "publisher_row_index": row[prefix + "row_index"],
                                      "publisher_partition": partition}}
    finally:
        csv.field_size_limit(previous)


def prepare_records(rows, *, links=(), matched_controls=(), seed=42, validation_fraction=0.1):
    """Exact grouping plus supplied audited near-duplicate links, with test priority.

    Links connect record IDs, not model outputs. All variants of a source and
    normalized identical source/target strings share a connected component.
    Training-origin records linked to any evaluation record are quarantined.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    by_id = {}
    parent = {}
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in by_id:
            raise ValueError("Every input row needs a unique nonempty id")
        if row.get("pool") not in {"training", "evaluation"}:
            raise ValueError("Every row needs a training/evaluation pool")
        by_id[row["id"]] = dict(row)
        parent[row["id"]] = row["id"]

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left, right):
        a, b = sorted((find(left), find(right)))
        parent[b] = a

    seen = {}
    for key, row in by_id.items():
        for text in (row["source"], row["target"]):
            norm = normalized(text)
            if not norm:
                continue
            token = hashlib.sha256(norm.encode()).hexdigest()
            if token in seen:
                union(key, seen[token])
            else:
                seen[token] = key
    for link in links:
        if (set(link) != {"left", "right", "reason"} or not isinstance(link["reason"], str)
                or not link["reason"].strip() or link["left"] not in by_id or link["right"] not in by_id):
            raise ValueError("Near-duplicate links need known left/right IDs and a reason")
        union(link["left"], link["right"])
    test_families = {find(k) for k, v in by_id.items() if v["pool"] == "evaluation"}
    outputs = {"train": [], "validation": [], "test": [], "quarantine": []}
    assignments = []
    for key, row in by_id.items():
        family = find(key)
        draw = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(), 16) / 2**256
        split = "test" if row["pool"] == "evaluation" else ("validation" if draw < validation_fraction else "train")
        reason = None
        if row["pool"] == "training" and family in test_families:
            reason = "overlaps_held_out_family"
        elif not eligible(row):
            reason = "not_unanimously_accepted_with_resolved_policy_and_group"
        value = {**row, "family_id": family, "split": split}
        if reason:
            outputs["quarantine"].append({**value, "exclusion_reason": reason})
        else:
            outputs[split].append(value)
        assignments.append({"id": key, "family_id": family, "split": split, "exclusion_reason": reason})
    accepted_test = {row["id"]: row for row in outputs["test"]}
    matched_pairs = set()
    for link in matched_controls:
        if set(link) != {"harmful_id", "benign_id", "reason"} or not isinstance(link["reason"], str) or not link["reason"].strip():
            raise ValueError("Matched controls need harmful_id, benign_id, and a reason")
        harmful = accepted_test.get(link["harmful_id"])
        benign = accepted_test.get(link["benign_id"])
        if (harmful is None or benign is None or harmful["selection_group"] != "harmful_benign_framing"
                or benign["selection_group"] != "benign_adversarial_control"):
            raise ValueError("Matched control links must join accepted harmful and benign test candidates")
        pair = (harmful["id"], benign["id"])
        if pair in matched_pairs:
            raise ValueError("Duplicate matched control link")
        matched_pairs.add(pair)
        benign["control_group"] = "matched"
        benign.setdefault("matched_harmful_ids", []).append(harmful["id"])
    for row in outputs["test"]:
        row.setdefault("control_group", "other")
    return outputs, assignments


def convert_evaluation(harmful, benign, destination, *, revision):
    """Lossless adapter for the existing reviewer, with original JSON in every row."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "eval.tsv"
        with staged.open("w", encoding="utf-8", newline="") as stream:
          writer = csv.DictWriter(stream, fieldnames=["vanilla", "adversarial", "completion", "data_type",
                      "original_file", "original_row_index", "original_json", "original_sha256", "original_revision"],
                      delimiter="\t")
          writer.writeheader()
          for path, label in ((Path(harmful), "harmful"), (Path(benign), "benign")):
              rows, digest = load_jsonl(path), sha256(path)
              counts[label] = len(rows)
              for index, row in enumerate(rows):
                  if not isinstance(row.get("vanilla"), str) or not isinstance(row.get("adversarial"), str):
                      raise ValueError("Expected AI2 vanilla/adversarial strings")
                  writer.writerow({"vanilla": row["vanilla"], "adversarial": row["adversarial"],
                      "completion": "", "data_type": "adversarial_" + label,
                      "original_file": path.name, "original_row_index": index,
                      "original_json": json.dumps(row, ensure_ascii=False), "original_sha256": digest,
                      "original_revision": revision})
          stream.flush()
          os.fsync(stream.fileno())
        os.link(staged, destination)
    return {"counts": counts, "sha256": sha256(destination), "revision": revision,
            "note": "Adapted AI2 evaluation JSONL, not a publisher TSV release; original JSON retained."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("evaluation-tsv")
    for name in ("harmful", "benign", "output"):
        convert.add_argument("--" + name, type=Path, required=True)
    convert.add_argument("--revision", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--training-export", type=Path, required=True)
    prepare.add_argument("--evaluation-export", type=Path, required=True)
    prepare.add_argument("--family-links", type=Path, required=True,
                         help="Audited near-duplicate links JSONL; empty file allowed after review")
    prepare.add_argument("--near-duplicate-review-complete", action="store_true")
    prepare.add_argument("--matched-controls", type=Path,
                         help="Audited harmful/benign test matches; optional, never inferred from topic")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.command == "evaluation-tsv":
        # This boundary consumes raw publisher files, unlike reviewed derived exports.
        # Reuse their existing byte/revision pins before publishing the adapted TSV.
        from .assets import verify_asset
        assets = json.loads(INPUT_ASSET_MANIFEST.read_text())["assets"]
        for group in ("harmful", "benign"):
            matches = [asset for asset in assets if asset["name"] == "wildjailbreak_" + group]
            if len(matches) != 1:
                raise ValueError(f"asset manifest must contain exactly one {group} evaluation input pin")
            asset = matches[0]
            verify_asset(getattr(args, group), asset["sha256"])
            revision = urlparse(asset["url"]).path.strip("/").split("/")[2]
            if args.revision != revision:
                raise ValueError("evaluation source revision differs from the official input pin")
        print(json.dumps(convert_evaluation(args.harmful, args.benign, args.output, revision=args.revision)))
        return
    if not args.near_duplicate_review_complete:
        parser.error("Complete near-duplicate review before freezing experiment splits")
    rows = [*read_review_export(args.training_export, pool="training"),
            *read_review_export(args.evaluation_export, pool="evaluation")]
    links = load_jsonl(args.family_links) if args.family_links.read_text().strip() else []
    matches = load_jsonl(args.matched_controls) if args.matched_controls and args.matched_controls.read_text().strip() else []
    outputs, assignments = prepare_records(rows, links=links, matched_controls=matches, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, values in outputs.items():
        atomic_write_jsonl(values, args.output_dir / (name + ".jsonl"))
    atomic_write_jsonl(assignments, args.output_dir / "assignments.jsonl")
    manifest = {"schema_version": 1, "seed": args.seed, "validation_fraction": 0.1,
                "near_duplicate_review_complete": True, "family_links_sha256": sha256(args.family_links),
                "matched_controls_sha256": sha256(args.matched_controls) if args.matched_controls else None,
                "inputs": {str(p): sha256(p) for p in (args.training_export, args.evaluation_export)},
                "counts": {k: len(v) for k, v in outputs.items()},
                "source_status_counts": dict(Counter(row["selection_status"] for row in rows)),
                "outputs": {p.name: sha256(p) for p in args.output_dir.glob("*.jsonl")}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
