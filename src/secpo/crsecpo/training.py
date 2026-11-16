"""Fixed-row CRSecPO comparison using the released SecPO trainer and loss.

No online filtering, attack generation, or evaluation-set loading happens here.
``prepare`` accepts an injected tokenizer and never loads model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data import BenignReferenceCollator, PreferenceSchemaError, _read_json_records

METHODS = ("sft", "dpo", "crsecpo", "rubric_only")


class ComparisonCollator(BenignReferenceCollator):
    """Keep original policy tensors; only select the frozen reference context."""

    def __init__(self, tokenizer: Any, max_length: int, *, method: str,
                 max_prompt_length: int | None = None):
        if method not in METHODS:
            raise ValueError(f"unknown comparison method: {method}")
        # prepare() validates exactly one native terminator in each completion.
        # Preserve those tokens through upstream collation without appending EOS.
        super().__init__(tokenizer, max_length, max_prompt_length, append_eos=False,
                         allow_arbitrary_reference=True, truncate_prompt=False)
        self.method = method

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        rows = []
        for feature in features:
            reference = feature["prompt"] if self.method in {"sft", "dpo"} else feature[
                "rubric_prompt" if self.method == "rubric_only" else "clean_prompt"]
            rows.append({key: feature[key] for key in ("id", "prompt", "chosen", "rejected")}
                        | {"clean_prompt": reference})
        batch = super().__call__(rows)
        if self.method == "sft":
            return {suffix: batch[f"policy_chosen_{suffix}"]
                    for suffix in ("input_ids", "attention_mask", "labels")}
        return batch


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PreferenceSchemaError(f"pair {row.get('id')!r}: {key} must be nonempty text")
    return value


def _validate_training_row(row, config, tokenizer, collator):
    if not isinstance(row, dict):
        raise PreferenceSchemaError("each pair must be an object")
    row_id = _required_text(row, "id")
    if row.get("split") != "train":
        raise PreferenceSchemaError(f"pair {row_id}: split must be train; final-test ingestion is forbidden")
    from .pairs import validate_candidate
    try:
        validate_candidate(row)
        if row["provenance"].get("publisher_partition") != "train":
            raise ValueError("final-test/evaluation source provenance is forbidden in training")
    except (ValueError, TypeError) as error:
        raise PreferenceSchemaError(f"pair {row_id}: invalid training candidate: {error}") from error
    metadata = row.get("pair_metadata")
    if not isinstance(metadata, dict):
        raise PreferenceSchemaError(f"pair {row_id}: pair_metadata must be an object")
    if metadata.get("deployment_system") != "":
        raise PreferenceSchemaError(f"pair {row_id}: the scientific protocol requires user-only deployment prompts")
    from .prompts import validate_training_pair
    try:
        validate_training_pair(row, tokenizer, config.model_name_or_path, config.model_revision,
                               tokenizer_id=config.model_name_or_path,
                               tokenizer_revision=config.model_revision)
    except (ValueError, TypeError) as error:
        raise PreferenceSchemaError(f"pair {row_id}: provenance validation failed: {error}") from error
    if row["chosen"] == row["rejected"]:
        raise PreferenceSchemaError(f"pair {row_id}: chosen and rejected must differ")
    # Validate every reference for *every* method, even SFT. The methods
    # therefore cannot differ in the rows admitted by their token budgets.
    try:
        response_ids = [collator._tokenize_response(row[key]) for key in ("chosen", "rejected")]
        if response_ids[0] == response_ids[1]:
            raise ValueError("chosen and rejected have identical response token IDs")
        for key in ("prompt", "clean_prompt", "rubric_prompt"):
            prompt_ids = collator._tokenize(row[key])
            if config.max_prompt_length is not None and len(prompt_ids) > config.max_prompt_length:
                raise ValueError(f"{key} has {len(prompt_ids)} tokens > max_prompt_length={config.max_prompt_length}")
            for response in response_ids:
                if len(prompt_ids) + len(response) > config.max_length:
                    raise ValueError(f"{key}+response has {len(prompt_ids) + len(response)} tokens > max_length={config.max_length}")
        collator([row])
    except (ValueError, TypeError) as error:
        raise PreferenceSchemaError(f"{row_id}: {error}") from error


def prepare(config: Any, method: str, *, tokenizer: Any, require_selection: bool = False) -> tuple[list[dict], ComparisonCollator, dict]:
    """Validate every row/context, preserving input order across all methods.

    A failure lists offending IDs instead of dropping rows or shortening prompts.
    Raw file hashes and ordered IDs let callers compare runs without loading a
    final-test set. Only explicit ``split=train`` pairs are accepted.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if config.online_exchange_dir is not None:
        raise ValueError("CRSecPO comparisons require fixed offline pairs; online exchange is forbidden")
    if not config.append_eos:
        raise ValueError("CRSecPO comparisons require append_eos=true for native termination")
    path = Path(config.data_file).expanduser()
    payload = path.read_bytes()
    rows = _read_json_records(path)
    if not rows:
        raise PreferenceSchemaError("training pair file is empty")
    seen = set()
    collator = ComparisonCollator(tokenizer, config.max_length, method=method,
                                  max_prompt_length=config.max_prompt_length)
    errors = []
    for row in rows:
        if not isinstance(row, dict):
            raise PreferenceSchemaError("each pair must be an object")
        row_id = _required_text(row, "id")
        if row_id in seen:
            raise PreferenceSchemaError(f"duplicate training pair id: {row_id}")
        seen.add(row_id)
        try:
            _validate_training_row(row, config, tokenizer, collator)
        except PreferenceSchemaError as error:
            errors.append(str(error))
    if errors:
        raise PreferenceSchemaError("training preparation failed (no rows removed):\n" + "\n".join(errors))
    selection_path = path.parent / "selection_manifest.json"
    selection = None
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        counts = Counter("harmful" if row["selection_group"] == "harmful_benign_framing" else "benign" for row in rows)
        if (selection.get("schema_version") != "crsecpo-selection-v1"
                or selection.get("output_sha256") != hashlib.sha256(payload).hexdigest()
                or selection.get("ordered_ids") != [row["id"] for row in rows]
                or selection.get("selected_counts") != dict(counts)
                or counts["harmful"] != counts["benign"] or not counts["harmful"]):
            raise PreferenceSchemaError("frozen common selection manifest does not match balanced pair file")
    elif require_selection:
        raise PreferenceSchemaError("scientific training requires --freeze-pairs output and selection_manifest.json")
    manifest = {
        "schema_version": "crsecpo-training-manifest-v1", "method": method,
        "input_path": str(path.resolve()), "input_sha256": hashlib.sha256(payload).hexdigest(),
        "ordered_ids": [row["id"] for row in rows], "rows": len(rows),
        "source_sha256_counts": dict(Counter(hashlib.sha256(row["source"].encode()).hexdigest() for row in rows)),
        "selection_group_counts": dict(Counter(row["selection_group"] for row in rows)),
        "source_provenance": [row["provenance"] for row in rows],
        "split_counts": {"train": len(rows)}, "final_test_ingested": False,
        "max_length": config.max_length, "max_prompt_length": config.max_prompt_length,
        "truncate_prompt": False, "seed": config.seed, "common_selection": selection,
        "model_name_or_path": config.model_name_or_path, "model_revision": config.model_revision,
    }
    return rows, collator, manifest


def freeze_pairs(config, *, tokenizer, output_dir, pair_limit=2048):
    """Publish one balanced common set; retain an audit entry for every input pair."""
    from .pairs import ordered_by_class_and_family
    from .records import sha256
    from secpo.evaluation.sep_judge import atomic_write_jsonl
    if type(pair_limit) is not int or pair_limit < 2 or pair_limit % 2:
        raise ValueError("pair_limit must be a positive even number")
    source = Path(config.data_file).expanduser()
    input_hash = sha256(source)
    rows = _read_json_records(source)
    collator = ComparisonCollator(tokenizer, config.max_length, method="crsecpo",
                                  max_prompt_length=config.max_prompt_length)
    eligible, audit, seen = [], [], set()
    for row in rows:
        row_id = _required_text(row, "id")
        if row_id in seen:
            raise PreferenceSchemaError(f"duplicate input pair id: {row_id}")
        seen.add(row_id)
        entry = {"id": row_id, "status": "valid_unselected"}
        try:
            _validate_training_row(row, config, tokenizer, collator)
            eligible.append(row)
        except PreferenceSchemaError as error:
            entry.update(status="excluded", reason=str(error))
        audit.append(entry)
    ordered = ordered_by_class_and_family(eligible, seed=config.seed)
    classes = {name: [row for row in ordered if
        (row["selection_group"] == "harmful_benign_framing") == (name == "harmful")]
        for name in ("harmful", "benign")}
    per_class = min(pair_limit // 2, *(len(values) for values in classes.values()))
    if per_class == 0:
        raise PreferenceSchemaError("no balanced common selection: both classes need usable pairs")
    selected = [row for pair in zip(classes["harmful"][:per_class], classes["benign"][:per_class]) for row in pair]
    selected_ids = {row["id"] for row in selected}
    for entry in audit:
        if entry["id"] in selected_ids:
            entry["status"] = "selected"
    if sha256(source) != input_hash:
        raise ValueError("input pair file changed while freezing the common selection")
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=False)
    output = out / "train_pairs.jsonl"
    atomic_write_jsonl(selected, output)
    atomic_write_jsonl(audit, out / "selection_audit.jsonl")
    manifest = {"schema_version": "crsecpo-selection-v1", "input_sha256": input_hash,
        "input_pairs": len(rows), "pair_limit": pair_limit, "seed": config.seed,
        "max_length": config.max_length, "max_prompt_length": config.max_prompt_length,
        "eligible_counts": {name: len(values) for name, values in classes.items()},
        "selected_counts": {name: per_class for name in classes},
        "ordered_ids": [row["id"] for row in selected], "output_sha256": sha256(output),
        "audit_sha256": sha256(out / "selection_audit.jsonl")}
    _write_manifest_once(out / "selection_manifest.json", manifest)
    return manifest


def build_trainer(config: Any, method: str, *, tokenizer: Any = None) -> tuple[Any, Any, dict]:
    """Reuse released model/LoRA/arguments construction for every method."""
    from datasets import Dataset
    from transformers import Trainer, set_seed
    from ..training_cli import _build_training_arguments, _load_policy, _load_tokenizer
    from ..trainer import SecPOTrainer

    # Must precede adapter creation, including its random initialization.
    set_seed(config.seed)
    tokenizer = _load_tokenizer(config) if tokenizer is None else tokenizer
    rows, collator, manifest = prepare(config, method, tokenizer=tokenizer)
    training_args = _build_training_arguments(config)
    if training_args.world_size != 1 or training_args.n_gpu > 1:
        raise ValueError("the fixed-batch CRSecPO protocol requires one process and at most one GPU")
    policy = _load_policy(config)
    if any(parameter.requires_grad for name, parameter in policy.named_parameters()
           if "lora_" not in name):
        raise ValueError("all non-LoRA base parameters must remain frozen")
    trainer_class = Trainer if method == "sft" else SecPOTrainer
    kwargs = {} if method == "sft" else {"beta": config.beta}
    trainer = trainer_class(model=policy, args=training_args,
                            train_dataset=Dataset.from_list(rows), data_collator=collator,
                            processing_class=tokenizer, **kwargs)
    manifest["training_config"] = asdict(config)
    return trainer, tokenizer, manifest


def _run_identity(manifest: Mapping[str, Any]) -> dict:
    """All scientific settings and pair bytes, independent of local paths."""
    identity = dict(manifest)
    identity.pop("input_path", None)
    config = dict(identity.get("training_config", {}))
    for name in ("output_dir", "data_file"):
        config.pop(name, None)
    identity["training_config"] = config
    return json.loads(json.dumps(identity, sort_keys=True))


def guard_output(config: Any, manifest: dict, *, resume_from_checkpoint: str | None = None,
                 prepare_only: bool = False) -> Path:
    """Reject collisions and mismatched checkpoint provenance before model loading."""
    output = Path(config.output_dir).expanduser()
    canonical = output / "training_manifest.json"

    def check(path):
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot verify existing training manifest: {path}") from error
        if _run_identity(existing) != _run_identity(manifest):
            raise ValueError(f"Training manifest differs in method, pairs, or settings: {path}")

    if canonical.exists():
        check(canonical)
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory contains unrelated files without a training manifest: {output}")
    artifacts = output.exists() and any(path != canonical for path in output.iterdir())
    if artifacts and not prepare_only and resume_from_checkpoint is None:
        raise ValueError("Output contains training artifacts; choose a new directory or explicitly resume")
    if resume_from_checkpoint is not None:
        checkpoint = Path(resume_from_checkpoint).expanduser().resolve()
        if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume requires a Trainer checkpoint directory with trainer_state.json")
        source_manifest = next((parent / "training_manifest.json" for parent in (checkpoint, *checkpoint.parents)
                                if (parent / "training_manifest.json").is_file()), None)
        if source_manifest is None:
            raise ValueError("Resume checkpoint has no enclosing training manifest")
        check(source_manifest)
        # Trainer otherwise silently starts fresh optimizer/scheduler state or
        # skips restoring RNG state. Require the complete single-process PEFT
        # checkpoint written by this protocol, not a final model-only export.
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt",
                    "rng_state.pth", "adapter_config.json")
        missing = [name for name in required if not (checkpoint / name).is_file()
                   or (checkpoint / name).stat().st_size == 0]
        weights = ("adapter_model.safetensors", "adapter_model.bin")
        if not any((checkpoint / name).is_file() and (checkpoint / name).stat().st_size
                   for name in weights):
            missing.append("adapter_model.safetensors or adapter_model.bin")
        if missing:
            raise ValueError("Resume requires a complete single-process Trainer/PEFT checkpoint; "
                             "missing or empty artifacts: " + ", ".join(missing))
    return canonical


def _write_manifest_once(destination: Path, manifest: dict) -> None:
    """Preserve existing manifests; never replace provenance during a resume."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except FileExistsError:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if _run_identity(existing) != _run_identity(manifest):
            raise ValueError(f"Refusing to overwrite a different manifest: {destination}")


def main(argv: Sequence[str] | None = None) -> None:
    from ..training_cli import load_training_config, _load_tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--freeze-pairs", action="store_true",
                        help="Validate and freeze one balanced common pair file; never load model weights")
    parser.add_argument("--pair-limit", type=int, default=2048)
    parser.add_argument("--data-file", help="Override the fixed training-pairs file")
    parser.add_argument("--output-dir", help="Default: CONFIG_OUTPUT_DIR/METHOD")
    parser.add_argument("--prepare", "--dry-run", action="store_true", dest="prepare_only")
    parser.add_argument("--manifest", help="Optional additional manifest path")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args(argv)
    if args.freeze_pairs:
        if args.method or args.prepare_only or args.resume_from_checkpoint or args.manifest or not args.output_dir:
            parser.error("--freeze-pairs requires --output-dir and cannot be combined with training options")
    elif not args.method:
        parser.error("--method is required unless freezing pairs")
    config = load_training_config(args.config)
    config = replace(config, output_dir=args.output_dir or str(Path(config.output_dir) / args.method),
                     data_file=args.data_file or config.data_file)
    tokenizer = _load_tokenizer(config)
    if args.freeze_pairs:
        print(json.dumps(freeze_pairs(config, tokenizer=tokenizer, output_dir=args.output_dir,
                                      pair_limit=args.pair_limit), indent=2))
        return
    _, _, manifest = prepare(config, args.method, tokenizer=tokenizer, require_selection=True)
    manifest["training_config"] = asdict(config)
    canonical = guard_output(config, manifest, resume_from_checkpoint=args.resume_from_checkpoint,
                             prepare_only=args.prepare_only)
    # Record provenance before model creation; preserve prior manifests verbatim.
    _write_manifest_once(canonical, manifest)
    if args.manifest:
        _write_manifest_once(Path(args.manifest).expanduser(), manifest)
    if not args.prepare_only:
        trainer, tokenizer, actual_manifest = build_trainer(config, args.method, tokenizer=tokenizer)
        if _run_identity(actual_manifest) != _run_identity(manifest):
            raise ValueError("Training pairs changed after output provenance was checked")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_state()
        trainer.save_model(config.output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
