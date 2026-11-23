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
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from secpo.evaluation.sep_adaptive import atomic_write_json

from ..data import BenignReferenceCollator, PreferenceSchemaError, _read_json_records

from .artifact_versions import logical_artifact_path
from .collection_exports import read_collection_export

METHODS = ("sft", "dpo", "crsecpo", "rubric_only")
COLLECTION_FIELDS = ("backend_identity", "generation_config", "judge_config")


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


def _validate_training_row(row, config, tokenizer, collator, *, expected_split="train"):
    if not isinstance(row, dict):
        raise PreferenceSchemaError("each pair must be an object")
    row_id = _required_text(row, "id")
    if row.get("split") != expected_split:
        raise PreferenceSchemaError(
            f"pair {row_id}: split must be {expected_split}; final-test ingestion is forbidden")
    from .pairs import validate_candidate
    try:
        # train and validation both live in the training pool (partition "train");
        # the held-out evaluation pool must never enter either.
        validate_candidate(row)
        if row["provenance"].get("publisher_partition") != "train":
            raise ValueError("final-test/evaluation source provenance is forbidden in train/validation")
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


def _pair_retry_limit(metadata):
    # Legacy HF collection never retried model outputs. Legacy vLLM files
    # could come from either policy; absence of retries is not evidence of a limit.
    return metadata.get("max_output_retries", 0 if metadata.get("backend_identity") is None else None)


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
    payload = read_collection_export(path, "pairs")
    rows = _read_json_records(path, payload=payload)
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
        recorded_settings = {field: selection[field] for field in COLLECTION_FIELDS if field in selection}
        if (selection.get("schema_version") != "crsecpo-selection-v1"
                or selection.get("output_sha256") != hashlib.sha256(payload).hexdigest()
                or selection.get("ordered_ids") != [row["id"] for row in rows]
                or selection.get("selected_counts") != dict(counts)
                or any(row["pair_metadata"].get(field) != value
                       for row in rows for field, value in recorded_settings.items())
                or ("max_output_retries" in selection and any(
                    _pair_retry_limit(row["pair_metadata"]) != selection["max_output_retries"] for row in rows))
                or counts["harmful"] != counts["benign"] or not counts["harmful"]):
            raise PreferenceSchemaError("frozen common selection manifest does not match balanced pair file")
    elif require_selection:
        raise PreferenceSchemaError("scientific training requires --freeze-pairs output and selection_manifest.json")
    manifest = {
        "schema_version": "crsecpo-training-manifest-v1", "method": method,
        "input_path": str(logical_artifact_path(path)), "input_sha256": hashlib.sha256(payload).hexdigest(),
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


def freeze_pairs(config, *, tokenizer, output_dir, pair_limit=2048, benign_fraction=0.5):
    """Publish one balanced common set; retain an audit entry for every input pair."""
    from .pairs import ordered_by_class_and_family
    from .records import sha256
    from secpo.evaluation.sep_judge import atomic_write_jsonl
    if type(pair_limit) is not int or pair_limit < 2 or pair_limit % 2:
        raise ValueError("pair_limit must be a positive even number")
    out = Path(output_dir).expanduser()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Frozen selection already exists: {out}")
    source = Path(config.data_file).expanduser()
    payload = read_collection_export(source, "pairs")
    input_hash = hashlib.sha256(payload).hexdigest()
    rows = _read_json_records(source, payload=payload)
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
    if not 0.0 < benign_fraction < 1.0:
        raise ValueError("benign_fraction must be strictly between 0 and 1")
    # Select the largest set (<= pair_limit) that holds the requested benign:harmful ratio
    # given how many of each are available. 0.5 exactly reproduces the historical 50/50
    # balance; raising it adds "help the suspicious benign request" examples to counter
    # over-refusal (at some cost to harmful safety).
    have = {name: len(values) for name, values in classes.items()}
    total = min(pair_limit, int(have["benign"] / benign_fraction),
                int(have["harmful"] / (1.0 - benign_fraction)))
    target = {"benign": round(benign_fraction * total)}
    target["harmful"] = total - target["benign"]
    if target["benign"] == 0 or target["harmful"] == 0:
        raise PreferenceSchemaError("no usable common selection: both classes need usable pairs")
    metadata = eligible[0]["pair_metadata"]
    collection_settings = {field: metadata.get(field) for field in COLLECTION_FIELDS}
    for field, value in collection_settings.items():
        if any(row["pair_metadata"].get(field) != value for row in eligible):
            label = "collection backend identities" if field == "backend_identity" else field
            raise PreferenceSchemaError(f"eligible pairs use different {label}")
    retry_limits = {_pair_retry_limit(row["pair_metadata"]) for row in eligible}
    if None in retry_limits:
        raise PreferenceSchemaError("eligible vLLM pairs lack a recorded retry limit; re-export from their collection journal")
    if len(retry_limits) != 1:
        raise PreferenceSchemaError("eligible pairs use different max_output_retries")
    collection_settings["max_output_retries"] = retry_limits.pop()
    chosen_ids = ({row["id"] for row in classes["harmful"][:target["harmful"]]}
                  | {row["id"] for row in classes["benign"][:target["benign"]]})
    selected = [row for row in ordered if row["id"] in chosen_ids]
    selected_ids = {row["id"] for row in selected}
    for entry in audit:
        if entry["id"] in selected_ids:
            entry["status"] = "selected"
    if hashlib.sha256(read_collection_export(source, "pairs")).hexdigest() != input_hash:
        raise ValueError("input pair file changed while freezing the common selection")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}.") as temporary:
        staged = Path(temporary) / "selection"
        output = staged / "train_pairs.jsonl"
        audit_path = staged / "selection_audit.jsonl"
        atomic_write_jsonl(selected, output)
        atomic_write_jsonl(audit, audit_path)
        manifest = {"schema_version": "crsecpo-selection-v1", "input_sha256": input_hash,
            "input_pairs": len(rows), "pair_limit": pair_limit, "seed": config.seed,
            "max_length": config.max_length, "max_prompt_length": config.max_prompt_length,
            **collection_settings,
            "eligible_counts": {name: len(values) for name, values in classes.items()},
            "selected_counts": dict(target),
            "ordered_ids": [row["id"] for row in selected], "output_sha256": sha256(output),
            "audit_sha256": sha256(audit_path)}
        _write_manifest_once(staged / "selection_manifest.json", manifest)
        # A competing freeze publishes a nonempty directory that rename cannot replace.
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"Frozen selection already exists: {out}")
        staged.rename(out)
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
    eval_dataset = None
    if getattr(config, "eval_data_file", None) is not None:
        eval_dataset = _prepare_eval_dataset(config, method, tokenizer, collator)
    trainer_class = Trainer if method == "sft" else SecPOTrainer
    kwargs = {} if method == "sft" else {"beta": config.beta}
    if method != "sft" and eval_dataset is not None:
        # SecPO reports preference accuracy on validation; plain-Trainer SFT keeps eval_loss.
        from ..trainer import secpo_preference_metrics
        kwargs["compute_metrics"] = secpo_preference_metrics
    trainer = trainer_class(model=policy, args=training_args,
                            train_dataset=Dataset.from_list(rows), eval_dataset=eval_dataset,
                            data_collator=collator, processing_class=tokenizer, **kwargs)
    manifest["training_config"] = asdict(config)
    return trainer, tokenizer, manifest


def _prepare_eval_dataset(config: Any, method: str, tokenizer: Any, collator: Any):
    """Validate the held-out validation pairs for best-checkpoint selection.

    Validation pairs must carry ``split='validation'`` and training-pool
    provenance; the final-test (evaluation pool) can never enter here.
    """
    from datasets import Dataset
    path = Path(config.eval_data_file).expanduser()
    payload = read_collection_export(path, "pairs")
    rows = _read_json_records(path, payload=payload)
    if not rows:
        raise PreferenceSchemaError("validation pair file is empty")
    seen, errors = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise PreferenceSchemaError("each pair must be an object")
        row_id = _required_text(row, "id")
        if row_id in seen:
            raise PreferenceSchemaError(f"duplicate validation pair id: {row_id}")
        seen.add(row_id)
        try:
            _validate_training_row(row, config, tokenizer, collator, expected_split="validation")
        except PreferenceSchemaError as error:
            errors.append(str(error))
    if errors:
        raise PreferenceSchemaError("validation preparation failed (no rows removed):\n" + "\n".join(errors))
    return Dataset.from_list(rows)


def _run_identity(manifest: Mapping[str, Any]) -> dict:
    """All scientific settings and pair bytes, independent of local paths."""
    identity = dict(manifest)
    identity.pop("input_path", None)
    config = dict(identity.get("training_config", {}))
    for name in ("output_dir", "data_file", "eval_data_file"):
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

    artifacts = output.exists() and any(
        path != canonical and not (
            path.name.startswith(_manifest_stage_prefix(canonical)) and path.is_dir() and not path.is_symlink()
        )
        for path in output.iterdir()
    )
    if canonical.exists() or canonical.is_symlink():
        check(canonical)
    elif artifacts:
        raise ValueError(f"Output directory contains unrelated files without a training manifest: {output}")
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


def _manifest_stage_prefix(destination: Path) -> str:
    return f".{destination.name}.stage-"


def _write_manifest_once(destination: Path, manifest: dict) -> None:
    """Atomically publish provenance without replacing an existing manifest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and not destination.is_symlink():
        with tempfile.TemporaryDirectory(dir=destination.parent,
                                         prefix=_manifest_stage_prefix(destination)) as temporary:
            staged = Path(temporary) / destination.name
            atomic_write_json(staged, manifest)
            try:
                os.link(staged, destination)
            except FileExistsError:
                pass  # Check the complete manifest published by the other writer.
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
    parser.add_argument("--benign-fraction", type=float, default=0.5,
                        help="Fraction of the frozen set that is benign/safe pairs (0.5 = 50/50). "
                             "Raise it to add 'help the suspicious benign request' examples and curb over-refusal.")
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
                                      pair_limit=args.pair_limit, benign_fraction=args.benign_fraction), indent=2))
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
