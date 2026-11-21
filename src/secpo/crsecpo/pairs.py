"""Frozen offline ordinary/combined preference collection, with no original completions."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import zip_longest
import json
import os
from pathlib import Path
from typing import Mapping

from secpo.evaluation.sep_judge import atomic_write_jsonl, load_jsonl
from .inference import (GenerationConfig, HFBackend, JournalBackend, ModelSpec,
                        make_request, render_messages, validate_response)
from .prompts import (PROMPT_VERSION, RUBRIC, answer_messages, digest,
                      judging_orders, native_terminator)

CANDIDATE_FIELDS = ("id", "source", "target", "family_id", "split", "target_policy",
                    "selection_group", "selection_status", "provenance")


def validate_candidate(row, *, allowed_splits=("train", "validation")):
    if not isinstance(row, Mapping):
        raise ValueError("candidate must be an object")
    for key in CANDIDATE_FIELDS:
        if key not in row:
            raise ValueError(f"candidate missing {key}")
        if key != "provenance" and (not isinstance(row[key], str) or not row[key].strip()):
            raise ValueError(f"candidate {key} must be nonempty text")
    if row["source"].strip() == row["target"].strip():
        raise ValueError("candidate source and rewrite must be distinct after stripping surrounding whitespace")
    if not isinstance(row["provenance"], Mapping) or not row["provenance"]:
        raise ValueError("candidate provenance must be a nonempty object")
    if row["selection_status"] != "accepted":
        raise ValueError("generation requires accepted candidates")
    if row["split"] not in allowed_splits:
        raise ValueError(f"candidate split must be one of {', '.join(sorted(allowed_splits))}")
    policy = row["target_policy"]
    if policy not in {"prohibited", "limited", "permissible"}:
        raise ValueError("target_policy must be resolved before generation")
    group = row["selection_group"]
    if group not in {"harmful_benign_framing", "benign_adversarial_control", "benign_goal_change"}:
        raise ValueError("unknown selection_group")
    if (group == "harmful_benign_framing") != (policy in {"prohibited", "limited"}):
        raise ValueError("target_policy is inconsistent with selection_group")


def _seed(seed, candidate_id, kind):
    return int(digest(seed, candidate_id, kind)[:8], 16)


def ordered_by_class_and_family(rows, *, seed):
    """Fixed, input-order-independent candidate order, before any model outcomes."""
    groups = {"harmful": defaultdict(list), "benign": defaultdict(list)}
    for row in rows:
        group = "harmful" if row["selection_group"] == "harmful_benign_framing" else "benign"
        groups[group][row["family_id"]].append(row)
    ordered = {}
    for group, families in groups.items():
        streams = [sorted(families[key], key=lambda row: digest(seed, "row", row["id"]))
                   for key in sorted(families, key=lambda key: digest(seed, "family", key))]
        ordered[group] = [row for batch in zip_longest(*streams) for row in batch if row is not None]
    return [row for batch in zip_longest(ordered["harmful"], ordered["benign"])
            for row in batch if row is not None]


def balanced_candidate_cap(rows):
    """Smallest prefix of the class-interleaved order that schedules every row of the
    smaller class and an equal number of the larger one.

    Frozen selection keeps equal harmful/benign counts, so candidates of the larger
    class beyond this prefix cannot enter the balanced training file.
    """
    harmful = sum(row["selection_group"] == "harmful_benign_framing" for row in rows)
    smaller = min(harmful, len(rows) - harmful)
    if smaller < 1:
        raise ValueError("a balanced candidate cap requires both harmful and benign candidates")
    return 2 * smaller


def _max_candidates(value):
    if value == "balanced":
        return value
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer or 'balanced'") from None
    if number < 1:
        raise argparse.ArgumentTypeError("--max-candidates must be positive")
    return number


def _validate_candidates(candidates):
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidate file is empty")
    seen = set()
    for row in candidates:
        validate_candidate(row)
        if row["id"] in seen:
            raise ValueError(f"duplicate candidate id: {row['id']}")
        seen.add(row["id"])
    return candidates


def _validate_distinct_paths(**paths):
    """Prevent inputs, completed exports and request journals from replacing one another."""
    items = [(label, Path(path).resolve()) for label, path in paths.items()]
    for index, (left_label, left) in enumerate(items):
        if left.is_dir():
            raise ValueError(f"{left_label} must use distinct files, not directories")
        for right_label, right in items[index + 1:]:
            if (left == right or left in right.parents or right in left.parents
                    or (left.exists() and right.exists() and left.samefile(right))):
                raise ValueError(f"{left_label} and {right_label} must use distinct files")


def _completion(response, terminator):
    if response["finish_reason"] != "stop":
        raise ValueError("truncated completion")
    text = response["text"]
    if response.get("completion_terminator") not in {None, terminator}:
        raise ValueError("unexpected completion terminator")
    # Callable backends may return answer text; HFBackend retains the generated eot.
    if not text.endswith(terminator):
        text += terminator
    if text.count(terminator) != 1:
        raise ValueError("completion contains an interior or duplicate native terminator")
    answer = text[:-len(terminator)]
    if not answer.strip():
        raise ValueError("empty completion")
    return text, answer


def generate_pairs(candidates, *, backend, tokenizer, model_spec: ModelSpec,
                   generation_config: GenerationConfig = GenerationConfig(),
                   judge_config: GenerationConfig = GenerationConfig(max_new_tokens=256),
                   seed: int = 0, deployment_system: str = "",
                   output_path: str | Path | None = None,
                   outcomes_path: str | Path | None = None):
    """Return (accepted pairs, all outcomes), atomically exporting completed runs.

    Wrap ``backend`` in JournalBackend for per-request restart without resampling.
    Judge requests carry only rubric/target/anonymous answers. Their seeds differ
    by display order, making them separate samples even with deterministic seeds.
    """
    candidates = _validate_candidates(candidates)
    terminator = native_terminator(tokenizer)
    pairs, outcomes = [], []
    for candidate in candidates:
        row = {key: candidate[key] for key in CANDIDATE_FIELDS}
        generations, judgments, completions, answers = {}, [], {}, {}
        outcome = {**row, "accepted": False, "status": "generation_error",
                   "generations": generations, "judgments": judgments}
        for condition in ("ordinary", "combined"):
            request = make_request(answer_messages(candidate, condition, deployment_system=deployment_system),
                tokenizer=tokenizer, model_spec=model_spec, config=generation_config,
                seed=_seed(seed, candidate["id"], condition), kind=condition)
            result = {"request_id": digest(request), "request": request}
            generations[condition] = result
            try:
                response = validate_response(backend(request))
                result["response"] = response
                completions[condition], answers[condition] = _completion(response, terminator)
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
        if len(completions) == 2:
            for order, messages in judging_orders(candidate["target"], answers["ordinary"],
                                                  answers["combined"], comparison_id=candidate["id"]):
                request = make_request(messages, tokenizer=tokenizer, model_spec=model_spec,
                    config=judge_config, seed=_seed(seed, candidate["id"], f"judge-{order[0]}"),
                    kind="preference_judgment")
                vote = {"order": list(order), "request_id": digest(request), "request": request}
                judgments.append(vote)
                try:
                    response = validate_response(backend(request))
                    vote["response"] = response
                    if response["finish_reason"] != "stop":
                        raise ValueError("truncated judgment")
                    raw = response.get("answer_text", response["text"])
                    if raw.endswith(terminator):
                        raw = raw[:-len(terminator)]
                    decision = json.loads(raw)
                    if not isinstance(decision, dict) or decision.get("verdict") not in {"A", "B", "tie", "uncertain", "unsure"}:
                        raise ValueError("invalid preference judgment JSON verdict")
                    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                        raise ValueError("judgment must include a reason")
                    if any(type(decision.get(key)) is not bool for key in ("A_acceptable", "B_acceptable")):
                        raise ValueError("judgment must include boolean A_acceptable and B_acceptable")
                    verdict = decision["verdict"]
                    combined_label = "A" if order[0] == 1 else "B"
                    vote["combined_acceptable"] = decision[combined_label + "_acceptable"]
                    vote.update(decision=decision, outcome=(
                        "combined_wins" if order[0 if verdict == "A" else 1] == 1 else "ordinary_wins"
                    ) if verdict in {"A", "B"} else verdict)
                except Exception as error:
                    vote.update(outcome="error", error=f"{type(error).__name__}: {error}")
            vote_outcomes = [vote["outcome"] for vote in judgments]
            accepted = (vote_outcomes == ["combined_wins", "combined_wins"]
                        and all(vote.get("combined_acceptable") is True for vote in judgments)
                        and completions["combined"] != completions["ordinary"])
            outcome.update(accepted=accepted, status="accepted" if accepted else "not_unanimous_combined",
                           vote_outcomes=vote_outcomes)
            if accepted:
                metadata = {"schema_version": "crsecpo-pair-v1", "model": model_spec.to_dict(),
                    "prompt_version": PROMPT_VERSION, "rubric": RUBRIC,
                    "deployment_system": deployment_system, "seed": seed,
                    "generation_config": generation_config.to_dict(), "judge_config": judge_config.to_dict(),
                    "completion_terminator": terminator, "generations": generations, "judgments": judgments,
                    "completion_normalization": "append native terminator when absent; no other text changes",
                    "acceptance": "two_independent_swapped_orders_both_combined_wins_and_acceptable"}
                pairs.append({**row, "chosen": completions["combined"], "rejected": completions["ordinary"],
                    "prompt": generations["ordinary"]["request"]["rendered_prompt"],
                    "clean_prompt": generations["combined"]["request"]["rendered_prompt"],
                    "rubric_prompt": render_messages(tokenizer, answer_messages(candidate, "empty_source", deployment_system=deployment_system)),
                    "pair_metadata": metadata})
        outcomes.append(outcome)
    if output_path is not None:
        atomic_write_jsonl(pairs, output_path)
    if outcomes_path is not None:
        atomic_write_jsonl(outcomes, outcomes_path)
    return pairs, outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--max-candidates", type=_max_candidates, required=True,
                        help="Maximum candidate prefix (at most four model calls per candidate), or "
                             "'balanced' for twice the smaller class: every candidate that can enter "
                             "the balanced frozen selection, and no others")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args(argv)
    # Validate all inputs before importing an optional ML runtime or loading weights.
    manifest_path = args.output.with_suffix(args.output.suffix + ".collection.json")
    _validate_distinct_paths(candidates=args.candidates, output=args.output,
                             outcomes=args.outcomes, journal=args.journal, manifest=manifest_path)
    generation_config = GenerationConfig(args.max_new_tokens, args.temperature, args.top_p)
    judge_config = GenerationConfig(args.judge_max_new_tokens, args.temperature, args.top_p)
    rows = ordered_by_class_and_family(_validate_candidates(load_jsonl(args.candidates)), seed=args.seed)
    cap_rule = None
    if args.max_candidates == "balanced":
        cap_rule, args.max_candidates = "balanced", balanced_candidate_cap(rows)
    spec = ModelSpec(args.model_id, args.model_revision, args.tokenizer_id, args.tokenizer_revision)
    from .records import sha256
    manifest = {"schema_version": "crsecpo-collection-v1", "input_sha256": sha256(args.candidates),
        "destinations": {name: os.path.relpath(path.resolve(), manifest_path.parent.resolve())
                         for name, path in (("outcomes", args.outcomes), ("journal", args.journal))},
        "seed": args.seed, "max_candidates": args.max_candidates, "model": spec.to_dict(),
        "generation_config": generation_config.to_dict(), "judge_config": judge_config.to_dict(),
        "prompt_version": PROMPT_VERSION,
        "scheduled_ids": [row["id"] for row in rows[:args.max_candidates]],
        "deferred_ids": [row["id"] for row in rows[args.max_candidates:]],
        "maximum_model_calls": 4 * min(len(rows), args.max_candidates)}
    if cap_rule is not None:
        manifest["max_candidates_rule"] = cap_rule
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("collection settings changed; choose new export paths and retain the journal")
    else:
        if args.output.exists() or args.outcomes.exists():
            raise ValueError("existing exports have no collection manifest; choose new export paths")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    rows = rows[:args.max_candidates]
    backend = HFBackend(spec, device=args.device)
    if getattr(backend.model.config, "model_type", None) != "llama":
        raise ValueError("CR-SecPO pair collection requires the pinned frozen Llama checkpoint")
    pairs, outcomes = generate_pairs(rows, backend=JournalBackend(backend, args.journal),
        tokenizer=backend.tokenizer, model_spec=spec, seed=args.seed,
        generation_config=generation_config,
        judge_config=judge_config,
        output_path=args.output, outcomes_path=args.outcomes)
    print(json.dumps({"candidates": len(outcomes), "accepted_pairs": len(pairs)}))


if __name__ == "__main__":
    main()
