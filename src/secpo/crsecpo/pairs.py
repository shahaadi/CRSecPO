"""Frozen offline ordinary/combined preference collection, with no original completions."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import ExitStack, nullcontext
import fcntl
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

from secpo.evaluation.sep_adaptive import atomic_write_json
from secpo.evaluation.sep_judge import load_jsonl
from .artifact_versions import logical_artifact_path, version_root
from .collection_exports import (claim_journal_destination, collection_manifest_path, export_lock_path,
                                 export_record_path, lock_data_paths,
                                 export_reference_path,
                                 export_reservation_path, validate_data_path,
                                 reserve_collection_exports, write_collection_exports,
                                 write_standalone_export)
from .inference import (GenerationConfig, HFBackend, JournalBackend, JournalStateError, ModelSpec, VLLMBackend,
                        format_error, make_request, render_messages, validate_response)
from .prompts import (PROMPT_VERSION, RUBRIC, answer_messages, digest,
                      judging_orders, native_terminator)
from .vllm_worker import DEFAULT_ENGINE_TIMEOUT, VLLMWorker

MAX_OUTPUT_RETRIES = 2

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

    This legacy budget cap can defer candidates that would make it into the
    frozen selection after generation, judging and length checks.
    """
    harmful = sum(row["selection_group"] == "harmful_benign_framing" for row in rows)
    smaller = min(harmful, len(rows) - harmful)
    if smaller < 1:
        raise ValueError("a balanced candidate cap requires both harmful and benign candidates")
    return 2 * smaller


def _max_candidates(value):
    if value in {"all", "balanced"}:
        return value
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer, 'all', or 'balanced'") from None
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
    items = [(label, logical_artifact_path(path)) for label, path in paths.items()]
    for index, (left_label, left) in enumerate(items):
        if left.is_dir() and not left_label.endswith("_versions"):
            raise ValueError(f"{left_label} must use distinct files, not directories")
        for right_label, right in items[index + 1:]:
            if (left == right or left in right.parents or right in left.parents
                    or (left.exists() and right.exists() and left.samefile(right))
                    or _case_alias(left, right)):
                raise ValueError(f"{left_label} and {right_label} must use distinct files")


def _case_alias(left, right):
    folded_left, folded_right = (Path(str(path).casefold()) for path in (left, right))
    if (folded_left != folded_right and folded_left not in folded_right.parents
            and folded_right not in folded_left.parents):
        return False
    # Neither destination needs to exist yet. Probe the nearest existing
    # destination directory so the check follows the volume, not the OS.
    parent = next(parent for parent in right.parents if parent.is_dir())
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        probe = Path(temporary) / "alias-test"
        probe.touch()
        return probe.with_name("ALIAS-TEST").exists()


def _export_paths(**destinations):
    paths = {}
    for role, value in destinations.items():
        validate_data_path(value)
        path = logical_artifact_path(value)
        paths.update({role: path, f"{role}_record": export_record_path(path),
                      f"{role}_reference": export_reference_path(path),
                      f"{role}_lock": export_lock_path(path),
                      f"{role}_reservation": export_reservation_path(path),
                      f"{role}_manifest": collection_manifest_path(path),
                      f"{role}_versions": version_root(path)})
    return paths


def _export_collection(pairs, outcomes, output_path, outcomes_path, *, verify_source=None):
    destinations = {role: path for role, path in (("output", output_path), ("outcomes", outcomes_path))
                    if path is not None}
    paths = _export_paths(**destinations)
    _validate_distinct_paths(**paths)
    if output_path is not None and outcomes_path is not None:
        write_collection_exports(pairs, outcomes, paths["output"], paths["outcomes"],
                                 verify_source=verify_source)
    elif output_path is not None:
        write_standalone_export(pairs, paths["output"], verify_source=verify_source)
    elif outcomes_path is not None:
        write_standalone_export(outcomes, paths["outcomes"], verify_source=verify_source)


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


def _unique_judgment_object(fields):
    decision = {}
    for key, value in fields:
        if key in decision:
            raise ValueError(f"duplicate judgment key: {key!r}")
        decision[key] = value
    return decision


def _judge(response, terminator, order):
    if response["finish_reason"] != "stop":
        raise ValueError("truncated judgment")
    raw = response.get("answer_text", response["text"])
    if raw.endswith(terminator):
        raw = raw[:-len(terminator)]
    decision = json.loads(raw, object_pairs_hook=_unique_judgment_object)
    if not isinstance(decision, dict) or decision.get("verdict") not in {"A", "B", "tie", "uncertain", "unsure"}:
        raise ValueError("invalid preference judgment JSON verdict")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        raise ValueError("judgment must include a reason")
    if any(type(decision.get(key)) is not bool for key in ("A_acceptable", "B_acceptable")):
        raise ValueError("judgment must include boolean A_acceptable and B_acceptable")
    # JSON decoding permits non-finite numbers and escaped lone surrogates.
    # Reject them here so an unusable vote takes the normal output-retry path.
    try:
        json.dumps(decision, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except ValueError as error:
        raise ValueError("judgment contains non-finite numbers or invalid Unicode") from error
    verdict = decision["verdict"]
    combined_label = "A" if order[0] == 1 else "B"
    outcome = verdict
    if verdict in {"A", "B"}:
        outcome = "combined_wins" if order[0 if verdict == "A" else 1] == 1 else "ordinary_wins"
    return {"decision": decision, "outcome": outcome,
            "combined_acceptable": decision[combined_label + "_acceptable"]}


def _sample(request, backend, parse_output, output_retries):
    """Retry unusable model outputs with deterministic seeds and keep their provenance."""
    prior_samples = []
    initial_seed = request["seed"]
    for retry in range(output_retries + 1):
        attempt = dict(request, seed=(initial_seed + retry) % 2**32)
        if retry:
            attempt["output_retry"] = retry
        record = {"request_id": digest(attempt), "request": attempt}
        parsed = None
        try:
            record["response"] = validate_response(backend(attempt))
        except JournalStateError:
            raise
        except Exception as error:
            record["error"] = format_error(error)
            break  # Backend failures and deferred batched calls use the existing journal policy.
        try:
            parsed = parse_output(record["response"])
        except (ValueError, TypeError, AttributeError) as error:
            record["error"] = format_error(error)
            if retry < output_retries:
                prior_samples.append(record)
                continue
        break
    if prior_samples:
        record["prior_samples"] = prior_samples
    return record, parsed


def generate_pairs(candidates, *, backend, tokenizer, model_spec: ModelSpec,
                   generation_config: GenerationConfig = GenerationConfig(),
                   judge_config: GenerationConfig = GenerationConfig(max_new_tokens=256),
                   seed: int = 0, deployment_system: str = "",
                   output_path: str | Path | None = None,
                   outcomes_path: str | Path | None = None, backend_identity=None, output_retries=0,
                   judge=True, conditions=("ordinary", "combined")):
    """Return (accepted pairs, all outcomes), atomically exporting completed runs.

    Wrap ``backend`` in JournalBackend for per-request restart without resampling.
    Judge requests carry only rubric/target/anonymous answers. Their seeds differ
    by display order, making them separate samples even with deterministic seeds.
    ``backend_identity`` (e.g. the vLLM engine) enters every request's identity and
    the pair metadata; None leaves requests and metadata exactly as before.

    With ``judge=False`` no preference judge runs: the rubric-conditioned
    (combined) answer is taken as ``chosen`` and the ordinary answer as
    ``rejected`` whenever both generate and differ. Relies on the combined answer
    being generally better rather than verifying it per pair.
    """
    # Provenance replay also supports pairs collected with the earlier three-retry policy.
    if type(output_retries) is not int or not 0 <= output_retries <= 3:
        raise ValueError("output_retries must be an integer from 0 to 3")
    candidates = _validate_candidates(candidates)
    terminator = native_terminator(tokenizer)
    pairs, outcomes = [], []
    for candidate in candidates:
        row = {key: candidate[key] for key in CANDIDATE_FIELDS}
        generations, judgments, completions, answers = {}, [], {}, {}
        outcome = {**row, "accepted": False, "status": "generation_error",
                   "generations": generations, "judgments": judgments,
                   "completion_terminator": terminator, "max_output_retries": output_retries}
        for condition in conditions:
            request = make_request(answer_messages(candidate, condition, deployment_system=deployment_system),
                tokenizer=tokenizer, model_spec=model_spec, config=generation_config,
                seed=_seed(seed, candidate["id"], condition), kind=condition,
                backend_identity=backend_identity)
            result, parsed = _sample(request, backend, lambda response: _completion(response, terminator),
                                     output_retries)
            generations[condition] = result
            if parsed is not None:
                completions[condition], answers[condition] = parsed
        if "combined" not in conditions:
            # Student-only pre-pass (optimization): journal the ordinary answers so a later
            # both-conditions run on a subset replays them; form no pairs here.
            outcome["status"] = "student_only" if "ordinary" in completions else "generation_error"
            outcomes.append(outcome)
            continue
        if len(completions) == 2:
            if judge:
                for order, messages in judging_orders(candidate["target"], answers["ordinary"],
                                                      answers["combined"], comparison_id=candidate["id"]):
                    request = make_request(messages, tokenizer=tokenizer, model_spec=model_spec,
                        config=judge_config, seed=_seed(seed, candidate["id"], f"judge-{order[0]}"),
                        kind="preference_judgment", backend_identity=backend_identity)
                    vote, parsed = _sample(request, backend, lambda response: _judge(response, terminator, order),
                                           output_retries)
                    vote["order"] = list(order)
                    vote.update(parsed if parsed is not None else {"outcome": "error"})
                    judgments.append(vote)
                vote_outcomes = [vote["outcome"] for vote in judgments]
                accepted = (vote_outcomes == ["combined_wins", "combined_wins"]
                            and all(vote.get("combined_acceptable") is True for vote in judgments)
                            and completions["combined"] != completions["ordinary"])
                status = "not_unanimous_combined"
                if "error" in vote_outcomes:
                    status = "judgment_error"
                elif accepted:
                    status = "accepted"
                acceptance = "two_independent_swapped_orders_both_combined_wins_and_acceptable"
                outcome.update(accepted=accepted, status=status, vote_outcomes=vote_outcomes)
            else:
                accepted = completions["combined"] != completions["ordinary"]
                status = "accepted" if accepted else "combined_equals_ordinary"
                acceptance = "combined_over_ordinary_without_judge"
                outcome.update(accepted=accepted, status=status, vote_outcomes=[])
            if accepted:
                metadata = {"schema_version": "crsecpo-pair-v1", "model": model_spec.to_dict(),
                    "prompt_version": PROMPT_VERSION, "rubric": RUBRIC,
                    "deployment_system": deployment_system, "seed": seed,
                    "generation_config": generation_config.to_dict(), "judge_config": judge_config.to_dict(),
                    "max_output_retries": output_retries,
                    "completion_terminator": terminator, "generations": generations, "judgments": judgments,
                    "completion_normalization": "append native terminator when absent; no other text changes",
                    "acceptance": acceptance}
                if backend_identity is not None:
                    metadata["backend_identity"] = backend_identity
                pairs.append({**row, "chosen": completions["combined"], "rejected": completions["ordinary"],
                    "prompt": generations["ordinary"]["request"]["rendered_prompt"],
                    "clean_prompt": generations["combined"]["request"]["rendered_prompt"],
                    "rubric_prompt": render_messages(tokenizer, answer_messages(candidate, "empty_source", deployment_system=deployment_system)),
                    "pair_metadata": metadata})
        outcomes.append(outcome)
    snapshot = backend.publication_snapshot() if isinstance(backend, JournalBackend) else nullcontext()
    with snapshot as verify_source:
        _export_collection(pairs, outcomes, output_path, outcomes_path, verify_source=verify_source)
    return pairs, outcomes


def _collection_errors(outcomes):
    generations = [result for row in outcomes for result in row["generations"].values()]
    judgments = [vote for row in outcomes for vote in row["judgments"]]
    return {
        "generation_requests": len(generations),
        "generation_errors": sum("error" in result for result in generations),
        "judgment_requests": len(judgments),
        "judgment_errors": sum("error" in vote for vote in judgments),
        "fully_judged_candidates": sum(row["status"] in {"accepted", "not_unanimous_combined"}
                                       for row in outcomes),
    }


class _Deferred(Exception):
    """A planning-pass request that has not been generated yet."""


def generate_pairs_batched(candidates, *, journal, backend=None, start_backend=None, chunk_size=256,
                           output_path=None, outcomes_path=None, output_retries=MAX_OUTPUT_RETRIES, **kwargs):
    """Batched driver around the unchanged sequential ``generate_pairs``.

    Acceptance logic is never duplicated. Each wave dry-runs ``generate_pairs``
    with a planner that replays journaled responses and collects every request
    still missing (answers, output retries, and judgments for complete answer
    pairs). Those run in backend batches of ``chunk_size``, journaled after each
    chunk so an interruption loses at most one chunk. The final planning pass
    replays every completed response from the journal. When given a startup
    callback, it starts the backend only if a request is missing.

    Failed engine batches are journaled and split to isolate request failures.
    A terminal per-request failure is reported and retried on resume.
    The output retry limit applies across restarts.
    A run with no candidate judged successfully in both orders fails before exporting pairs.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if type(output_retries) is not int or not 0 <= output_retries <= MAX_OUTPUT_RETRIES:
        raise ValueError(f"vLLM output_retries must be an integer from 0 to {MAX_OUTPUT_RETRIES}")
    if (backend is None) == (start_backend is None):
        raise ValueError("provide either a backend or a backend startup callback")
    candidates = _validate_candidates(candidates)
    failed = {}

    def replay_or(on_missing):
        def call(request):
            request = json.loads(json.dumps(request, allow_nan=False))
            request_id = digest(request)
            if request_id in failed:
                raise failed[request_id]  # the original exception, as the sequential path records it
            response = journal.completed(request)
            if response is not None:
                return response
            return on_missing(request_id, request)
        return call

    while True:
        pending = {}
        def plan(request_id, request):
            pending.setdefault(request_id, request)
            raise _Deferred(request_id)
        pairs, outcomes = generate_pairs(candidates, backend=replay_or(plan),
                                         output_retries=output_retries, **kwargs)
        if not pending:
            break
        if backend is None:
            backend = start_backend()
        batch = list(pending.values())
        for start in range(0, len(batch), chunk_size):
            failed.update(journal.record_batch(batch[start:start + chunk_size], backend.generate_many))

    counts = _collection_errors(outcomes)
    # A student-only pre-pass (conditions without "combined") intentionally forms no pairs,
    # so the "no accepted pair / no judgment" aborts below do not apply to it.
    both = "combined" in kwargs.get("conditions", ("ordinary", "combined"))
    for stage in ("generation", "judgment"):
        requests = counts[f"{stage}_requests"]
        errors = counts[f"{stage}_errors"]
        if requests and errors == requests:
            raise RuntimeError(f"vLLM collection failed: {errors}/{requests} {stage} requests "
                               f"produced no usable results. Inspect {journal.path}.")
    if kwargs.get("judge", True) and both:
        if not counts["judgment_requests"]:
            raise RuntimeError(f"vLLM collection failed: no candidate had two usable answers "
                               f"({counts['generation_errors']}/{counts['generation_requests']} generation errors). "
                               f"Inspect {journal.path}.")
        if not counts["fully_judged_candidates"]:
            raise RuntimeError(f"vLLM collection failed: no candidate had two valid judgments "
                               f"({counts['judgment_errors']}/{counts['judgment_requests']} judgment errors). "
                               f"Inspect {journal.path}.")
    elif both and not any(row["accepted"] for row in outcomes):
        raise RuntimeError(f"vLLM collection failed: no candidate produced two differing answers "
                           f"({counts['generation_errors']}/{counts['generation_requests']} generation errors). "
                           f"Inspect {journal.path}.")
    with journal.publication_snapshot() as verify_source:
        _export_collection(pairs, outcomes, output_path, outcomes_path, verify_source=verify_source)
    return pairs, outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--max-candidates", type=_max_candidates, required=True,
                        help="Maximum candidate prefix, 'all' for the full input, or 'balanced' for the legacy budget cap "
                             f"(up to 4 base requests per candidate and {4 * (1 + MAX_OUTPUT_RETRIES)} vLLM samples "
                             "with output retries, excluding backend error recovery). 'balanced' schedules "
                             "twice the smaller class size, which may reduce usable pair yield")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-judge", dest="judge", action="store_false",
                        help="Skip the preference judge: accept chosen=combined/rejected=ordinary "
                             "whenever the two answers differ (relies on combined being generally better)")
    parser.set_defaults(judge=True)
    parser.add_argument("--conditions", choices=("both", "ordinary"), default="both",
                        help="both: generate ordinary+combined and form pairs (default). "
                             "ordinary: student-only pre-pass that journals the ordinary answers "
                             "(forms no pairs) so a later both-conditions run on a subset replays them")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf",
                        help="hf: sequential transformers generate (default). vllm: batched vLLM; "
                             "a separate experiment whose requests carry the engine identity")
    parser.add_argument("--vllm-max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-timeout-seconds", type=float, default=DEFAULT_ENGINE_TIMEOUT,
                        help="Wall-clock limit for vLLM startup or one engine submission (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="vLLM requests per journaled engine batch")
    args = parser.parse_args(argv)
    args.conditions = ("ordinary", "combined") if args.conditions == "both" else ("ordinary",)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not math.isfinite(args.vllm_timeout_seconds) or args.vllm_timeout_seconds <= 0:
        parser.error("--vllm-timeout-seconds must be positive and finite")
    if not 0 < args.vllm_gpu_memory_utilization <= 1:
        parser.error("--vllm-gpu-memory-utilization must be in (0, 1]")
    # Validate all inputs before importing an optional ML runtime or loading weights.
    validate_data_path(args.journal)
    export_paths = _export_paths(output=args.output, outcomes=args.outcomes)
    old_manifest = collection_manifest_path(args.output).absolute()
    args.output = export_paths["output"]
    args.outcomes = export_paths["outcomes"]
    manifest_path = export_paths["output_manifest"]
    args.journal = logical_artifact_path(args.journal)
    lock_path = args.journal.with_name(args.journal.name + ".lock")
    _validate_distinct_paths(candidates=args.candidates, journal=args.journal,
                             journal_lock=lock_path, journal_path_lock=export_lock_path(args.journal),
                             journal_reservation=export_reservation_path(args.journal), **export_paths)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        # The sidecar stays locked across journal creation and repair.
        _lock_journal(locks.enter_context(lock_path.open("a")), args.journal)
        locks.enter_context(lock_data_paths(args.journal))

        def lock_journal_file(path):
            _lock_journal(locks.enter_context(path.open("r+b")), args.journal)

        try:
            lock_journal_file(args.journal)
        except FileNotFoundError:
            journal_exists = False
        else:
            journal_exists = True
        _collect(args, manifest_path, old_manifest=old_manifest, lock_journal_file=lock_journal_file,
                 journal_exists=journal_exists)


def _lock_journal(handle, path):
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ValueError(f"another collector is using journal {path}") from error


def _collect(args, manifest_path, *, old_manifest, lock_journal_file, journal_exists):
    generation_config = GenerationConfig(args.max_new_tokens, args.temperature, args.top_p)
    judge_config = GenerationConfig(args.judge_max_new_tokens, args.temperature, args.top_p)
    try:
        candidate_payload = args.candidates.read_bytes()
    except OSError as error:
        raise ValueError(f"could not read input JSONL {args.candidates}: {error}") from error
    rows = ordered_by_class_and_family(
        _validate_candidates(load_jsonl(args.candidates, payload=candidate_payload)), seed=args.seed)
    input_hash = hashlib.sha256(candidate_payload).hexdigest()
    cap_rule = None
    if args.max_candidates == "all":
        cap_rule, args.max_candidates = "all", len(rows)
    elif args.max_candidates == "balanced":
        cap_rule, args.max_candidates = "balanced", balanced_candidate_cap(rows)
    spec = ModelSpec(args.model_id, args.model_revision, args.tokenizer_id, args.tokenizer_revision)
    backend_identity = None
    if args.backend == "vllm":
        # Computed from local files and the installed engine version, before loading weights.
        backend_identity = VLLMBackend.identity_for(spec, max_model_len=args.vllm_max_model_len)
    manifest = {"schema_version": "crsecpo-collection-v1", "input_sha256": input_hash,
        "destinations": {name: os.path.relpath(logical_artifact_path(path), manifest_path.parent.resolve())
                         for name, path in (("outcomes", args.outcomes), ("journal", args.journal))},
        "seed": args.seed, "max_candidates": args.max_candidates, "model": spec.to_dict(),
        "generation_config": generation_config.to_dict(), "judge_config": judge_config.to_dict(),
        "prompt_version": PROMPT_VERSION,
        "scheduled_ids": [row["id"] for row in rows[:args.max_candidates]],
        "deferred_ids": [row["id"] for row in rows[args.max_candidates:]],
        "maximum_distinct_requests": (len(args.conditions)
            + (2 if (args.judge and len(args.conditions) == 2) else 0)) * min(len(rows), args.max_candidates)}
    # Only recorded when non-default, so existing both-conditions manifests compare unchanged.
    if list(args.conditions) != ["ordinary", "combined"]:
        manifest["conditions"] = list(args.conditions)
    if cap_rule is not None:
        manifest["max_candidates_rule"] = cap_rule
    if not args.judge:
        # Absent on judged runs, so existing manifests compare unchanged; a judge-free
        # resume cannot silently reuse a judged journal and vice versa.
        manifest["judge"] = False
    if backend_identity is not None:
        manifest["backend_identity"] = backend_identity
        manifest["max_output_retries"] = MAX_OUTPUT_RETRIES

    def prepare_manifest():
        if manifest_path.exists():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            _collection_retry_limit(recorded, manifest, backend_identity)
            return recorded, None
        initial = manifest
        if old_manifest != manifest_path and old_manifest.exists():
            initial = json.loads(old_manifest.read_text(encoding="utf-8"))
            initial["destinations"] = {
                name: os.path.relpath(logical_artifact_path(old_manifest.parent / path), manifest_path.parent)
                for name, path in initial["destinations"].items()
            }
            _collection_retry_limit(initial, manifest, backend_identity)
        elif args.output.exists() or args.outcomes.exists():
            raise ValueError("existing exports have no collection manifest; choose new export paths")

        def publish():
            with tempfile.TemporaryDirectory(dir=manifest_path.parent) as temporary:
                staged = Path(temporary) / manifest_path.name
                atomic_write_json(staged, initial)
                os.link(staged, manifest_path)

        return initial, publish

    if manifest_path.exists():
        # Report changed settings before inspecting a newly specified journal.
        prepare_manifest()
    journal = JournalBackend(None, args.journal, lock_replacement=lock_journal_file)
    recorded = reserve_collection_exports(args.output, args.outcomes, prepare_manifest=prepare_manifest)
    output_retries = _collection_retry_limit(recorded, manifest, backend_identity)
    claim_journal_destination(args.journal)
    if not journal_exists:
        # Publish only after taking the inode lock, so a new alias cannot race startup.
        with tempfile.TemporaryDirectory(dir=args.journal.parent) as temporary:
            staged = Path(temporary) / args.journal.name
            staged.touch()
            lock_journal_file(staged)
            os.link(staged, args.journal)
    rows = rows[:args.max_candidates]
    if args.backend == "vllm":
        tokenizer = VLLMBackend.load_tokenizer(spec)
        with ExitStack() as engine:
            def start_backend():
                backend = engine.enter_context(VLLMWorker(
                    spec, timeout_seconds=args.vllm_timeout_seconds,
                    max_model_len=args.vllm_max_model_len,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    identity=backend_identity))
                if backend.model_type != "llama":
                    raise ValueError("CR-SecPO pair collection requires the pinned frozen Llama checkpoint")
                return backend

            pairs, outcomes = generate_pairs_batched(rows, start_backend=start_backend,
                journal=journal, chunk_size=args.batch_size,
                tokenizer=tokenizer, model_spec=spec, seed=args.seed,
                generation_config=generation_config, judge_config=judge_config,
                backend_identity=backend_identity, output_retries=output_retries, judge=args.judge,
                conditions=args.conditions)
        with journal.publication_snapshot() as verify_source:
            write_collection_exports(pairs, outcomes, args.output, args.outcomes, manifest=recorded,
                                     verify_source=verify_source)
        print(json.dumps({"candidates": len(outcomes), "accepted_pairs": len(pairs),
                          **_collection_errors(outcomes)}))
        return
    backend = HFBackend(spec, device=args.device)
    if getattr(backend.model.config, "model_type", None) != "llama":
        raise ValueError("CR-SecPO pair collection requires the pinned frozen Llama checkpoint")
    journal.backend = backend
    pairs, outcomes = generate_pairs(rows, backend=journal,
        tokenizer=backend.tokenizer, model_spec=spec, seed=args.seed,
        generation_config=generation_config,
        judge_config=judge_config, judge=args.judge, conditions=args.conditions)
    with journal.publication_snapshot() as verify_source:
        write_collection_exports(pairs, outcomes, args.output, args.outcomes, manifest=recorded,
                                 verify_source=verify_source)
    print(json.dumps({"candidates": len(outcomes), "accepted_pairs": len(pairs)}))


def _collection_retry_limit(recorded, manifest, backend_identity):
    recorded, manifest = dict(recorded), dict(manifest)
    if "maximum_model_calls" in recorded and "maximum_distinct_requests" not in recorded:
        recorded["maximum_distinct_requests"] = recorded.pop("maximum_model_calls")
    output_retries = MAX_OUTPUT_RETRIES
    if backend_identity is not None and "backend_identity" in recorded and "max_output_retries" not in recorded:
        # Before this field existed, vLLM made no additional samples for unusable outputs.
        manifest.pop("max_output_retries")
        output_retries = 0
    if recorded != manifest:
        raise ValueError("collection settings changed; choose new export paths and retain the journal")
    return output_retries


if __name__ == "__main__":
    main()
