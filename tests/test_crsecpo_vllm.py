"""CPU-only tests for batched (vLLM) pair collection; no GPU, vLLM install or weights needed."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import pytest

from secpo.crsecpo.inference import (BatchEngineUnavailable, JournalBackend, VLLMBackend, effective_hf_sampling,
                                     make_request, GenerationConfig)
from secpo.crsecpo.pairs import MAX_OUTPUT_RETRIES, generate_pairs, generate_pairs_batched
from secpo.crsecpo.prompts import digest, validate_training_pair
from secpo.crsecpo.vllm_worker import VLLMWorker

from test_crsecpo_pairs import SPEC, Tokenizer, _cli_args, candidate, make_legacy_export

IDENTITY = {"schema_version": "crsecpo-vllm-backend-v1", "engine": "vllm", "engine_version": "0.0-test",
            "dtype": "bfloat16", "max_model_len": 8192,
            "sampling_defaults": {"top_k": 50, "repetition_penalty": 1.0, "min_p": 0.0}}


def respond(request):
    """Deterministic fake model: a pure function of the request, so any driver must agree."""
    case = request["messages"][-1]["content"]
    marker = digest(case)
    if request["kind"] in {"ordinary", "combined"}:
        if "TRUNCATE" in case:
            return {"text": "partial", "finish_reason": "length"}
        if "EXPLODE" in case and request["kind"] == "combined":
            raise ValueError("backend exploded")
        return {"text": f"{request['kind']} answer {marker[:6]}<|eot_id|>", "finish_reason": "stop"}
    display = json.loads(case)
    # Mix of outcomes: most rows favor combined, some favor ordinary, some tie or are malformed.
    pick = int(digest(display["A"], display["B"])[:2], 16) % 5
    if pick == 4:
        return {"text": "not json", "finish_reason": "stop"}
    if pick == 3:
        verdict = "tie"
    else:
        winner = "ordinary" if pick == 2 else "combined"
        verdict = "A" if display["A"].startswith(winner) else "B"
    return {"text": json.dumps({"verdict": verdict, "reason": "substantive advantage",
                                "A_acceptable": True, "B_acceptable": True}), "finish_reason": "stop"}


class BatchBackend:
    def __init__(self, fail_on_call=None, failure=None):
        self.calls, self.fail_on_call = [], fail_on_call
        self.failure = failure
    def generate_many(self, requests):
        self.calls.append([digest(request) for request in requests])
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise self.failure
        results = []
        for request in requests:
            assert request["backend_identity"] == IDENTITY
            try:
                results.append(respond(request))
            except Exception as error:  # noqa: BLE001
                results.append(error)
        return results


def rows(count=24):
    out = []
    for index in range(count):
        target = f"TARGET {index}" + (" TRUNCATE" if index == 5 else "") + (" EXPLODE" if index == 7 else "")
        harmful = index % 2 == 0
        out.append(candidate(id=f"c{index}", family_id=f"f{index}", target=target,
                             source=f"SOURCE {index}",
                             target_policy="prohibited" if harmful else "permissible",
                             selection_group="harmful_benign_framing" if harmful else "benign_adversarial_control"))
    return out


def common(identity=IDENTITY):
    return dict(tokenizer=Tokenizer(), model_spec=SPEC, seed=42, backend_identity=identity)


def sequential(candidates, identity=IDENTITY):
    return generate_pairs(candidates, backend=respond, output_retries=MAX_OUTPUT_RETRIES, **common(identity))


def test_batched_cached_collection_cannot_publish_after_journal_loss(tmp_path):
    path = tmp_path / "requests.jsonl"
    candidates = rows()
    generate_pairs_batched(candidates, backend=BatchBackend(), journal=JournalBackend(None, path), **common())
    journal = JournalBackend(None, path)
    path.unlink()
    pairs, outcomes = tmp_path / "pairs.jsonl", tmp_path / "outcomes.jsonl"
    with pytest.raises(ValueError, match="journal changed after loading"):
        generate_pairs_batched(candidates, journal=journal,
                               start_backend=lambda: pytest.fail("must not start engine"),
                               output_path=pairs, outcomes_path=outcomes, **common())
    assert not pairs.exists() and not outcomes.exists()


def test_batched_driver_matches_sequential_byte_for_byte(tmp_path):
    candidates = rows()
    expected_pairs, expected_outcomes = sequential(candidates)
    backend = BatchBackend()
    journal = JournalBackend(None, tmp_path / "journal.jsonl")
    pairs, outcomes = generate_pairs_batched(candidates, backend=backend, journal=journal,
                                             chunk_size=7, **common())
    assert pairs and len(pairs) < len(candidates)  # the fixture exercises accepted and rejected rows
    statuses = {outcome["status"] for outcome in expected_outcomes}
    assert {"accepted", "not_unanimous_combined", "generation_error"} <= statuses
    assert pairs == expected_pairs
    assert outcomes == expected_outcomes
    # Each distinct sample, including a format retry, is generated once.
    generated = [request_id for call in backend.calls for request_id in call]
    assert len(generated) == len(set(generated))
    sizes = [len(call) for call in backend.calls]
    assert all(size <= 7 for size in sizes)
    samples = [sample for outcome in expected_outcomes
               for record in [*outcome["generations"].values(), *outcome["judgments"]]
               for sample in [*record.get("prior_samples", ()), record]]
    assert set(generated) == {sample["request_id"] for sample in samples}


def test_batched_pairs_validate_for_training_only_with_their_engine_identity(tmp_path):
    journal = JournalBackend(None, tmp_path / "journal.jsonl")
    pairs, _ = generate_pairs_batched(rows(), backend=BatchBackend(), journal=journal, **common())
    assert pairs[0]["pair_metadata"]["backend_identity"] == IDENTITY
    validate_training_pair(pairs[0], Tokenizer(), "llama", "a" * 40)
    stripped = json.loads(json.dumps(pairs[0]))
    del stripped["pair_metadata"]["backend_identity"]
    with pytest.raises(ValueError):
        validate_training_pair(stripped, Tokenizer(), "llama", "a" * 40)


def test_no_identity_keeps_requests_and_metadata_unchanged():
    pairs, outcomes = sequential(rows(), identity=None)
    assert "backend_identity" not in pairs[0]["pair_metadata"]
    for outcome in outcomes:
        for generation in outcome["generations"].values():
            assert "backend_identity" not in generation["request"]


def test_failed_chunk_is_journaled_and_retried_in_smaller_batches(tmp_path):
    path = tmp_path / "journal.jsonl"
    candidates = rows(4)
    backend = BatchBackend(fail_on_call=1, failure=RuntimeError("batch failed"))
    result = generate_pairs_batched(candidates, backend=backend, journal=JournalBackend(None, path),
                                    chunk_size=4, **common())
    assert result == sequential(candidates)
    assert backend.calls[0] == backend.calls[1] + backend.calls[2]
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    for request_id in backend.calls[0]:
        attempts = [entry for entry in entries if entry["request_id"] == request_id]
        assert [(entry["attempt"], entry["status"], entry["batch_size"]) for entry in attempts] == [
            (1, "error", 4), (2, "ok", 2)]
    assert JournalBackend(None, path).attempt_counts[backend.calls[0][0]] == 2


def test_batch_recovery_isolates_a_persistently_failing_request(tmp_path):
    def generate_many(requests):
        if any(item["seed"] == 12 for item in requests):
            raise ValueError("bad request")
        return [{"text": "ok", "finish_reason": "stop"} for _ in requests]

    requests = [request(seed=seed) for seed in (11, 12, 13, 14)]
    journal = JournalBackend(None, tmp_path / "journal.jsonl")
    failures = journal.record_batch(requests, generate_many)
    assert list(failures) == [digest(requests[1])]
    assert str(failures[digest(requests[1])]) == "bad request"
    assert journal.completed(requests[1]) is None
    assert all(journal.completed(item) is not None for item in (requests[0], *requests[2:]))


@pytest.mark.parametrize("stage", ["generation", "judgment"])
def test_all_failed_stage_preserves_journal_and_existing_exports(tmp_path, stage):
    class FailingBackend(BatchBackend):
        def generate_many(self, requests):
            current_stage = "judgment" if requests[0]["kind"] == "preference_judgment" else "generation"
            if current_stage == stage:
                raise RuntimeError("systematic engine failure")
            return super().generate_many(requests)

    candidates = rows(4)
    path = tmp_path / "journal.jsonl"
    pair_path, outcome_path = tmp_path / "pairs.jsonl", tmp_path / "outcomes.jsonl"
    pair_path.write_text("original pairs")
    kwargs = dict(chunk_size=8, output_path=pair_path, outcomes_path=outcome_path, **common())
    with pytest.raises(RuntimeError, match=f"8/8 {stage} requests"):
        generate_pairs_batched(candidates, backend=FailingBackend(), journal=JournalBackend(None, path),
                               **kwargs)

    assert pair_path.read_text() == "original pairs"
    assert not outcome_path.exists()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({entry["request_id"] for entry in entries if entry["status"] == "error"}) == 8
    recovered = BatchBackend()
    assert generate_pairs_batched(candidates, backend=recovered, journal=JournalBackend(None, path),
                                  **kwargs) == sequential(candidates)
    if stage == "judgment":
        assert len(recovered.calls[0]) == 8  # Only judgments are pending.
        assert set(recovered.calls[0]) == {entry["request_id"] for entry in entries if entry["status"] == "error"}


def test_no_complete_answer_pairs_fails_before_export(tmp_path):
    class PartialBackend(BatchBackend):
        def generate_many(self, requests):
            return [ValueError("combined generation failed") if item["kind"] == "combined"
                    else respond(item) for item in requests]

    path = tmp_path / "pairs.jsonl"
    with pytest.raises(RuntimeError, match="no candidate had two usable answers.*2/4 generation errors"):
        generate_pairs_batched(rows(2), backend=PartialBackend(),
                               journal=JournalBackend(None, tmp_path / "journal.jsonl"),
                               output_path=path, **common())
    assert not path.exists()


def test_partially_judged_candidates_cannot_complete_collection(tmp_path):
    class PartialJudgeBackend(BatchBackend):
        def generate_many(self, requests):
            if requests[0]["kind"] != "preference_judgment":
                return super().generate_many(requests)
            vote = {"verdict": "tie", "reason": "both answers equivalent",
                    "A_acceptable": True, "B_acceptable": True}
            response = {"text": json.dumps(vote), "finish_reason": "stop"}
            return [response, ValueError("second vote failed")] * (len(requests) // 2)

    path = tmp_path / "journal.jsonl"
    pair_path = tmp_path / "pairs.jsonl"
    outcome_path = tmp_path / "outcomes.jsonl"
    kwargs = dict(output_path=pair_path, outcomes_path=outcome_path, **common())
    with pytest.raises(RuntimeError, match="no candidate had two valid judgments.*2/4 judgment errors"):
        generate_pairs_batched(rows(2), backend=PartialJudgeBackend(),
                               journal=JournalBackend(None, path), **kwargs)
    assert not pair_path.exists() and not outcome_path.exists()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    failed = [entry["request_id"] for entry in entries if entry["status"] == "error"]

    recovered = BatchBackend()
    _, outcomes = generate_pairs_batched(rows(2), backend=recovered,
                                          journal=JournalBackend(None, path), **kwargs)
    assert recovered.calls[0] == failed
    assert any(row["status"] == "not_unanimous_combined" for row in outcomes)


@pytest.mark.parametrize("first_error", ["nonfinite", "duplicate_key"])
def test_recovered_samples_keep_valid_training_provenance(tmp_path, first_error):
    class RecoveringBackend:
        def __init__(self):
            self.seeds_by_prompt = {}

        def generate_many(self, requests):
            responses = []
            for item in requests:
                key = item["kind"], item["rendered_prompt"]
                seeds = self.seeds_by_prompt.setdefault(key, [])
                seeds.append(item["seed"])
                if item["kind"] == "preference_judgment":
                    display = json.loads(item["messages"][-1]["content"])
                    decision = {"verdict": "A" if display["A"].startswith("combined") else "B",
                                "reason": "combined is better", "A_acceptable": True, "B_acceptable": True}
                    text = json.dumps(decision)
                    if len(seeds) == 1:
                        if first_error == "duplicate_key":
                            key = decision["verdict"] + "_acceptable"
                            text = '{"' + key + '":false,' + text[1:]
                        else:
                            text = text[:-1] + ',"extra":NaN}'
                    elif len(seeds) == 2:
                        text = text.replace("combined is better", r"\ud800")
                    responses.append({"text": text, "finish_reason": "stop"})
                else:
                    responses.append({"text": item["kind"], "finish_reason": "length" if len(seeds) == 1 else "stop"})
            return responses

    backend = RecoveringBackend()
    journal = JournalBackend(None, tmp_path / "journal.jsonl")
    pairs, _ = generate_pairs_batched(rows(1), backend=backend, journal=journal,
                                      output_path=tmp_path / "pairs.jsonl",
                                      outcomes_path=tmp_path / "outcomes.jsonl", **common())
    (pair,) = pairs
    assert sorted(len(seeds) for seeds in backend.seeds_by_prompt.values()) == [2, 2, 3, 3]
    for vote in pair["pair_metadata"]["judgments"]:
        assert len(vote["prior_samples"]) == 2
        first, second = vote["prior_samples"]
        expected = ("duplicate judgment key" if first_error == "duplicate_key"
                    else "judgment contains non-finite numbers or invalid Unicode")
        assert expected in first["error"]
        assert "judgment contains non-finite numbers or invalid Unicode" in second["error"]
    assert all(len(seeds) == len(set(seeds)) for seeds in backend.seeds_by_prompt.values())
    assert len(journal.entries) == 10
    validate_training_pair(pair, Tokenizer(), "llama", "a" * 40)

    pair["pair_metadata"]["judgments"][0]["prior_samples"].pop(0)
    with pytest.raises(ValueError, match="pair differs from accepted recorded"):
        validate_training_pair(pair, Tokenizer(), "llama", "a" * 40)


def test_output_retry_limit_survives_resume(tmp_path):
    class MalformedJudgeBackend(BatchBackend):
        def generate_many(self, requests):
            if requests[0]["kind"] == "preference_judgment":
                return [{"text": "invalid JSON", "finish_reason": "stop"} for _ in requests]
            return super().generate_many(requests)

    path = tmp_path / "journal.jsonl"
    journal = JournalBackend(None, path)
    with pytest.raises(RuntimeError, match="2/2 judgment requests"):
        generate_pairs_batched(rows(1), backend=MalformedJudgeBackend(), journal=journal, **common())
    samples = [row for row in journal.entries.values() if row["request"]["kind"] == "preference_judgment"]
    assert len(samples) == 6

    resumed = BatchBackend()
    with pytest.raises(RuntimeError, match="2/2 judgment requests"):
        generate_pairs_batched(rows(1), backend=resumed, journal=JournalBackend(None, path), **common())
    assert resumed.calls == []


def test_valid_rejections_can_export_zero_pairs(tmp_path):
    class TiedBackend(BatchBackend):
        def generate_many(self, requests):
            if requests[0]["kind"] == "preference_judgment":
                vote = {"verdict": "tie", "reason": "both answers equivalent",
                        "A_acceptable": True, "B_acceptable": True}
                return [{"text": json.dumps(vote), "finish_reason": "stop"} for _ in requests]
            return super().generate_many(requests)

    output_path = tmp_path / "pairs.jsonl"
    pairs, outcomes = generate_pairs_batched(rows(2), backend=TiedBackend(),
                                             journal=JournalBackend(None, tmp_path / "journal.jsonl"),
                                             output_path=output_path, **common())
    assert pairs == [] and output_path.read_text() == ""
    assert all(row["vote_outcomes"] == ["tie", "tie"] for row in outcomes)


def test_interrupted_batch_resumes_without_regenerating(tmp_path):
    candidates = rows()
    path = tmp_path / "journal.jsonl"
    crashing = BatchBackend(fail_on_call=3, failure=BatchEngineUnavailable("engine crashed"))
    with pytest.raises(BatchEngineUnavailable, match="engine crashed"):
        generate_pairs_batched(candidates, backend=crashing, journal=JournalBackend(None, path),
                               chunk_size=5, **common())
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [entry["request_id"] for entry in entries if entry["status"] == "error"] == crashing.calls[2]
    done_before = sum(len(call) for call in crashing.calls[:2])
    resumed = BatchBackend()
    pairs, outcomes = generate_pairs_batched(candidates, backend=resumed, journal=JournalBackend(None, path),
                                             chunk_size=5, **common())
    regenerated = {request_id for call in resumed.calls for request_id in call}
    assert not regenerated & {request_id for call in crashing.calls[:2] for request_id in call}
    assert done_before == 10
    reference = generate_pairs_batched(candidates, backend=BatchBackend(),
                                       journal=JournalBackend(None, tmp_path / "fresh.jsonl"), **common())
    assert (pairs, outcomes) == reference


def test_failed_request_is_journaled_once_then_retried_on_resume(tmp_path):
    path = tmp_path / "journal.jsonl"
    backend = BatchBackend()
    generate_pairs_batched(rows(), backend=backend, journal=JournalBackend(None, path), **common())
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    errors = [entry for entry in entries if entry["status"] == "error"]
    assert len(errors) == 1 and errors[0]["error"] == "ValueError: backend exploded"
    assert all(entry["batch_size"] >= 1 for entry in entries)
    again = BatchBackend()
    generate_pairs_batched(rows(), backend=again, journal=JournalBackend(None, path), **common())
    assert [len(call) for call in again.calls] == [1]  # only the failed request is retried
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(entry["request_id"] == errors[0]["request_id"] for entry in entries) == 2


def test_record_batch_rejects_duplicate_or_completed_requests(tmp_path):
    journal = JournalBackend(None, tmp_path / "journal.jsonl")
    request = make_request([{"role": "user", "content": "hi"}], tokenizer=Tokenizer(), model_spec=SPEC,
                           config=GenerationConfig(), seed=1, kind="ordinary", backend_identity=IDENTITY)
    with pytest.raises(ValueError, match="distinct"):
        journal.record_batch([request, request], lambda requests: [])
    with pytest.raises(ValueError, match="number of results"):
        journal.record_batch([request], lambda requests: [])
    journal.record_batch([request], lambda requests: [{"text": "ok<|eot_id|>", "finish_reason": "stop"}])
    assert journal.completed(request)["text"] == "ok<|eot_id|>"
    with pytest.raises(ValueError, match="already be completed"):
        journal.record_batch([request], lambda requests: [{"text": "x", "finish_reason": "stop"}])
    reloaded = JournalBackend(None, tmp_path / "journal.jsonl")
    assert reloaded.completed(request)["text"] == "ok<|eot_id|>"


# ---- Sampling parity with HFBackend -------------------------------------------------

DEFAULTS = {"top_k": 50, "top_p": 1.0, "repetition_penalty": 1.0, "typical_p": 1.0, "num_beams": 1,
            "no_repeat_ngram_size": 0, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
            "encoder_repetition_penalty": 1.0, "min_length": 0, "bad_words_ids": None,
            "suppress_tokens": None, "begin_suppress_tokens": None, "forced_bos_token_id": None,
            "forced_eos_token_id": None, "exponential_decay_length_penalty": None}


def test_effective_sampling_uses_transformers_top_k_default_when_model_leaves_it_unset():
    llama = SimpleNamespace(temperature=0.6, top_p=0.9, do_sample=True, eos_token_id=[1, 2])
    assert effective_hf_sampling(llama, DEFAULTS) == {"top_k": 50, "repetition_penalty": 1.0, "min_p": 0.0}
    assert effective_hf_sampling(SimpleNamespace(top_k=20, min_p=0.05), DEFAULTS)["top_k"] == 20


def test_real_transformers_defaults_resolve_top_k_50():
    from transformers import GenerationConfig as HFGenerationConfig
    unset = HFGenerationConfig(temperature=0.6, top_p=0.9, do_sample=True)
    resolved = effective_hf_sampling(unset, HFGenerationConfig._get_default_generation_params())
    assert resolved == {"top_k": 50, "repetition_penalty": 1.0, "min_p": 0.0}


@pytest.mark.parametrize("field,value", [("typical_p", 0.9), ("num_beams", 4),
                                         ("suppress_tokens", [1, 2]), ("no_repeat_ngram_size", 3)])
def test_effective_sampling_rejects_settings_vllm_would_not_reproduce(field, value):
    with pytest.raises(ValueError, match=field):
        effective_hf_sampling(SimpleNamespace(**{field: value}), DEFAULTS)


# ---- VLLMBackend request/response handling, with a fake vllm module -----------------

class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_vllm(monkeypatch):
    module = types.ModuleType("vllm")
    module.SamplingParams = FakeSamplingParams
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    module.inputs = inputs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    return module


class TokenTokenizer(Tokenizer):
    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text) // 10 + 1))}
    def decode(self, tokens, **kwargs):
        return "".join("<|eot_id|>" if token == 7 else f"t{token}" for token in tokens)


class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.seen = outputs, []
    def generate(self, prompts, params, use_tqdm):
        self.seen.append((prompts, params))
        return [SimpleNamespace(outputs=[self.outputs.pop(0)]) for _ in prompts]


def make_backend(outputs, context_limit=None, max_model_len=8192):
    backend = object.__new__(VLLMBackend)
    backend.spec, backend.tokenizer, backend.identity = SPEC, TokenTokenizer(), IDENTITY
    backend.sampling_defaults = IDENTITY["sampling_defaults"]
    backend.stop_ids, backend.context_limit, backend.max_model_len = {7, 9}, context_limit, max_model_len
    backend.llm = FakeLLM(outputs)
    return backend


def request(temperature=0.7, max_new_tokens=64, identity=IDENTITY, seed=11):
    return make_request([{"role": "user", "content": "x" * 50}], tokenizer=Tokenizer(), model_spec=SPEC,
                        config=GenerationConfig(max_new_tokens, temperature, 0.95), seed=seed,
                        kind="ordinary", backend_identity=identity)


def output(tokens, finish="stop", stop_reason=None):
    return SimpleNamespace(token_ids=tokens, finish_reason=finish, stop_reason=stop_reason)


def test_sampling_params_match_hf_effective_settings_and_per_request_seed(fake_vllm):
    backend = make_backend([output([3, 4, 7]), output([5, 9])])
    sampled, greedy = backend.generate_many([request(seed=11), request(temperature=0.0, seed=12)])
    (_, params), = backend.llm.seen
    assert params[0].kwargs == {"n": 1, "max_tokens": 64, "seed": 11, "stop_token_ids": [7, 9],
                                "ignore_eos": False, "repetition_penalty": 1.0, "temperature": 0.7,
                                "top_p": 0.95, "top_k": 50, "min_p": 0.0}
    assert params[1].kwargs["temperature"] == 0.0 and "top_k" not in params[1].kwargs
    assert sampled["text"] == "t3t4<|eot_id|>" and sampled["answer_text"] == "t3t4"
    assert sampled["finish_reason"] == "stop" and sampled["completion_terminator"] == "<|eot_id|>"
    assert sampled["seed"] == 11 and sampled["backend_identity"] == IDENTITY
    assert greedy["completion_terminator"] == "t9" and greedy["generated_token_ids"] == [5, 9]


@pytest.mark.parametrize("queued,engine_dead", [(False, False), (True, False), (False, True)])
def test_vllm_batch_failure_checks_engine_before_retry(fake_vllm, monkeypatch, queued, engine_dead):
    class EngineDeadError(RuntimeError):
        pass

    exceptions = types.ModuleType("vllm.v1.engine.exceptions")
    exceptions.EngineDeadError = EngineDeadError
    monkeypatch.setitem(sys.modules, exceptions.__name__, exceptions)

    def fail(*args, **kwargs):
        if engine_dead:
            raise EngineDeadError("engine died")
        raise ValueError("generation failed")

    backend = make_backend([])
    backend.llm = SimpleNamespace(generate=fail,
                                  llm_engine=SimpleNamespace(has_unfinished_requests=lambda: queued))
    if engine_dead or queued:
        with pytest.raises(BatchEngineUnavailable, match="engine died" if engine_dead else "unfinished requests"):
            backend.generate_many([request()])
    else:
        with pytest.raises(ValueError, match="generation failed"):
            backend.generate_many([request()])


def test_stop_id_reported_but_not_retained_is_restored(fake_vllm):
    backend = make_backend([output([3, 4], stop_reason=7)])
    (response,) = backend.generate_many([request()])
    assert response["generated_token_ids"] == [3, 4, 7] and response["finish_reason"] == "stop"


def test_length_limit_and_unexpected_stops(fake_vllm):
    backend = make_backend([output([3, 4], finish="length"), output([3], stop_reason="STOP"),
                            output([3], finish="abort")])
    truncated, odd_stop, aborted = backend.generate_many([request(), request(seed=2), request(seed=3)])
    assert truncated["finish_reason"] == "length" and truncated["completion_terminator"] is None
    assert isinstance(odd_stop, ValueError) and isinstance(aborted, ValueError)


def test_invalid_requests_fail_individually_without_reaching_engine(fake_vllm):
    backend = make_backend([output([3, 7])], max_model_len=40)
    other = dict(IDENTITY, engine_version="other")
    good, wrong_engine, hf_request, too_long = backend.generate_many(
        [request(max_new_tokens=10), request(identity=other), request(identity=None),
         request(max_new_tokens=100, seed=4)])
    assert good["finish_reason"] == "stop"
    assert "identity" in str(wrong_engine) and "identity" in str(hf_request)
    assert "max_model_len" in str(too_long)
    assert len(backend.llm.seen[0][0]) == 1


class SyntheticWorkerBackend:
    model_type = "llama"

    def __init__(self, spec, *, stalled_seed=None):
        self.stalled_seed = stalled_seed

    @staticmethod
    def load_tokenizer(spec):
        return Tokenizer()

    def generate_many(self, requests):
        if any(item["seed"] == self.stalled_seed for item in requests):
            time.sleep(60)
        return [{"text": f"response {item['seed']}", "finish_reason": "stop"} for item in requests]


def test_vllm_worker_timeout_stops_process_and_preserves_journal(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    completed, stalled = {"seed": 1}, {"seed": 2}
    with VLLMWorker(SPEC, timeout_seconds=10, backend_factory=SyntheticWorkerBackend,
                    stalled_seed=2) as worker:
        journal = JournalBackend(None, journal_path)
        assert journal.record_batch([completed], worker.generate_many) == {}
        worker.timeout_seconds = 0.2
        with pytest.raises(BatchEngineUnavailable, match="batch timed out.*worker stopped"):
            journal.record_batch([stalled], worker.generate_many)
        assert not worker.process.is_alive()

    resumed = JournalBackend(None, journal_path)
    assert resumed.completed(completed)["text"] == "response 1"
    assert resumed.completed(stalled) is None
    attempts = list(resumed.entries.values())
    assert [attempt["status"] for attempt in attempts] == ["ok", "error"]
    assert "timed out" in attempts[-1]["error"]

    with VLLMWorker(SPEC, timeout_seconds=10, backend_factory=SyntheticWorkerBackend) as worker:
        assert resumed.record_batch([stalled], worker.generate_many) == {}
    assert resumed.completed(stalled)["text"] == "response 2"


@pytest.mark.parametrize("dies", ["collector", "worker"])
def test_vllm_watchdog_stops_engine_on_process_death(tmp_path, dies):
    driver = tmp_path / "collector.py"
    started = tmp_path / "worker.started"
    driver.write_text("""
import os
from pathlib import Path
import signal
import sys
import threading
import time

from secpo.crsecpo.vllm_worker import VLLMWorker


class StalledBackend:
    model_type = "llama"

    def __init__(self, spec):
        if sys.argv[2] == "worker" and os.fork() == 0:
            # An engine child can inherit the watchdog's control pipe.
            mark_started()
            time.sleep(60)
            os._exit(0)

    @staticmethod
    def load_tokenizer(spec):
        return None

    def generate_many(self, requests):
        mark_started()
        time.sleep(60)
        return []


def mark_started():
    temporary = Path(sys.argv[1]).with_suffix(".tmp")
    temporary.write_text(str(os.getpgrp()))
    temporary.rename(sys.argv[1])


def wait_until_started():
    while not Path(sys.argv[1]).exists():
        time.sleep(0.01)


def exit_during_generation():
    wait_until_started()
    os._exit(0)


if __name__ == "__main__":
    with VLLMWorker(None, timeout_seconds=30, backend_factory=StalledBackend) as worker:
        if sys.argv[2] == "collector":
            threading.Thread(target=exit_during_generation, daemon=True).start()
            worker.generate_many([{"seed": 1}])
        else:
            wait_until_started()
            os.kill(worker.process.pid, signal.SIGKILL)
            worker.process.join(timeout=2)
            assert worker.process.exitcode == -signal.SIGKILL
""")
    source = Path(__file__).resolve().parents[1] / "src"
    try:
        # Inherited stdout/stderr keep capture open until the worker has exited.
        subprocess.run([sys.executable, str(driver), str(started), dies],
                       capture_output=True, timeout=10, check=True,
                       env={**os.environ, "PYTHONPATH": str(source)})
    except subprocess.TimeoutExpired:
        if started.exists():
            try:
                os.killpg(int(started.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    assert started.is_file()


# ---- CLI ----------------------------------------------------------------------------

def test_cli_vllm_path_records_identity_and_never_mixes_with_hf(tmp_path, monkeypatch):
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text("".join(json.dumps(row) + "\n" for row in rows(6)))

    class FakeVLLM(BatchBackend):
        identity_for = staticmethod(lambda spec, max_model_len: IDENTITY)
        load_tokenizer = staticmethod(lambda spec: Tokenizer())
        def __init__(self, spec, *, max_model_len, gpu_memory_utilization, identity):
            super().__init__()
            assert identity == IDENTITY
            self.tokenizer, self.model_type = Tokenizer(), "llama"
            created.append(self)
    created = []
    monkeypatch.setattr(module, "VLLMBackend", FakeVLLM)
    @contextmanager
    def fake_worker(spec, *, timeout_seconds, **options):
        assert timeout_seconds == 12.5
        yield FakeVLLM(spec, **options)
    monkeypatch.setattr(module, "VLLMWorker", fake_worker)
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("hf must not load"))
    args = _cli_args(paths) + ["--backend", "vllm", "--batch-size", "4", "--seed", "42",
                               "--vllm-timeout-seconds", "12.5"]
    module.main(args)
    manifest_path = paths["output"].with_suffix(".jsonl.collection.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["backend_identity"] == IDENTITY
    assert manifest["max_output_retries"] == 2
    assert all(len(call) <= 4 for call in created[0].calls)
    output = paths["output"].read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(module, "VLLMWorker", lambda *a, **kw: pytest.fail("complete replay needs no engine"))
        module.main(args)
    assert len(created) == 1 and paths["output"].read_bytes() == output
    with pytest.raises(ValueError, match="collection settings changed"):
        module.main(_cli_args(paths) + ["--seed", "42"])  # same paths, default hf backend

    manifest_path.write_text(json.dumps(dict(manifest, max_output_retries=3)))
    with pytest.raises(ValueError, match="collection settings changed"):
        module.main(args)
    assert len(created) == 1 and paths["output"].read_bytes() == output

    manifest_path.write_text(json.dumps(manifest))
    entries = [json.loads(line) for line in paths["journal"].read_text().splitlines()]
    missing = next(entry for entry in entries if entry["status"] == "ok"
                   and entry["request"]["kind"] == "preference_judgment"
                   and '"verdict"' in entry["response"]["text"])
    paths["journal"].write_text("".join(json.dumps(entry) + "\n" for entry in entries
                                       if entry["request_id"] != missing["request_id"]))
    module.main(args)
    assert len(created) == 2 and created[1].calls == [[missing["request_id"]]]
    assert paths["output"].read_bytes() == output

    # Recreate a pre-retry run: its manifest has no retry policy and its journal
    # has only base samples, including responses with unusable output.
    make_legacy_export(paths)
    manifest.pop("max_output_retries")
    legacy_manifest = json.dumps(manifest)
    manifest_path.write_text(legacy_manifest)
    entries = [json.loads(line) for line in paths["journal"].read_text().splitlines()]
    assert any("output_retry" in entry["request"] for entry in entries)
    paths["journal"].write_text("".join(json.dumps(entry) + "\n" for entry in entries
                                       if "output_retry" not in entry["request"]))
    module.main(args)
    assert len(created) == 2
    assert manifest_path.read_text() == legacy_manifest
    outcomes = [json.loads(line) for line in paths["outcomes"].read_text().splitlines()]
    assert len(outcomes) == 6 and {row["max_output_retries"] for row in outcomes} == {0}


@pytest.mark.parametrize("extra", [["--batch-size", "0"], ["--vllm-gpu-memory-utilization", "0"],
                                   ["--vllm-timeout-seconds", "0"], ["--vllm-timeout-seconds", "nan"],
                                   ["--backend", "other"]])
def test_cli_rejects_bad_vllm_options_before_loading(tmp_path, monkeypatch, extra):
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    monkeypatch.setattr(module, "VLLMBackend", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(SystemExit):
        module.main(_cli_args(paths) + ["--backend", "vllm"] + extra)
