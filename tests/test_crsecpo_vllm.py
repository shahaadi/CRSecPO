"""CPU-only tests for batched (vLLM) pair collection; no GPU, vLLM install or weights needed."""
import json
import sys
import types
from types import SimpleNamespace

import pytest

from secpo.crsecpo.inference import (JournalBackend, VLLMBackend, effective_hf_sampling,
                                     make_request, GenerationConfig)
from secpo.crsecpo.pairs import generate_pairs, generate_pairs_batched
from secpo.crsecpo.prompts import digest, validate_training_pair

from test_crsecpo_pairs import SPEC, Tokenizer, _cli_args, candidate

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
    def __init__(self, fail_on_call=None):
        self.calls, self.fail_on_call = [], fail_on_call
    def generate_many(self, requests):
        self.calls.append([digest(request) for request in requests])
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("engine crashed")
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
    return generate_pairs(candidates, backend=respond, **common(identity))


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
    # Every request was generated exactly once, in two waves (answers, then judgments).
    generated = [request_id for call in backend.calls for request_id in call]
    assert len(generated) == len(set(generated))
    sizes = [len(call) for call in backend.calls]
    assert all(size <= 7 for size in sizes)
    assert len(generated) == 2 * len(candidates) + 2 * sum(
        len(outcome["generations"]) == 2 and all("response" in g and g["response"]["finish_reason"] == "stop"
                                                  for g in outcome["generations"].values())
        for outcome in expected_outcomes)


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


def test_interrupted_batch_resumes_without_regenerating(tmp_path):
    candidates = rows()
    path = tmp_path / "journal.jsonl"
    crashing = BatchBackend(fail_on_call=3)
    with pytest.raises(RuntimeError, match="engine crashed"):
        generate_pairs_batched(candidates, backend=crashing, journal=JournalBackend(None, path),
                               chunk_size=5, **common())
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


# ---- CLI ----------------------------------------------------------------------------

def test_cli_vllm_path_records_identity_and_never_mixes_with_hf(tmp_path, monkeypatch):
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text("".join(json.dumps(row) + "\n" for row in rows(6)))

    class FakeVLLM(BatchBackend):
        identity_for = staticmethod(lambda spec, max_model_len: IDENTITY)
        def __init__(self, spec, *, max_model_len, gpu_memory_utilization, identity):
            super().__init__()
            assert identity == IDENTITY
            self.tokenizer, self.model_type = Tokenizer(), "llama"
            created.append(self)
    created = []
    monkeypatch.setattr(module, "VLLMBackend", FakeVLLM)
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("hf must not load"))
    args = _cli_args(paths) + ["--backend", "vllm", "--batch-size", "4", "--seed", "42"]
    module.main(args)
    manifest = json.loads(paths["output"].with_suffix(".jsonl.collection.json").read_text())
    assert manifest["backend_identity"] == IDENTITY
    assert all(len(call) <= 4 for call in created[0].calls)
    output = paths["output"].read_bytes()
    module.main(args)  # resume replays everything
    assert created[1].calls == [] and paths["output"].read_bytes() == output
    with pytest.raises(ValueError, match="collection settings changed"):
        module.main(_cli_args(paths) + ["--seed", "42"])  # same paths, default hf backend


@pytest.mark.parametrize("extra", [["--batch-size", "0"], ["--vllm-gpu-memory-utilization", "0"],
                                   ["--backend", "other"]])
def test_cli_rejects_bad_vllm_options_before_loading(tmp_path, monkeypatch, extra):
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    monkeypatch.setattr(module, "VLLMBackend", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(SystemExit):
        module.main(_cli_args(paths) + ["--backend", "vllm"] + extra)
