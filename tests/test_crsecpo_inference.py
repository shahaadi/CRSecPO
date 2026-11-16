"""Journal and optional HF integration tested without loading weights."""
import json
import sys
from types import SimpleNamespace

import pytest

from secpo.crsecpo.inference import HFBackend, JournalBackend, ModelSpec, validate_response


def test_errors_checkpoint_and_retry_on_resume(tmp_path):
    path = tmp_path / "journal.jsonl"
    request = {"messages": [], "model": {"revision": "a" * 40}, "seed": 3}
    def failing(_):
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError):
        JournalBackend(failing, path)(request)
    entry = json.loads(path.read_text())
    assert entry["status"] == "error" and entry["request"] == request
    recovered = JournalBackend(lambda _: {"text": "ok", "finish_reason": "stop"}, path)
    assert recovered(request)["text"] == "ok"
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [entry["status"] for entry in entries] == ["error", "ok"]
    assert [entry["attempt"] for entry in entries] == [1, 2]
    assert all(entry["elapsed_seconds"] >= 0 for entry in entries)
    assert entries[0] == entry
    def forbidden(_):
        raise AssertionError("successful request cannot resample")
    assert JournalBackend(forbidden, path)(request)["text"] == "ok"
    assert len(path.read_text().splitlines()) == 2


def test_journal_fails_closed_on_corrupted_request_hash(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text(json.dumps({"request_id": "bad", "request": {}, "status": "ok",
                                "response": {"text": "ok", "finish_reason": "stop"}}))
    with pytest.raises(ValueError, match="hash mismatch"):
        JournalBackend(lambda _: None, path)


@pytest.mark.parametrize("tail", [b'{"request_id":"unfinished",', b'{"text":"split \xe2\x82'])
def test_torn_final_journal_write_preserves_bytes_and_replays_completed_calls(tmp_path, tail):
    path = tmp_path / "journal.jsonl"
    request = {"kind": "ordinary", "seed": 3}
    answer = {"text": "completed answer", "finish_reason": "stop"}
    JournalBackend(lambda _: answer, path)(request)
    committed = path.read_bytes()
    path.write_bytes(committed + tail)
    original = path.read_bytes()
    calls = []

    def backend(value):
        calls.append(value)
        return answer

    resumed = JournalBackend(backend, path)
    assert resumed(request) == answer and calls == []
    assert path.read_bytes() == committed
    backups = list(tmp_path.glob("journal.jsonl.interrupted-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    next_request = {"kind": "combined", "seed": 4}
    assert resumed(next_request) == answer and calls == [next_request]
    assert JournalBackend(lambda _: pytest.fail("resampled"), path)(next_request) == answer
    assert backups[0].read_bytes() == original


def test_complete_final_record_without_newline_is_replayed_not_discarded(tmp_path):
    path = tmp_path / "journal.jsonl"
    request = {"seed": 3}
    answer = {"text": "completed answer", "finish_reason": "stop"}
    JournalBackend(lambda _: answer, path)(request)
    original = path.read_bytes().removesuffix(b"\n")
    path.write_bytes(original)
    resumed = JournalBackend(lambda _: pytest.fail("resampled completed answer"), path)
    assert resumed(request) == answer
    assert path.read_bytes() == original + b"\n"
    assert next(tmp_path.glob("journal.jsonl.interrupted-*")).read_bytes() == original


@pytest.mark.parametrize("corruption", ["json", "utf8", "hash", "status", "attempt", "response", "final_hash"])
def test_tail_recovery_never_hides_corrupt_complete_records(tmp_path, corruption):
    path = tmp_path / "journal.jsonl"
    JournalBackend(lambda _: {"text": "ok", "finish_reason": "stop"}, path)({"seed": 3})
    row = json.loads(path.read_text())
    if corruption in {"hash", "final_hash"}:
        row["request_id"] = "bad"
    elif corruption == "status":
        row["status"] = "unknown"
    elif corruption == "attempt":
        row["attempt"] = 2
    elif corruption == "response":
        row["response"]["finish_reason"] = "unknown"
    complete = b'{invalid}\n' if corruption == "json" else b'\xff\n' if corruption == "utf8" else json.dumps(row).encode() + b"\n"
    original = complete[:-1] if corruption == "final_hash" else complete + b'{"unfinished":'
    path.write_bytes(original)
    with pytest.raises(ValueError):
        JournalBackend(lambda _: pytest.fail("called backend"), path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("journal.jsonl.interrupted-*"))


def test_first_interrupted_record_and_interrupted_recovery_can_resume(tmp_path, monkeypatch):
    from secpo.crsecpo import inference
    path = tmp_path / "journal.jsonl"
    original = b'{"first":"unfinished'
    path.write_bytes(original)
    writer = inference.atomic_write_jsonl
    monkeypatch.setattr(inference, "atomic_write_jsonl", lambda *a: (_ for _ in ()).throw(OSError("interrupted repair")))
    with pytest.raises(OSError, match="interrupted repair"):
        JournalBackend(lambda _: None, path)
    assert path.read_bytes() == original
    monkeypatch.setattr(inference, "atomic_write_jsonl", writer)
    resumed = JournalBackend(lambda _: {"text": "ok", "finish_reason": "stop"}, path)
    assert resumed({"seed": 1})["text"] == "ok"
    assert next(tmp_path.glob("journal.jsonl.interrupted-*")).read_bytes() == original


@pytest.mark.parametrize("response", [{"text": "x"}, {"text": 1, "finish_reason": "stop"},
    {"text": "x", "finish_reason": "unknown"}])
def test_backend_must_expose_truncation(response):
    with pytest.raises(ValueError):
        validate_response(response)


def test_optional_hf_backend_uses_only_pinned_local_frozen_weights(monkeypatch):
    calls = []
    class Model:
        peft_config = None
        def to(self, device):
            calls.append(("device", device))
            return self
        def eval(self):
            calls.append(("eval",))
        def requires_grad_(self, enabled):
            calls.append(("grad", enabled))
    class Loader:
        @classmethod
        def from_pretrained(cls, identifier, **kwargs):
            calls.append((cls.__name__, identifier, kwargs))
            return Model() if cls.__name__ == "ModelLoader" else SimpleNamespace(
                apply_chat_template=lambda *args, **kwargs: "native prompt")
    class ModelLoader(Loader):
        pass
    class TokenizerLoader(Loader):
        pass
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=ModelLoader, AutoTokenizer=TokenizerLoader))
    spec = ModelSpec("llama", "a" * 40, "tokenizer", "b" * 40)
    backend = HFBackend(spec)
    assert backend.spec == spec
    model_load = [call for call in calls if call[0] == "ModelLoader"][0]
    tokenizer_load = [call for call in calls if call[0] == "TokenizerLoader"][0]
    assert model_load[2]["revision"] == "a" * 40
    assert tokenizer_load[2]["revision"] == "b" * 40
    assert model_load[2]["local_files_only"] is True
    assert tokenizer_load[2]["local_files_only"] is True
    assert model_load[2]["trust_remote_code"] is False
    assert ("grad", False) in calls and ("eval",) in calls


def test_optional_hf_rejects_adapter_directory_before_loading(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=object, AutoTokenizer=object))
    spec = ModelSpec(str(adapter), "a" * 40, "llama", "b" * 40)
    with pytest.raises(ValueError, match="not an adapter"):
        HFBackend(spec)


def test_missing_native_chat_template_fails_before_weights(monkeypatch):
    def no_template(*args, **kwargs):
        raise ValueError("native chat template missing")
    tokenizer = SimpleNamespace(apply_chat_template=no_template)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: pytest.fail("loaded weights"))))
    with pytest.raises(ValueError, match="native chat template missing"):
        HFBackend(ModelSpec("llama", "a" * 40, "llama", "a" * 40))


@pytest.mark.parametrize("generated,finish,text,answer", [
    ([10, 7], "stop", "answer<|eot_id|>", "answer"),
    ([10, 10], "length", "answeranswer", "answeranswer"),
])
def test_hf_native_continuation_tokens_and_explicit_truncation(monkeypatch, generated, finish, text, answer):
    from contextlib import nullcontext
    from secpo.crsecpo.inference import GenerationConfig, make_request
    class Tensor:
        def __init__(self, tokens):
            self.tokens = tokens
            self.shape = (1, len(tokens))
        def to(self, _):
            return self
        def __getitem__(self, key):
            return Tensor(self.tokens[key[1]])
        def tolist(self):
            return self.tokens
    class Tokenizer:
        eos_token_id = 9
        pad_token_id = None
        def apply_chat_template(self, messages, **kwargs):
            return "native-prompt"
        def __call__(self, text, **kwargs):
            assert text == "native-prompt" and kwargs["add_special_tokens"] is False
            return {"input_ids": Tensor([1, 2, 3]), "attention_mask": Tensor([1, 1, 1])}
        def decode(self, tokens, **kwargs):
            assert kwargs["skip_special_tokens"] is False
            return "".join({10: "answer", 7: "<|eot_id|>"}[token] for token in tokens)
    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=8)
        generation_config = SimpleNamespace(eos_token_id=[7, 9])
        def generate(self, **kwargs):
            assert kwargs["eos_token_id"] == [7, 9]
            assert kwargs["do_sample"] is False
            return Tensor([1, 2, 3] + generated)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    seeds = []
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=seeds.append))
    spec = ModelSpec("llama", "a" * 40, "llama", "b" * 40)
    backend = object.__new__(HFBackend)
    backend.spec, backend.tokenizer, backend.model = spec, Tokenizer(), Model()
    request = make_request([{"role": "user", "content": "hello"}], tokenizer=backend.tokenizer,
                           model_spec=spec, config=GenerationConfig(2, 0, 1), seed=4, kind="ordinary")
    response = backend(request)
    assert response["finish_reason"] == finish
    assert response["text"] == text and response["answer_text"] == answer
    assert response["generated_token_ids"] == generated
    assert seeds == [4]
    request["max_new_tokens"] = 6
    with pytest.raises(ValueError, match="input truncation is forbidden"):
        backend(request)


def test_adapter_digest_changes_request_and_requires_matching_identity(tmp_path):
    from secpo.crsecpo.inference import AdapterBackend, adapter_content_sha256
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"weights1")
    original = adapter_content_sha256(adapter)
    weights.write_bytes(b"weights2")
    assert adapter_content_sha256(adapter) != original
    backend = object.__new__(AdapterBackend)
    backend.fingerprint = {"schema_version": "crsecpo-local-adapter-v1", "adapter_sha256": original}
    with pytest.raises(ValueError, match="adapter identity"):
        backend._validate_backend_identity({})
    backend._validate_backend_identity({"backend_identity": backend.fingerprint})


def test_tiny_real_local_hf_adapter_generation(tmp_path, monkeypatch):
    """Construct random tiny weights locally; test genuine PEFT generation on CPU."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    from secpo.crsecpo.inference import (AdapterBackend, GenerationConfig, adapter_content_sha256,
                                        make_request)
    from secpo.crsecpo.prompts import digest
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.manual_seed(7)
    base_dir = tmp_path / ("a" * 40)
    tokenizer_dir = tmp_path / ("b" * 40)
    vocab = {token: i for i, token in enumerate([
        "<pad>", "<bos>", "<eos>", "<|eot_id|>", "<unk>", "hello", "assistant", "user", "system"])}
    tokenizer_core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer_core.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer_core,
        bos_token="<bos>", eos_token="<|eot_id|>", unk_token="<unk>", pad_token="<pad>",
        additional_special_tokens=["<|eot_id|>"])
    tokenizer.chat_template = "{{ bos_token }} {% for m in messages %}{{m['role']}} {{m['content']}} {{eos_token}} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    tokenizer.save_pretrained(tokenizer_dir)
    config = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=128, bos_token_id=1,
        eos_token_id=3, pad_token_id=0)
    model = transformers.LlamaForCausalLM(config)
    model.generation_config.forced_eos_token_id = 3
    model.save_pretrained(base_dir)
    model = transformers.LlamaForCausalLM.from_pretrained(base_dir, local_files_only=True)
    lora = peft.get_peft_model(model, peft.LoraConfig(task_type="CAUSAL_LM", r=2,
        lora_alpha=4, target_modules=["q_proj", "v_proj"]))
    adapter_dir = tmp_path / "adapter"
    lora.save_pretrained(adapter_dir)
    spec = ModelSpec(str(base_dir), "a" * 40, str(tokenizer_dir), "b" * 40)
    backend = AdapterBackend(spec, adapter_dir, expected_adapter_sha256=adapter_content_sha256(adapter_dir))
    request = make_request([{"role": "user", "content": "hello"}], tokenizer=backend.tokenizer,
        model_spec=spec, config=GenerationConfig(4, 0, 1), seed=11, kind="evaluation",
        backend_identity=backend.fingerprint)
    response = backend(request)
    assert response["finish_reason"] == "stop"
    assert response["generated_token_ids"][-1] == 3
    assert response["text"].endswith("<|eot_id|>")
    assert response["backend_identity"] == backend.fingerprint
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())
    altered = dict(request, backend_identity={"adapter_sha256": "0" * 64})
    assert digest(altered) != digest(request)
    with pytest.raises(ValueError, match="adapter identity"):
        backend(altered)
    with pytest.raises(ValueError, match="adapter identity"):
        backend({key: value for key, value in request.items() if key != "backend_identity"})
