"""Local-only, revision-pinned inference and resumable request journals.

Backends are synchronous callables taking a JSON-compatible request dictionary and
returning a dictionary with ``text`` and ``finish_reason`` (stop or length).
No provider SDK, credential, download, or API billing is involved.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Mapping

from secpo.evaluation.sep_judge import atomic_write_jsonl, load_jsonl
from .prompts import digest


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str

    def __post_init__(self):
        for name in ("model_id", "tokenizer_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        for name in ("model_revision", "tokenizer_revision"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)):
                raise ValueError(f"{name} must be an immutable 40-character commit SHA")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95

    def __post_init__(self):
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not 0 <= self.temperature < float("inf") or not 0 < self.top_p <= 1:
            raise ValueError("invalid temperature/top_p")

    def to_dict(self):
        return asdict(self)


def render_messages(tokenizer, messages):
    """Render the native template, retaining the exact assistant generation header."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer must return a nonempty rendered chat string")
    return rendered


def make_request(messages, *, tokenizer, model_spec: ModelSpec,
                 config: GenerationConfig, seed: int, kind: str, backend_identity=None):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a nonempty string")
    messages = [dict(message) for message in messages]
    request = {"schema_version": "crsecpo-generation-request-v1", "messages": messages,
            "rendered_prompt": render_messages(tokenizer, messages), "seed": seed,
            "kind": kind, "model": model_spec.to_dict(), **config.to_dict()}
    if backend_identity is not None:
        request["backend_identity"] = json.loads(json.dumps(backend_identity, allow_nan=False))
    return request


def validate_response(value):
    if not isinstance(value, Mapping) or not isinstance(value.get("text"), str):
        raise ValueError("backend response must contain string text")
    if value.get("finish_reason") not in {"stop", "length"}:
        raise ValueError("backend must report finish_reason stop or length")
    # Ensure the entire response is persistable, including backend provenance.
    json.dumps(dict(value), allow_nan=False)
    return dict(value)


class JournalBackend:
    """Append and fsync every request attempt; errors can be retried on resume.

    Request identity includes every input, rendered prompt, model/tokenizer
    revision and seed. Every retry remains in the audit log. This is a single-
    writer journal: do not share one path between concurrent processes.
    """
    def __init__(self, backend: Callable, path: str | Path):
        self.backend, self.path = backend, Path(path)
        self.entries = {}
        self.attempt_counts = {}
        if self.path.exists() and self.path.stat().st_size:
            original = self.path.read_bytes()
            interrupted = not original.endswith(b"\n")
            if interrupted:
                tail = original.rsplit(b"\n", 1)[-1]
                try:
                    json.loads(tail.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    recovered = original[:-len(tail)]
                else:
                    # A complete final response must replay even if its newline
                    # was the only byte lost in the interrupted write.
                    recovered = original + b"\n"
                with tempfile.NamedTemporaryFile(dir=self.path.parent) as staged:
                    staged.write(recovered)
                    staged.flush()
                    rows = load_jsonl(staged.name) if recovered.strip() else []
            else:
                rows = load_jsonl(self.path)
            for row in rows:
                request_id = digest(row["request"])
                if request_id != row.get("request_id"):
                    raise ValueError("journal request hash mismatch")
                if row.get("status") not in {"ok", "error"}:
                    raise ValueError("journal contains unknown attempt status")
                if row.get("status") == "ok":
                    validate_response(row["response"])
                next_attempt = self.attempt_counts.get(request_id, 0) + 1
                if row.get("attempt", next_attempt) != next_attempt:
                    raise ValueError("journal attempt sequence is corrupt")
                self.attempt_counts[request_id] = next_attempt
                # A completed successful sample is immutable, even if a later
                # audit row records an error for that request.
                if row["status"] == "ok" or self.entries.get(request_id, {}).get("status") != "ok":
                    self.entries[request_id] = row
            if interrupted:
                # Validate the entire retained prefix before changing anything.
                # A hard link preserves every original byte across replacement,
                # including a partial UTF-8 character or unfinished JSON object.
                backup = self.path.with_name(self.path.name + ".interrupted-" +
                                             hashlib.sha256(original).hexdigest())
                try:
                    os.link(self.path, backup)
                except FileExistsError:
                    if backup.read_bytes() != original:
                        raise ValueError("interrupted journal backup differs from original")
                atomic_write_jsonl(rows, self.path)

    def _append(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        request_id = row["request_id"]
        row["attempt"] = self.attempt_counts.get(request_id, 0) + 1
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.attempt_counts[request_id] = row["attempt"]
        self.entries[request_id] = row

    def __call__(self, request):
        request = json.loads(json.dumps(request, allow_nan=False))
        request_id = digest(request)
        previous = self.entries.get(request_id)
        if previous and previous["status"] == "ok":
            return dict(previous["response"])
        started = time.perf_counter()
        try:
            response = validate_response(self.backend(request))
        except Exception as error:
            self._append({"request_id": request_id, "request": request,
                          "status": "error", "error": f"{type(error).__name__}: {error}",
                          "elapsed_seconds": time.perf_counter() - started})
            raise
        self._append({"request_id": request_id, "request": request,
                      "status": "ok", "response": response,
                      "elapsed_seconds": time.perf_counter() - started})
        return dict(response)


class HFBackend:
    """Optional frozen local Hugging Face model; never downloads or attaches adapters.

    Local snapshots must retain their Hub commit identity via ``ModelSpec``.
    Importing this module does not import torch/transformers or load weights.
    """
    def __init__(self, spec: ModelSpec, *, device: str = "cpu", dtype: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.spec = spec
        # Reject a local adapter directory before AutoModel can auto-discover PEFT.
        if (Path(spec.model_id) / "adapter_config.json").exists():
            raise ValueError("pair inference requires a frozen base model, not an adapter")
        for identifier, revision in ((spec.model_id, spec.model_revision),
                                     (spec.tokenizer_id, spec.tokenizer_revision)):
            local_path = Path(identifier)
            if local_path.is_dir() and local_path.resolve().name != revision:
                raise ValueError("local weights/tokenizer must use their immutable commit snapshot directory")
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.tokenizer_id, revision=spec.tokenizer_revision,
            local_files_only=True, trust_remote_code=False)
        # Reject a missing/incompatible native template before allocating weights.
        render_messages(self.tokenizer, [{"role": "user", "content": "Ready?"}])
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, revision=spec.model_revision, local_files_only=True,
            trust_remote_code=False, dtype=dtype).to(device)
        if getattr(self.model, "peft_config", None):
            raise ValueError("pair inference forbids adapters")
        actual_commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        if actual_commit is not None and actual_commit != spec.model_revision:
            raise ValueError("loaded model commit differs from requested immutable revision")
        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(self, request):
        import torch
        from transformers import set_seed
        self._validate_backend_identity(request)
        if request["model"] != self.spec.to_dict():
            raise ValueError("request model/tokenizer identity differs from loaded backend")
        rendered = render_messages(self.tokenizer, request["messages"])
        if request["rendered_prompt"] != rendered:
            raise ValueError("request rendered prompt differs from the native chat template")
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and input_length + request["max_new_tokens"] > context_limit:
            raise ValueError("request exceeds model context; input truncation is forbidden")
        set_seed(request["seed"])
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        if not stop_ids:
            raise ValueError("model has no native completion terminator")
        kwargs = {"max_new_tokens": request["max_new_tokens"],
                  "do_sample": request["temperature"] > 0,
                  "eos_token_id": sorted(stop_ids),
                  "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(stop_ids)}
        if kwargs["do_sample"]:
            kwargs.update(temperature=request["temperature"], top_p=request["top_p"])
        with torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        tokens = output[0, input_length:].tolist()
        stopped = bool(tokens) and tokens[-1] in stop_ids
        text = self.tokenizer.decode(tokens, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        answer_tokens = tokens[:-1] if stopped else tokens
        answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
        terminator = self.tokenizer.decode([tokens[-1]], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False) if stopped else None
        response = {"text": text, "answer_text": answer_text,
                "finish_reason": "stop" if stopped else "length",
                "completion_terminator": terminator, "generated_token_ids": tokens,
                "input_tokens": input_length, "output_tokens": len(tokens),
                "model": self.spec.to_dict(), "seed": request["seed"]}
        if request.get("backend_identity") is not None:
            response["backend_identity"] = request["backend_identity"]
        return response

    def _validate_backend_identity(self, request):
        if request.get("backend_identity") is not None:
            raise ValueError("frozen base backend does not accept adapter identity")
        if getattr(self.model, "peft_config", None):
            raise ValueError("frozen pair backend forbids attached adapters")


def adapter_content_sha256(directory: str | Path) -> str:
    """Fingerprint filenames and contents of a local adapter, with stable ordering."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir() or not (root / "adapter_config.json").is_file():
        raise ValueError("adapter_path must contain a local adapter_config.json")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not any(path.name in {"adapter_model.safetensors", "adapter_model.bin"} for path in files):
        raise ValueError("adapter_path must contain local adapter weights")
    entries = []
    for path in files:
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError("adapter contents must be local regular files, not symlinks")
        with path.open("rb") as handle:
            content_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        entries.append({"path": path.relative_to(root).as_posix(), "sha256": content_hash})
    return digest("crsecpo-local-adapter-v1", entries)


class AdapterBackend(HFBackend):
    """Evaluation-only frozen PEFT adapter over the same pinned local base model.

    The adapter's content digest must be in every request, and therefore in every
    journal/cache key. An optional expected digest pins an already frozen adapter.
    Pair collection continues to use HFBackend, which forbids attached adapters.
    """
    def __init__(self, spec: ModelSpec, adapter_path: str | Path, *, device="cpu",
                 dtype="auto", expected_adapter_sha256: str | None = None):
        adapter_path = Path(adapter_path).resolve(strict=True)
        adapter_sha256 = adapter_content_sha256(adapter_path)
        if expected_adapter_sha256 is not None and adapter_sha256 != expected_adapter_sha256:
            raise ValueError("local adapter content differs from expected immutable digest")
        adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
        if adapter_config.get("revision") not in {None, spec.model_revision}:
            raise ValueError("adapter base revision differs from pinned base revision")
        adapter_base = adapter_config.get("base_model_name_or_path")
        if adapter_base and adapter_base != spec.model_id:
            if not (Path(adapter_base).is_dir() and Path(spec.model_id).is_dir()
                    and Path(adapter_base).resolve() == Path(spec.model_id).resolve()):
                raise ValueError("adapter base model differs from pinned base model")
        super().__init__(spec, device=device, dtype=dtype)
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, str(adapter_path),
                                               is_trainable=False, local_files_only=True)
        self.model.eval()
        self.model.requires_grad_(False)
        if adapter_content_sha256(adapter_path) != adapter_sha256:
            raise ValueError("adapter contents changed while loading")
        self.fingerprint = {"schema_version": "crsecpo-local-adapter-v1",
                            "adapter_sha256": adapter_sha256}

    def _validate_backend_identity(self, request):
        if request.get("backend_identity") != self.fingerprint:
            raise ValueError("request is missing or mismatches immutable adapter identity")
