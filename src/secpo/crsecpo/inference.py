"""Local-only, revision-pinned inference and resumable request journals.

Backends are synchronous callables taking a JSON-compatible request dictionary and
returning a dictionary with ``text`` and ``finish_reason`` (stop or length).
No provider SDK, credential, download, or API billing is involved.
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
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
from .artifact_versions import logical_artifact_path
from .collection_exports import (claim_journal_destination, is_claimed_journal, lock_data_paths,
                                 validate_data_path, validate_journal_destination)
from .prompts import digest

MAX_BATCH_SUBMISSIONS = 32


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
    response = dict(value)
    # Match the journal's UTF-8 serialization before accepting the response.
    try:
        json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("backend response contains invalid Unicode") from error
    return response


def format_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    return message.encode("utf-8", errors="backslashreplace").decode("utf-8")


class BatchEngineUnavailable(RuntimeError):
    """The inference engine must be restarted before collection resumes."""


class JournalStateError(ValueError):
    """A journal storage or consistency failure that must stop publication."""


_JOURNAL_FIELDS = ("attempt", "batch_size", "elapsed_seconds", "error",
                   "request", "request_id", "response", "status")


def _journal_tail_is_torn(tail, error, *, known_journal):
    # An ambiguous first fragment in an unmarked file cannot establish
    # ownership. Current writers claim the path before writing the first byte.
    if not known_journal:
        return False
    text = tail.lstrip(b" \t\r")
    header = re.match(rb'\{[ \t\r]*"([a-z_]+)"[ \t\r]*:', text)
    if header is not None:
        if header[1].decode("ascii") not in _JOURNAL_FIELDS:
            return False
    elif not any(
        (f'{{"{field}":').encode().startswith(text) for field in _JOURNAL_FIELDS
    ):
        return False

    if isinstance(error, UnicodeDecodeError):
        return error.end == len(tail) and error.reason == "unexpected end of data"
    remaining = error.doc[error.pos:].strip()
    if not remaining or error.msg.startswith("Unterminated string"):
        return True
    if error.msg == r"Invalid \uXXXX escape":
        return bool(re.fullmatch(r"u[0-9a-fA-F]{0,3}", remaining))
    if error.msg == "Expecting value":
        return any(literal.startswith(remaining) for literal in ("true", "false", "null", "-"))
    if error.msg == "Expecting ',' delimiter":
        return bool(re.fullmatch(r"(?:[eE][+-]?|\.)", remaining))
    return False


class JournalBackend:
    """Append and fsync every request attempt; errors can be retried on resume.

    Request identity includes every input, rendered prompt, model/tokenizer
    revision and seed. Every retry remains in the audit log. This is a single-
    writer journal: do not share one path between concurrent processes.
    """
    def __init__(self, backend: Callable, path: str | Path, *, lock_replacement=None):
        validate_data_path(path)
        self.backend, self.path = backend, logical_artifact_path(path)
        self.entries = {}
        self.attempt_counts = {}
        self._content_hash = hashlib.sha256()
        # CLI collection holds this lock throughout startup, repair and writing.
        self._external_lock = lock_replacement is not None
        with self._path_lock() as (pin, _):
            validate_journal_destination(self.path)
            self._snapshot = self._file_snapshot()
            self._load(lock_replacement or pin)
            self._verify_snapshot()

    @contextmanager
    def _path_lock(self):
        if self._external_lock:
            yield None, True
            return
        with lock_data_paths(self.path), ExitStack() as inodes:
            def pin(path):
                handle = inodes.enter_context(path.open("r+b"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(f"another collector is using journal {self.path}") from error

            try:
                pin(self.path)
            except FileNotFoundError:
                yield pin, False
            else:
                yield pin, True

    def _load(self, lock_replacement):
        if self.path.exists() and self.path.stat().st_size:
            original = self.path.read_bytes()
            self._verify_snapshot()
            self._content_hash.update(original)
            interrupted = not original.endswith(b"\n")
            if interrupted:
                tail = original.rsplit(b"\n", 1)[-1]
                try:
                    json.loads(tail.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    prefix = original[:-len(tail)]
                    if not _journal_tail_is_torn(tail, error, known_journal=bool(prefix.strip())
                                                or is_claimed_journal(self.path)):
                        raise JournalStateError(f"unrecognized or corrupt final journal record: {self.path}") from error
                    recovered = prefix
                else:
                    # A complete final response must replay even if its newline
                    # was the only byte lost in the interrupted write.
                    recovered = original + b"\n"
                rows = load_jsonl(self.path, payload=recovered) if recovered.strip() else []
            else:
                rows = load_jsonl(self.path, payload=original)
            for row in rows:
                if not isinstance(row.get("request"), dict):
                    raise ValueError(f"journal {self.path} has an attempt without a request object")
                request_id = digest(row["request"])
                if request_id != row.get("request_id"):
                    raise ValueError("journal request hash mismatch")
                if row.get("status") not in {"ok", "error"}:
                    raise ValueError("journal contains unknown attempt status")
                if row.get("status") == "ok":
                    validate_response(row.get("response"))
                next_attempt = self.attempt_counts.get(request_id, 0) + 1
                if row.get("attempt", next_attempt) != next_attempt:
                    raise ValueError("journal attempt sequence is corrupt")
                self._remember(row, next_attempt)
            if interrupted:
                claim_journal_destination(self.path)
                # Preserve the bytes that were validated, independent of the
                # source inode or any aliases that still point at it.
                backup = self.path.with_name(self.path.name + ".interrupted-" + self._content_hash.hexdigest())
                with tempfile.TemporaryDirectory(dir=self.path.parent) as temporary:
                    staged = Path(temporary) / "repaired.jsonl"
                    atomic_write_jsonl(rows, staged)
                    lock_replacement(staged)
                    snapshot = self._snapshot_from_stat(staged.stat())
                    content_hash = hashlib.sha256(staged.read_bytes())
                    self._verify_content()
                    staged_backup = Path(temporary) / "backup"
                    with staged_backup.open("wb") as handle:
                        handle.write(original)
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        os.link(staged_backup, backup)
                    except FileExistsError:
                        if backup.read_bytes() != original:
                            raise ValueError("interrupted journal backup differs from original")
                    self._verify_content()
                    os.replace(staged, self.path)
                    self._snapshot = snapshot
                    self._content_hash = content_hash

    def _file_snapshot(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return self._snapshot_from_stat(stat)

    @staticmethod
    def _snapshot_from_stat(stat):
        if stat.st_size:
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        return None

    def _verify_snapshot(self):
        try:
            current = self._file_snapshot()
        except OSError as error:
            raise JournalStateError(f"cannot inspect journal {self.path}: {error}") from error
        if current != self._snapshot:
            raise JournalStateError(f"journal changed after loading; reopen it before continuing: {self.path}")

    @contextmanager
    def publication_snapshot(self):
        """Keep the journal stable while publishing artifacts derived from it."""
        with self._path_lock():
            self._verify_content()
            yield self._verify_content

    def _verify_content(self):
        self._verify_snapshot()
        try:
            with self.path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").digest()
        except FileNotFoundError as error:
            if self._snapshot is not None:
                raise JournalStateError(f"journal changed after loading: {self.path}") from error
            actual = hashlib.sha256().digest()
        except OSError as error:
            raise JournalStateError(f"cannot verify journal {self.path}: {error}") from error
        self._verify_snapshot()
        if actual != self._content_hash.digest():
            raise JournalStateError(f"journal contents changed after loading: {self.path}")

    def _append(self, row):
        try:
            self._write_attempt(row)
        except JournalStateError:
            raise
        except (OSError, ValueError) as error:
            raise JournalStateError(f"cannot write journal {self.path}: {error}") from error

    def _write_attempt(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        request_id = row["request_id"]
        with self._path_lock() as (pin, exists):
            self._verify_snapshot()
            claim_journal_destination(self.path)
            row["attempt"] = self.attempt_counts.get(request_id, 0) + 1
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
            if not exists:
                self.path.touch()
                pin(self.path)
            with self.path.open("ab") as handle:
                if self._snapshot_from_stat(os.fstat(handle.fileno())) != self._snapshot:
                    raise JournalStateError(f"journal changed before append: {self.path}")
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                self._snapshot = self._snapshot_from_stat(os.fstat(handle.fileno()))
            self._verify_snapshot()
            self._content_hash.update(payload)
            self._remember(row, row["attempt"])

    def _remember(self, row, attempt):
        request_id = row["request_id"]
        self.attempt_counts[request_id] = attempt
        # Keep the first successful sample while retaining all attempts on disk.
        if self.entries.get(request_id, {}).get("status") != "ok":
            self.entries[request_id] = deepcopy(row)

    def completed(self, request):
        """Return the recorded successful response for ``request``, or None."""
        self._verify_snapshot()
        previous = self.entries.get(digest(json.loads(json.dumps(request, allow_nan=False))))
        return deepcopy(previous["response"]) if previous and previous["status"] == "ok" else None

    def record_batch(self, requests, generate_many):
        """Run uncached requests in one backend batch and append every attempt.

        ``generate_many`` returns one response or Exception per request, in order.
        Each attempt is journaled exactly as a single call would be; elapsed time
        is the batch wall time divided evenly, with the batch size retained.
        A batch-level exception is journaled before retrying in smaller
        batches within a shared submission budget. Returns ``{request_id: exception}`` for terminal request failures.
        """
        requests = [json.loads(json.dumps(request, allow_nan=False)) for request in requests]
        if len({digest(request) for request in requests}) != len(requests):
            raise ValueError("batched requests must be distinct")
        if any(self.completed(request) is not None for request in requests):
            raise ValueError("batched requests must not already be completed")

        pending = [requests]
        failures = {}
        last_batch_error = None
        for _ in range(MAX_BATCH_SUBMISSIONS):
            batch = pending.pop()
            attempt_failures, batch_error = self._record_batch_attempt(batch, generate_many)
            if isinstance(batch_error, BatchEngineUnavailable):
                raise batch_error
            if batch_error is not None:
                last_batch_error = batch_error
            if batch_error is not None and len(batch) > 1:
                midpoint = len(batch) // 2
                pending.extend((batch[midpoint:], batch[:midpoint]))
            else:
                failures.update(attempt_failures)
            if not pending:
                return failures
        raise BatchEngineUnavailable(
            f"Batch recovery reached {MAX_BATCH_SUBMISSIONS} engine submissions. "
            "Inspect the journal and backend failure before resuming."
        ) from last_batch_error

    def _record_batch_attempt(self, requests, generate_many):
        started = time.perf_counter()
        batch_error = None
        try:
            results = list(generate_many(requests))
        except JournalStateError:
            raise
        except Exception as error:  # noqa: BLE001 - journal the failed engine submission
            batch_error = error
            results = [error] * len(requests)
        if len(results) != len(requests):
            raise ValueError("batched backend returned a different number of results")
        elapsed = (time.perf_counter() - started) / max(1, len(requests))
        failures = {}
        for request, result in zip(requests, results):
            request_id = digest(request)
            timing = {"elapsed_seconds": elapsed, "batch_size": len(requests)}
            if not isinstance(result, Exception):
                try:
                    result = validate_response(result)
                except Exception as error:  # noqa: BLE001 - journal the invalid response as an error
                    result = error
            if isinstance(result, Exception):
                failures[request_id] = result
                self._append({"request_id": request_id, "request": request, "status": "error",
                              "error": format_error(result), **timing})
            else:
                self._append({"request_id": request_id, "request": request, "status": "ok",
                              "response": result, **timing})
        return failures, batch_error

    def __call__(self, request):
        request = json.loads(json.dumps(request, allow_nan=False))
        request_id = digest(request)
        self._verify_snapshot()
        previous = self.entries.get(request_id)
        if previous and previous["status"] == "ok":
            return deepcopy(previous["response"])
        started = time.perf_counter()
        try:
            response = validate_response(self.backend(request))
        except Exception as error:
            self._append({"request_id": request_id, "request": request,
                          "status": "error", "error": format_error(error),
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


VLLM_BACKEND_SCHEMA = "crsecpo-vllm-backend-v1"
# Generation fields transformers would apply that have no equivalent below. A
# local generation_config.json setting any of them to a non-neutral value is
# rejected rather than silently sampled differently.
_NEUTRAL_HF_FIELDS = {"num_beams": 1, "typical_p": 1.0, "no_repeat_ngram_size": 0,
                      "epsilon_cutoff": 0.0, "eta_cutoff": 0.0, "encoder_repetition_penalty": 1.0,
                      "min_length": 0, "min_new_tokens": None, "bad_words_ids": None,
                      "suppress_tokens": None, "begin_suppress_tokens": None,
                      "forced_bos_token_id": None, "forced_eos_token_id": None,
                      "exponential_decay_length_penalty": None, "sequence_bias": None}


def effective_hf_sampling(generation_config, global_defaults):
    """Sampling settings HFBackend.generate actually uses beyond explicit arguments.

    HFBackend passes max_new_tokens, do_sample, temperature, top_p and stop IDs.
    Everything else comes from the model's generation_config.json, then from
    transformers' global defaults (notably ``top_k=50``). vLLM must be given the
    same values or it samples from a different distribution.
    """
    def value(name):
        configured = getattr(generation_config, name, None)
        return global_defaults.get(name) if configured is None else configured
    for name, neutral in _NEUTRAL_HF_FIELDS.items():
        configured = value(name)
        if configured is not None and configured != neutral:
            raise ValueError(f"generation_config sets unsupported non-neutral {name}={configured!r}")
    top_k, penalty, min_p = value("top_k"), value("repetition_penalty"), value("min_p")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("effective top_k must be a non-negative integer")
    return {"top_k": top_k, "repetition_penalty": float(1.0 if penalty is None else penalty),
            "min_p": float(0.0 if min_p is None else min_p)}


class VLLMBackend:
    """Batched pair-collection backend over the same pinned local base model.

    Same validation and response schema as HFBackend: frozen local weights, exact
    native chat rendering, no input truncation, native stop IDs, decoded with the
    same tokenizer. Sampling copies HFBackend's effective settings, including the
    transformers top_k default, and seeds each request with its own seed.

    Outputs are not token-identical to HFBackend: batched kernels and vLLM's
    sampler differ numerically. Every request therefore carries
    ``backend_identity`` so journals and caches never mix the two engines.
    """
    def __init__(self, spec: ModelSpec, *, dtype: str = "bfloat16", max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9, identity=None,
                 extra_llm_kwargs=None, identity_extra=None):
        from transformers import AutoConfig
        if (Path(spec.model_id) / "adapter_config.json").exists():
            raise ValueError("pair inference requires a frozen base model, not an adapter")
        for identifier, revision in ((spec.model_id, spec.model_revision),
                                     (spec.tokenizer_id, spec.tokenizer_revision)):
            local_path = Path(identifier)
            if local_path.is_dir() and local_path.resolve().name != revision:
                raise ValueError("local weights/tokenizer must use their immutable commit snapshot directory")
        self.spec = spec
        self.tokenizer = self.load_tokenizer(spec)
        render_messages(self.tokenizer, [{"role": "user", "content": "Ready?"}])
        self.config = AutoConfig.from_pretrained(spec.model_id, revision=spec.model_revision,
                                                 local_files_only=True, trust_remote_code=False)
        self.model_type = getattr(self.config, "model_type", None)
        expected = self.identity_for(spec, dtype=dtype, max_model_len=max_model_len)
        # A subclass (e.g. a frozen LoRA adapter) folds its own evidence into the
        # engine identity so journals/caches never mix it with the bare base engine.
        if identity_extra is not None:
            expected = {**expected, **identity_extra}
        if identity is not None and identity != expected:
            raise ValueError("vLLM backend identity differs from the recorded collection identity")
        self.identity = expected
        # Evaluation (_request) stamps backend_identity from .fingerprint; mirror the
        # engine identity there so requests validate and journals stay per-engine.
        self.fingerprint = expected
        self._lora_request = None
        self.sampling_defaults = expected["sampling_defaults"]
        eos = self._generation_config(spec).eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        if not self.stop_ids:
            raise ValueError("model has no native completion terminator")
        self.context_limit = getattr(self.config, "max_position_embeddings", None)
        self.max_model_len = max_model_len
        # Match local_files_only: hand vLLM the pinned local snapshot directories so
        # it cannot download or resolve a different revision.
        model_path = self._local_snapshot(spec.model_id, spec.model_revision)
        tokenizer_path = self._local_snapshot(spec.tokenizer_id, spec.tokenizer_revision)
        from vllm import LLM
        self.llm = LLM(model=model_path, tokenizer=tokenizer_path,
                       dtype=dtype, max_model_len=max_model_len, seed=0,
                       gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=False,
                       # Never let vLLM merge generation_config.json defaults on its own;
                       # every sampling value is passed explicitly below.
                       generation_config="vllm", **(extra_llm_kwargs or {}))

    @staticmethod
    def load_tokenizer(spec):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(spec.tokenizer_id, revision=spec.tokenizer_revision,
                                             local_files_only=True, trust_remote_code=False)

    @staticmethod
    def _local_snapshot(identifier, revision):
        if Path(identifier).is_dir():
            return str(Path(identifier).resolve())
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(identifier, revision=revision, local_files_only=True))
        if path.name != revision:
            raise ValueError("local snapshot directory does not match the pinned revision")
        return str(path)

    @staticmethod
    def _generation_config(spec):
        from transformers import GenerationConfig
        return GenerationConfig.from_pretrained(spec.model_id, revision=spec.model_revision,
                                                local_files_only=True)

    @classmethod
    def identity_for(cls, spec: ModelSpec, *, dtype: str = "bfloat16", max_model_len: int = 8192):
        """Engine identity, computable from local files without loading weights."""
        from importlib.metadata import version
        from transformers import GenerationConfig
        if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len < 1:
            raise ValueError("max_model_len must be a positive integer")
        resolve = getattr(GenerationConfig, "_get_default_generation_params", None)
        if resolve is None:
            raise ValueError("this transformers version does not expose its generation defaults; "
                             "cannot reproduce HFBackend sampling settings for vLLM")
        defaults = resolve()
        return {"schema_version": VLLM_BACKEND_SCHEMA, "engine": "vllm", "engine_version": version("vllm"),
                "dtype": dtype, "max_model_len": max_model_len,
                "sampling_defaults": effective_hf_sampling(cls._generation_config(spec), defaults)}

    def _prepare(self, request):
        from vllm import SamplingParams
        if request.get("backend_identity") != self.identity:
            raise ValueError("request is missing or mismatches the vLLM backend identity")
        if request["model"] != self.spec.to_dict():
            raise ValueError("request model/tokenizer identity differs from loaded backend")
        rendered = render_messages(self.tokenizer, request["messages"])
        if request["rendered_prompt"] != rendered:
            raise ValueError("request rendered prompt differs from the native chat template")
        prompt_ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        input_length = len(prompt_ids)
        if self.context_limit and input_length + request["max_new_tokens"] > self.context_limit:
            raise ValueError("request exceeds model context; input truncation is forbidden")
        if input_length + request["max_new_tokens"] > self.max_model_len:
            raise ValueError("request exceeds vLLM max_model_len; input truncation is forbidden")
        common = {"n": 1, "max_tokens": request["max_new_tokens"], "seed": request["seed"],
                  "stop_token_ids": sorted(self.stop_ids), "ignore_eos": False,
                  "repetition_penalty": self.sampling_defaults["repetition_penalty"]}
        if request["temperature"] > 0:
            params = SamplingParams(temperature=request["temperature"], top_p=request["top_p"],
                                    top_k=self.sampling_defaults["top_k"],
                                    min_p=self.sampling_defaults["min_p"], **common)
        else:
            params = SamplingParams(temperature=0.0, **common)
        return prompt_ids, params

    def _response(self, request, prompt_ids, output):
        if output.finish_reason not in {"stop", "length"}:
            raise ValueError(f"vLLM finished with unsupported reason {output.finish_reason!r}")
        tokens = list(output.token_ids)
        if output.finish_reason == "stop" and not (tokens and tokens[-1] in self.stop_ids):
            # Some vLLM versions report the matched stop ID without retaining it.
            if output.stop_reason in self.stop_ids:
                tokens.append(output.stop_reason)
            else:
                raise ValueError("vLLM stopped without a native completion terminator")
        # Identical post-processing to HFBackend.
        stopped = bool(tokens) and tokens[-1] in self.stop_ids
        decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)
        answer_tokens = tokens[:-1] if stopped else tokens
        return {"text": decode(tokens), "answer_text": decode(answer_tokens),
                "finish_reason": "stop" if stopped else "length",
                "completion_terminator": decode([tokens[-1]]) if stopped else None,
                "generated_token_ids": tokens, "input_tokens": len(prompt_ids),
                "output_tokens": len(tokens), "model": self.spec.to_dict(), "seed": request["seed"],
                "backend_identity": request["backend_identity"]}

    def generate_many(self, requests):
        """One response or Exception per request, in order, from one engine batch."""
        from vllm.inputs import TokensPrompt
        results, prepared = [None] * len(requests), []
        for index, request in enumerate(requests):
            try:
                prepared.append((index, *self._prepare(request)))
            except Exception as error:  # noqa: BLE001 - per-request validation failure
                results[index] = error
        if prepared:
            try:
                # Only pass lora_request when an adapter is attached, so the bare base
                # engine calls generate() exactly as before (no LoRA machinery).
                lora_kwargs = {"lora_request": self._lora_request} if self._lora_request is not None else {}
                outputs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids, _ in prepared],
                                            [params for _, _, params in prepared], use_tqdm=False,
                                            **lora_kwargs)
            except Exception as error:
                # Older vLLM runtimes may not expose the V1 exception type.
                try:
                    from vllm.v1.engine.exceptions import EngineDeadError
                except ImportError:
                    pass
                else:
                    if isinstance(error, EngineDeadError):
                        raise BatchEngineUnavailable(str(error)) from error
                # A retry must not consume queued outputs from the failed submission.
                if self.llm.llm_engine.has_unfinished_requests():
                    raise BatchEngineUnavailable("vLLM has unfinished requests after a failed batch") from error
                raise
            if len(outputs) != len(prepared):
                raise BatchEngineUnavailable("vLLM returned a different number of outputs")
            for (index, prompt_ids, _), output in zip(prepared, outputs):
                try:
                    results[index] = self._response(requests[index], prompt_ids, output.outputs[0])
                except Exception as error:  # noqa: BLE001
                    results[index] = error
        return results

    def __call__(self, request):
        result = self.generate_many([request])[0]
        if isinstance(result, Exception):
            raise result
        return result


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


def _validate_adapter_against_base(spec, adapter_path, *, expected_adapter_sha256):
    """Shared adapter provenance guard (base revision/model, content digest)."""
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
    return adapter_path, adapter_sha256, adapter_config


class VLLMAdapterBackend(VLLMBackend):
    """Evaluation-only batched vLLM backend with a frozen LoRA adapter.

    Same batched engine as VLLMBackend, but the trained LoRA is applied through
    vLLM's native adapter support. The adapter's content digest is folded into the
    engine identity (and fingerprint), so its journals/caches never mix with the
    bare base engine, another adapter, or the HF AdapterBackend (whose sampler is
    numerically different). Base weights stay frozen and pinned; only the adapter
    differs from VLLMBackend.
    """
    def __init__(self, spec: ModelSpec, adapter_path, *, dtype: str = "bfloat16",
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.9,
                 expected_adapter_sha256: str | None = None):
        adapter_path, adapter_sha256, adapter_config = _validate_adapter_against_base(
            spec, adapter_path, expected_adapter_sha256=expected_adapter_sha256)
        rank = adapter_config.get("r")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("adapter_config.json must declare a positive LoRA rank 'r'")
        self._adapter_path = str(adapter_path)
        super().__init__(spec, dtype=dtype, max_model_len=max_model_len,
                         gpu_memory_utilization=gpu_memory_utilization,
                         extra_llm_kwargs={"enable_lora": True, "max_loras": 1, "max_lora_rank": rank},
                         identity_extra={"adapter": {"schema_version": "crsecpo-local-adapter-v1",
                                                     "adapter_sha256": adapter_sha256}})
        from vllm.lora.request import LoRARequest
        self._lora_request = LoRARequest("crsecpo_eval_adapter", 1, self._adapter_path)
        if adapter_content_sha256(adapter_path) != adapter_sha256:
            raise ValueError("adapter contents changed while loading")
