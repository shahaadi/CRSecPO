"""Optional OpenAI chat judge for CRSecPO evaluation grading.

A drop-in judge backend: given the blinded grading request it calls a (cheap)
OpenAI chat model in strict JSON-schema mode and returns the grade exactly as the
local judge would (``prohibited_assistance`` / ``refusal`` / ``permitted_help`` /
``rationale``). The API key is read from the OPENAI_API_KEY environment variable;
it is never written to configs, journals, or code.

Grading is journaled by the caller, so if the run stops on a quota or rate error,
re-running resumes and the already-completed grades are replayed without being
re-charged. Transient rate/connection errors are retried here with exponential
backoff so they don't halt the whole run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import time

SCHEMA_VERSION = "crsecpo-openai-judge-v1"

# Strict schema so the model must return exactly the fields parse_grade expects.
_GRADE_SCHEMA = {
    "name": "crsecpo_grade",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "prohibited_assistance": {"type": ["boolean", "null"]},
            "refusal": {"type": ["boolean", "null"]},
            "permitted_help": {"type": ["boolean", "null"]},
            "rationale": {"type": "string"},
        },
        "required": ["prohibited_assistance", "refusal", "permitted_help", "rationale"],
    },
}


class _StubTokenizer:
    """Deterministic chat rendering so request hashes are stable (no model)."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize or not add_generation_prompt:
            raise ValueError("unexpected chat-template arguments for the OpenAI judge")
        return json.dumps(messages, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class _OpenAIJudgeSpec:
    """Judge identity. A dataclass so the caller's dataclasses.asdict() works,
    mirroring the local ModelSpec; also exposes to_dict() for make_request."""

    provider: str
    model: str

    def to_dict(self):
        return asdict(self)


class OpenAIJudgeBackend:
    """Judge backend that grades via an OpenAI chat model instead of a local LM."""

    def __init__(self, model, *, temperature=0.0, max_tokens=512, max_retries=6, base_delay=2.0):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI judge requires a non-empty model name")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set; export it (e.g. from Colab secrets) to use the OpenAI judge")
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError("the OpenAI judge requires the 'openai' package (pip install openai)") from error
        self.model = model.strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.base_delay = float(base_delay)
        identity = {"schema_version": SCHEMA_VERSION, "provider": "openai",
                    "model": self.model, "temperature": self.temperature,
                    "response_format": "json_schema", "grade_schema": _GRADE_SCHEMA}
        self.spec = _OpenAIJudgeSpec(provider="openai", model=self.model)
        self.tokenizer = _StubTokenizer()
        self.fingerprint = identity
        self._client = OpenAI()

    def __call__(self, request):
        import openai
        messages = request["messages"]
        last_error = None
        for attempt in range(self.max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_schema", "json_schema": _GRADE_SCHEMA},
                    seed=request.get("seed"),
                )
            except (openai.RateLimitError, openai.APITimeoutError,
                    openai.APIConnectionError, openai.InternalServerError) as error:
                last_error = error
                time.sleep(self.base_delay * (2 ** attempt))  # exponential backoff
                continue
            choice = completion.choices[0]
            text = choice.message.content or ""
            return {"text": text, "answer_text": text,
                    "finish_reason": "stop" if choice.finish_reason == "stop" else "length",
                    "model": self.spec.to_dict(), "backend_identity": self.fingerprint}
        raise RuntimeError(
            f"OpenAI judge '{self.model}' failed after {self.max_retries} retries: {last_error}")
