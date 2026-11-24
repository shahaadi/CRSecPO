"""Shared grading for the general-safety filter: the same blinded judge used at eval
time, with a resumable on-disk cache and two backends.

- sync  (default): OpenAIJudgeBackend.generate_many, concurrent chunks.
- batch (GSAFE_USE_BATCH=1): the OpenAI Batch API -- ~50% cheaper and no rate-limit
  babysitting, async (up to 24h). Resumable: the batch id is persisted, so re-running
  after a disconnect polls/retrieves the same batch instead of resubmitting.

Both paths grade identically (same prompt, same parse_grade) and write to the same
cache keyed by '<id>:<student|teacher>', so the student-selection and teacher-check
steps never re-charge a resolved answer.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from secpo.crsecpo.prompts import RUBRIC
from secpo.crsecpo.evaluation import blinded_judge_messages, parse_grade

GRADE_ERROR = {"prohibited_assistance": None, "refusal": None, "permitted_help": None,
               "status": "grade_error", "rationale": ""}


def strip_terminator(text: str, terminator: str) -> str:
    text = text or ""
    if terminator and text.endswith(terminator):
        text = text[: -len(terminator)]
    return text.strip()


def judge_messages(target: str, answer: str):
    return blinded_judge_messages(target, RUBRIC, answer)


class GradeCache:
    """Append-only JSONL cache keyed by '<id>:<student|teacher>'."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["grade"]
        self._handle = self.path.open("a", encoding="utf-8")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, grade):
        if key in self.data:
            return
        self.data[key] = grade
        self._handle.write(json.dumps({"key": key, "grade": grade}, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()


def _grade_sync(pending, cache, *, model, concurrency, chunk=64):
    from secpo.crsecpo.openai_judge import OpenAIJudgeBackend
    backend = OpenAIJudgeBackend(model, concurrency=concurrency)
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        requests = [{"messages": msgs} for (_, _, msgs) in batch]
        responses = backend.generate_many(requests)
        for (key, _, _), response in zip(batch, responses):
            cache.put(key, parse_grade(response) if isinstance(response, dict) else dict(GRADE_ERROR))
        print(f"[grade:sync] {min(start + chunk, len(pending))}/{len(pending)}", flush=True)
    return True


def _grade_batch(pending, cache, *, model, state_path, max_tokens, poll_hours):
    """Submit (or resume) one OpenAI batch for the pending requests. Returns True when
    all results are retrieved and cached; False if the batch is still running (re-run)."""
    from openai import OpenAI
    import io
    client = OpenAI()
    state_path = Path(state_path)
    batch_id = json.loads(state_path.read_text()).get("batch_id") if state_path.exists() else None

    if not batch_id:
        lines = [json.dumps({"custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                             "body": {"model": model, "messages": msgs,
                                      "max_completion_tokens": max_tokens,
                                      "response_format": {"type": "json_object"}}})
                 for (key, _, msgs) in pending]
        upload = client.files.create(file=io.BytesIO("\n".join(lines).encode("utf-8")), purpose="batch")
        batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions",
                                      completion_window="24h")
        batch_id = batch.id
        state_path.write_text(json.dumps({"batch_id": batch_id, "n": len(pending)}))
        print(f"[grade:batch] submitted {batch_id} ({len(pending)} requests)", flush=True)

    start = time.time()
    batch = client.batches.retrieve(batch_id)
    while batch.status not in {"completed", "failed", "expired", "cancelled"}:
        if time.time() - start > poll_hours * 3600:
            print(f"[grade:batch] still '{batch.status}' after {poll_hours}h; re-run this step to resume",
                  flush=True)
            return False
        print(f"[grade:batch] status={batch.status} counts={batch.request_counts}", flush=True)
        time.sleep(30)
        batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise SystemExit(f"[grade:batch] batch {batch_id} ended '{batch.status}'; "
                         f"delete {state_path} and retry")

    seen = set()
    if batch.output_file_id:
        for line in client.files.content(batch.output_file_id).text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec["custom_id"]
            try:
                choice = rec["response"]["body"]["choices"][0]
                text = choice["message"]["content"] or ""
                finish = "stop" if choice["finish_reason"] == "stop" else "length"
                cache.put(key, parse_grade({"text": text, "answer_text": text, "finish_reason": finish}))
            except (KeyError, IndexError, TypeError):
                cache.put(key, dict(GRADE_ERROR))
            seen.add(key)
    for key, _, _ in pending:                      # anything the batch dropped/errored
        if key not in seen:
            cache.put(key, dict(GRADE_ERROR))
    state_path.unlink(missing_ok=True)
    print(f"[grade:batch] retrieved {len(seen)} results", flush=True)
    return True


def grade_jobs(jobs, cache, *, label):
    """jobs: list of (key, target, answer). Grade only uncached keys (via whichever backend
    the env selects) and write them to `cache`. Returns True when complete, False if a batch
    is still running. `label` names the pass (student/teacher) for the batch-state file."""
    pending = [(key, target, judge_messages(target, answer))
               for (key, target, answer) in jobs if cache.get(key) is None]
    if not pending:
        return True
    model = os.environ.get("GSAFE_FILTER_MODEL", "gpt-4o-mini")
    if os.environ.get("GSAFE_USE_BATCH", "0") == "1":
        state = cache.path.with_name(f"batch_{label}.json")
        return _grade_batch(pending, cache, model=model, state_path=state,
                            max_tokens=int(os.environ.get("GSAFE_JUDGE_MAX_TOKENS", "4096")),
                            poll_hours=float(os.environ.get("GSAFE_BATCH_POLL_HOURS", "6")))
    return _grade_sync(pending, cache, model=model,
                       concurrency=int(os.environ.get("FILTER_CONCURRENCY", "8")))
