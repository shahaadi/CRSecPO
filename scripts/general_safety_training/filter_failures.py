"""Keep only the base model's safety FAILURES from the generated pairs.

Reads the merged pair set, grades each candidate's *student* (bare-prompt) answer
with the OpenAI judge -- reusing the exact grading used at eval time -- and keeps a
row only when the student failed in the direction we want to correct:

    unsafe prompt (harmful_benign_framing) and prohibited_assistance == True   (complied)
    safe   prompt (everything else)        and refusal             == True     (over-refused)

By default it then verifies the *teacher* (chosen) answer is correct on the kept
rows, so the training target is trustworthy:
    unsafe: teacher prohibited_assistance == False
    safe:   teacher refusal == False and permitted_help == True

The kept rows overwrite merged_pairs.jsonl (the pre-filter set is preserved once as
merged_pairs.raw.jsonl); stage 2 then freezes the filtered set unchanged. Grades are
cached in filter_grades.jsonl, so reruns never re-charge resolved rows.

Env: PAIRS_DIR, GSAFE_FILTER_MODEL, FILTER_CONCURRENCY, GSAFE_REQUIRE_TEACHER_CORRECT,
     GSAFE_REFILTER.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from secpo.crsecpo.prompts import RUBRIC, safety_label
from secpo.crsecpo.evaluation import blinded_judge_messages, parse_grade
from secpo.crsecpo.openai_judge import OpenAIJudgeBackend

HARMFUL_GROUP = "harmful_benign_framing"


def _answer_text(row, key):
    """The raw completion for grading: strip the single native chat terminator."""
    text = row.get(key) or ""
    term = (row.get("pair_metadata") or {}).get("completion_terminator") or ""
    if term and text.endswith(term):
        text = text[: -len(term)]
    return text.strip()


class GradeCache:
    """Append-only JSONL cache keyed by '<candidate id>:<student|teacher>'."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["grade"]
        self._handle = path.open("a", encoding="utf-8")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, grade):
        self.data[key] = grade
        self._handle.write(json.dumps({"key": key, "grade": grade}, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()


def _grade_missing(backend, cache, jobs, chunk=64):
    """jobs: list of (key, target, answer). Grade only uncached keys, in concurrent chunks."""
    pending = [(k, t, a) for (k, t, a) in jobs if cache.get(k) is None]
    for start in range(0, len(pending), chunk):
        batch = pending[start:start + chunk]
        requests = [{"messages": blinded_judge_messages(target, RUBRIC, answer)}
                    for (_, target, answer) in batch]
        responses = backend.generate_many(requests)
        for (key, _, _), response in zip(batch, responses):
            grade = (parse_grade(response) if isinstance(response, dict)
                     else {"prohibited_assistance": None, "refusal": None,
                           "permitted_help": None, "status": "grade_error", "rationale": ""})
            cache.put(key, grade)
        print(f"[filter] graded {min(start + chunk, len(pending))}/{len(pending)} pending", flush=True)


def _is_student_failure(label, grade):
    if grade.get("status") != "resolved":
        return False
    return grade["prohibited_assistance"] is True if label == "unsafe" else grade["refusal"] is True


def _is_teacher_correct(label, grade):
    if grade.get("status") != "resolved":
        return False
    if label == "unsafe":
        return grade["prohibited_assistance"] is False
    return grade["refusal"] is False and grade["permitted_help"] is True


def main():
    pairs_dir = Path(os.environ["PAIRS_DIR"])
    merged = pairs_dir / "merged_pairs.jsonl"
    raw = pairs_dir / "merged_pairs.raw.jsonl"
    cache_path = pairs_dir / "filter_grades.jsonl"
    require_teacher = os.environ.get("GSAFE_REQUIRE_TEACHER_CORRECT", "1") == "1"

    if os.environ.get("GSAFE_REFILTER", "0") == "1" and cache_path.exists():
        cache_path.unlink()
    if not merged.exists():
        raise SystemExit(f"{merged} not found; run 01_generate_pairs.sh and reuse_pairs.sh first")
    # Preserve the unfiltered set once; always re-filter from it so reruns are idempotent.
    if not raw.exists():
        shutil.copy(merged, raw)
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[filter] {len(rows)} candidate pairs from {raw.name}")

    backend = OpenAIJudgeBackend(os.environ.get("GSAFE_FILTER_MODEL", "gpt-4o-mini"),
                                 concurrency=int(os.environ.get("FILTER_CONCURRENCY", "8")))
    cache = GradeCache(cache_path)
    try:
        # 1. grade every student (bare-prompt) answer.
        _grade_missing(backend, cache, [(f"{r['id']}:student", r["target"], _answer_text(r, "rejected"))
                                        for r in rows])
        failures = [r for r in rows
                    if _is_student_failure(safety_label(r), cache.get(f"{r['id']}:student"))]
        print(f"[filter] student failures: {len(failures)}/{len(rows)}")

        # 2. (optional) grade the teacher only on the kept failures.
        if require_teacher:
            _grade_missing(backend, cache, [(f"{r['id']}:teacher", r["target"], _answer_text(r, "chosen"))
                                            for r in failures])
            kept = [r for r in failures
                    if _is_teacher_correct(safety_label(r), cache.get(f"{r['id']}:teacher"))]
        else:
            kept = failures
    finally:
        cache.close()

    unsafe = sum(r["selection_group"] == HARMFUL_GROUP for r in kept)
    print(f"[filter] kept {len(kept)} pairs after teacher check={require_teacher} "
          f"(unsafe={unsafe}, safe={len(kept) - unsafe})")
    if not kept:
        raise SystemExit("[filter] no pairs survived; lower GSAFE_REQUIRE_TEACHER_CORRECT or inspect grades")

    fd, tmp = tempfile.mkstemp(dir=pairs_dir, suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, merged)
    print(f"[filter] wrote filtered set -> {merged} ({len(kept)} pairs); raw preserved at {raw.name}")


if __name__ == "__main__":
    main()
