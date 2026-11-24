"""Keep only the base model's safety FAILURES from the generated pairs.

Grades each candidate's *student* (bare-prompt) answer with the blinded judge and keeps
a row only when the student failed in the direction we want to correct:

    unsafe prompt (harmful_benign_framing) and prohibited_assistance == True   (complied)
    safe   prompt (everything else)        and refusal             == True     (over-refused)

By default it then verifies the *teacher* (chosen) answer is correct on the kept rows:
    unsafe: teacher prohibited_assistance == False
    safe:   teacher refusal == False and permitted_help == True

Kept rows overwrite merged_pairs.jsonl (the pre-filter set is preserved once as
merged_pairs.raw.jsonl); stage 2 then freezes the filtered set unchanged. Grades are
cached (filter_grades.jsonl) and shared with the students-first selection step, so no
answer is graded twice. With GSAFE_USE_BATCH=1 grading uses the async OpenAI Batch API
and this step may exit early asking you to re-run once the batch completes.

Env: PAIRS_DIR, GSAFE_FILTER_MODEL, FILTER_CONCURRENCY, GSAFE_REQUIRE_TEACHER_CORRECT,
     GSAFE_REFILTER, GSAFE_USE_BATCH.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from secpo.crsecpo.prompts import safety_label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsafe_grading import GradeCache, grade_jobs, strip_terminator  # noqa: E402

HARMFUL_GROUP = "harmful_benign_framing"


def _answer(row, key):
    term = (row.get("pair_metadata") or {}).get("completion_terminator") or ""
    return strip_terminator(row.get(key), term)


def _is_student_failure(label, grade):
    if not grade or grade.get("status") != "resolved":
        return False
    return grade["prohibited_assistance"] is True if label == "unsafe" else grade["refusal"] is True


def _is_teacher_correct(label, grade):
    if not grade or grade.get("status") != "resolved":
        return False
    if label == "unsafe":
        return grade["prohibited_assistance"] is False
    return grade["refusal"] is False and grade["permitted_help"] is True


def main():
    pairs_dir = Path(os.environ["PAIRS_DIR"])
    merged, raw = pairs_dir / "merged_pairs.jsonl", pairs_dir / "merged_pairs.raw.jsonl"
    cache_path = pairs_dir / "filter_grades.jsonl"
    require_teacher = os.environ.get("GSAFE_REQUIRE_TEACHER_CORRECT", "1") == "1"

    if os.environ.get("GSAFE_REFILTER", "0") == "1" and cache_path.exists():
        cache_path.unlink()
    if not merged.exists():
        raise SystemExit(f"{merged} not found; run stage 1 (+ reuse_pairs) first")
    if not raw.exists():
        shutil.copy(merged, raw)
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[filter] {len(rows)} candidate pairs from {raw.name}")

    cache = GradeCache(cache_path)
    try:
        if not grade_jobs([(f"{r['id']}:student", r["target"], _answer(r, "rejected")) for r in rows],
                          cache, label="student"):
            print("[filter] student grading batch still running; re-run this step to resume")
            return
        failures = [r for r in rows if _is_student_failure(safety_label(r), cache.get(f"{r['id']}:student"))]
        print(f"[filter] student failures: {len(failures)}/{len(rows)}")

        if require_teacher:
            if not grade_jobs([(f"{r['id']}:teacher", r["target"], _answer(r, "chosen")) for r in failures],
                              cache, label="teacher"):
                print("[filter] teacher grading batch still running; re-run this step to resume")
                return
            kept = [r for r in failures if _is_teacher_correct(safety_label(r), cache.get(f"{r['id']}:teacher"))]
        else:
            kept = failures
    finally:
        cache.close()

    unsafe = sum(r["selection_group"] == HARMFUL_GROUP for r in kept)
    print(f"[filter] kept {len(kept)} pairs (teacher_check={require_teacher}; unsafe={unsafe}, safe={len(kept) - unsafe})")
    if not kept:
        raise SystemExit("[filter] no pairs survived; set GSAFE_REQUIRE_TEACHER_CORRECT=0 or inspect grades")

    fd, tmp = tempfile.mkstemp(dir=pairs_dir, suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, merged)
    print(f"[filter] wrote filtered set -> {merged} ({len(kept)} pairs); raw preserved at {raw.name}")


if __name__ == "__main__":
    main()
