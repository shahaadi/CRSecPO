"""Students-first selection (optimization #1).

Reads the student-only pre-pass outcomes (produced by 01a_student_pass.sh, which ran
pairs.py --conditions ordinary), grades each student (bare-prompt) answer with the
blinded judge, and writes the REDUCED candidate files containing only the base model's
failures:

    unsafe prompt and prohibited_assistance == True   (model complied)
    safe   prompt and refusal             == True     (model over-refused)

Only these failures then go through the expensive teacher (combined) generation in the
normal stage 1, so we never pay for a teacher answer on a prompt the model already
handles correctly. The student grades are written to the SAME cache the teacher-check
filter reads (filter_grades.jsonl), so no student answer is graded twice.

Env: PAIRS_DIR, SPLIT_DIR (writes failures_<split>.jsonl there), GSAFE_FILTER_MODEL,
     FILTER_CONCURRENCY, GSAFE_USE_BATCH.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from secpo.crsecpo.pairs import CANDIDATE_FIELDS
from secpo.crsecpo.prompts import safety_label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsafe_grading import GradeCache, grade_jobs, strip_terminator  # noqa: E402

HARMFUL_GROUP = "harmful_benign_framing"
SPLITS = ("train", "validation")


def _student_answer(outcome):
    gen = (outcome.get("generations") or {}).get("ordinary")
    if not gen or "response" not in gen or not isinstance(gen["response"].get("text"), str):
        return None  # generation errored -> cannot determine a failure
    return strip_terminator(gen["response"]["text"], outcome.get("completion_terminator", ""))


def _is_failure(label, grade):
    if not grade or grade.get("status") != "resolved":
        return False
    return grade["prohibited_assistance"] is True if label == "unsafe" else grade["refusal"] is True


def main():
    pairs_dir, split_dir = Path(os.environ["PAIRS_DIR"]), Path(os.environ["SPLIT_DIR"])
    cache = GradeCache(pairs_dir / "filter_grades.jsonl")
    per_split, jobs = {}, []
    for split in SPLITS:
        path = pairs_dir / f"studentpass_{split}_outcomes.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found; run 01a_student_pass.sh first")
        outcomes = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        per_split[split] = outcomes
        for o in outcomes:
            answer = _student_answer(o)
            if answer is not None:
                jobs.append((f"{o['id']}:student", o["target"], answer))
    print(f"[select] grading {len(jobs)} student answers across {', '.join(SPLITS)}")

    try:
        if not grade_jobs(jobs, cache, label="student"):
            print("[select] student grading batch still running; re-run this step to resume")
            return
    finally:
        cache.close()

    total = kept = 0
    for split, outcomes in per_split.items():
        failures = [o for o in outcomes
                    if _student_answer(o) is not None
                    and _is_failure(safety_label(o), cache.get(f"{o['id']}:student"))]
        out = split_dir / f"failures_{split}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for o in failures:
                handle.write(json.dumps({k: o[k] for k in CANDIDATE_FIELDS}, ensure_ascii=False) + "\n")
        unsafe = sum(o["selection_group"] == HARMFUL_GROUP for o in failures)
        total += len(outcomes); kept += len(failures)
        print(f"[select] {split:<10} {len(failures):>6}/{len(outcomes)} failures "
              f"(unsafe={unsafe}, safe={len(failures) - unsafe}) -> {out.name}")
    if not kept:
        raise SystemExit("[select] no failures found; nothing to train on")
    print(f"[select] total failures {kept}/{total} -> teacher generation runs on these only")


if __name__ == "__main__":
    main()
