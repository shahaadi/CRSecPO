"""Explain why stage-4 grades are unresolved (which collapses the contrasts).

For each checkpoint it tallies the victim finish_reason and the grade status, then
prints a few raw judge responses. Read it like this:

- Lots of victim finish_reason == "length"  -> generations are truncated; raise
  the victim max_new_tokens in configs/crsecpo_evaluation.json ("generation").
- grade.status mostly "truncated_or_incomplete" -> the JUDGE response was cut off
  (common with a reasoning model: hidden reasoning eats the token budget and the
  JSON never lands). Raise the judge token budget / lower reasoning effort.
- grade.status mostly "invalid_judgment" -> the judge emitted the wrong JSON shape.
- grade.status mostly "resolved" -> grades are fine; the contrast issue is elsewhere.

Env in: EVAL_DIR (set by config.sh).
"""
import collections
import json
import os
from pathlib import Path

EVAL = Path(os.environ["EVAL_DIR"])
CHECKPOINTS = ("base", "sft", "dpo", "crsecpo")


def rows(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            yield json.loads(line)


for ck in CHECKPOINTS:
    results = EVAL / ck / "results.jsonl"
    if not results.exists():
        print(f"[{ck}] no results.jsonl yet")
        continue
    finish = collections.Counter()
    status = collections.Counter()
    total = 0
    for row in rows(results):
        total += 1
        finish[row.get("finish_reason")] += 1
        status[(row.get("grade") or {}).get("status")] += 1
    print(f"[{ck}] {total} results")
    print(f"     victim finish_reason: {dict(finish)}")
    print(f"     grade.status:         {dict(status)}")

# Show a few raw judge responses from the first checkpoint that has them.
for ck in CHECKPOINTS:
    judge = EVAL / ck / "judge_requests.jsonl"
    if not judge.exists():
        continue
    print(f"\n--- sample judge responses ({ck}) ---")
    shown = 0
    for entry in rows(judge):
        if entry.get("status") != "ok":
            print(f"  [error] {entry.get('error')}")
        else:
            response = entry["response"]
            text = response.get("answer_text", response.get("text", ""))
            print(f"  finish={response.get('finish_reason')!r} text={text[:240]!r}")
        shown += 1
        if shown >= 4:
            break
    break
