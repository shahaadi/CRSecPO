"""Print how many results in a stage-4 results.jsonl still need repair.

A job needs repair when its victim answer truncated (finish_reason != "stop") or
its grade never resolved because the judge truncated/erred. Deliberate judge
nulls ("unknown") and clean grades ("resolved") are NOT failures. Arg: results path.
"""
import json
import sys
from pathlib import Path

FAILED_GRADES = {"truncated_or_incomplete", "invalid_judgment",
                 "generation_incomplete", "judge_incomplete"}

path = Path(sys.argv[1])
count = 0
if path.exists():
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        status = (row.get("grade") or {}).get("status")
        if status == "degenerate_repetition":
            continue  # manually resolved as a non-response; not a failure, do not re-repair it
        if row.get("finish_reason") != "stop" or status in FAILED_GRADES:
            count += 1
print(count)
