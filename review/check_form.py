"""Check a completed CRSecPO review form before handing it back.

Uses the same validation as the coordinator's ingest step, but needs only the
blank form shipped in this folder (no answer key).

    python review/check_form.py audit review/human-audit/human_1.jsonl my_human_1.jsonl
    python review/check_form.py relations review/relation-review/relations.jsonl my_relations.jsonl
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secpo.crsecpo.preparation import AUDIT_FIELDS, _filled_form, _validate_audit  # noqa: E402
from secpo.evaluation.sep_judge import load_jsonl  # noqa: E402


def main(argv):
    if len(argv) != 3 or argv[0] not in {"audit", "relations"}:
        sys.exit(__doc__)
    kind, blank, filled = argv[0], load_jsonl(Path(argv[1])), Path(argv[2])
    fields = {*AUDIT_FIELDS, "reason"} if kind == "audit" else {"decision", "reason"}
    try:
        rows = _filled_form(blank, filled, fields)
        for row in rows:
            if kind == "audit":
                try:
                    _validate_audit(row)
                except ValueError as error:
                    raise ValueError(f"{row['review_id'][:12]}: {error}") from None
            elif row["decision"] not in {"link", "no_link", "unsure"}:
                raise ValueError(f"{row['review_id'][:12]}: decision must be link, no_link, or unsure")
    except ValueError as error:
        sys.exit(f"NOT READY: {error}")
    print(f"OK: {len(rows)} rows complete and valid.", dict(Counter(row["decision"] for row in rows)))


if __name__ == "__main__":
    main(sys.argv[1:])
