#!/usr/bin/env bash
# Reuse already-generated pairs instead of re-running stage 1.
#
# If you already ran stage 1 and have BOTH train_pairs.jsonl and
# validation_pairs.jsonl, this folds them into one merged training set
# (PAIRS_DIR/merged_pairs.jsonl) without regenerating any answers. Stage 2 then
# freezes merged_pairs.jsonl automatically. Run this in place of 01, then 02, 03.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "== Reuse: merge existing train + validation pairs (no generation) =="
python "${SCRIPT_DIR}/_merge_pairs.py"
backup_work
echo "Done. Stage 2 will freeze ${PAIRS_DIR}/merged_pairs.jsonl automatically."
