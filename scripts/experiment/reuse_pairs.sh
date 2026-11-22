#!/usr/bin/env bash
# Reuse already-generated pairs instead of re-running stage 1.
#
# Stage 1 produces train_pairs.jsonl and validation_pairs.jsonl as SEPARATE
# exports; this folds BOTH into one merged training set
# (PAIRS_DIR/merged_pairs.jsonl) without regenerating any answers. It is the
# REQUIRED bridge between stage 1 and stage 2: stage 2 freezes merged_pairs.jsonl
# and refuses to proceed without it (so validation is never silently dropped).
# Order:  bash 01_generate_pairs.sh && bash reuse_pairs.sh && bash 02_freeze_pairs.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "== Reuse: merge existing train + validation pairs (no generation) =="
MERGED="${PAIRS_DIR}/merged_pairs.jsonl"
if [ -s "${MERGED}" ]; then
  echo "merged set already present: ${MERGED} ($(wc -l < "${MERGED}") pairs); skipping merge"
else
  python "${SCRIPT_DIR}/_merge_pairs.py"
  backup_work
fi
echo "Done. Stage 2 freezes ${MERGED}."
