#!/usr/bin/env bash
# Stage 2: freeze ONE balanced 50/50 (safe vs harmful) training set from the
# generated pairs. harmful = harmful_benign_framing; everything else is benign.
# This is the frozen common selection every arm trains on.
#
# Input: merged_pairs.jsonl ONLY -- the one set reuse_pairs.sh produces by folding
# train + validation together. Freezing from train_pairs.jsonl alone would silently
# drop the validation pairs, so reuse_pairs.sh is a hard prerequisite here.
set -euo pipefail
# Capture the script dir before config.sh changes the working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "== Stage 2: freeze balanced train selection =="
python "${SCRIPT_DIR}/_make_train_config.py"

# Idempotent: a frozen selection already here is the one every arm trains on; keep it.
if [ -s "${FROZEN_DIR}/train_pairs.jsonl" ]; then
  echo "frozen balanced set already present: ${FROZEN_DIR}/train_pairs.jsonl; skipping freeze"
  exit 0
fi

FREEZE_INPUT="${PAIRS_DIR}/merged_pairs.jsonl"
if [ ! -s "${FREEZE_INPUT}" ]; then
  echo "ERROR: ${FREEZE_INPUT} is missing. Run 'bash reuse_pairs.sh' first to fold" >&2
  echo "       train + validation pairs into one set before freezing." >&2
  exit 1
fi
echo "freezing from: ${FREEZE_INPUT}"

# --freeze-pairs reads config.data_file; we override it to the pair export.
# FROZEN_DIR must NOT already exist (freeze refuses to overwrite a selection).
rm -rf "${FROZEN_DIR}"
python -m secpo.crsecpo.training \
  --config "${CFG_DIR}/train_experiment.json" \
  --freeze-pairs \
  --data-file "${FREEZE_INPUT}" \
  --output-dir "${FROZEN_DIR}" \
  --pair-limit "${PAIR_LIMIT}"

backup_work
echo "Frozen balanced set: ${FROZEN_DIR}/train_pairs.jsonl (+ selection_manifest.json)"
