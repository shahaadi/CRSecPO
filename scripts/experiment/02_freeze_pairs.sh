#!/usr/bin/env bash
# Stage 2: freeze ONE balanced 50/50 (safe vs harmful) training set from the
# generated pairs. harmful = harmful_benign_framing; everything else is benign.
# This is the frozen common selection every arm trains on.
#
# Input: merged_pairs.jsonl if present (from reuse_pairs.sh, which folds existing
# train+validation pairs without regeneration), otherwise train_pairs.jsonl (the
# merged set stage 1 produces).
set -euo pipefail
# Capture the script dir before config.sh changes the working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "== Stage 2: freeze balanced train selection =="
python "${SCRIPT_DIR}/_make_train_config.py"

FREEZE_INPUT="${PAIRS_DIR}/train_pairs.jsonl"
[ -f "${PAIRS_DIR}/merged_pairs.jsonl" ] && FREEZE_INPUT="${PAIRS_DIR}/merged_pairs.jsonl"
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
