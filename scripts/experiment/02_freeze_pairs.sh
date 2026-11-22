#!/usr/bin/env bash
# Stage 2: freeze ONE balanced 50/50 (safe vs harmful) training set from the
# generated train pairs. harmful = harmful_benign_framing; everything else is
# benign. This is the frozen common selection every arm trains on.
#
# Validation pairs are NOT frozen or balanced -- they are used as-is for
# best-checkpoint selection in stage 3.
set -euo pipefail
# Capture the script dir before config.sh changes the working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "== Stage 2: freeze balanced train selection =="
python "${SCRIPT_DIR}/_make_train_config.py"

# --freeze-pairs reads config.data_file; we override it to the raw train pairs.
# FROZEN_DIR must NOT already exist (freeze refuses to overwrite a selection).
rm -rf "${FROZEN_DIR}"
python -m secpo.crsecpo.training \
  --config "${CFG_DIR}/train_experiment.json" \
  --freeze-pairs \
  --data-file "${PAIRS_DIR}/train_pairs.jsonl" \
  --output-dir "${FROZEN_DIR}" \
  --pair-limit "${PAIR_LIMIT}"

backup_work
echo "Frozen balanced set: ${FROZEN_DIR}/train_pairs.jsonl (+ selection_manifest.json)"
