#!/usr/bin/env bash
# Stage 3: train the three comparable arms on the SAME frozen pairs.
#   sft     - supervised on the chosen answer
#   dpo     - ordinary DPO (reference = plain prompt)
#   crsecpo - benign-reference SecPO (reference = rubric/combined prompt)
# base is NOT trained; it is the frozen model, evaluated directly in stage 4.
#
# All three use the SAME config (only --method differs) so they stay comparable.
# Validation is on: each arm keeps the checkpoint with the best eval_loss.
set -euo pipefail
# Capture the script dir before config.sh changes the working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "== Stage 3: train sft / dpo / crsecpo =="
python "${SCRIPT_DIR}/_make_train_config.py"

for METHOD in sft dpo crsecpo; do
  echo "--- training ${METHOD} ---"
  python -m secpo.crsecpo.training \
    --config "${CFG_DIR}/train_experiment.json" \
    --method "${METHOD}" \
    --data-file "${FROZEN_DIR}/train_pairs.jsonl" \
    --output-dir "${TRAIN_OUT}/${METHOD}"
  echo "adapter: ${TRAIN_OUT}/${METHOD}"
done

echo "Trained adapters under ${TRAIN_OUT}/{sft,dpo,crsecpo}."
