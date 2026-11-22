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

train_method () {  # $1 = method; remaining args passed through (e.g. resume)
  local m="$1"; shift
  python -m secpo.crsecpo.training \
    --config "${CFG_DIR}/train_experiment.json" \
    --method "${m}" \
    --data-file "${FROZEN_DIR}/train_pairs.jsonl" \
    --output-dir "${TRAIN_OUT}/${m}" "$@"
}

for METHOD in sft dpo crsecpo; do
  OUT="${TRAIN_OUT}/${METHOD}"
  # Final adapter at the dir root means this arm already finished -> skip it.
  if [ -f "${OUT}/adapter_config.json" ]; then
    echo "--- ${METHOD}: already complete, skipping ---"
    continue
  fi
  # Otherwise resume from the newest saved checkpoint if one exists.
  latest="$(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1 || true)"
  if [ -n "${latest}" ]; then
    echo "--- ${METHOD}: resuming from checkpoint-${latest} ---"
    train_method "${METHOD}" --resume-from-checkpoint "${OUT}/checkpoint-${latest}"
  else
    # No finished adapter and no resumable checkpoint: anything here is a stale
    # manifest from a failed earlier attempt (e.g. different pairs/settings).
    # Clear it so the run isn't blocked by the manifest-mismatch guard.
    rm -rf "${OUT}"
    echo "--- ${METHOD}: training from scratch ---"
    train_method "${METHOD}"
  fi
  echo "adapter: ${OUT}"
  backup_work
done

echo "Trained adapters under ${TRAIN_OUT}/{sft,dpo,crsecpo}."
