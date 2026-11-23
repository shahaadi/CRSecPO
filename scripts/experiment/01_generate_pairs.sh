#!/usr/bin/env bash
# Stage 1: generate preference pairs for TRAIN and VALIDATION (as separate
# exports). Test is NOT generated here -- it is the held-out eval set.
#
# For each candidate the model produces an "ordinary" answer (bare framed query)
# and a "combined" answer (rubric + the vanilla source in context); judge-free,
# the pair becomes (chosen=combined, rejected=ordinary).
#
# These two exports are then folded into one training set by reuse_pairs.sh (no
# regeneration) before stage 2 freezes it:
#     bash 01_generate_pairs.sh && bash reuse_pairs.sh && bash 02_freeze_pairs.sh
#
# The journal makes this resumable AND reuse-aware: re-running with the same
# candidates and existing journals replays them instead of regenerating.
set -euo pipefail
# Capture the script dir before config.sh changes the working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

JUDGE_ARG="--no-judge"
[ "${USE_JUDGE}" = "1" ] && JUDGE_ARG=""

echo "== Stage 1: generate pairs (backend=${PAIRS_BACKEND}, max_candidates=${MAX_CANDIDATES}, judge=${USE_JUDGE}) =="

# Pair generation is the long stage. If a Drive backup is configured, mirror the
# journal to it every 2 min (and once more on exit) so a disconnect loses little.
if [ -n "${DRIVE_BACKUP:-}" ]; then
  ( while true; do sleep 120; backup_work; done ) & SYNC_PID=$!
  trap 'kill ${SYNC_PID} 2>/dev/null || true; backup_work' EXIT
fi

gen () {  # $1 = split name, $2 = candidates file
  local split="$1" cand="$2"
  local out="${PAIRS_DIR}/${split}_pairs.jsonl"
  local journal="${PAIRS_DIR}/${split}_journal.jsonl"
  # Skip ONLY migrated data: a pair export with NO generation journal means you dropped
  # in pre-made pairs (reuse_pairs reads them raw, immune to Drive-corrupted sidecars),
  # so there is no reason to re-enter the reservation path. But if a journal EXISTS this
  # split was generated here, so run pairs.py to RESUME: it replays every finished answer
  # and retries only the ones that errored. A fully complete journal re-exports in seconds
  # without even loading the model (the backend starts only if a request is still missing).
  if [ -s "${out}" ] && [ ! -e "${journal}" ]; then
    echo "--- ${split}: ${out} present with no journal (migrated); skipping generation ---"
    return 0
  fi
  if [ -s "${out}" ]; then
    echo "--- ${split}: resuming (replay finished answers, retry errored ones) ---"
  fi
  echo "--- ${split} pairs from ${cand} ---"
  python -m secpo.crsecpo.pairs "${cand}" \
    --output   "${PAIRS_DIR}/${split}_pairs.jsonl" \
    --outcomes "${PAIRS_DIR}/${split}_outcomes.jsonl" \
    --journal  "${PAIRS_DIR}/${split}_journal.jsonl" \
    --max-candidates "${MAX_CANDIDATES}" \
    --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
    --tokenizer-id "${TOKENIZER_ID}" --tokenizer-revision "${TOKENIZER_REVISION}" \
    --seed "${SEED}" --device "${DEVICE}" --backend "${PAIRS_BACKEND}" \
    --vllm-gpu-memory-utilization "${PAIRS_GPU_MEM}" --vllm-max-model-len "${PAIRS_MAX_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    ${JUDGE_ARG}
}

gen train      "${TRAIN_CANDIDATES}"
gen validation "${VAL_CANDIDATES}"

echo "Train pairs:      ${PAIRS_DIR}/train_pairs.jsonl"
echo "Validation pairs: ${PAIRS_DIR}/validation_pairs.jsonl"
echo "Next: bash reuse_pairs.sh (folds these two into merged_pairs.jsonl without"
echo "      regenerating), then bash 02_freeze_pairs.sh."
