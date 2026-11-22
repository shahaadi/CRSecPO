#!/usr/bin/env bash
# Stage 1: generate preference pairs for the TRAIN+VALIDATION set.
#
# Full-epoch training uses no validation-based selection, so the train and
# validation splits are merged into one training set here (test stays held out).
# For each candidate the model produces an "ordinary" answer (bare framed query)
# and a "combined" answer (rubric + the vanilla counterfactual in context);
# judge-free, the pair becomes (chosen=combined, rejected=ordinary).
#
# The journal makes this resumable: re-run after an interruption and it continues.
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

# Merge train + validation into one training candidate set (test stays held out).
python "${SCRIPT_DIR}/_make_trainval.py"
gen train "${WORK}/trainval_candidates.jsonl"

echo "Training pairs (train+validation): ${PAIRS_DIR}/train_pairs.jsonl"
echo "NOTE: with judge-free acceptance, accepted pairs ~= candidates whose two"
echo "      answers differ. If too few, raise MAX_CANDIDATES in config.sh ('all')."
