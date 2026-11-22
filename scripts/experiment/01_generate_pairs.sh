#!/usr/bin/env bash
# Stage 1: generate preference pairs for TRAIN and VALIDATION.
#
# For each candidate the model produces an "ordinary" answer (bare framed query)
# and a "combined" answer (rubric + the vanilla counterfactual in context), then
# a swapped-order judge decides. Accepted pairs become (chosen=combined,
# rejected=ordinary). Test is NOT generated here -- it is the held-out eval set.
#
# The journal makes this resumable: re-run after an interruption and it continues.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "== Stage 1: generate pairs (backend=${PAIRS_BACKEND}, max_candidates=${MAX_CANDIDATES}) =="

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
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEM}" --vllm-max-model-len "${VLLM_MAX_LEN}"
}

gen train      "${TRAIN_CANDIDATES}"
gen validation "${VAL_CANDIDATES}"

echo "Train pairs:      ${PAIRS_DIR}/train_pairs.jsonl"
echo "Validation pairs: ${PAIRS_DIR}/validation_pairs.jsonl"
echo "NOTE: acceptance is strict, so accepted pairs << candidates. If a split yields"
echo "      too few, raise MAX_CANDIDATES in config.sh (use 'all' for the full run)."
