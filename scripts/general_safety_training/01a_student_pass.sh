#!/usr/bin/env bash
# Students-first optimization, step 1: generate ONLY the student (bare-prompt) answer
# for every candidate, so we can discover the base model's failures before paying for
# any teacher (combined) generations. Uses pairs.py --conditions ordinary (forms no
# pairs; just journals the ordinary answers + writes outcomes we grade next).
#
# Own journal (studentpass_<split>_journal.jsonl), separate from the normal stage-1
# journal, so there is no cross-run journal sharing to reason about.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
source "${REPO_ROOT}/scripts/experiment/config.sh"

echo "== Stage 1a: student-only pre-pass (no pairs; journals ordinary answers) =="

gen_students () {
  local split="$1" cand="$2"
  [ -s "${cand}" ] || { echo "ERROR: missing candidates ${cand}" >&2; exit 1; }
  echo "--- ${split}: student answers from ${cand} ---"
  python -m secpo.crsecpo.pairs "${cand}" \
    --output   "${PAIRS_DIR}/studentpass_${split}.jsonl" \
    --outcomes "${PAIRS_DIR}/studentpass_${split}_outcomes.jsonl" \
    --journal  "${PAIRS_DIR}/studentpass_${split}_journal.jsonl" \
    --max-candidates "${MAX_CANDIDATES}" \
    --conditions ordinary --no-judge \
    --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}" \
    --tokenizer-id "${TOKENIZER_ID}" --tokenizer-revision "${TOKENIZER_REVISION}" \
    --seed "${SEED}" --device "${DEVICE}" --backend "${PAIRS_BACKEND}" \
    --vllm-gpu-memory-utilization "${PAIRS_GPU_MEM}" --vllm-max-model-len "${PAIRS_MAX_LEN}" \
    --batch-size "${BATCH_SIZE}"
}

gen_students train      "${TRAIN_CANDIDATES}"
gen_students validation "${VAL_CANDIDATES}"
backup_work
echo "Student pre-pass done. Next: select_student_failures.py -> failures_<split>.jsonl"
