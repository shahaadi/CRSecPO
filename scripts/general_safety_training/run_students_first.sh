#!/usr/bin/env bash
# Students-first driver (optimization #1): generate student answers for ALL candidates,
# keep only the base model's failures, and run the expensive teacher (combined) generation
# ONLY on those failures. Saves the teacher generations for every prompt the model already
# handles -- typically the majority -- cutting stage-1 generation substantially.
#
# Same reused stages as run.sh, re-ordered so filtering happens before teacher generation.
# Each stage is idempotent/resumable. Prereqs: HF_TOKEN + OPENAI_API_KEY in the env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
EXP="${REPO_ROOT}/scripts/experiment"
export PYTHONPATH="${REPO_ROOT}/src"
export PAIRS_DIR="${WORK}/pairs"

echo "== [sf] 00: prepare WildJailbreak candidate splits =="
python "${SCRIPT_DIR}/00_prepare_wjb.py"

echo "== [sf] 01a: student-only pre-pass (cheap bare-prompt answers for ALL candidates) =="
bash "${SCRIPT_DIR}/01a_student_pass.sh"

echo "== [sf] select failures -> reduced failures_<split>.jsonl =="
python "${SCRIPT_DIR}/select_student_failures.py"   # if GSAFE_USE_BATCH=1 may ask you to re-run

echo "== [sf] 01: teacher + pair generation on FAILURES ONLY =="
TRAIN_CANDIDATES="${SPLIT_DIR}/failures_train.jsonl" \
VAL_CANDIDATES="${SPLIT_DIR}/failures_validation.jsonl" \
  bash "${EXP}/01_generate_pairs.sh"

echo "== [sf] merge train + validation pairs =="
bash "${EXP}/reuse_pairs.sh"

echo "== [sf] filter: teacher-correctness (student grades reused from the cache) =="
python "${SCRIPT_DIR}/filter_failures.py"

echo "== [sf] 02-05: freeze / train / eval / aggregate =="
bash "${EXP}/02_freeze_pairs.sh"
bash "${EXP}/03_train.sh"
bash "${EXP}/04_eval_safety.sh"
bash "${EXP}/05_aggregate.sh"
echo "== [sf] done; artifacts in ${WORK} =="
