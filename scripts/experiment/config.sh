#!/usr/bin/env bash
# Shared configuration for the CRSecPO experiment scripts. Edit values here;
# every other script sources this file. Nothing below runs a model.
set -euo pipefail

# ---- repo / python ----
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"

# ---- pinned base model (the frozen generator, judge, and policy base) ----
# Must be Llama; pair generation refuses anything else. Revision is an immutable
# 40-char commit SHA. Update both if you move to a different pinned snapshot.
export MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_REVISION="${MODEL_REVISION:-0e9e39f249a16976918f6564b8830bc894c89659}"
export TOKENIZER_ID="${TOKENIZER_ID:-$MODEL_ID}"
export TOKENIZER_REVISION="${TOKENIZER_REVISION:-$MODEL_REVISION}"

# ---- data (the 70/10/20 preview splits already in the repo) ----
export SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/crsecpo_preview}"
export TRAIN_CANDIDATES="${SPLIT_DIR}/train.jsonl"
export VAL_CANDIDATES="${SPLIT_DIR}/validation.jsonl"
export TEST_CANDIDATES="${SPLIT_DIR}/test.jsonl"

# ---- outputs ----
export WORK="${WORK:-${REPO_ROOT}/experiment_out}"
export PAIRS_DIR="${WORK}/pairs"
export FROZEN_DIR="${WORK}/frozen_train"      # holds the balanced 50/50 train_pairs.jsonl
export TRAIN_OUT="${WORK}/checkpoints"        # per-method adapters land in $TRAIN_OUT/<method>
export EVAL_DIR="${WORK}/eval"                # per-checkpoint safety runs
export AGG_DIR="${WORK}/aggregate"
export CFG_DIR="${WORK}/configs"              # generated configs

# ---- compute knobs ----
export PAIRS_BACKEND="${PAIRS_BACKEND:-vllm}" # 'vllm' (batched, GPU) or 'hf' (sequential)
export DEVICE="${DEVICE:-cuda}"
# Start SMALL to prove the pipeline end to end, then set to 'all' for the real run.
export MAX_CANDIDATES="${MAX_CANDIDATES:-20}"
export PAIR_LIMIT="${PAIR_LIMIT:-2048}"       # even; cap on the balanced train set
export SEED="${SEED:-42}"
# save/eval cadence for best-checkpoint selection (save_steps must be a multiple of eval_steps)
export EVAL_STEPS="${EVAL_STEPS:-8}"
export SAVE_STEPS="${SAVE_STEPS:-8}"

mkdir -p "${WORK}" "${PAIRS_DIR}" "${TRAIN_OUT}" "${EVAL_DIR}" "${AGG_DIR}" "${CFG_DIR}"

echo "[config] repo=${REPO_ROOT}"
echo "[config] model=${MODEL_ID}@${MODEL_REVISION:0:12} backend=${PAIRS_BACKEND} max_candidates=${MAX_CANDIDATES}"
echo "[config] work=${WORK}"
