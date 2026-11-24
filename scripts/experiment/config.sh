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
# The RAW splits, before any API filtering. 00b reads these and writes filtered ones.
export ORIG_SPLIT_DIR="${SPLIT_DIR}"
# Default to the split dir, but respect an explicit override (the general-safety
# students-first pass points stage 1 at reduced failures_<split>.jsonl candidate files).
export TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${SPLIT_DIR}/train.jsonl}"
export VAL_CANDIDATES="${VAL_CANDIDATES:-${SPLIT_DIR}/validation.jsonl}"
export TEST_CANDIDATES="${TEST_CANDIDATES:-${SPLIT_DIR}/test.jsonl}"

# ---- outputs ----
# IMPORTANT: the pipeline's crash-safe writers use hard links (os.link) for the
# journals and manifests that make every stage resumable. Google Drive's FUSE
# mount does NOT support hard links, so WORK must be on real local disk. If you
# point WORK at Drive, we transparently run on local disk and use the Drive path
# as a backup target (restored on a fresh runtime, synced after each stage), so
# you still survive disconnects without feeding Drive to the atomic writers.
export WORK="${WORK:-${REPO_ROOT}/experiment_out}"
export DRIVE_BACKUP="${DRIVE_BACKUP:-}"
if [[ "${WORK}" == */drive/* ]]; then
  DRIVE_BACKUP="${WORK}"
  WORK="/content/crsecpo_out"
  echo "[config] WORK was on Google Drive (no hard-link support); running on ${WORK} and backing up to ${DRIVE_BACKUP}"
fi
if [ -n "${DRIVE_BACKUP}" ] && [ ! -d "${WORK}" ] && [ -d "${DRIVE_BACKUP}" ]; then
  echo "[config] restoring previous run from ${DRIVE_BACKUP} ..."
  mkdir -p "${WORK}" && rsync -a "${DRIVE_BACKUP}/" "${WORK}/" || true
fi
# Mirror local work to the Drive backup (no-op when DRIVE_BACKUP is unset).
backup_work () {
  [ -n "${DRIVE_BACKUP:-}" ] || return 0
  mkdir -p "${DRIVE_BACKUP}" 2>/dev/null || true
  rsync -a "${WORK}/" "${DRIVE_BACKUP}/" 2>/dev/null || true
}
export -f backup_work 2>/dev/null || true
export WORK DRIVE_BACKUP
export PAIRS_DIR="${WORK}/pairs"
export FROZEN_DIR="${WORK}/frozen_train"      # holds the balanced 50/50 train_pairs.jsonl
export TRAIN_OUT="${WORK}/checkpoints"        # per-method adapters land in $TRAIN_OUT/<method>
export EVAL_DIR="${WORK}/eval"                # per-checkpoint safety runs
export AGG_DIR="${WORK}/aggregate"
export CFG_DIR="${WORK}/configs"              # generated configs
export FILTERED_DIR="${WORK}/filtered_splits" # 00b writes API-validated splits here
# Once 00b has produced filtered splits, every later stage uses them automatically.
if [ -s "${FILTERED_DIR}/train.jsonl" ]; then
  SPLIT_DIR="${FILTERED_DIR}"
  TRAIN_CANDIDATES="${SPLIT_DIR}/train.jsonl"
  VAL_CANDIDATES="${SPLIT_DIR}/validation.jsonl"
  TEST_CANDIDATES="${SPLIT_DIR}/test.jsonl"
  echo "[config] using API-filtered splits in ${FILTERED_DIR}"
fi

# ---- compute knobs ----
# Reduce CUDA allocator fragmentation (frees reserved-but-unallocated memory);
# this is what lets CRSecPO's four forward passes fit on a 24 GB card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PAIRS_BACKEND="${PAIRS_BACKEND:-vllm}" # 'vllm' (batched, GPU) or 'hf' (sequential)
export DEVICE="${DEVICE:-cuda}"
# 0 = judge-free: accept chosen=combined/rejected=ordinary whenever they differ
# (relies on the combined answer being generally better). 1 = run the preference judge.
export USE_JUDGE="${USE_JUDGE:-0}"
# Stage-4 grading judge. Empty = local 8B judge. Set to a cheap OpenAI model
# (e.g. gpt-4o-mini) to grade via API -- also set OPENAI_API_KEY in the env.
export OPENAI_JUDGE_MODEL="${OPENAI_JUDGE_MODEL:-}"
# Stage-0b pair filter (validates that each target is an equivalent reframing of its
# source and keeps the safe/unsafe label). A cheap classifier is plenty -- gpt-4o-mini.
export OPENAI_FILTER_MODEL="${OPENAI_FILTER_MODEL:-gpt-4o-mini}"
export FILTER_CONCURRENCY="${FILTER_CONCURRENCY:-8}"
# Stage-4 answer generation engine. 'vllm' batches all test prompts through one
# vLLM engine (base + LoRA adapters), far faster than 'hf' (one prompt at a time).
# Outputs differ numerically from HF, so each engine keeps its own journal/identity.
export EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
# How many generations/grades to journal per chunk during stage-4 eval (vLLM path):
# a mid-run disconnect loses at most one chunk, then resumes. Bigger = fewer flushes.
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
# Stage-4 repair budgets. After the normal eval, 04 regenerates ONLY the answers
# that truncated (finish_reason=length) at this larger victim budget, and re-grades
# ONLY the answers whose judge call truncated/erred with this larger judge budget --
# reusing every answer/grade that already succeeded (completed greedy answers are
# budget-invariant, so reuse is exact).
export REPAIR_MAX_NEW_TOKENS="${REPAIR_MAX_NEW_TOKENS:-4096}"
export REPAIR_JUDGE_TOKENS="${REPAIR_JUDGE_TOKENS:-16000}"
# Tokens reserved for the prompt when repairing under vLLM: the regenerated answer
# budget is capped at (vllm_max_model_len - this) so prompt+answer fit the context.
# An item whose prompt is longer than this reserve is left unrepaired (not aborted).
export REPAIR_PROMPT_RESERVE="${REPAIR_PROMPT_RESERVE:-1024}"
# Set EVAL_REGRADE=1 to re-grade every checkpoint with the CURRENT judge prompt,
# reusing the existing generations (no regeneration). Use after editing the judge
# rubric/prompt: run `EVAL_REGRADE=1 bash 04_eval_safety.sh` then `bash 05_aggregate.sh`.
export EVAL_REGRADE="${EVAL_REGRADE:-0}"
# Start SMALL to prove the pipeline end to end, then set to 'all' for the real run.
export MAX_CANDIDATES="${MAX_CANDIDATES:-20}"
export PAIR_LIMIT="${PAIR_LIMIT:-2048}"       # even; cap on the balanced train set
export SEED="${SEED:-42}"
# save/eval cadence for best-checkpoint selection (save_steps must be a multiple of eval_steps)
export EVAL_STEPS="${EVAL_STEPS:-8}"
export SAVE_STEPS="${SAVE_STEPS:-8}"
# ---- over-refusal tuning dials (empty = template defaults: beta 0.1, lr 1.6e-4, 1 epoch) ----
# These move crsecpo ALONG the tradeoff curve: lower beta/lr or fewer epochs -> closer to
# base -> less over-refusal but a smaller safety gain. Applied to all three arms equally.
export TRAIN_BETA="${TRAIN_BETA:-}"      # e.g. 0.05 for a softer preference (less over-refusal)
export TRAIN_LR="${TRAIN_LR:-}"          # e.g. 0.00008 for gentler updates
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"  # e.g. 1 (default); fractional allowed via max_steps upstream
# Fraction of the balanced train set that should be benign/safe pairs (the other part is
# harmful). 0.5 = the default 50/50. Raise it (e.g. 0.6) to show more "help the suspicious
# benign request" examples, directly countering over-refusal (at some cost to harmful safety).
export BENIGN_FRACTION="${BENIGN_FRACTION:-0.5}"

# ---- GPU auto-tuning (automatic; you don't set anything) ----
# Reads the card's VRAM and picks safe memory settings. A100/40GB+ gets the full
# setup; 24GB cards (L4/4090/3090) get the conservative one so the 8B model fits.
# Any value can still be forced by exporting it before sourcing this file.
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' || true)"
[[ "${GPU_MEM_MB}" =~ ^[0-9]+$ ]] || GPU_MEM_MB=0
if [ "${GPU_MEM_MB}" -ge 38000 ]; then
  GPU_TIER="large (A100/40GB+)"; _VMEM=0.90; _VLEN=8192; _BATCH=2; _ACCUM=32; _CHUNK=256
else
  # On a 24GB card the 8B weights (~15GB) dominate, so a HIGHER fraction is what
  # leaves room for the KV cache (and thus batching throughput), not a lower one.
  # Small chunk so the journal flushes (= visible progress + fine resume) often;
  # vLLM concurrency is KV-limited anyway, so this barely affects throughput.
  GPU_TIER="small (<=24GB, e.g. L4)"; _VMEM=0.92; _VLEN=4096; _BATCH=1; _ACCUM=64; _CHUNK=48
fi
# Named without a VLLM_ prefix so vLLM does not flag them as unknown env vars.
export PAIRS_GPU_MEM="${PAIRS_GPU_MEM:-$_VMEM}"  # vLLM pair-gen GPU fraction
export PAIRS_MAX_LEN="${PAIRS_MAX_LEN:-$_VLEN}"  # vLLM context budget
export BATCH_SIZE="${BATCH_SIZE:-$_CHUNK}"       # vLLM requests per journaled chunk
export TRAIN_BATCH="${TRAIN_BATCH:-$_BATCH}"     # per-device train batch
export GRAD_ACCUM="${GRAD_ACCUM:-$_ACCUM}"       # accumulation (effective batch = BATCH*ACCUM = 64)
echo "[config] gpu=${GPU_MEM_MB}MB tier=${GPU_TIER}; vllm_mem=${PAIRS_GPU_MEM} vllm_len=${PAIRS_MAX_LEN} chunk=${BATCH_SIZE} batch=${TRAIN_BATCH} accum=${GRAD_ACCUM}"

mkdir -p "${WORK}" "${PAIRS_DIR}" "${TRAIN_OUT}" "${EVAL_DIR}" "${AGG_DIR}" "${CFG_DIR}"

echo "[config] repo=${REPO_ROOT}"
echo "[config] model=${MODEL_ID}@${MODEL_REVISION:0:12} backend=${PAIRS_BACKEND} max_candidates=${MAX_CANDIDATES}"
echo "[config] work=${WORK}"
