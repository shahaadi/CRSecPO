#!/usr/bin/env bash
# Stage 4: run the blinded safety evaluation on the held-out TEST split, once per
# checkpoint (base + the three trained arms), then REPAIR only the items that
# failed. Each run generates answers on the test prompts and grades them
# (prohibited_assistance / refusal / permitted_help) with the frozen judge. base
# uses no adapter; the others point at their adapter.
#
# Repair pass: an answer that truncated (finish_reason=length) or a grade the judge
# truncated/erred on is regenerated/re-graded at a larger budget, reusing every
# answer and grade that already succeeded. Completed greedy answers are
# budget-invariant, so that reuse is exact -- only the failed items are recomputed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "== Stage 4: safety evaluation on test =="

# Each checkpoint can take tens of minutes, so mirror the journals to the Drive
# backup every 2 min (and on exit), not just after each checkpoint -- a disconnect
# mid-checkpoint then loses little. The eval is journaled, so the resumed run
# replays completed generations/grades.
if [ -n "${DRIVE_BACKUP:-}" ]; then
  ( while true; do sleep 120; backup_work; done ) & SYNC_PID=$!
  trap 'kill ${SYNC_PID} 2>/dev/null || true; backup_work' EXIT
fi

make_eval_config () {  # $1 = checkpoint, $2 = adapter_dir (""), $3 = mode (normal|repair) -> prints config path
  REPO_ROOT="${REPO_ROOT}" CK="$1" ADAPTER="$2" MODE="${3:-normal}" \
  OUT="${CFG_DIR}/eval_$1${3:+_$3}.json" CFG_DIR="${CFG_DIR}" EVAL_DIR="${EVAL_DIR}" \
  OPENAI_JUDGE_MODEL="${OPENAI_JUDGE_MODEL:-}" EVAL_BACKEND="${EVAL_BACKEND:-hf}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}" PAIRS_GPU_MEM="${PAIRS_GPU_MEM:-0.9}" \
  PAIRS_MAX_LEN="${PAIRS_MAX_LEN:-8192}" REPAIR_MAX_NEW_TOKENS="${REPAIR_MAX_NEW_TOKENS:-4096}" \
  REPAIR_JUDGE_TOKENS="${REPAIR_JUDGE_TOKENS:-16000}" REPAIR_PROMPT_RESERVE="${REPAIR_PROMPT_RESERVE:-1024}" \
  python "${SCRIPT_DIR}/_make_eval_config.py"
}

eval_safety () {  # $1 = checkpoint, $2 = config, $3 = output-dir, plus any extra flags
  local ck="$1" cfg="$2" out="$3"; shift 3
  python -m secpo.crsecpo.evaluation safety \
    --config "${cfg}" --input "${TEST_CANDIDATES}" --checkpoint "${ck}" --output-dir "${out}" "$@"
}

count_failed () { python "${SCRIPT_DIR}/_count_failed.py" "$1/results.jsonl"; }

run_ck () {  # $1 = checkpoint label, $2 = adapter_dir ("" for base)
  local ck="$1" adapter="$2"
  local cfg; cfg="$(make_eval_config "${ck}" "${adapter}" normal)"
  local out="${EVAL_DIR}/${ck}" log="${CFG_DIR}/eval_${ck}.log"
  local applied="${CFG_DIR}/eval_${ck}.applied.json"
  echo "--- safety: ${ck} ---"
  # Skip (no model load) only if this checkpoint already finished under the SAME
  # config AND has nothing left to repair.
  if [ -s "${out}/summary.json" ] && [ -f "${applied}" ] && cmp -s "${cfg}" "${applied}" \
     && [ "$(count_failed "${out}")" = "0" ]; then
    echo "    already evaluated with the current config; skipping"
    return 0
  fi
  # Journaled resume: a matching dir REPLAYS (no regeneration). If the stored
  # protocol does NOT match (e.g. vLLM/transformers was upgraded, the judge model
  # changed, or the eval config was edited), the dir cannot be resumed. We NEVER
  # silently delete it -- doing so would scrap finished generations. By default we
  # stop and explain; regeneration from scratch is strictly opt-in.
  if ! eval_safety "${ck}" "${cfg}" "${out}" 2>&1 | tee "${log}"; then
    if grep -q "resume protocol hash mismatch" "${log}"; then
      if [ "${EVAL_FORCE_FRESH:-0}" = "1" ]; then
        echo "    ${ck}: protocol changed AND EVAL_FORCE_FRESH=1 -> archiving the old dir and regenerating" >&2
        mv "${out}" "${out}.superseded.$(ls -d "${out}".superseded.* 2>/dev/null | wc -l | tr -d ' ')"
        eval_safety "${ck}" "${cfg}" "${out}"
      else
        echo "" >&2
        echo "!! ${ck}: the stored eval protocol does not match the current one, so this dir" >&2
        echo "   CANNOT be resumed. NOTHING was deleted -- your finished generations are intact." >&2
        echo "   This almost always means the environment changed since you built it: a" >&2
        echo "   different vLLM/transformers version, a different OPENAI_JUDGE_MODEL, or an" >&2
        echo "   edited eval config. Restore the matching versions/model and re-run -- it will" >&2
        echo "   then replay. ONLY if you intend to regenerate from scratch, re-run with" >&2
        echo "   EVAL_FORCE_FRESH=1 (which ARCHIVES the old dir to ${out}.superseded.*, not delete)." >&2
        exit 1
      fi
    else
      echo "    (eval failed for ${ck}; see ${log})" >&2
      exit 1
    fi
  fi
  cp -f "${cfg}" "${applied}"
}

common_budget () {  # the single victim budget every checkpoint must end at, so aggregate accepts them
  if [ "${EVAL_BACKEND:-hf}" = "vllm" ]; then
    local cap=$(( PAIRS_MAX_LEN - REPAIR_PROMPT_RESERVE ))
    if [ "${REPAIR_MAX_NEW_TOKENS}" -lt "${cap}" ]; then echo "${REPAIR_MAX_NEW_TOKENS}"; else echo "${cap}"; fi
  else
    echo "${REPAIR_MAX_NEW_TOKENS}"
  fi
}
dir_budget () {  # the max_new_tokens recorded in a dir's protocol.json ("" if none)
  [ -f "$1/protocol.json" ] || { echo ""; return; }
  python -c "import json,sys; print(json.load(open(sys.argv[1]+'/protocol.json'))['generation_config'].get('max_new_tokens',''))" "$1"
}

repair_ck () {  # $1 = checkpoint label, $2 = adapter_dir ("" for base)
  local ck="$1" adapter="$2"
  local out="${EVAL_DIR}/${ck}"
  [ -s "${out}/results.jsonl" ] || return 0
  local failed target current
  failed="$(count_failed "${out}")"
  target="$(common_budget)"
  current="$(dir_budget "${out}")"
  # Every checkpoint must end at the SAME victim budget or the aggregate rejects the
  # run ("different frozen evaluation protocols"). So repair runs when there are
  # failures OR when this dir is still at a different budget; a 0-failure checkpoint
  # already at the common budget is left untouched.
  if [ "${failed}" = "0" ] && [ "${current}" = "${target}" ]; then
    echo "    ${ck}: complete and at the common budget (${target}); nothing to do"
    return 0
  fi
  if [ "${failed}" = "0" ]; then
    echo "    ${ck}: aligning to the common budget ${target} (was ${current:-?}); re-keys finished answers, no regeneration"
  else
    echo "    ${ck}: ${failed} failed item(s); regenerating those + aligning to the common budget ${target} (reusing the rest)"
  fi
  local cfg; cfg="$(make_eval_config "${ck}" "${adapter}" repair)"
  local repair="${out}.repair"; rm -rf "${repair}"
  # The repair writes to a SEPARATE dir and only reads the original (via adopt). The
  # original is replaced solely after the repaired dir is confirmed to cover the exact
  # same jobs, so a bad repair can never scrap the finished work.
  if eval_safety "${ck}" "${cfg}" "${repair}" --adopt-completed-from "${out}"; then
    local before after
    before="$(wc -l < "${out}/results.jsonl")"
    after="$(wc -l < "${repair}/results.jsonl" 2>/dev/null || echo 0)"
    if [ "${after}" != "${before}" ]; then
      echo "    ${ck}: repaired dir covers ${after} jobs vs ${before}; keeping the ORIGINAL (inspect ${repair})" >&2
      return 1
    fi
    # Crash-safe swap: the original is moved aside first, so the dir is never absent.
    mv "${out}" "${out}.replacing" && mv "${repair}" "${out}" && rm -rf "${out}.replacing"
    cp -f "${cfg}" "${CFG_DIR}/eval_${ck}.applied.json"
    echo "    ${ck}: repaired; $(count_failed "${out}") item(s) still unresolved (raise REPAIR_MAX_NEW_TOKENS if any remain)"
  else
    echo "    ${ck}: repair run failed; keeping the ORIGINAL dir unchanged (inspect ${repair})" >&2
    return 1
  fi
}

eval_ck () { run_ck "$1" "$2"; repair_ck "$1" "$2"; backup_work; }

eval_ck base    ""
eval_ck sft     "${TRAIN_OUT}/sft"
eval_ck dpo     "${TRAIN_OUT}/dpo"
eval_ck crsecpo "${TRAIN_OUT}/crsecpo"

echo "Per-checkpoint results under ${EVAL_DIR}/{base,sft,dpo,crsecpo}."
