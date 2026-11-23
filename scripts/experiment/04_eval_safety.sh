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
  REPAIR_JUDGE_TOKENS="${REPAIR_JUDGE_TOKENS:-16000}" \
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
  # Journaled resume. If the dir holds a protocol from a DIFFERENT configuration
  # (e.g. a different backend/judge), the hash won't match and resume is impossible;
  # restart just this checkpoint fresh rather than aborting the whole stage.
  if ! eval_safety "${ck}" "${cfg}" "${out}" 2>&1 | tee "${log}"; then
    if grep -q "resume protocol hash mismatch" "${log}"; then
      echo "    (eval config changed since the last run; restarting ${ck} from a fresh dir)"
      rm -rf "${out}"
      eval_safety "${ck}" "${cfg}" "${out}"
    else
      echo "    (eval failed for ${ck}; see ${log})" >&2
      exit 1
    fi
  fi
  cp -f "${cfg}" "${applied}"
}

repair_ck () {  # $1 = checkpoint label, $2 = adapter_dir ("" for base)
  local ck="$1" adapter="$2" out="${EVAL_DIR}/${ck}"
  [ -s "${out}/results.jsonl" ] || return 0
  local failed; failed="$(count_failed "${out}")"
  if [ "${failed}" = "0" ]; then
    echo "    ${ck}: all items evaluated; nothing to repair"
    return 0
  fi
  echo "    ${ck}: ${failed} failed item(s); regenerating ONLY those at ${REPAIR_MAX_NEW_TOKENS} tok (reusing the rest)"
  local cfg; cfg="$(make_eval_config "${ck}" "${adapter}" repair)"
  local repair="${out}.repair"; rm -rf "${repair}"
  if eval_safety "${ck}" "${cfg}" "${repair}" --adopt-completed-from "${out}"; then
    rm -rf "${out}"; mv "${repair}" "${out}"
    cp -f "${cfg}" "${CFG_DIR}/eval_${ck}.applied.json"
    echo "    ${ck}: repaired; $(count_failed "${out}") item(s) still unresolved (raise REPAIR_MAX_NEW_TOKENS if any remain)"
  else
    echo "    ${ck}: repair run failed; keeping the original dir" >&2
    rm -rf "${repair}"
    return 1
  fi
}

eval_ck () { run_ck "$1" "$2"; repair_ck "$1" "$2"; backup_work; }

eval_ck base    ""
eval_ck sft     "${TRAIN_OUT}/sft"
eval_ck dpo     "${TRAIN_OUT}/dpo"
eval_ck crsecpo "${TRAIN_OUT}/crsecpo"

echo "Per-checkpoint results under ${EVAL_DIR}/{base,sft,dpo,crsecpo}."
