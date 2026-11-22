#!/usr/bin/env bash
# Stage 4: run the blinded safety evaluation on the held-out TEST split, once per
# checkpoint (base + the three trained arms). Each run generates answers on the
# test prompts and grades them (prohibited_assistance / refusal / permitted_help)
# with the frozen judge. base uses no adapter; the others point at their adapter.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "== Stage 4: safety evaluation on test =="

# Each checkpoint can take tens of minutes, so mirror the journals to the Drive
# backup every 2 min (and on exit), not just after each checkpoint -- a disconnect
# mid-checkpoint then loses little. The eval is journaled, so the resumed run
# replays completed generations/grades.
if [ -n "${DRIVE_BACKUP:-}" ]; then
  ( while true; do sleep 120; backup_work; done ) & SYNC_PID=$!
  trap 'kill ${SYNC_PID} 2>/dev/null || true; backup_work' EXIT
fi

make_eval_config () {  # $1 = checkpoint, $2 = adapter_dir ("" for base) -> prints config path
  local ck="$1" adapter="$2" out="${CFG_DIR}/eval_${ck}.json"
  REPO_ROOT="${REPO_ROOT}" CK="${ck}" ADAPTER="${adapter}" OUT="${out}" \
  OPENAI_JUDGE_MODEL="${OPENAI_JUDGE_MODEL:-}" EVAL_BACKEND="${EVAL_BACKEND:-hf}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}" \
  PAIRS_GPU_MEM="${PAIRS_GPU_MEM:-0.9}" PAIRS_MAX_LEN="${PAIRS_MAX_LEN:-8192}" python - <<'PY'
import json, os
from pathlib import Path
base = json.loads((Path(os.environ["REPO_ROOT"]) / "configs/crsecpo_evaluation.json").read_text())
if os.environ["ADAPTER"]:
    base["adapter_path"] = os.environ["ADAPTER"]   # trained arm
judge_model = os.environ.get("OPENAI_JUDGE_MODEL", "").strip()
if judge_model:                                     # grade via a cheap OpenAI model
    base["judge_model"] = {"provider": "openai", "model": judge_model}
if os.environ.get("EVAL_BACKEND", "hf").strip() == "vllm":   # batch generation through vLLM
    base["generation_backend"] = "vllm"
    base["vllm_gpu_memory_utilization"] = float(os.environ["PAIRS_GPU_MEM"])
    base["vllm_max_model_len"] = int(os.environ["PAIRS_MAX_LEN"])
    # Flush the journal every this-many generations/grades so a mid-run disconnect
    # loses at most one chunk (and resumes from there).
    base["eval_batch_size"] = int(os.environ.get("EVAL_BATCH_SIZE", "128"))
Path(os.environ["OUT"]).write_text(json.dumps(base, indent=2) + "\n")
print(os.environ["OUT"])
PY
}

eval_safety () {  # run one checkpoint; args: checkpoint, config, output-dir
  python -m secpo.crsecpo.evaluation safety \
    --config "$2" --input "${TEST_CANDIDATES}" --checkpoint "$1" --output-dir "$3"
}

run_ck () {  # $1 = checkpoint label, $2 = adapter_dir ("" for base)
  local ck="$1" adapter="$2"
  local cfg; cfg="$(make_eval_config "${ck}" "${adapter}")"
  local out="${EVAL_DIR}/${ck}" log="${CFG_DIR}/eval_${ck}.log"
  echo "--- safety: ${ck} ---"
  # Journaled: a prior run with the SAME protocol replays/resumes. But if this dir
  # holds a protocol from a DIFFERENT configuration (e.g. you switched generation to
  # vLLM or added the OpenAI judge), its hash won't match and resume is impossible.
  # Rather than abort the whole stage, restart just THIS checkpoint from a fresh dir.
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
}

run_ck base    "";                backup_work
run_ck sft     "${TRAIN_OUT}/sft";     backup_work
run_ck dpo     "${TRAIN_OUT}/dpo";     backup_work
run_ck crsecpo "${TRAIN_OUT}/crsecpo"; backup_work

echo "Per-checkpoint results under ${EVAL_DIR}/{base,sft,dpo,crsecpo}."
