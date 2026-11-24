#!/usr/bin/env bash
# One-shot driver for the general-safety (WildJailbreak failure-mining) pipeline.
# Reuses every scripts/experiment/*.sh stage; only 00 (prepare) and the failure
# filter are new. Each stage is idempotent, so a disconnect just resumes.
#
# Prereqs (Colab): export HF_TOKEN (allenai/wildjailbreak is gated) and
# OPENAI_API_KEY (for the filter; read from env only, never written anywhere).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

EXP="${REPO_ROOT}/scripts/experiment"
export PYTHONPATH="${REPO_ROOT}/src"
export PAIRS_DIR="${WORK}/pairs"   # filter_failures.py reads this; the stages derive the same from config.sh

echo "== [gsafe] 00: prepare WildJailbreak candidate splits =="
python "${SCRIPT_DIR}/00_prepare_wjb.py"

echo "== [gsafe] 01: generate ordinary (student) + teacher answers  [reused] =="
bash "${EXP}/01_generate_pairs.sh"

echo "== [gsafe] merge train + validation pairs  [reused] =="
bash "${EXP}/reuse_pairs.sh"

echo "== [gsafe] filter to the base model's safety FAILURES =="
python "${SCRIPT_DIR}/filter_failures.py"

echo "== [gsafe] 02: freeze balanced set from the filtered pairs  [reused] =="
bash "${EXP}/02_freeze_pairs.sh"

echo "== [gsafe] 03: train sft / dpo / crsecpo  [reused] =="
bash "${EXP}/03_train.sh"

echo "== [gsafe] 04: eval safety on the held-out WildJailbreak test split  [reused] =="
bash "${EXP}/04_eval_safety.sh"

echo "== [gsafe] 05: aggregate + decision  [reused] =="
bash "${EXP}/05_aggregate.sh"

echo "== [gsafe] done. per-model rates + decision above; artifacts in ${WORK} =="
