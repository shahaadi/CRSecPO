#!/usr/bin/env bash
# Environment overrides for the general-safety (WildJailbreak failure-mining) run.
# SOURCE this (run.sh does) before any stage. It redirects the data + work paths so
# this pipeline never collides with the framing (crsecpo_preview) experiment, then
# every reused scripts/experiment/*.sh stage picks these up via its own config.sh
# (which honours ${VAR:-default}).
#
# No `set -e` here: this file is sourced, sometimes into an interactive shell.

# Repo root = two levels up from this file (scripts/general_safety_training/).
GSAFE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${GSAFE_DIR}/../.." && pwd)}"
export GSAFE_DIR

# --- path redirects (override the base experiment defaults) ---
# Candidate splits written by 00_prepare_wjb.py live here; SPLIT_DIR drives every
# downstream stage (train/val/test candidates are derived from it in config.sh).
export GSAFE_SPLIT_DIR="${GSAFE_SPLIT_DIR:-${REPO_ROOT}/data/wildjailbreak_general}"
export SPLIT_DIR="${GSAFE_SPLIT_DIR}"
# Separate work dir so pairs/checkpoints/eval never overwrite the framing run.
export WORK="${GSAFE_WORK:-${REPO_ROOT}/gsafe_out}"

# --- generation / freeze sizing for the general run ---
# 00_prepare_wjb.py already caps per class (GSAFE_MAX_PER_CLASS), so generate over all
# prepared candidates and let the freeze take everything that survives the filter.
export MAX_CANDIDATES="${MAX_CANDIDATES:-all}"
export PAIR_LIMIT="${PAIR_LIMIT:-100000}"

# --- failure filter knobs ---
export GSAFE_MAX_PER_CLASS="${GSAFE_MAX_PER_CLASS:-5000}"      # per class before splitting; 'all' = entire set
export GSAFE_REQUIRE_TEACHER_CORRECT="${GSAFE_REQUIRE_TEACHER_CORRECT:-1}"
export GSAFE_FILTER_MODEL="${GSAFE_FILTER_MODEL:-${OPENAI_FILTER_MODEL:-gpt-4o-mini}}"
export FILTER_CONCURRENCY="${FILTER_CONCURRENCY:-8}"
export GSAFE_FORCE_PREP="${GSAFE_FORCE_PREP:-0}"
export GSAFE_REFILTER="${GSAFE_REFILTER:-0}"

echo "[gsafe] repo=${REPO_ROOT}"
echo "[gsafe] splits=${SPLIT_DIR}  work=${WORK}"
echo "[gsafe] max_per_class=${GSAFE_MAX_PER_CLASS} filter_model=${GSAFE_FILTER_MODEL} require_teacher=${GSAFE_REQUIRE_TEACHER_CORRECT}"
