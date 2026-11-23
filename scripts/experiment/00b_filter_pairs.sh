#!/usr/bin/env bash
# Stage 0b (optional, runs after 00_setup): validate the (source, target) pairs with a
# cheap API agent and write FILTERED splits that every later stage uses automatically.
#
# A pair is kept only if the agent confirms BOTH:
#   - the target is an equivalent, relevant reframing of the source (same underlying ask), and
#   - the safe/unsafe label still holds (an unsafe pair's target is still harmful; a safe
#     pair's target is still legitimate).
# It is calibrated NOT to over-reject on sensitive topics and NOT to accept flipped or
# off-topic pairs. Resumable (per-split verdict journal) and concurrent.
#
# Needs OPENAI_API_KEY in the environment (e.g. Colab secrets). Model: OPENAI_FILTER_MODEL
# (default gpt-4o-mini). Re-run to resume; it only calls the API for pairs not yet judged.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set; export it (e.g. from Colab secrets) to filter." >&2
  exit 1
fi

echo "== Stage 0b: filter pairs with ${OPENAI_FILTER_MODEL} (concurrency ${FILTER_CONCURRENCY}) =="
# Read the RAW splits (ORIG_SPLIT_DIR) and write validated ones to FILTERED_DIR. config.sh
# redirects SPLIT_DIR there on the next run, so we always read the originals here.
ORIG_SPLIT_DIR="${ORIG_SPLIT_DIR}" FILTERED_DIR="${FILTERED_DIR}" \
OPENAI_FILTER_MODEL="${OPENAI_FILTER_MODEL}" FILTER_CONCURRENCY="${FILTER_CONCURRENCY}" \
python "${SCRIPT_DIR}/_filter_pairs.py"
backup_work

echo "Filtered splits in ${FILTERED_DIR}; verdicts in ${FILTERED_DIR}/*.verdicts.jsonl."
echo "Later stages (01, 04) now use the filtered splits automatically."
