#!/usr/bin/env bash
# Stage 5: combine the four checkpoint runs and apply the prespecified contrasts.
#
# THE OVER-REFUSAL SWITCH: by default the over-refusal contrast only uses rows
# tagged control_group="matched", which this dataset does not have, so it would
# fail closed. Because the whole benign_adversarial_control group is curated to
# be the benign-framing counterfactual, we pass:
#     --benign-control-basis benign_adversarial_control_group
# so the contrast actually runs (as a group-level claim).
#
# NOTE on the headline joint_claim: it ALSO requires the two utility contrasts
# (ifeval/gsm8k, via --utility-scores) and the two attestations below. Without
# utility scores the safety contrasts still compute; joint_claim stays False
# until utility + attestations are supplied. Only pass the attestation flags once
# the independent-families and grading-audit steps are genuinely done.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "== Stage 5: aggregate + decision =="

ATTEST="${ATTEST:-}"   # set to "--independent-families-attested --grading-audit-passed" only when actually done
UTILITY="${UTILITY:-}" # set UTILITY="--utility-scores /path/to/utility.json" when available

python -m secpo.crsecpo.evaluation aggregate \
  --run-dir "${EVAL_DIR}/base" \
  --run-dir "${EVAL_DIR}/sft" \
  --run-dir "${EVAL_DIR}/dpo" \
  --run-dir "${EVAL_DIR}/crsecpo" \
  --input "${TEST_CANDIDATES}" \
  --output-dir "${AGG_DIR}" \
  --benign-control-basis benign_adversarial_control_group \
  ${UTILITY} ${ATTEST}

echo "Aggregate report: ${AGG_DIR}/summary.json"
echo "Read contrasts[*].passed and the top-level joint_claim there."
