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

backup_work
echo "Aggregate report: ${AGG_DIR}/summary.json"
echo
# Print the decision right here in the cell output: the headline joint_claim, the
# eligibility gates it depends on, and every prespecified contrast (one-sided
# upper bound vs. its limit). Reads only the just-written summary.
AGG_DIR="${AGG_DIR}" python - <<'PY'
import json, os
from pathlib import Path
summary = json.loads((Path(os.environ["AGG_DIR"]) / "summary.json").read_text())
decision = summary.get("decision", summary)
yn = lambda b: "PASS" if b else "fail"
print("================= CRSecPO DECISION =================")
print(f"joint_claim: {decision.get('joint_claim')}")
gates = ("coverage_complete", "training_comparable",
         "independent_families_attested", "grading_audit_passed")
print("eligibility gates: " + "  ".join(f"{g}={decision.get(g)}" for g in gates))
noncommittal = decision.get("noncommittal_responses") or {}
if any(noncommittal.values()):
    print("non-committal responses (neither refused, complied, nor helped -- usually a benign "
          "deflection of a harmful ask): " + "  ".join(f"{k}={v}" for k, v in noncommittal.items()))
repetition = decision.get("degenerate_repetition_responses") or {}
if any(repetition.values()):
    print("  of which true degenerate repetition loops (hand-marked): "
          + "  ".join(f"{k}={v}" for k, v in repetition.items() if v))
print("---- contrasts (one-sided upper bound must be <= limit) ----")
for name, c in decision.get("contrasts", {}).items():
    upper = c.get("upper")
    upper_s = "n/a" if upper is None else f"{upper:.4f}"
    extra = f" [{c['status']}]" if c.get("status") not in (None, "ok", "bounded") else ""
    basis = f" basis={c['control_basis']}" if c.get("control_basis") else ""
    print(f"  [{yn(c.get('passed'))}] {name}: upper={upper_s} limit={c.get('limit')}"
          f" families={c.get('families')}{basis}{extra}")
print("===================================================")
if not decision.get("joint_claim"):
    print("joint_claim is False: see the gates/contrasts above. Utility contrasts and the")
    print("two attestations (UTILITY=..., ATTEST=...) are required for the full headline claim.")
PY
