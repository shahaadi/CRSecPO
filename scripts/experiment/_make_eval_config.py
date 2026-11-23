"""Write one stage-4 eval config from the template + environment.

Two modes (MODE env):
  normal  - the config used to generate/grade. Victim/judge budgets are "sticky":
            if this checkpoint was already evaluated (an applied-config snapshot or
            a protocol.json exists), reproduce the budgets it was built with, so a
            re-run REPLAYS the existing dir instead of regenerating at a different
            budget (which would force a full wipe).
  repair  - larger victim budget (REPAIR_MAX_NEW_TOKENS) and larger judge budget
            (REPAIR_JUDGE_TOKENS), used to regenerate only the failed items.

Env in: REPO_ROOT, CK, ADAPTER, OUT, CFG_DIR, EVAL_DIR, MODE, OPENAI_JUDGE_MODEL,
        EVAL_BACKEND, EVAL_BATCH_SIZE, PAIRS_GPU_MEM, PAIRS_MAX_LEN,
        REPAIR_MAX_NEW_TOKENS, REPAIR_JUDGE_TOKENS.
"""
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
ck = os.environ["CK"]
out = Path(os.environ["OUT"])
mode = os.environ.get("MODE", "normal")

config = json.loads((repo / "configs/crsecpo_evaluation.json").read_text())
config.setdefault("generation", {})

if os.environ.get("ADAPTER"):
    config["adapter_path"] = os.environ["ADAPTER"]
judge_model = os.environ.get("OPENAI_JUDGE_MODEL", "").strip()
if judge_model:
    config["judge_model"] = {"provider": "openai", "model": judge_model}
if os.environ.get("EVAL_BACKEND", "hf").strip() == "vllm":
    config["generation_backend"] = "vllm"
    config["vllm_gpu_memory_utilization"] = float(os.environ["PAIRS_GPU_MEM"])
    config["vllm_max_model_len"] = int(os.environ["PAIRS_MAX_LEN"])
    config["eval_batch_size"] = int(os.environ.get("EVAL_BATCH_SIZE", "128"))

if mode == "repair":
    config["generation"]["max_new_tokens"] = int(os.environ["REPAIR_MAX_NEW_TOKENS"])
    config["judge_max_completion_tokens"] = int(os.environ["REPAIR_JUDGE_TOKENS"])
else:
    # Sticky budgets: reproduce what this dir was last built with so a re-run
    # replays it. Prefer the applied-config snapshot (it also carries the judge
    # budget); otherwise fall back to the generation budget recorded in protocol.json.
    applied = Path(os.environ["CFG_DIR"]) / f"eval_{ck}.applied.json"
    protocol = Path(os.environ["EVAL_DIR"]) / ck / "protocol.json"
    if applied.exists():
        prev = json.loads(applied.read_text())
        if prev.get("generation", {}).get("max_new_tokens") is not None:
            config["generation"]["max_new_tokens"] = prev["generation"]["max_new_tokens"]
        if "judge_max_completion_tokens" in prev:
            config["judge_max_completion_tokens"] = prev["judge_max_completion_tokens"]
    elif protocol.exists():
        recorded = json.loads(protocol.read_text()).get("generation_config", {}).get("max_new_tokens")
        if recorded is not None:
            config["generation"]["max_new_tokens"] = recorded

out.write_text(json.dumps(config, indent=2) + "\n")
print(str(out))
