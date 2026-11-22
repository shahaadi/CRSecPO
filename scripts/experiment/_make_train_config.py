"""Write the shared experiment training config from the released template.

One config drives all three arms (sft/dpo/crsecpo): the only thing that differs
between arms is --method, so their training configs stay identical and the runs
count as "comparable". Validation selection uses eval_loss because SFT (a plain
LM run) cannot produce the preference-accuracy metric; eval_loss is computable
for every arm, which is what comparability requires.

Env in: REPO_ROOT, CFG_DIR, MODEL_ID, MODEL_REVISION, TRAIN_OUT, FROZEN_DIR,
        PAIRS_DIR, EVAL_STEPS, SAVE_STEPS.
"""
import json, os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
template = json.loads((repo / "configs/train_crsecpo_llama31.json").read_text())

template.update({
    "model_name_or_path": os.environ["MODEL_ID"],
    "model_revision": os.environ["MODEL_REVISION"],
    # Overridden on the CLI per stage, but must be present and valid here.
    "data_file": str(Path(os.environ["FROZEN_DIR"]) / "train_pairs.jsonl"),
    "output_dir": os.environ["TRAIN_OUT"],
    # ---- validation-based best-checkpoint selection (comparable across arms) ----
    "eval_data_file": str(Path(os.environ["PAIRS_DIR"]) / "validation_pairs.jsonl"),
    "eval_strategy": "steps",
    "eval_steps": int(os.environ["EVAL_STEPS"]),
    "save_strategy": "steps",
    "save_steps": int(os.environ["SAVE_STEPS"]),
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
})

out = Path(os.environ["CFG_DIR"]) / "train_experiment.json"
out.write_text(json.dumps(template, indent=2) + "\n")
print("wrote", out)
