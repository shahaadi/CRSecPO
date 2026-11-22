"""Write the shared experiment training config from the released template.

One config drives all three arms (sft/dpo/crsecpo): the only thing that differs
between arms is --method, so their training configs stay identical and the runs
count as "comparable". Training runs one full epoch and keeps the end-of-epoch
checkpoint (no validation-based selection). save_strategy="steps" still writes
periodic checkpoints so a disconnect can resume; the final model saved after the
epoch is the one used.

Env in: REPO_ROOT, CFG_DIR, MODEL_ID, MODEL_REVISION, TRAIN_OUT, FROZEN_DIR,
        SAVE_STEPS.
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
    # Auto-tuned for the detected GPU (set in config.sh). Effective batch is kept
    # constant (batch * accum = 64), so the optimization is unchanged across cards.
    "per_device_train_batch_size": int(os.environ["TRAIN_BATCH"]),
    "gradient_accumulation_steps": int(os.environ["GRAD_ACCUM"]),
    # Periodic checkpoints for resume; the final (end-of-epoch) model is kept.
    "save_strategy": "steps",
    "save_steps": int(os.environ["SAVE_STEPS"]),
})

out = Path(os.environ["CFG_DIR"]) / "train_experiment.json"
out.write_text(json.dumps(template, indent=2) + "\n")
print("wrote", out)
