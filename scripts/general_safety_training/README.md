# General-safety training (WildJailbreak failure-mining)

This pipeline shifts the threat model from **benign framing specifically** to
**safety in general**, by training on the base model's *own failures* across the
full WildJailbreak distribution instead of a curated framing subset.

## The idea

1. Convert the WildJailbreak **adversarial** split into candidates (`00_prepare_wjb.py`).
2. Generate, for every candidate, the model's ordinary (bare-prompt) answer and the
   rubric-informed teacher answer — **this reuses the existing `01_generate_pairs.sh`
   unchanged**, which already produces `(chosen=teacher, rejected=student)` pairs.
3. **Keep only the failures** (`filter_failures.py`): grade each candidate's *student*
   (bare) answer with the OpenAI judge and keep a row only when the student failed —
   - `unsafe prompt → prohibited content` (the model complied with a harmful request), or
   - `safe prompt → refusal` (the model over-refused a benign request).
   Rows where the student already behaves correctly carry no training signal and are dropped.
   (By default we also verify the *teacher* answer is correct on the kept rows, so the
   `chosen` target is trustworthy — see `GSAFE_REQUIRE_TEACHER_CORRECT`.)
4. Freeze → train the three arms (sft / dpo / crsecpo) → eval → aggregate, **all reused
   verbatim** from `scripts/experiment/`.

This is hard-negative mining: the training set is exactly the places the model gets
safety wrong, across the whole WildJailbreak tactic distribution — not just framing.

## Why only the adversarial WildJailbreak split

`validate_candidate` (shared, in `pairs.py`) requires a **distinct `source` (de-framed)
and `target` (as-presented)** per row. WildJailbreak's *adversarial* examples carry both
a `vanilla` and an `adversarial` field, so they map directly (source=vanilla, target=adversarial)
and cover **both** classes (adversarial_harmful → unsafe, adversarial_benign → safe) over
the full tactic distribution (~262k rows). Pure-`vanilla_*` rows have no distinct source and
would need a one-line relaxation of the candidate schema; they're skipped here and noted as
a follow-up. The adversarial set is already a large generalization beyond the ~2k preview.

## Run it

```bash
# 0. (Colab) accept the WildJailbreak terms on HF and export a token with access:
#    export HF_TOKEN=hf_xxx         # allenai/wildjailbreak is gated
#    export OPENAI_API_KEY=sk-...   # for the failure filter (read from env only)

# 1. one command runs every stage with the right path overrides:
bash scripts/general_safety_training/run.sh
```

`run.sh` just sources `env.sh` (which redirects `SPLIT_DIR` and `WORK` so this run never
collides with the framing experiment) and then calls, in order:
`00_prepare_wjb.py → 01_generate_pairs.sh → reuse_pairs.sh → filter_failures.py →
02_freeze_pairs.sh → 03_train.sh → 04_eval_safety.sh → 05_aggregate.sh`.
Every stage is idempotent, so a disconnect just resumes.

## Knobs (set before `run.sh`, all have defaults in `env.sh`)

| var | default | meaning |
|---|---|---|
| `GSAFE_WORK` | `<repo>/gsafe_out` | work dir (set `/content/gsafe_out` on Colab) |
| `GSAFE_SPLIT_DIR` | `<repo>/data/wildjailbreak_general` | where the prepared candidate splits go |
| `GSAFE_MAX_PER_CLASS` | `5000` | cap per class before splitting (`all` = entire set — big cost) |
| `GSAFE_REQUIRE_TEACHER_CORRECT` | `1` | also verify the teacher (chosen) is correct on kept rows |
| `GSAFE_FILTER_MODEL` | `gpt-4o-mini` | OpenAI model for the failure filter |
| `FILTER_CONCURRENCY` | `8` | concurrent grading calls |
| `GSAFE_FORCE_PREP` | `0` | `1` re-prepares splits even if present |
| `GSAFE_REFILTER` | `0` | `1` re-runs the filter from scratch (re-grades) |

**Cost note:** the filter grades one student answer per generated pair (then one teacher
answer per *kept* failure). Start with the default `GSAFE_MAX_PER_CLASS=5000` to size the
cost before setting `all`. Completed grades are cached (`<work>/pairs/filter_grades.jsonl`),
so reruns never re-charge resolved rows.
