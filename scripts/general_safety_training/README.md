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
| `GSAFE_MAX_PER_CLASS` | `5000` | cap per class before splitting (`all` = entire ~160K adversarial set — big cost) |
| `GSAFE_TEST_SIZE` | `2000` | FIXED eval holdout (not a %); keeps stage-4 tractable even at full scale |
| `GSAFE_VAL_SIZE` | `1000` | fixed validation holdout; everything else → train |
| `GSAFE_REQUIRE_TEACHER_CORRECT` | `1` | also verify the teacher (chosen) is correct on kept rows |
| `GSAFE_FILTER_MODEL` | `gpt-4o-mini` | OpenAI model for the failure filter |
| `FILTER_CONCURRENCY` | `8` | concurrent grading calls (sync path) |
| `GSAFE_USE_BATCH` | `0` | `1` grades via the async OpenAI Batch API (opt #2) |
| `GSAFE_BATCH_POLL_HOURS` | `6` | per-invocation poll budget before the step asks you to re-run |
| `GSAFE_FORCE_PREP` | `0` | `1` re-prepares splits even if present |
| `GSAFE_REFILTER` | `0` | `1` re-runs the filter from scratch (re-grades) |

**Cost note:** the filter grades one student answer per generated pair (then one teacher
answer per *kept* failure). Start with the default `GSAFE_MAX_PER_CLASS=5000` to size the
cost before setting `all`. Completed grades are cached (`<work>/pairs/filter_grades.jsonl`),
so reruns never re-charge resolved rows.

## Two speed/cost optimizations (opt-in)

**#1 Students-first generation — `run_students_first.sh` (instead of `run.sh`).**
Generates only the cheap student (bare-prompt) answer for *all* candidates, grades them,
and runs the expensive teacher (combined) generation **only on the failures** (`01a_student_pass.sh`
→ `select_student_failures.py` → normal stage 1 on `failures_<split>.jsonl`). Since the base
model already handles the majority of prompts, this skips most teacher generations — roughly a
third off stage-1 wall-clock at full scale (more in token terms, because teacher generations are
the long ones). The student grades it computes are reused by the later teacher-check (shared
cache), so nothing is graded twice. Uses a separate journal from the normal stage 1, so there is
no cross-run journal sharing to worry about. **Validate on a tiny slice first**
(`GSAFE_MAX_PER_CLASS=20 bash run_students_first.sh`) before a full run — the student-only
pass uses the new `pairs.py --conditions ordinary` path.

**#2 Batch grading — `GSAFE_USE_BATCH=1`.** Routes all judge calls through the async OpenAI
Batch API (~50% cheaper, no rate-limit babysitting), at the cost of up-to-24h turnaround. It is
resumable: the batch id is persisted, so if the grading step exits with "batch still running",
just re-run that same step/cell later and it polls and retrieves the finished results. Works with
both `run.sh` and `run_students_first.sh`.
