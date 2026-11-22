# CRSecPO experiment — run order

Scripts to take the 70/10/20 preview splits through the full pipeline:
**generate pairs → freeze a balanced train set → train the arms → evaluate on
test → aggregate the contrasts.** Each stage is resumable and writes under
`experiment_out/` (override with `WORK`).

**Google Drive note:** the crash-safe writers use hard links, which Drive's FUSE
mount does not support, so `WORK` must be on local disk. If you set `WORK` to a
Drive path (common on Colab for persistence), `config.sh` detects it, runs on
local disk anyway, and uses the Drive path as a **backup** — restored on a fresh
runtime and synced after each stage (every 2 min during pair generation). So
you get both: working hard links and disconnect-survival, with nothing to do.

## Order

```bash
cd scripts/experiment
bash 00_setup.sh          # install, HF login, download the pinned Llama
bash 01_generate_pairs.sh # train + validation preference pairs (NOT test); replays if present
bash reuse_pairs.sh       # fold train+validation pairs into one set (no regeneration)
bash 02_freeze_pairs.sh   # one balanced 50/50 safe-vs-harmful train set
bash 03_train.sh          # sft, dpo, crsecpo on the SAME frozen pairs
bash 04_eval_safety.sh    # blinded safety eval on test for base/sft/dpo/crsecpo
bash 05_aggregate.sh      # contrasts + decision (over-refusal switch flipped)
```

`01` writes `train_pairs.jsonl` and `validation_pairs.jsonl`; re-running it with
the same candidates and existing journals **replays them instead of
regenerating**. `reuse_pairs.sh` then merges those two into `merged_pairs.jsonl`
(every row `split=train`), which `02` freezes.

Edit **`config.sh`** first — it holds every path, the pinned model id/revision,
the compute knobs, and is sourced by all stages.

## Colab bootstrap

```python
!git clone <your-repo-url> CRSecPO && cd CRSecPO && git checkout crsecpo-preview-splits
import os; os.environ["HF_TOKEN"] = "hf_..."   # a token for the gated Llama license
!cd CRSecPO/scripts/experiment && bash 00_setup.sh
# run MAX_CANDIDATES=20 first to prove the pipeline, then set it to 'all':
!cd CRSecPO/scripts/experiment && MAX_CANDIDATES=20 bash 01_generate_pairs.sh
!cd CRSecPO/scripts/experiment && bash reuse_pairs.sh && bash 02_freeze_pairs.sh && bash 03_train.sh
!cd CRSecPO/scripts/experiment && bash 04_eval_safety.sh && bash 05_aggregate.sh
```

## Read these before the real run (the gotchas we discussed)

1. **Start small.** `config.sh` ships `MAX_CANDIDATES=20` so you can prove the
   whole chain end to end cheaply. Set it to `all` only for the real run.
2. **The over-refusal switch is already flipped** in `05_aggregate.sh`
   (`--benign-control-basis benign_adversarial_control_group`). Without it that
   contrast fails closed because your data has no `control_group="matched"` rows.
   The resulting claim is group-level, which is the right claim for your
   curated counterfactual controls.
3. **Full-epoch training, train+validation merged.** There is no validation-based
   checkpoint selection: each arm trains one full epoch and keeps the end-of-epoch
   model (periodic step checkpoints remain only for resume). Since validation is
   unused for selection, `reuse_pairs.sh` folds the train and validation pair
   exports into one training set (all `split=train`) — without regenerating —
   which `02` freezes. All three arms share one config (only `--method` differs),
   so the runs stay comparable.
4. **Test never gets pairs.** Pairs are generated for train and validation only;
   `test.jsonl` is the held-out eval set used in stage 4.
5. **joint_claim needs more than safety.** `05` computes the safety + over-refusal
   contrasts, but the headline `joint_claim` also needs the two utility contrasts
   (ifeval 541 / gsm8k 400, via `UTILITY="--utility-scores ..."`) and the two
   attestations (`ATTEST="--independent-families-attested --grading-audit-passed"`).
   Pass the attestation flags ONLY when those steps are genuinely done — they are
   claims you are making, not switches to flip for a green result. Utility
   generation/scoring is a separate heavy step (`evaluation utility` /
   `score-ifeval` / `score-gsm8k`); add it once the safety loop is working.
6. **Compute reality.** This loads an 8B model for generation, training, and
   judging. The full `all` run is many thousands of generations plus three LoRA
   trainings — realistically an A100-class GPU and hours, not a free T4 session.
   The `MAX_CANDIDATES=20` smoke path is what fits a small session.
   GPU memory settings are **auto-detected** in `config.sh` (an L4/24GB card gets
   conservative vLLM + batch settings, an A100 gets the full ones) and the
   effective training batch is held constant across cards, so you don't tune
   anything by hand.
7. **Not run here.** These were authored against the code but not executed in
   this environment (no GPU libs). Run the smoke path first and watch stage 1
   actually accept a few pairs before launching the full job.
