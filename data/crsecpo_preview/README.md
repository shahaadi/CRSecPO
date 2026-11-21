# CRSecPO preview splits (accepted rows only)

Accepted WildJailbreak-derived rows from `outputs/crsecpo/preview_splits/`
(seed 42, validation fraction 0.1). **Preview only:** the human audit and
near-duplicate relation review have not been completed, so labels may change.

| Split | Rows | harmful_benign_framing | benign_adversarial_control |
|---|---|---|---|
| train | 2,716 | 1,077 | 1,639 |
| validation | 272 | 113 | 159 |
| test | 24 | 18 | 6 |

Each row has `source` (direct request), `target` (framed rewrite),
`selection_group`, `target_policy`, `family_id`, and WildJailbreak provenance
(revision and row index). Train/validation come from the WildJailbreak train
partition; test comes from the WildJailbreak eval partition. No IDs, families,
or normalized texts overlap across splits.

Text is derived from [allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)
and remains subject to its license and terms; the repository code license does
not apply to it. Contains adversarial harmful prompts for safety research only.
