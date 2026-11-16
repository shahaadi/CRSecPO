"""Exercise the real adapter interfaces together using inert scripted responses."""
import json

import torch
import pytest

from secpo.crsecpo.inference import JournalBackend, ModelSpec
from secpo.crsecpo.pairs import generate_pairs, validate_candidate
from secpo.crsecpo.records import eligible, prepare_records
from secpo.crsecpo.training import prepare
from test_crsecpo_pairs import ScriptedBackend
from test_crsecpo_training import Tokenizer, config


@pytest.mark.parametrize("target", ["Museum opening hours?", " \tMuseum opening hours?\n"])
def test_absent_rewrite_is_quarantined_and_rejected_at_generation_boundaries(target):
    candidate = dict(id="no-rewrite", source="Museum opening hours?", target=target,
        has_rewrite=True, pool="training", selection_status="accepted", target_policy="permissible",
        selection_group="benign_adversarial_control", provenance={"publisher_partition": "train"})
    original = dict(candidate)
    assert not eligible(candidate)
    prepared, assignments = prepare_records([candidate])
    assert len(prepared["quarantine"]) == len(assignments) == 1
    assert not prepared["train"] and not prepared["validation"] and not prepared["test"]
    for split in ("train", "validation", "test"):
        with pytest.raises(ValueError, match="source and rewrite must be distinct"):
            validate_candidate(dict(candidate, family_id="family", split=split), allowed_splits=(split,))
    assert candidate == original


def test_filtered_record_to_shared_sft_dpo_crsecpo_inputs(tmp_path):
    candidate = dict(id="fixture", source="Museum source task", target="Museum target task",
        has_rewrite=True, pool="training", selection_status="accepted", target_policy="limited",
        selection_group="harmful_benign_framing", provenance={"publisher_partition": "train"})
    # Deterministic training-only fixture; the production splitter is unchanged.
    prepared, _ = prepare_records([candidate], validation_fraction=1e-12)
    backend = ScriptedBackend()
    tokenizer = Tokenizer()
    pairs, outcomes = generate_pairs(prepared["train"], backend=JournalBackend(backend, tmp_path / "journal.jsonl"),
        tokenizer=tokenizer, model_spec=ModelSpec("org/model", "a"*40, "org/model", "a"*40),
        output_path=tmp_path / "pairs.jsonl", outcomes_path=tmp_path / "outcomes.jsonl")
    assert outcomes[0]["accepted"] and len(pairs) == 1
    cfg = config(tmp_path, pairs)
    batches, manifests = {}, []
    for method in ("sft", "dpo", "crsecpo", "rubric_only"):
        rows, collator, manifest = prepare(cfg, method, tokenizer=tokenizer)
        batches[method] = collator(rows)
        manifests.append(manifest)
    assert len({m["input_sha256"] for m in manifests}) == 1
    assert all(m["ordered_ids"] == ["fixture"] for m in manifests)
    for key in ("input_ids", "attention_mask", "labels"):
        assert torch.equal(batches["sft"][key], batches["dpo"]["policy_chosen_" + key])
        assert torch.equal(batches["dpo"]["policy_chosen_" + key], batches["crsecpo"]["policy_chosen_" + key])
        assert torch.equal(batches["dpo"]["policy_rejected_" + key], batches["crsecpo"]["policy_rejected_" + key])
    for answer in ("chosen", "rejected"):
        policy = batches["crsecpo"]["policy_" + answer + "_labels"][0]
        reference = batches["crsecpo"]["reference_" + answer + "_labels"][0]
        assert torch.equal(policy[policy != -100], reference[reference != -100])
    expected = tokenizer(pairs[0]["pair_metadata"]["generations"]["combined"]["request"]["rendered_prompt"])["input_ids"]
    ref = batches["crsecpo"]["reference_chosen_input_ids"][0]
    mask = batches["crsecpo"]["reference_chosen_labels"][0] == -100
    assert ref[mask].tolist() == expected
    assert len(backend.requests) == 4
    assert json.loads((tmp_path / "outcomes.jsonl").read_text())["accepted"]
