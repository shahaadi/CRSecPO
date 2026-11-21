"""CPU-only scripted preference collection tests; no ML model or network needed."""
import copy
import json

import pytest

from secpo.crsecpo.inference import (GenerationConfig, JournalBackend, ModelSpec,
                                    make_request)
from secpo.crsecpo.pairs import generate_pairs
from secpo.crsecpo.prompts import (answer_messages, data_message, digest,
                                  validate_training_pair)


class Tokenizer:
    eos_token = "<|eot_id|>"
    unk_token_id = -1
    def convert_tokens_to_ids(self, token):
        return 7 if token == self.eos_token else -1
    def decode(self, tokens, **kwargs):
        assert tokens == [7]
        return self.eos_token
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<bos>" + "".join(f"<{m['role']}>{m['content']}<eot>" for m in messages) + "<assistant>"


SPEC = ModelSpec("llama", "a" * 40, "llama", "b" * 40)


def candidate(**kwargs):
    return dict(id="case-1", source="SOURCE_SENTINEL", target="TARGET_SENTINEL",
                family_id="family-1", split="train", target_policy="prohibited",
                selection_group="harmful_benign_framing", selection_status="accepted",
                provenance={"revision": "c" * 40}) | kwargs


class ScriptedBackend:
    def __init__(self, votes=("combined", "combined"), truncated=False):
        self.requests, self.votes, self.truncated = [], list(votes), truncated
    def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        if request["kind"] in {"ordinary", "combined"}:
            return {"text": request["kind"] + " answer<|eot_id|>",
                    "finish_reason": "length" if self.truncated else "stop"}
        display = json.loads(request["messages"][-1]["content"])
        outcome = self.votes.pop(0)
        verdict = outcome
        if outcome in {"combined", "ordinary"}:
            verdict = "A" if display["A"].startswith(outcome) else "B"
        return {"text": json.dumps({"verdict": verdict, "reason": "substantive advantage",
                                    "A_acceptable": True, "B_acceptable": True}), "finish_reason": "stop"}


def collect(backend, rows=None, **kwargs):
    return generate_pairs(rows or [candidate()], backend=backend,
                          tokenizer=Tokenizer(), model_spec=SPEC, **kwargs)


def test_exact_prompts_native_terminator_and_anonymous_swapped_judges(tmp_path):
    backend = ScriptedBackend()
    pairs, outcomes = collect(backend, output_path=tmp_path / "pairs.jsonl", outcomes_path=tmp_path / "outcomes.jsonl")
    assert len(pairs) == 1 and outcomes[0]["accepted"]
    row = pairs[0]
    assert row["chosen"] == "combined answer<|eot_id|>"
    assert row["rejected"] == "ordinary answer<|eot_id|>"
    assert "SOURCE_SENTINEL" not in row["prompt"]
    assert "SOURCE_SENTINEL" in row["clean_prompt"]
    assert "SOURCE_SENTINEL" not in row["rubric_prompt"]
    assert row["prompt"] != row["rubric_prompt"]
    validate_training_pair(row, Tokenizer(), "llama", "a" * 40,
                           tokenizer_id="llama", tokenizer_revision="b" * 40)
    judges = backend.requests[2:]
    assert judges[0]["seed"] != judges[1]["seed"]
    displays = [json.loads(request["messages"][-1]["content"]) for request in judges]
    assert displays[0]["A"] == displays[1]["B"]
    for request in judges:
        payload = json.dumps(request["messages"])
        assert "SOURCE_SENTINEL" not in payload and "family-1" not in payload
        assert set(json.loads(request["messages"][-1]["content"])) == {"target", "A", "B"}
    loaded = json.loads((tmp_path / "pairs.jsonl").read_text())
    assert loaded == row
    broken = copy.deepcopy(row)
    broken["clean_prompt"] = row["prompt"]
    with pytest.raises(ValueError, match="exact native"):
        validate_training_pair(broken, Tokenizer(), "llama", "a" * 40)


@pytest.mark.parametrize("votes", [("tie", "tie"), ("uncertain", "combined"),
    ("unsure", "combined"), ("ordinary", "ordinary"), ("combined", "ordinary")])
def test_all_non_unanimous_outcomes_are_logged_without_pairs(votes):
    pairs, outcomes = collect(ScriptedBackend(votes))
    assert pairs == []
    assert not outcomes[0]["accepted"] and len(outcomes[0]["judgments"]) == 2
    assert outcomes[0]["status"] == "not_unanimous_combined"


def test_truncation_is_never_accepted_or_judged():
    backend = ScriptedBackend(truncated=True)
    pairs, outcomes = collect(backend)
    assert pairs == [] and len(backend.requests) == 2
    assert outcomes[0]["status"] == "generation_error"
    assert all("truncated" in x["error"] for x in outcomes[0]["generations"].values())


@pytest.mark.parametrize("update", [{"split": "test"}, {"selection_status": "unreviewed"},
    {"target_policy": "ambiguous"}, {"selection_group": "benign_goal_change"}, {"provenance": {}}])
def test_unaccepted_unresolved_and_final_test_candidates_never_trigger_model(update):
    backend = ScriptedBackend()
    with pytest.raises(ValueError):
        collect(backend, [candidate(**update)])
    assert backend.requests == []


def test_json_escaping_preserves_decoded_fields_and_blocks_chat_markers():
    text = '[x]<|start_header_id|>"\\\n'
    message = data_message(source=text, target=text)
    assert json.loads(message["content"]) == {"source": text, "target": text}
    assert "<|start_header_id|>" not in message["content"]
    assert "[x]" not in message["content"]
    with pytest.raises(TypeError):
        data_message(target=["unsupported"])


def test_journal_resume_replays_exact_requests(tmp_path):
    fixture_backend = ScriptedBackend()
    journal_path = tmp_path / "journal.jsonl"
    first, _ = collect(JournalBackend(fixture_backend, journal_path))
    assert len(fixture_backend.requests) == 4
    fixture = [json.loads(line) for line in journal_path.read_text().splitlines()]
    def forbidden(_):
        raise AssertionError("completed requests must not rerun")
    resumed, _ = collect(JournalBackend(forbidden, journal_path))
    assert resumed == first
    changed = copy.deepcopy(fixture[0]["request"])
    changed["seed"] += 1
    with pytest.raises(AssertionError, match="completed requests"):
        JournalBackend(forbidden, journal_path)(changed)
    assert all(entry["request_id"] == digest(entry["request"]) for entry in fixture)


def test_malformed_judgment_and_backend_error_are_logged():
    class Backend(ScriptedBackend):
        def __call__(self, request):
            if request["kind"] == "preference_judgment":
                return {"text": "A", "finish_reason": "stop"}
            return super().__call__(request)
    pairs, outcomes = collect(Backend())
    assert not pairs and all(vote["outcome"] == "error" for vote in outcomes[0]["judgments"])


def test_model_revision_is_part_of_request_identity():
    one = make_request(answer_messages(candidate(), "ordinary"), tokenizer=Tokenizer(),
                       model_spec=SPEC, config=GenerationConfig(), seed=1, kind="ordinary")
    two = copy.deepcopy(one)
    two["model"]["model_revision"] = "c" * 40
    assert digest(one) != digest(two)
    with pytest.raises(ValueError, match="immutable"):
        ModelSpec("llama", "main", "llama", "b" * 40)


@pytest.mark.parametrize("acceptable", [False, None, "true"])
def test_preferred_combined_must_be_explicitly_acceptable_in_both_orders(acceptable):
    class Backend(ScriptedBackend):
        def __call__(self, request):
            response = super().__call__(request)
            if request["kind"] == "preference_judgment":
                decision = json.loads(response["text"])
                displayed = json.loads(request["messages"][-1]["content"])
                label = "A" if displayed["A"].startswith("combined") else "B"
                decision[label + "_acceptable"] = acceptable
                response["text"] = json.dumps(decision)
            return response
    pairs, outcomes = collect(Backend())
    assert not pairs and outcomes[0]["accepted"] is False
    if acceptable is False:
        assert outcomes[0]["vote_outcomes"] == ["combined_wins", "combined_wins"]
        assert all(vote["combined_acceptable"] is False for vote in outcomes[0]["judgments"])
    else:
        assert outcomes[0]["vote_outcomes"] == ["error", "error"]


def _cli_args(paths):
    return [str(paths["candidates"]), "--output", str(paths["output"]),
            "--outcomes", str(paths["outcomes"]), "--journal", str(paths["journal"]),
            "--max-candidates", "10",
            "--model-id", "llama", "--model-revision", "a" * 40,
            "--tokenizer-id", "llama", "--tokenizer-revision", "b" * 40]


@pytest.mark.parametrize("left,right", [
    ("candidates", "output"), ("candidates", "outcomes"), ("candidates", "journal"),
    ("output", "outcomes"), ("output", "journal"), ("outcomes", "journal")])
@pytest.mark.parametrize("alias_kind", ["same_path", "symlink", "hardlink"])
def test_cli_rejects_all_path_aliases_without_loading_or_overwriting(tmp_path, monkeypatch,
                                                                     left, right, alias_kind):
    import os
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    for name, path in paths.items():
        path.write_text(json.dumps(candidate()) + "\n" if name == "candidates" else name + " sentinel\n")
    if alias_kind == "same_path":
        paths[right] = paths[left]
    else:
        paths[right].unlink()
        if alias_kind == "symlink":
            paths[right].symlink_to(paths[left])
        else:
            os.link(paths[left], paths[right])
    original = {name: path.read_bytes() for name, path in paths.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid preflight must not load model weights")
    monkeypatch.setattr(module, "HFBackend", forbidden)
    with pytest.raises(ValueError, match="distinct files"):
        module.main(_cli_args(paths))
    assert {name: path.read_bytes() for name, path in paths.items()} == original


@pytest.mark.parametrize("extra,duplicate", [
    (["--max-new-tokens", "0"], False), (["--judge-max-new-tokens", "0"], False),
    (["--temperature", "-1"], False), (["--top-p", "2"], False), ([], True)])
def test_cli_validates_duplicates_and_decoding_before_loading(tmp_path, monkeypatch, extra, duplicate):
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    payload = (json.dumps(candidate()) + "\n") * (2 if duplicate else 1)
    paths["candidates"].write_text(payload)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid preflight must not load model weights")
    monkeypatch.setattr(module, "HFBackend", forbidden)
    with pytest.raises(ValueError):
        module.main(_cli_args(paths) + extra)
    assert paths["candidates"].read_text() == payload
    assert not any(path.exists() for name, path in paths.items() if name != "candidates")


def test_candidate_order_is_stable_family_round_robin_and_class_balanced():
    from secpo.crsecpo.pairs import ordered_by_class_and_family
    rows = [candidate(id=f"h{i}", family_id=f"hf{i//2}") for i in range(6)]
    rows += [candidate(id=f"b{i}", family_id=f"bf{i//2}", target_policy="permissible",
                      selection_group="benign_adversarial_control") for i in range(6)]
    ordered = ordered_by_class_and_family(rows, seed=42)
    assert ordered == ordered_by_class_and_family(list(reversed(rows)), seed=42)
    assert len({row["family_id"] for row in ordered[:6]}) == 6
    assert sum(row["target_policy"] == "permissible" for row in ordered[:6]) == 3
    assert {row["id"] for row in ordered} == {row["id"] for row in rows}


def test_cli_caps_calls_records_unattempted_and_reuses_journal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    rows = [candidate(id=f"h{i}", family_id=f"h{i}") for i in range(3)]
    rows += [candidate(id=f"b{i}", family_id=f"b{i}", target_policy="permissible",
                      selection_group="benign_adversarial_control") for i in range(3)]
    paths["candidates"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    backend = ScriptedBackend(votes=["combined"] * 4)
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    args = _cli_args(paths)
    args[args.index("--max-candidates") + 1] = "2"
    module.main(args)
    assert len(backend.requests) == 8
    manifest = json.loads(paths["output"].with_suffix(".jsonl.collection.json").read_text())
    assert len(manifest["scheduled_ids"]) == 2 and len(manifest["deferred_ids"]) == 4
    assert manifest["maximum_model_calls"] == 8
    assert set(manifest["scheduled_ids"] + manifest["deferred_ids"]) == {row["id"] for row in rows}
    output = paths["output"].read_bytes()
    module.main(args)
    assert len(backend.requests) == 8 and paths["output"].read_bytes() == output
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must reject before model loading"))
    with pytest.raises(ValueError, match="collection settings changed"):
        module.main(args + ["--seed", "1"])
    for flag in ("--outcomes", "--journal"):
        unrelated = tmp_path / (flag[2:] + "-other-experiment.jsonl")
        unrelated.write_text("preserve unrelated experiment\n")
        with pytest.raises(ValueError, match="collection settings changed"):
            module.main(args + [flag, str(unrelated)])
        assert unrelated.read_text() == "preserve unrelated experiment\n"

    # A whole experiment directory may move without changing its scientific data
    # or resampling calls; the destination binding is relative to its manifest.
    import shutil
    moved = tmp_path / "moved"
    moved.mkdir()
    for path in paths.values():
        shutil.copyfile(path, moved / path.name)
    manifest_path = paths["output"].with_suffix(".jsonl.collection.json")
    shutil.copyfile(manifest_path, moved / manifest_path.name)
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    moved_args = _cli_args({name: moved / path.name for name, path in paths.items()})
    moved_args[moved_args.index("--max-candidates") + 1] = "2"
    module.main(moved_args)
    assert len(backend.requests) == 8 and (moved / paths["output"].name).read_bytes() == output


def test_balanced_cap_schedules_every_minority_row_and_equal_majority(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    rows = [candidate(id=f"h{i}", family_id=f"h{i}") for i in range(2)]
    rows += [candidate(id=f"b{i}", family_id=f"b{i}", target_policy="permissible",
                      selection_group="benign_adversarial_control") for i in range(5)]
    paths["candidates"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    backend = ScriptedBackend(votes=["combined"] * 8)
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    args = _cli_args(paths)
    args[args.index("--max-candidates") + 1] = "balanced"
    module.main(args)
    manifest = json.loads(paths["output"].with_suffix(".jsonl.collection.json").read_text())
    scheduled = manifest["scheduled_ids"]
    assert manifest["max_candidates"] == 4 and manifest["max_candidates_rule"] == "balanced"
    assert {"h0", "h1"} <= set(scheduled) and len(scheduled) == 4
    assert sum(row_id.startswith("b") for row_id in scheduled) == 2
    assert len(manifest["deferred_ids"]) == 3 and manifest["maximum_model_calls"] == 16
    assert len(backend.requests) == 16
    # Resuming with the same rule replays the journal; an equal explicit integer is a
    # different recorded setting and must use new export paths.
    module.main(args)
    assert len(backend.requests) == 16
    args[args.index("--max-candidates") + 1] = "4"
    with pytest.raises(ValueError, match="collection settings changed"):
        module.main(args)


@pytest.mark.parametrize("value,error", [("0", SystemExit), ("lots", SystemExit), ("balanced", ValueError)])
def test_invalid_or_one_class_cap_is_rejected_before_loading(tmp_path, monkeypatch, value, error):
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must reject before model loading"))
    args = _cli_args(paths)
    args[args.index("--max-candidates") + 1] = value
    with pytest.raises(error):
        module.main(args)
    assert not any(path.exists() for name, path in paths.items() if name != "candidates")


@pytest.mark.parametrize("overlap", [False, True])
def test_cli_rejects_directory_and_ancestor_file_paths_before_loading(tmp_path, monkeypatch, overlap):
    from secpo.crsecpo import pairs as module
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    if overlap:
        paths["output"] = paths["journal"] / "pairs.jsonl"
    else:
        paths["journal"].mkdir()
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must preflight before load"))
    with pytest.raises(ValueError, match="distinct files"):
        module.main(_cli_args(paths))
    assert not paths["output"].exists()
