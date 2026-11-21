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


def test_unicode_separators_round_trip_in_journal_and_pair_exports(tmp_path):
    from secpo.data import _read_json_records
    from secpo.evaluation.sep_judge import load_jsonl

    class UnicodeAnswers(ScriptedBackend):
        def __call__(self, request):
            response = super().__call__(request)
            if request["kind"] in {"ordinary", "combined"}:
                response["text"] = response["text"].replace(" answer", " answer\u0085\u2028\u2029continued")
            return response

    candidates = [candidate(id="first"), candidate(id="second")]
    journal_path = tmp_path / "requests.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    outcomes_path = tmp_path / "outcomes.jsonl"
    backend = UnicodeAnswers(votes=["combined"] * 4)
    options = dict(output_path=pairs_path, outcomes_path=outcomes_path)
    pairs, outcomes = collect(JournalBackend(backend, journal_path), candidates, **options)
    assert len(pairs) == 2 and "\u2028" in pairs[0]["chosen"]
    assert load_jsonl(outcomes_path) == outcomes
    assert _read_json_records(pairs_path) == pairs
    assert collect(JournalBackend(None, journal_path), candidates, **options) == (pairs, outcomes)


def test_malformed_judgment_and_backend_error_are_logged():
    class Backend(ScriptedBackend):
        def __call__(self, request):
            if request["kind"] == "preference_judgment":
                return {"text": "A", "finish_reason": "stop"}
            return super().__call__(request)
    pairs, outcomes = collect(Backend())
    assert not pairs and all(vote["outcome"] == "error" for vote in outcomes[0]["judgments"])
    assert outcomes[0]["status"] == "judgment_error"


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


def make_legacy_export(paths):
    from secpo.crsecpo.collection_exports import export_record_path, export_reservation_path

    for sidecar in (export_record_path(paths["output"]),
                    export_reservation_path(paths["output"]), export_reservation_path(paths["outcomes"])):
        payload = json.loads(sidecar.read_text())
        del payload["manifest_sha256"]
        sidecar.write_text(json.dumps(payload))


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


def test_cli_rejects_case_aliased_journal_before_creating_files(tmp_path, monkeypatch):
    from secpo.crsecpo import pairs as module

    probe = tmp_path / "case-probe"
    probe.touch()
    if not probe.with_name("CASE-PROBE").exists():
        pytest.skip("requires a case-insensitive volume")
    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["journal"] = paths["output"].with_name("OUTPUT.JSONL")
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="distinct files"):
        module.main(_cli_args(paths))
    assert not paths["output"].exists() and not paths["journal"].exists()


@pytest.mark.parametrize("hardlink", [False, True])
def test_cli_locks_journal_before_recovery_or_model_startup(tmp_path, monkeypatch, hardlink):
    import fcntl
    import os
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    paths["journal"].write_text('{"attempt":1,"request":')
    from secpo.crsecpo.collection_exports import export_reservation_path
    export_reservation_path(paths["journal"]).write_text('{"role":"journal"}')
    if hardlink:
        lock_path = tmp_path / "alias.jsonl"
        os.link(paths["journal"], lock_path)
    else:
        lock_path = tmp_path / "journal.jsonl.lock"
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must not load"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another collector is using journal"):
            module.main(_cli_args(paths))
        assert paths["journal"].read_text() == '{"attempt":1,"request":'
        assert not paths["output"].with_suffix(".jsonl.collection.json").exists()

    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    module.main(_cli_args(paths))
    assert JournalBackend(None, paths["journal"]).entries
    assert paths["output"].exists()


@pytest.mark.parametrize("recover", [False, True])
def test_cli_locks_new_and_repaired_journal_inodes(tmp_path, monkeypatch, recover):
    import os
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    if recover:
        paths["journal"].write_text('{"attempt":1,"request":')
        from secpo.crsecpo.collection_exports import export_reservation_path
        export_reservation_path(paths["journal"]).write_text('{"role":"journal"}')
    alias = tmp_path / "alias.jsonl"
    contender = dict(paths, journal=alias, output=tmp_path / "other-pairs.jsonl",
                     outcomes=tmp_path / "other-outcomes.jsonl")
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    def interleave(*args, **kwargs):
        os.link(paths["journal"], alias)
        with monkeypatch.context() as patch:
            patch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must not load"))
            with pytest.raises(ValueError, match="another collector is using journal"):
                module.main(_cli_args(contender))
        return backend

    monkeypatch.setattr(module, "HFBackend", interleave)
    module.main(_cli_args(paths))
    assert len(JournalBackend(None, alias).entries) == 4
    assert not contender["output"].exists()


def test_cli_reserves_shared_outcomes_before_starting_backend(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module
    from secpo.crsecpo.collection_exports import read_collection_export

    first, second = [{field: tmp_path / f"{name}-{field}.jsonl"
                      for field in ("candidates", "journal", "output", "outcomes")}
                     for name in ("first", "second")]
    second["outcomes"] = first["outcomes"]
    for name, paths in (("first", first), ("second", second)):
        paths["candidates"].write_text(json.dumps(candidate(id=name)) + "\n")

    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    def interleave(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("outcomes already reserved"))
            with pytest.raises(ValueError, match="outcomes path belongs to another collection"):
                module.main(_cli_args(second))
        return backend

    monkeypatch.setattr(module, "HFBackend", interleave)
    module.main(_cli_args(first))
    for role, path in (("pairs", first["output"]), ("outcomes", first["outcomes"])):
        assert json.loads(read_collection_export(path, role))["id"] == "first"
    assert not second["output"].exists() and not second["journal"].exists()
    assert not second["output"].with_suffix(".jsonl.collection.json").exists()
    second["outcomes"] = tmp_path / "free-outcomes.jsonl"
    fresh = ScriptedBackend()
    fresh.tokenizer = Tokenizer()
    fresh.model = backend.model
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: fresh)
    module.main(_cli_args(second))
    assert json.loads(read_collection_export(second["output"], "pairs"))["id"] == "second"


def test_exports_and_journals_cannot_claim_each_others_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module
    from secpo.crsecpo.collection_exports import read_collection_export

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    def interleave(*args, **kwargs):
        with pytest.raises(ValueError, match="another writer is using export lock"):
            collect(ScriptedBackend(), output_path=paths["journal"])
        other = dict(paths, journal=paths["outcomes"], output=tmp_path / "other-pairs.jsonl",
                     outcomes=tmp_path / "other-outcomes.jsonl")
        with pytest.raises(ValueError, match="journal path belongs to an export"):
            module.main(_cli_args(other))
        assert not other["output"].with_suffix(".jsonl.collection.json").exists()
        return backend

    monkeypatch.setattr(module, "HFBackend", interleave)
    module.main(_cli_args(paths))
    before = paths["journal"].read_bytes()
    assert len(JournalBackend(None, paths["journal"]).entries) == 4
    for destination in ({"output_path": paths["journal"]}, {"outcomes_path": paths["journal"]},
                        {"output_path": tmp_path / "paired.jsonl", "outcomes_path": paths["journal"]}):
        with pytest.raises(ValueError, match="export path belongs to a journal"):
            collect(ScriptedBackend(), **destination)
    assert paths["journal"].read_bytes() == before
    assert read_collection_export(paths["outcomes"], "outcomes")


def test_empty_standalone_export_cannot_be_reused_as_journal(tmp_path):
    from secpo.crsecpo.collection_exports import read_collection_export

    path = tmp_path / "pairs.jsonl"
    collect(ScriptedBackend(votes=("ordinary", "ordinary")), output_path=path)
    assert read_collection_export(path, "pairs") == b""
    with pytest.raises(ValueError, match="journal path belongs to an export"):
        JournalBackend(None, path)


def test_direct_export_cannot_replace_cli_results(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module
    from secpo.crsecpo.collection_exports import read_collection_export

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate(id="cli")) + "\n")
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    module.main(_cli_args(paths))
    with pytest.raises(ValueError, match="belongs to a CLI collection"):
        collect(ScriptedBackend(), [candidate(id="direct")], output_path=paths["output"],
                outcomes_path=paths["outcomes"])
    assert json.loads(read_collection_export(paths["output"], "pairs"))["id"] == "cli"


def test_bad_journal_does_not_reserve_output_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import collection_exports as exports, pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    malformed = '{"status":"ok"}\n'
    paths["journal"].write_text(malformed)
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="attempt without a request object"):
        module.main(_cli_args(paths))
    assert paths["journal"].read_text() == malformed
    assert not exports.collection_manifest_path(paths["output"]).exists()
    assert not any(exports.export_reservation_path(paths[name]).exists() for name in ("output", "outcomes"))
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    module.main(_cli_args(dict(paths, journal=tmp_path / "valid-journal.jsonl")))
    assert paths["output"].exists() and paths["outcomes"].exists()


@pytest.mark.parametrize("change", ["different", "missing"])
def test_collection_readers_verify_manifest(tmp_path, monkeypatch, change):
    from types import SimpleNamespace
    from secpo.crsecpo import collection_exports as exports, pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    module.main(_cli_args(paths))
    manifest = exports.collection_manifest_path(paths["output"])
    recorded = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(recorded, sort_keys=True))
    for role, key in (("pairs", "output"), ("outcomes", "outcomes")):
        assert exports.read_collection_export(paths[key], role)
    if change == "different":
        manifest.write_text(json.dumps(dict(recorded, scheduled_ids=["unrelated"])))
    else:
        manifest.unlink()
    for role, key in (("pairs", "output"), ("outcomes", "outcomes")):
        with pytest.raises(ValueError, match="collection manifest does not match|collection.json"):
            exports.read_collection_export(paths[key], role)


def test_cli_pair_file_with_missing_sidecars_is_not_read_as_standalone(tmp_path):
    from secpo.crsecpo import collection_exports as exports

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text('{}\n')
    exports.collection_manifest_path(pairs).write_text('{}')
    with pytest.raises(ValueError, match="incomplete"):
        exports.read_collection_export(pairs, "pairs")


def test_cli_symlink_aliases_share_collection_manifest(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module
    from secpo.crsecpo.collection_exports import read_collection_export

    canonical = tmp_path / "pairs.jsonl"
    first, second = [{field: tmp_path / f"{name}-{field}.jsonl"
                      for field in ("candidates", "output", "journal", "outcomes")}
                     for name in ("first", "second")]
    second["outcomes"] = first["outcomes"]
    for name, paths in (("first", first), ("second", second)):
        paths["candidates"].write_text(json.dumps(candidate(id=name)) + "\n")
        paths["output"].symlink_to(canonical)

    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    def interleave(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(module, "HFBackend", lambda *a, **kw: pytest.fail("must not load"))
            with pytest.raises(ValueError, match="collection settings changed"):
                module.main(_cli_args(second))
        return backend

    monkeypatch.setattr(module, "HFBackend", interleave)
    module.main(_cli_args(first))
    assert json.loads(read_collection_export(canonical, "pairs"))["id"] == "first"
    assert canonical.with_suffix(".jsonl.collection.json").exists()
    for paths in (first, second):
        assert not paths["output"].with_suffix(".jsonl.collection.json").exists()


def test_cli_migrates_legacy_symlink_manifest(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    module.main(_cli_args(paths))
    before = paths["output"].read_bytes()
    canonical_manifest = paths["output"].with_suffix(".jsonl.collection.json")
    manifest = json.loads(canonical_manifest.read_text())
    alias = tmp_path / "aliases" / "linked.jsonl"
    alias.parent.mkdir()
    alias.symlink_to(paths["output"])
    legacy_manifest = alias.with_suffix(".jsonl.collection.json")
    legacy = {**manifest, "destinations": {name: os.path.relpath(paths[name], alias.parent)
                                          for name in ("journal", "outcomes")}}
    legacy_manifest.write_text(json.dumps(legacy))
    canonical_manifest.unlink()

    module.main(_cli_args(dict(paths, output=alias)))
    assert json.loads(canonical_manifest.read_text()) == manifest
    assert legacy_manifest.read_text() == json.dumps(legacy)
    assert len(backend.requests) == 4 and paths["output"].read_bytes() == before


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
    assert manifest["maximum_distinct_requests"] == 8
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
    from secpo.crsecpo.collection_exports import (export_record_path, export_reference_path,
                                                 read_collection_export)
    for sidecar in (export_record_path(paths["output"]), export_reference_path(paths["outcomes"])):
        shutil.copyfile(sidecar, moved / sidecar.name)
    assert read_collection_export(moved / paths["output"].name, "pairs") == output
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    moved_args = _cli_args({name: moved / path.name for name, path in paths.items()})
    moved_args[moved_args.index("--max-candidates") + 1] = "2"
    module.main(moved_args)
    assert len(backend.requests) == 8 and (moved / paths["output"].name).read_bytes() == output


def test_candidate_hash_matches_parsed_bytes_when_input_changes(tmp_path, monkeypatch):
    from hashlib import sha256
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    payload = (json.dumps(candidate(target="First synthetic target")) + "\n").encode()
    paths["candidates"].write_bytes(payload)
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    original_load = module.load_jsonl

    def replace_during_parse(path, **kwargs):
        paths["candidates"].write_text(json.dumps(candidate(target="Second synthetic target")) + "\n")
        return original_load(path, **kwargs)

    monkeypatch.setattr(module, "load_jsonl", replace_during_parse)
    module.main(_cli_args(paths))
    manifest = json.loads(paths["output"].with_suffix(".jsonl.collection.json").read_text())
    assert manifest["input_sha256"] == sha256(payload).hexdigest()
    assert json.loads(paths["output"].read_text())["target"] == "First synthetic target"


def test_direct_collection_cannot_rebind_pair_export(tmp_path):
    from secpo.crsecpo.collection_exports import export_reference_path, read_collection_export

    pairs = tmp_path / "pairs.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    replacement = tmp_path / "other-outcomes.jsonl"
    collect(ScriptedBackend(), [candidate(id="first")], output_path=pairs, outcomes_path=outcomes)
    with pytest.raises(ValueError, match="pair path belongs to another collection"):
        collect(ScriptedBackend(), [candidate(id="second")], output_path=pairs, outcomes_path=replacement)
    assert json.loads(read_collection_export(pairs, "pairs"))["id"] == "first"
    assert json.loads(read_collection_export(outcomes, "outcomes"))["id"] == "first"
    assert not replacement.exists() and not export_reference_path(replacement).exists()


@pytest.mark.parametrize("mode", ["pair_only", "outcomes_only", "swap"])
def test_direct_collection_cannot_overwrite_reserved_paths(tmp_path, mode):
    from secpo.crsecpo.collection_exports import read_collection_export

    pairs = tmp_path / "pairs.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    collect(ScriptedBackend(), [candidate(id="first")], output_path=pairs, outcomes_path=outcomes)
    paths = {"pair_only": (pairs, None), "outcomes_only": (None, outcomes),
             "swap": (outcomes, pairs)}
    output_path, outcomes_path = paths[mode]
    with pytest.raises(ValueError, match="standalone export would overwrite|cannot switch export roles"):
        collect(ScriptedBackend(), [candidate(id="second")], output_path=output_path,
                outcomes_path=outcomes_path)
    assert json.loads(read_collection_export(pairs, "pairs"))["id"] == "first"
    assert json.loads(read_collection_export(outcomes, "outcomes"))["id"] == "first"


@pytest.mark.parametrize("suffix", [".exports.json", ".exports-ref.json", ".exports-reservation.json",
                                    ".exports.lock", ".collection.json"])
def test_direct_collection_cannot_replace_metadata(tmp_path, suffix):
    destination = tmp_path / f"existing.jsonl{suffix}"
    destination.write_bytes(b"preserve")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(destination)
    uppercase = destination.with_name(destination.name.upper())
    for path in (destination, alias, uppercase):
        with pytest.raises(ValueError, match="reserved metadata filename"):
            collect(ScriptedBackend(), [candidate()], output_path=path)
    assert destination.read_bytes() == b"preserve"


def test_interrupted_export_reservation_keeps_original_pair_binding(tmp_path, monkeypatch):
    from secpo.crsecpo import collection_exports as exports

    pairs = tmp_path / "pairs.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    reference = exports.export_reference_path(outcomes)
    replacement = tmp_path / "other-outcomes.jsonl"
    original_write = exports.atomic_write_json

    def interrupt_reference(path, payload):
        if path == reference:
            raise OSError("interrupted outcomes reservation")
        original_write(path, payload)

    rows = [candidate(id="first")]
    with monkeypatch.context() as patch:
        patch.setattr(exports, "atomic_write_json", interrupt_reference)
        with pytest.raises(OSError, match="interrupted outcomes reservation"):
            collect(ScriptedBackend(), rows, output_path=pairs, outcomes_path=outcomes)
    assert exports.export_record_path(pairs).exists() and not reference.exists()
    with pytest.raises(ValueError, match="pair path belongs to another collection"):
        collect(ScriptedBackend(), rows, output_path=pairs, outcomes_path=replacement)
    with pytest.raises(ValueError, match="outcomes path belongs to another collection"):
        collect(ScriptedBackend(), rows, output_path=tmp_path / "other-pairs.jsonl", outcomes_path=outcomes)
    with pytest.raises(ValueError, match="standalone export would overwrite"):
        collect(ScriptedBackend(), rows, outcomes_path=outcomes)
    collect(ScriptedBackend(), rows, output_path=pairs, outcomes_path=outcomes)
    assert json.loads(exports.read_collection_export(pairs, "pairs"))["id"] == "first"
    assert json.loads(exports.read_collection_export(outcomes, "outcomes"))["id"] == "first"


def test_interruption_before_both_reservations_allows_safe_reuse(tmp_path, monkeypatch):
    from secpo.crsecpo import collection_exports as exports

    pairs = tmp_path / "pairs.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    other_pairs = tmp_path / "other-pairs.jsonl"
    other_outcomes = tmp_path / "other-outcomes.jsonl"
    original_write = exports.atomic_write_json

    def interrupt_second_reservation(path, payload):
        if path == exports.export_reservation_path(outcomes):
            raise OSError("interrupted before commit")
        original_write(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(exports, "atomic_write_json", interrupt_second_reservation)
        with pytest.raises(OSError, match="interrupted before commit"):
            collect(ScriptedBackend(), [candidate(id="first")], output_path=pairs, outcomes_path=outcomes)
    assert exports.export_reservation_path(pairs).exists()
    assert not exports.export_record_path(pairs).exists()
    assert not pairs.exists() and not outcomes.exists()
    collect(ScriptedBackend(), [candidate(id="second")], output_path=other_pairs, outcomes_path=outcomes)
    collect(ScriptedBackend(), [candidate(id="first")], output_path=pairs, outcomes_path=other_outcomes)
    assert json.loads(exports.read_collection_export(pairs, "pairs"))["id"] == "first"
    assert json.loads(exports.read_collection_export(outcomes, "outcomes"))["id"] == "second"


@pytest.mark.parametrize("role", ["pairs", "outcomes"])
def test_reader_rejects_partial_publication_during_payload_open(tmp_path, monkeypatch, role):
    from pathlib import Path
    from secpo.crsecpo import artifact_versions, collection_exports as exports

    pairs = tmp_path / "pairs.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    requested = pairs if role == "pairs" else outcomes
    pointer = artifact_versions.version_root(pairs) / "current"
    real_read = Path.read_bytes
    real_replace = artifact_versions.os.replace
    started = False

    def interrupt(source, destination):
        if destination == pointer:
            raise OSError("interrupted publication")
        return real_replace(source, destination)

    def publish_before_read(path):
        nonlocal started
        if path == requested and not started:
            started = True
            with monkeypatch.context() as patch:
                patch.setattr(artifact_versions.os, "replace", interrupt)
                with pytest.raises(OSError, match="interrupted publication"):
                    collect(ScriptedBackend(), output_path=pairs, outcomes_path=outcomes)
        return real_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", publish_before_read)
        with pytest.raises(ValueError, match="incomplete|Resume collection"):
            exports.read_collection_export(requested, role)
    assert started and not requested.exists()
    collect(ScriptedBackend(), output_path=pairs, outcomes_path=outcomes)
    assert exports.read_collection_export(requested, role)


@pytest.mark.parametrize("when", ["before_replay", "while_staging"])
@pytest.mark.parametrize("standalone", [False, True])
def test_pair_publication_stops_if_source_journal_disappears(tmp_path, monkeypatch, when, standalone):
    from secpo.crsecpo import collection_exports as exports

    journal = tmp_path / "journal.jsonl"
    rows = [candidate()]
    collect(JournalBackend(ScriptedBackend(), journal), rows)
    replay = JournalBackend(None, journal)
    pairs = tmp_path / "pairs.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    if when == "before_replay":
        journal.unlink()
        with pytest.raises(ValueError, match="journal changed after loading"):
            replay.completed(next(iter(replay.entries.values()))["request"])
    else:
        real_write = exports.atomic_write_jsonl

        def remove_during_staging(values, path, **kwargs):
            journal.unlink(missing_ok=True)
            real_write(values, path, **kwargs)

        monkeypatch.setattr(exports, "atomic_write_jsonl", remove_during_staging)
    with pytest.raises(ValueError, match="journal changed after loading"):
        collect(replay, rows, output_path=pairs, outcomes_path=None if standalone else outcomes)
    assert not pairs.exists() and not outcomes.exists()


def test_collection_checks_journal_bytes_even_when_size_and_mtime_match(tmp_path):
    import os

    path = tmp_path / "journal.jsonl"
    collect(JournalBackend(ScriptedBackend(), path))
    journal = JournalBackend(None, path)
    original = path.read_bytes()
    stat = path.stat()
    modified = original.replace(b"combined answer", b"ordinary answer", 1)
    assert modified != original and len(modified) == len(original)
    path.write_bytes(modified)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    pairs, outcomes = tmp_path / "pairs.jsonl", tmp_path / "outcomes.jsonl"
    with pytest.raises(ValueError, match="journal contents changed after loading"):
        collect(journal, output_path=pairs, outcomes_path=outcomes)
    assert not pairs.exists() and not outcomes.exists()


def test_journal_write_failure_aborts_collection_instead_of_becoming_an_outcome(tmp_path, monkeypatch):
    pairs, outcomes = tmp_path / "pairs.jsonl", tmp_path / "outcomes.jsonl"
    journal = JournalBackend(ScriptedBackend(), tmp_path / "journal.jsonl")

    def fail_write(row):
        raise OSError("simulated journal storage failure")

    monkeypatch.setattr(journal, "_write_attempt", fail_write)
    with pytest.raises(ValueError, match="cannot write journal.*storage failure"):
        collect(journal, output_path=pairs, outcomes_path=outcomes)
    assert not pairs.exists() and not outcomes.exists()


@pytest.mark.parametrize("mode", ["new", "legacy", "versioned"])
def test_interrupted_collection_refresh_preserves_last_committed_export(tmp_path, monkeypatch, mode):
    from secpo.crsecpo import artifact_versions, collection_exports as exports

    pair_path = tmp_path / "train" / "pairs.jsonl"
    outcome_path = tmp_path / "audit" / "outcomes.jsonl"
    options = {"output_path": pair_path, "outcomes_path": outcome_path}
    if mode != "new":
        collect(ScriptedBackend(), [candidate(target="First synthetic target")], **options)
        if mode == "legacy":
            import shutil

            for path in (pair_path, outcome_path, exports.export_record_path(pair_path)):
                payload = path.read_bytes()
                path.unlink()
                path.write_bytes(payload)
            shutil.rmtree(artifact_versions.version_root(pair_path))

    pointer = artifact_versions.version_root(pair_path) / "current"
    real_replace = artifact_versions.os.replace
    next_target = "Second synthetic target"

    def interrupt_commit(source, destination):
        # Legacy migration may first point at a copy of the old generation.
        if destination == pointer and (mode == "new" or pointer.exists()):
            raise OSError("interrupted commit")
        return real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_versions.os, "replace", interrupt_commit)
        with pytest.raises(OSError, match="interrupted commit"):
            collect(ScriptedBackend(), [candidate(target=next_target)], **options)

    for path, role in ((pair_path, "pairs"), (outcome_path, "outcomes")):
        if mode == "new":
            assert not path.exists()
            with pytest.raises(ValueError, match="Resume collection"):
                exports.read_collection_export(path, role)
        else:
            assert json.loads(path.read_bytes())["target"] == "First synthetic target"
            assert json.loads(exports.read_collection_export(path, role))["target"] == "First synthetic target"

    collect(ScriptedBackend(), [candidate(target=next_target)], **options)
    assert json.loads(exports.read_collection_export(pair_path, "pairs"))["target"] == next_target
    assert json.loads(exports.read_collection_export(outcome_path, "outcomes"))["target"] == next_target


def test_legacy_import_verifies_the_copied_generation_before_switching(tmp_path, monkeypatch):
    import shutil
    from secpo.crsecpo import artifact_versions, collection_exports as exports

    pairs, outcomes = tmp_path / "pairs.jsonl", tmp_path / "outcomes.jsonl"
    options = {"output_path": pairs, "outcomes_path": outcomes}
    collect(ScriptedBackend(), [candidate(id="old")], **options)
    for path in (pairs, outcomes, exports.export_record_path(pairs)):
        payload = path.read_bytes()
        path.unlink()
        path.write_bytes(payload)
    shutil.rmtree(artifact_versions.version_root(pairs))
    real_copy = shutil.copyfile

    def copy_changed_file(source, destination):
        if source == pairs:
            destination.write_bytes(source.read_bytes().replace(b'"old"', b'"bad"', 1))
            return
        real_copy(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfile", copy_changed_file)
        with pytest.raises(ValueError, match="failed verification during migration"):
            collect(ScriptedBackend(), [candidate(id="new")], **options)
    assert artifact_versions.committed_version(pairs) is None
    assert not pairs.is_symlink()
    assert json.loads(exports.read_collection_export(pairs, "pairs"))["id"] == "old"
    collect(ScriptedBackend(), [candidate(id="new")], **options)
    assert json.loads(exports.read_collection_export(pairs, "pairs"))["id"] == "new"


def test_collection_reader_pins_the_committed_generation(tmp_path, monkeypatch):
    from pathlib import Path
    from secpo.crsecpo import artifact_versions, collection_exports as exports

    pair_path = tmp_path / "pairs.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    options = {"output_path": pair_path, "outcomes_path": outcomes}
    collect(ScriptedBackend(), [candidate(id="first")], **options)
    old = artifact_versions.committed_version(pair_path)
    read_text = Path.read_text

    def refresh_after_pin(path, *args, **kwargs):
        if path == old / "record.json":
            with monkeypatch.context() as patch:
                patch.setattr(Path, "read_text", read_text)
                collect(ScriptedBackend(), [candidate(id="second")], **options)
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", refresh_after_pin)
        assert json.loads(exports.read_collection_export(pair_path, "pairs"))["id"] == "first"
    assert json.loads(exports.read_collection_export(pair_path, "pairs"))["id"] == "second"


@pytest.fixture
def freeze_input(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import training

    class SyntheticCollator:
        def __init__(self, *args, **kwargs):
            pass

        def _tokenize(self, value):
            return list(value.encode())

        _tokenize_response = _tokenize

        def __call__(self, rows):
            return {}

    monkeypatch.setattr(training, "ComparisonCollator", SyntheticCollator)
    revision = SPEC.model_revision
    spec = ModelSpec(SPEC.model_id, revision, SPEC.model_id, revision)
    source = tmp_path / "eligible.jsonl"
    config = SimpleNamespace(data_file=source, max_length=4096, max_prompt_length=None, seed=42,
                             model_name_or_path=spec.model_id, model_revision=revision)

    def collect_and_freeze(identities, output_dir, *, second_pair_options=None, legacy_second_pair=False):
        candidates = [candidate(id="h", provenance={"publisher_partition": "train"}),
                      candidate(id="b", selection_group="benign_adversarial_control",
                                target_policy="permissible", provenance={"publisher_partition": "train"})]
        pairs = []
        for index, (row, identity) in enumerate(zip(candidates, identities)):
            options = second_pair_options if index == 1 and second_pair_options else {}
            accepted, _ = generate_pairs([row], backend=ScriptedBackend(), tokenizer=Tokenizer(),
                                         model_spec=spec, backend_identity=identity, **options)
            if index == 1 and legacy_second_pair:
                del accepted[0]["pair_metadata"]["max_output_retries"]
            pairs.extend(accepted)
        source.write_text("".join(json.dumps(row) + "\n" for row in pairs))
        return training.freeze_pairs(config, tokenizer=Tokenizer(), output_dir=output_dir, pair_limit=2)

    return collect_and_freeze


@pytest.mark.parametrize("left,right", [(None, {"engine": "vllm"}),
                                       ({"engine": "vllm", "dtype": "float16"},
                                        {"engine": "vllm", "dtype": "bfloat16"})])
def test_freeze_rejects_mixed_backend_identities(tmp_path, freeze_input, left, right):
    from secpo.data import PreferenceSchemaError

    output = tmp_path / "frozen"
    with pytest.raises(PreferenceSchemaError, match="different collection backend identities"):
        freeze_input((left, right), output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["generation_config", "judge_config"])
def test_freeze_rejects_mixed_sampling_configs(tmp_path, freeze_input, field):
    from secpo.data import PreferenceSchemaError

    output = tmp_path / "frozen"
    identity = {"engine": "vllm", "dtype": "bfloat16"}
    options = {field: GenerationConfig(temperature=0.9)}
    with pytest.raises(PreferenceSchemaError, match=f"different {field}"):
        freeze_input((identity, identity), output, second_pair_options=options)
    assert not output.exists()


@pytest.mark.parametrize("legacy_second_pair", [False, True])
def test_freeze_checks_configured_retry_policy_even_without_resampling(tmp_path, freeze_input, legacy_second_pair):
    from secpo.data import PreferenceSchemaError

    output = tmp_path / "frozen"
    identity = {"engine": "vllm", "dtype": "bfloat16"}
    reason = "lack a recorded retry limit" if legacy_second_pair else "different max_output_retries"
    with pytest.raises(PreferenceSchemaError, match=reason):
        freeze_input((identity, identity), output, second_pair_options={"output_retries": 2},
                     legacy_second_pair=legacy_second_pair)
    assert not output.exists()


def test_interrupted_training_manifest_can_be_published_on_retry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import training

    config = SimpleNamespace(output_dir=tmp_path / "training")
    manifest = {"method": "sft", "training_config": {}}
    canonical = training.guard_output(config, manifest)

    def interrupted_write(staged_path, value):
        staged_path.write_text('{"method":')
        assert not canonical.exists()
        raise OSError("interrupted training manifest")

    with monkeypatch.context() as patch:
        patch.setattr(training, "atomic_write_json", interrupted_write)
        with pytest.raises(OSError, match="interrupted training manifest"):
            training._write_manifest_once(canonical, manifest)
    assert not canonical.exists()

    # Simulate the private staging directory left by a killed process.
    orphan = canonical.parent / (training._manifest_stage_prefix(canonical) + "orphan")
    orphan.mkdir()
    (orphan / canonical.name).write_text('{"method":')
    assert training.guard_output(config, manifest) == canonical
    training._write_manifest_once(canonical, manifest)
    assert json.loads(canonical.read_text()) == manifest
    assert training.guard_output(config, manifest) == canonical
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        training._write_manifest_once(canonical, dict(manifest, method="dpo"))
    assert json.loads(canonical.read_text()) == manifest


@pytest.mark.parametrize("field,value", [
    ("backend_identity", {"engine": "other"}),
    ("generation_config", {"temperature": 0.1}),
    ("judge_config", {"temperature": 0.1}),
])
def test_prepare_rejects_incorrect_collection_provenance(tmp_path, freeze_input, field, value):
    from types import SimpleNamespace
    from secpo.crsecpo import training
    from secpo.data import PreferenceSchemaError

    output = tmp_path / "frozen"
    identity = {"engine": "vllm"}
    manifest = freeze_input((identity, identity), output)
    selection = output / "selection_manifest.json"
    config = SimpleNamespace(data_file=output / "train_pairs.jsonl", online_exchange_dir=None,
        append_eos=True, max_length=4096, max_prompt_length=None, seed=42,
        model_name_or_path=SPEC.model_id, model_revision=SPEC.model_revision)
    training.prepare(config, "sft", tokenizer=Tokenizer(), require_selection=True)
    selection.write_text(json.dumps({**manifest, field: value}))
    with pytest.raises(PreferenceSchemaError, match="manifest does not match"):
        training.prepare(config, "sft", tokenizer=Tokenizer(), require_selection=True)


def test_interrupted_freeze_can_publish_to_same_directory(tmp_path, monkeypatch, freeze_input):
    from secpo.evaluation import sep_judge

    output = tmp_path / "frozen"
    identity = {"engine": "vllm", "dtype": "bfloat16"}
    original_write = sep_judge.atomic_write_jsonl

    def interrupt_audit(rows, path):
        assert not output.exists()
        if path.name == "selection_audit.jsonl":
            raise OSError("interrupted selection audit")
        return original_write(rows, path)

    with monkeypatch.context() as patch:
        patch.setattr(sep_judge, "atomic_write_jsonl", interrupt_audit)
        with pytest.raises(OSError, match="interrupted selection audit"):
            freeze_input((identity, identity), output)

    assert not output.exists()
    manifest = freeze_input((identity, identity), output)
    assert manifest["backend_identity"] == identity
    assert manifest["generation_config"] == GenerationConfig().to_dict()
    assert manifest["judge_config"] == GenerationConfig(max_new_tokens=256).to_dict()
    assert manifest["max_output_retries"] == 0
    assert manifest["selected_counts"] == {"harmful": 1, "benign": 1}
    assert {path.name for path in output.iterdir()} == {
        "train_pairs.jsonl", "selection_audit.jsonl", "selection_manifest.json",
    }
    assert json.loads((output / "selection_manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        freeze_input((identity, identity), output)


def test_interrupted_manifest_write_can_be_retried(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    paths["candidates"].write_text(json.dumps(candidate()) + "\n")
    manifest_path = paths["output"].with_suffix(".jsonl.collection.json")
    backend = ScriptedBackend()
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)

    def interrupted_write(staged_path, manifest):
        staged_path.write_text('{"schema_version":')
        assert not manifest_path.exists()
        raise OSError("interrupted manifest write")

    args = _cli_args(paths)
    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_write_json", interrupted_write)
        with pytest.raises(OSError, match="interrupted manifest write"):
            module.main(args)

    assert not manifest_path.exists()
    assert not backend.requests
    competing = dict(paths, journal=tmp_path / "another-journal.jsonl")
    with pytest.raises(ValueError, match="belongs to another collection"):
        module.main(_cli_args(competing) + ["--seed", "9"])
    assert not manifest_path.exists() and not competing["journal"].exists()
    with pytest.raises(ValueError, match="belongs to another collection"):
        collect(ScriptedBackend(), output_path=paths["output"], outcomes_path=paths["outcomes"])
    with pytest.raises(ValueError, match="standalone export would overwrite"):
        collect(ScriptedBackend(), outcomes_path=paths["outcomes"])
    upper = paths["outcomes"].with_name(paths["outcomes"].name.upper())
    from secpo.crsecpo.collection_exports import export_reservation_path
    if export_reservation_path(upper).exists():
        with pytest.raises(ValueError, match="standalone export would overwrite"):
            collect(ScriptedBackend(), outcomes_path=upper)
    module.main(args)
    assert json.loads(manifest_path.read_text())["maximum_distinct_requests"] == 4
    assert len(backend.requests) == 4


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
    assert len(manifest["deferred_ids"]) == 3 and manifest["maximum_distinct_requests"] == 16
    assert len(backend.requests) == 16
    # Old manifests use the previous field name for the same bound.
    make_legacy_export(paths)
    manifest["maximum_model_calls"] = manifest.pop("maximum_distinct_requests")
    paths["output"].with_suffix(".jsonl.collection.json").write_text(json.dumps(manifest))
    # Resuming with the same rule replays the journal; an equal explicit integer is a
    # different recorded setting and must use new export paths.
    module.main(args)
    assert len(backend.requests) == 16
    args[args.index("--max-candidates") + 1] = "4"
    with pytest.raises(ValueError, match="collection settings changed"):
        module.main(args)


def test_all_candidates_preserves_eligible_majority_candidate_after_rejection(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from secpo.crsecpo import pairs as module

    paths = {name: tmp_path / (name + ".jsonl")
             for name in ("candidates", "output", "outcomes", "journal")}
    rows = [candidate(id="harmful", family_id="h")]
    rows += [candidate(id=f"b{i}", family_id=f"b{i}", target_policy="permissible",
                      selection_group="benign_adversarial_control") for i in range(2)]
    paths["candidates"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    backend = ScriptedBackend(votes=["combined", "combined", "ordinary", "ordinary",
                                     "combined", "combined"])
    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    monkeypatch.setattr(module, "HFBackend", lambda *a, **kw: backend)
    args = _cli_args(paths)
    args[args.index("--max-candidates") + 1] = "all"
    module.main(args)

    manifest = json.loads(paths["output"].with_suffix(".jsonl.collection.json").read_text())
    assert manifest["max_candidates"] == 3 and manifest["max_candidates_rule"] == "all"
    assert manifest["deferred_ids"] == []
    accepted_ids = [json.loads(line)["id"] for line in paths["output"].read_text().splitlines()]
    assert accepted_ids == [manifest["scheduled_ids"][0], manifest["scheduled_ids"][2]]
    module.main(args)
    assert len(backend.requests) == 12


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
