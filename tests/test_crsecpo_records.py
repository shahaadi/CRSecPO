import csv
import json

import pytest

from secpo.crsecpo.records import convert_evaluation, eligible, prepare_records, read_review_export
from secpo.crsecpo.assets import fetch_asset, verify_asset


def row(key, source="source", target="target", pool="training", **kwargs):
    return dict(id=key, source=source, target=target, has_rewrite=True,
                selection_status="accepted", target_policy="limited",
                selection_group="harmful_benign_framing", pool=pool, provenance={}, **kwargs)


def test_family_grouping_prevents_leakage_without_deleting_records():
    rows = [row("a", "Same SOURCE", "t1"), row("b", "same source", "t2"),
            row("c", "different", "t3", pool="evaluation"),
            row("d", "other", "t4")]
    outputs, assignments = prepare_records(rows, links=[{"left": "b", "right": "c", "reason": "paraphrase"}])
    assert {v["id"] for v in outputs["quarantine"]} == {"a", "b"}
    assert [v["id"] for v in outputs["test"]] == ["c"]
    assert len(assignments) == len(rows) == sum(map(len, outputs.values()))
    kept = outputs["train"] + outputs["validation"]
    assert not {v["family_id"] for v in kept} & {v["family_id"] for v in outputs["test"]}


def test_split_is_stable_and_source_variants_stay_together():
    rows = [row(str(i), f"source-{i//2}", f"target-{i}") for i in range(30)]
    _, a = prepare_records(rows)
    _, b = prepare_records(list(reversed(rows)))
    assert {x["id"]: x for x in a} == {x["id"]: x for x in b}
    for i in range(0, 30, 2):
        assert a[i]["family_id"] == a[i+1]["family_id"]
        assert a[i]["split"] == a[i+1]["split"]


def test_unanimous_accept_does_not_resolve_policy_disagreement():
    value = row("a")
    value["target_policy"] = "disputed"
    assert not eligible(value)
    outputs, assignments = prepare_records([value])
    assert len(outputs["quarantine"]) == len(assignments) == 1
    value["target_policy"] = "permissible"
    assert not eligible(value)
    value["selection_group"] = "benign_adversarial_control"
    assert eligible(value)


def test_bad_links_and_duplicate_ids_fail():
    with pytest.raises(ValueError, match="unique"):
        prepare_records([row("a"), row("a")])
    with pytest.raises(ValueError, match="known"):
        prepare_records([row("a")], links=[dict(left="a", right="missing", reason="same")])
    assert not eligible(row("same", source="unchanged", target="unchanged"))


def test_matched_controls_require_explicit_valid_test_pairs():
    harmful = row("h", source="h source", target="h target", pool="evaluation")
    benign = row("b", source="b source", target="b target", pool="evaluation")
    benign.update(target_policy="permissible", selection_group="benign_adversarial_control")
    outputs, _ = prepare_records([harmful, benign], matched_controls=[
        dict(harmful_id="h", benign_id="b", reason="audited topic and style match")])
    control = next(r for r in outputs["test"] if r["id"] == "b")
    assert control["control_group"] == "matched" and control["matched_harmful_ids"] == ["h"]
    with pytest.raises(ValueError, match="accepted harmful and benign"):
        prepare_records([harmful, benign], matched_controls=[dict(harmful_id="b", benign_id="h", reason="wrong")])
    with pytest.raises(ValueError, match="Duplicate matched control link"):
        prepare_records([harmful, benign], matched_controls=[
            dict(harmful_id="h", benign_id="b", reason="first audit"),
            dict(harmful_id="h", benign_id="b", reason="repeated audit")])
    assert "matched_harmful_ids" not in benign


def test_evaluation_adapter_preserves_original_records(tmp_path):
    original = {"vanilla": "Direct\nrequest", "adversarial": "Rewritten\trequest", "extra": [1, 2]}
    harmful, benign = tmp_path / "harmful.jsonl", tmp_path / "benign.jsonl"
    for path in (harmful, benign):
        path.write_text(json.dumps(original) + "\n")
    output = tmp_path / "eval.tsv"
    result = convert_evaluation(harmful, benign, output, revision="a" * 40)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert result["counts"] == {"harmful": 1, "benign": 1}
    assert all(json.loads(r["original_json"]) == original for r in rows)
    assert all(r["vanilla"] == original["vanilla"] for r in rows)
    with pytest.raises(FileExistsError):
        convert_evaluation(harmful, benign, output, revision="a" * 40)


def evaluation_cli_fixture(tmp_path, monkeypatch):
    from secpo.crsecpo import records
    sources = {group: tmp_path / (group + ".jsonl") for group in ("harmful", "benign")}
    for group, path in sources.items():
        path.write_text(json.dumps({"vanilla": "Museum source " + group, "adversarial": "Museum target " + group}) + "\n")
    manifest = tmp_path / "fixture_asset_pins.json"
    manifest.write_text(json.dumps({"assets": [{"name": "wildjailbreak_" + group,
        "sha256": records.sha256(path),
        "url": f"https://raw.githubusercontent.com/fixture/evaluation/{'a' * 40}/{group}.jsonl"}
        for group, path in sources.items()]}))
    monkeypatch.setattr(records, "INPUT_ASSET_MANIFEST", manifest)
    output = tmp_path / "new" / "eval.tsv"
    args = ["evaluation-tsv", "--harmful", str(sources["harmful"]), "--benign", str(sources["benign"]),
            "--revision", "a" * 40, "--output", str(output)]
    return sources, output, args


def test_evaluation_tsv_cli_verifies_source_pins_and_preserves_originals(tmp_path, monkeypatch):
    from secpo.crsecpo import records
    sources, output, args = evaluation_cli_fixture(tmp_path, monkeypatch)
    original = {path: path.read_bytes() for path in sources.values()}
    records.main(args)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2
    for row, path in zip(rows, sources.values()):
        assert row["original_sha256"] == records.sha256(path)
        assert row["original_revision"] == "a" * 40
        assert json.loads(row["original_json"]) == json.loads(original[path])
    assert all(path.read_bytes() == content for path, content in original.items())
    with pytest.raises(FileExistsError):
        records.main(args)


@pytest.mark.parametrize("changed", ["harmful", "benign", "revision"])
def test_evaluation_tsv_cli_rejects_mixed_or_changed_release_before_publication(tmp_path, monkeypatch, changed):
    from secpo.crsecpo import records
    sources, output, args = evaluation_cli_fixture(tmp_path, monkeypatch)
    if changed == "revision":
        args[args.index("--revision") + 1] = "b" * 40
    else:
        sources[changed].write_text(json.dumps({"vanilla": "Different museum source", "adversarial": "Different museum target"}) + "\n")
    with pytest.raises(ValueError, match="revision differs|mismatched pinned asset"):
        records.main(args)
    assert not output.parent.exists()


def test_review_export_hides_completions_and_rejects_wrong_partition(tmp_path):
    path = tmp_path / "master.tsv"
    source = {"vanilla": "s", "adversarial": "t", "completion": "HIDDEN",
              **{"crsecpo_" + k: v for k, v in dict(record_id="id", consensus="accepted",
                target_policy="permissible", selection_group="benign_adversarial_control",
                has_rewrite="1", revision="a" * 40, row_index="1", partition="train").items()}}
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source), delimiter="\t")
        writer.writeheader(); writer.writerow(source)
    result = list(read_review_export(path, pool="training"))
    assert eligible(result[0])
    assert "HIDDEN" not in json.dumps(result)
    with pytest.raises(ValueError, match="partition"):
        list(read_review_export(path, pool="evaluation"))


def test_review_export_namespace_collision_and_ambiguity(tmp_path):
    path = tmp_path / "master.tsv"
    audit = dict(record_id="review-id", consensus="accepted", target_policy="permissible",
                 selection_group="benign_adversarial_control", has_rewrite="1",
                 revision="a" * 40, row_index="1", partition="train")
    source = {"vanilla": "s", "adversarial": "t", "crsecpo_record_id": "publisher-id",
              **{"crsecpo_2_" + k: v for k, v in audit.items()}}
    def write():
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(source), delimiter="\t")
            writer.writeheader(); writer.writerow(source)
    write()
    assert list(read_review_export(path, pool="training"))[0]["id"] == "review-id"
    source.update({"crsecpo_" + k: v for k, v in audit.items()})
    write()
    with pytest.raises(ValueError, match="one namespaced review export"):
        list(read_review_export(path, pool="training"))


def test_bad_evaluation_input_does_not_publish_partial_tsv(tmp_path):
    harmful, benign = tmp_path / "h.jsonl", tmp_path / "b.jsonl"
    harmful.write_text(json.dumps(dict(vanilla="s", adversarial="t")) + "\n")
    benign.write_text(json.dumps(dict(vanilla="missing target")) + "\n")
    output = tmp_path / "eval.tsv"
    with pytest.raises(ValueError):
        convert_evaluation(harmful, benign, output, revision="a"*40)
    assert not output.exists()


def test_asset_hash_mismatch_never_downloads_over_existing_file(tmp_path):
    path = tmp_path / "asset"
    path.write_text("wrong")
    with pytest.raises(ValueError, match="mismatched"):
        fetch_asset({"filename": "asset", "sha256": "a"*64, "url": "https://invalid.example"}, tmp_path)
    assert path.read_text() == "wrong"


def test_generated_asset_pins_verify_against_source_without_copying_readiness(tmp_path, capsys):
    from secpo.crsecpo.assets import main, project_source_lock
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "secpo": {"revision": "a" * 40},
        "wildjailbreak_evaluation": {"files": [dict(
            path="test/harmful.jsonl", url="https://example.org/harmful", sha256="b" * 64)]},
        "utility_evaluation": {"ifeval": dict(repository="https://github.com/org/repo",
            revision="c" * 40, path="data/input.jsonl", sha256="d" * 64)},
        "base_model": dict(repository="org/model", revision="e" * 40,
            weight_files=["model.safetensors"], support_files=["tokenizer.json"],
            access=dict(inference_verified=True, file_probes=[
                dict(path="model.safetensors", etag="f" * 64)]),
            downloaded_support_files=[dict(path="tokenizer.json", sha256="1" * 64)])}))
    manifest = tmp_path / "manifest.json"
    projected = project_source_lock(source)
    manifest.write_text(json.dumps(projected))
    assert projected["base_model"]["file_sha256"]["tokenizer.json"] == "1" * 64
    assert projected["assets"][1]["url"] == (
        "https://raw.githubusercontent.com/org/repo/" + "c" * 40 + "/data/input.jsonl")
    assert "inference_verified" not in json.dumps(projected)
    main(["--manifest", str(manifest), "--source-lock", str(source)])
    assert json.loads(capsys.readouterr().out)["source_lock_verified"]
    projected["base_model"]["revision"] = "2" * 40
    manifest.write_text(json.dumps(projected))
    with pytest.raises(ValueError, match="authoritative source lock"):
        main(["--manifest", str(manifest), "--source-lock", str(source)])


@pytest.mark.parametrize("finish,tokens,verified", [("stop", 2, True), ("length", 16, False), ("stop", 0, False)])
def test_host_smoke_uses_the_production_backend(tmp_path, monkeypatch, capsys, finish, tokens, verified):
    import sys
    from types import SimpleNamespace
    from secpo.crsecpo import assets, inference
    revision = "a" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(assets=[], base_model=dict(repository="org/model",
        revision=revision, weight_files=[], support_files=[], file_sha256={}))))
    monkeypatch.setattr(assets, "host_report", lambda: {"cuda_available": True})
    downloads, requests = [], []
    def local_snapshot(**kwargs):
        downloads.append(kwargs)
        return str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=local_snapshot))
    class Backend:
        def __init__(self, spec, *, device, dtype):
            assert spec.model_id == spec.tokenizer_id == str(snapshot)
            assert spec.model_revision == spec.tokenizer_revision == revision
            assert device == "cuda" and dtype == "bfloat16"
            self.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "native prompt")
        def __call__(self, request):
            requests.append(request)
            return dict(finish_reason=finish, output_tokens=tokens)
    monkeypatch.setattr(inference, "HFBackend", Backend)
    assets.main(["--manifest", str(manifest), "--smoke-model"])
    assert json.loads(capsys.readouterr().out)["model_inference_verified"] is verified
    assert downloads[0]["local_files_only"] is True
    assert requests[0]["rendered_prompt"] == "native prompt"
    assert requests[0]["max_new_tokens"] == 16 and requests[0]["temperature"] == 0
