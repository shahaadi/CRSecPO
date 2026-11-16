"""Read-only preparation reports must not trigger downloads or inference."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from secpo.crsecpo import assets


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    directory = tmp_path / "assets"
    directory.mkdir()
    (directory / "valid.jsonl").write_bytes(b"valid")
    (directory / "changed.jsonl").write_bytes(b"changed")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"fixture weights")
    (snapshot / "tokenizer.json").write_bytes(b"changed tokenizer")
    (snapshot / "generation_config.json").write_bytes(b"{}")
    checksum = lambda value: hashlib.sha256(value).hexdigest()
    manifest = {
        "assets": [dict(name=name, filename=name + ".jsonl", sha256=checksum(b"valid"))
                   for name in ("valid", "changed", "missing")],
        "base_model": dict(repository="org/model", revision="a" * 40,
            weight_files=["model.safetensors", "missing.safetensors"],
            support_files=["tokenizer.json", "generation_config.json"],
            file_sha256={"model.safetensors": checksum(b"fixture weights"),
                         "missing.safetensors": checksum(b"missing weights"),
                         "tokenizer.json": checksum(b"original tokenizer")})}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    calls = []

    def lookup(repository, filename, *, revision):
        assert repository == "org/model" and revision == "a" * 40
        calls.append(filename)
        path = snapshot / filename
        return str(path) if path.is_file() else None

    hub = SimpleNamespace(try_to_load_from_cache=lookup)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(assets, "host_report", lambda: dict(
        packages={}, cuda_available=False, cuda_devices=[], cuda_memory_bytes=[],
        model_inference_verified=False))
    return manifest, manifest_path, directory, hub, calls


def test_inventory_reports_every_input_without_network_loading_or_writes(inventory, tmp_path, capsys):
    manifest, manifest_path, directory, _, calls = inventory
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    # The fake Hub has no download or metadata methods, so either would fail.
    assets.main(["--manifest", str(manifest_path), "--output-dir", str(directory), "--report-only"])
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "read_only_inventory"
    assert report["model_inference_verified"] is False
    assert report["asset_manifest_sha256"] == assets.sha256(manifest_path)
    assert {key: value["status"] for key, value in report["assets"].items()} == {
        "valid": "verified", "changed": "checksum_mismatch", "missing": "missing"}
    files = report["base_model"]["files"]
    assert {key: value["local"]["status"] for key, value in files.items()} == {
        "model.safetensors": "verified", "missing.safetensors": "missing",
        "tokenizer.json": "checksum_mismatch", "generation_config.json": "present_without_checksum"}
    assert all(row["access"]["status"] == "not_checked" for row in files.values())
    assert calls == manifest["base_model"]["weight_files"] + manifest["base_model"]["support_files"]
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_optional_access_check_uses_existing_login_metadata_and_redacts_errors(inventory, capsys):
    manifest, manifest_path, directory, hub, _ = inventory
    secret = "hf_secret_must_not_appear"
    requests = []

    class AccessError(Exception):
        response = SimpleNamespace(status_code=403)

    def url(repository, filename, *, revision):
        assert repository == "org/model" and revision == "a" * 40
        return filename

    def metadata(filename, *, token, timeout):
        requests.append(filename)
        assert token is True and timeout == 10
        if filename == "missing.safetensors":
            raise AccessError("Authorization: Bearer " + secret)
        return SimpleNamespace(commit_hash="a" * 40, size=42,
            etag=manifest["base_model"]["file_sha256"].get(filename, "b" * 40),
            location="https://example.invalid/signed?token=" + secret)

    hub.hf_hub_url, hub.get_hf_file_metadata = url, metadata
    assets.main(["--manifest", str(manifest_path), "--output-dir", str(directory),
                 "--report-only", "--check-model-access"])
    output = capsys.readouterr().out
    assert secret not in output and "signed?" not in output and "Authorization" not in output
    report = json.loads(output)
    files = report["base_model"]["files"]
    assert files["model.safetensors"]["access"]["status"] == "accessible"
    assert files["model.safetensors"]["access"]["checksum_matches"] is True
    assert files["missing.safetensors"]["access"] == {
        "status": "unverified", "error_type": "AccessError", "http_status": 403}
    assert files["generation_config.json"]["access"]["checksum_matches"] is None
    assert len(requests) == 4
    assert report["model_inference_verified"] is False


@pytest.mark.parametrize("revision,etag", [("b" * 40, None), ("a" * 40, "c" * 64)])
def test_remote_identity_mismatch_does_not_report_access_as_verified(inventory, revision, etag):
    manifest, _, directory, hub, _ = inventory
    hub.hf_hub_url = lambda repository, filename, **kwargs: filename
    hub.get_hf_file_metadata = lambda filename, **kwargs: SimpleNamespace(
        commit_hash=revision, size=42,
        etag=etag or manifest["base_model"]["file_sha256"].get(filename, "d" * 40))
    report = assets.readiness_report(manifest, directory, check_model_access=True)
    assert report["base_model"]["files"]["model.safetensors"]["access"]["status"] == "metadata_mismatch"


def test_inventory_still_reports_public_files_when_hub_is_missing(inventory, monkeypatch):
    manifest, _, directory, _, _ = inventory
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    report = assets.readiness_report(manifest, directory)
    assert report["assets"]["valid"]["status"] == "verified"
    assert report["base_model"]["status"] == "huggingface_hub_unavailable"
    assert report["model_inference_verified"] is False


@pytest.mark.parametrize("failure", [OSError, RuntimeError])
def test_broken_host_runtime_does_not_hide_files_or_claim_no_gpu(inventory, monkeypatch, failure):
    manifest, _, directory, _, _ = inventory

    def broken_host():
        raise failure("private driver or environment diagnostic")

    monkeypatch.setattr(assets, "host_report", broken_host)
    report = assets.readiness_report(manifest, directory)
    assert report["host_status"] == "unverified"
    assert report["host_error_type"] == failure.__name__
    assert "cuda_available" not in report
    assert "cuda_devices" not in report
    assert report["assets"]["valid"]["status"] == "verified"
    assert report["base_model"]["files"]["model.safetensors"]["local"]["status"] == "verified"
    assert report["model_inference_verified"] is False
    assert "private driver" not in json.dumps(report)


def test_unreadable_cache_entry_does_not_stop_the_remaining_inventory(inventory):
    manifest, _, directory, hub, calls = inventory
    lookup = hub.try_to_load_from_cache

    def failing_lookup(repository, filename, *, revision):
        if filename == "model.safetensors":
            raise PermissionError("private cache location")
        return lookup(repository, filename, revision=revision)

    hub.try_to_load_from_cache = failing_lookup
    report = assets.readiness_report(manifest, directory)
    files = report["base_model"]["files"]
    assert files["model.safetensors"]["local"] == {
        "status": "cache_lookup_error", "error_type": "PermissionError"}
    assert files["tokenizer.json"]["local"]["status"] == "checksum_mismatch"
    assert calls == ["missing.safetensors", "tokenizer.json", "generation_config.json"]
    assert "private cache" not in json.dumps(report)


@pytest.mark.parametrize("conflict", ["--fetch", "--download-model", "--smoke-model", "--source-lock"])
def test_report_rejects_actions_before_reading_manifest(conflict):
    args = ["--manifest", "/missing/manifest.json", "--report-only", conflict]
    if conflict == "--source-lock":
        args.append("/missing/lock.json")
    with pytest.raises(SystemExit) as error:
        assets.main(args)
    assert error.value.code == 2


def test_access_check_requires_report_mode():
    with pytest.raises(SystemExit) as error:
        assets.main(["--manifest", "/missing/manifest.json", "--check-model-access"])
    assert error.value.code == 2


def test_host_inventory_includes_device_memory_and_missing_packages(monkeypatch):
    def version(name):
        if name == "peft":
            raise assets.importlib.metadata.PackageNotFoundError(name)
        return "fixture-version"

    monkeypatch.setattr(assets.importlib.metadata, "version", version)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1,
        get_device_name=lambda index: "Fixture GPU",
        get_device_properties=lambda index: SimpleNamespace(total_memory=80 * 1024 ** 3))))
    report = assets.host_report()
    assert report["packages"]["peft"] is None
    assert report["cuda_devices"] == ["Fixture GPU"]
    assert report["cuda_memory_bytes"] == [80 * 1024 ** 3]
    assert report["model_inference_verified"] is False
