"""Pinned asset download and GPU-host checks; no compute is rented here."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from .records import sha256


def project_source_lock(path):
    """Portable download pins derived from the existing research input lock."""
    source = json.loads(Path(path).read_text())
    model = source["base_model"]
    assets = [{"name": "wildjailbreak_" + Path(row["path"]).stem,
               "url": row["url"], "sha256": row["sha256"],
               "filename": Path(row["path"]).name}
              for row in source["wildjailbreak_evaluation"]["files"]]
    for name, row in source["utility_evaluation"].items():
        assets.append({"name": name, "url": row["repository"].replace(
            "https://github.com/", "https://raw.githubusercontent.com/")
            + "/" + row["revision"] + "/" + row["path"],
            "sha256": row["sha256"], "filename": name + ".jsonl"})
    hashes = {row["path"]: row["etag"] for row in model["access"]["file_probes"]
              if row["path"] in model["weight_files"]}
    hashes.update({row["path"]: row["sha256"]
                   for row in model["downloaded_support_files"]})
    return {"schema_version": 1, "source_lock": {
        "path": "fall-2026-research/03-benign-framing-llm-safety/experiment/configs/crsecpo_inputs.lock.json",
        "sha256": sha256(path)}, "secpo_revision": source["secpo"]["revision"],
        "assets": assets, "base_model": {
            **{key: model[key] for key in ("repository", "revision", "weight_files", "support_files")},
            "file_sha256": hashes}}


def verify_asset(path, expected):
    if not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f"Missing or mismatched pinned asset: {path}")


def fetch_asset(asset, directory):
    destination = Path(directory) / asset["filename"]
    if destination.exists():
        verify_asset(destination, asset["sha256"])
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        downloaded = Path(temporary) / "download"
        with urllib.request.urlopen(asset["url"], timeout=60) as response, downloaded.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        verify_asset(downloaded, asset["sha256"])
        os.link(downloaded, destination)
    return destination


def host_report():
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"packages": packages, "model_inference_verified": False}
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        result["cuda_memory_bytes"] = [torch.cuda.get_device_properties(i).total_memory
                                       for i in range(torch.cuda.device_count())]
    except ImportError:
        result.update(cuda_available=False, cuda_devices=[], cuda_memory_bytes=[])
    return result


def _local_file_report(path, expected=None):
    if path is None:
        return {"status": "missing"}
    path = Path(path)
    result = {"path": str(path.absolute())}
    try:
        if not path.is_file():
            return {**result, "status": "missing"}
        if expected is None:
            return {**result, "status": "present_without_checksum"}
        verify_asset(path, expected)
        return {**result, "status": "verified"}
    except ValueError:
        return {**result, "status": "checksum_mismatch"}
    except OSError as error:
        return {**result, "status": "unreadable", "error_type": type(error).__name__}


def readiness_report(manifest, directory, *, check_model_access=False):
    """Inspect every local input without downloading files or loading a model.

    Optional Hub HEAD requests use the existing login. Report only allowlisted
    metadata: exception messages and redirected URLs may contain credentials.
    This is an inventory, not evidence of inference, throughput or scientific readiness.
    """
    try:
        result = host_report()
    except (OSError, RuntimeError) as error:
        result = {"model_inference_verified": False, "host_status": "unverified",
                  "host_error_type": type(error).__name__}
    result["mode"] = "read_only_inventory"
    result["assets"] = {asset["name"]: _local_file_report(
        Path(directory) / asset["filename"], asset["sha256"]) for asset in manifest["assets"]}
    spec = manifest["base_model"]
    model = {"repository": spec["repository"], "revision": spec["revision"], "files": {}}
    result["base_model"] = model
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        model["status"] = "huggingface_hub_unavailable"
        return result
    if check_model_access:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
    for filename in dict.fromkeys(spec["weight_files"] + spec["support_files"]):
        expected = spec["file_sha256"].get(filename)
        try:
            cached = try_to_load_from_cache(spec["repository"], filename, revision=spec["revision"])
            entry = {"local": _local_file_report(cached if isinstance(cached, str) else None, expected)}
        except OSError as error:
            entry = {"local": {"status": "cache_lookup_error", "error_type": type(error).__name__}}
        if check_model_access:
            try:
                metadata = get_hf_file_metadata(hf_hub_url(spec["repository"], filename,
                    revision=spec["revision"]), token=True, timeout=10)
                revision_matches = metadata.commit_hash == spec["revision"]
                # LFS etags are content SHA256; ordinary Git etags are not.
                checksum_matches = (metadata.etag == expected if expected and
                    isinstance(metadata.etag, str) and len(metadata.etag) == 64 else None)
                entry["access"] = {"status": "accessible" if revision_matches and
                    checksum_matches is not False else "metadata_mismatch",
                    "revision_matches": revision_matches, "checksum_matches": checksum_matches,
                    "size_bytes": metadata.size}
            except Exception as error:
                entry["access"] = {"status": "unverified", "error_type": type(error).__name__}
                status = getattr(getattr(error, "response", None), "status_code", None)
                if type(status) is int:
                    entry["access"]["http_status"] = status
        else:
            entry["access"] = {"status": "not_checked"}
        model["files"][filename] = entry
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/crsecpo_inputs.json"))
    parser.add_argument("--source-lock", type=Path,
                        help="Verify the generated manifest against the research input lock and exit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crsecpo/assets"))
    parser.add_argument("--fetch", action="store_true", help="Download pinned public evaluation files")
    parser.add_argument("--download-model", action="store_true", help="Download official weights with existing HF login")
    parser.add_argument("--smoke-model", action="store_true", help="Load local pinned model and generate one harmless reply")
    parser.add_argument("--report-only", action="store_true",
                        help="Report all local inputs and host details; do not download or load models")
    parser.add_argument("--check-model-access", action="store_true",
                        help="With --report-only, make authenticated metadata-only requests using the existing HF login")
    args = parser.parse_args(argv)
    if args.report_only and (args.fetch or args.download_model or args.smoke_model or args.source_lock):
        parser.error("--report-only cannot be combined with fetch, download, smoke, or source-lock actions")
    if args.check_model_access and not args.report_only:
        parser.error("--check-model-access requires --report-only")
    manifest = json.loads(args.manifest.read_text())
    if args.report_only:
        report = readiness_report(manifest, args.output_dir, check_model_access=args.check_model_access)
        report["asset_manifest_sha256"] = sha256(args.manifest)
        print(json.dumps(report, indent=2))
        return
    if args.source_lock:
        if manifest != project_source_lock(args.source_lock):
            raise ValueError("Asset manifest differs from its authoritative source lock")
        print(json.dumps({"source_lock_verified": True, "sha256": sha256(args.source_lock)}))
        return
    result = host_report()
    result["asset_manifest_sha256"] = sha256(args.manifest)
    result["assets"] = {}
    for asset in manifest["assets"]:
        path = args.output_dir / asset["filename"]
        if args.fetch:
            fetch_asset(asset, args.output_dir)
        verify_asset(path, asset["sha256"])
        result["assets"][asset["name"]] = str(path.resolve())
    if args.download_model or args.smoke_model:
        from huggingface_hub import snapshot_download
        spec = manifest["base_model"]
        model_path = Path(snapshot_download(repo_id=spec["repository"], revision=spec["revision"],
            allow_patterns=spec["weight_files"] + spec["support_files"],
            local_files_only=not args.download_model))
        for filename, digest in spec["file_sha256"].items():
            verify_asset(model_path / filename, digest)
        result["model_snapshot"] = str(model_path)
        if args.smoke_model:
            from .inference import GenerationConfig, HFBackend, ModelSpec, make_request
            identity = ModelSpec(str(model_path), spec["revision"], str(model_path), spec["revision"])
            cuda = result["cuda_available"]
            backend = HFBackend(identity, device="cuda" if cuda else "cpu",
                                dtype="bfloat16" if cuda else "float32")
            request = make_request([{"role": "user", "content": "Reply with the word ready."}],
                tokenizer=backend.tokenizer, model_spec=identity,
                config=GenerationConfig(max_new_tokens=16, temperature=0, top_p=1),
                seed=42, kind="host_smoke")
            response = backend(request)
            result["model_inference_verified"] = (response["finish_reason"] == "stop"
                                                   and response["output_tokens"] > 0)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
