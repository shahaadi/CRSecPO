"""Publish and verify the pair and outcome files as one collection result."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path

from secpo.evaluation.sep_adaptive import atomic_write_json
from secpo.evaluation.sep_judge import atomic_write_jsonl
from .artifact_versions import (VERSION_SUFFIX, committed_version, is_version_view,
                                logical_artifact_path, publish_version, version_view_target)
from .prompts import digest
from .records import sha256

EXPORT_SCHEMA = "crsecpo-collection-exports-v1"


def collection_manifest_path(pairs_path):
    return pairs_path.with_name(pairs_path.name + ".collection.json")


def export_record_path(pairs_path):
    return pairs_path.with_name(pairs_path.name + ".exports.json")


def export_reference_path(outcomes_path):
    return outcomes_path.with_name(outcomes_path.name + ".exports-ref.json")


def export_reservation_path(path):
    return path.with_name(path.name + ".exports-reservation.json")


def validate_data_path(path):
    path = Path(path)
    for candidate in (path, logical_artifact_path(path)):
        name = candidate.name.casefold()
        if (name.endswith((".exports.json", ".exports-ref.json",
                                     ".exports-reservation.json", ".collection.json", ".lock"))
                or _is_journal_backup(name)
                or any(part.casefold().endswith(VERSION_SUFFIX) for part in candidate.parts)):
            raise ValueError(f"data path uses a reserved metadata filename: {candidate}")


def _is_journal_backup(name):
    _, marker, digest = name.rpartition(".interrupted-")
    return bool(marker and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest))


def export_lock_path(pairs_path):
    return pairs_path.with_name(pairs_path.name + ".exports.lock")


@contextmanager
def lock_data_paths(*paths):
    with ExitStack() as locks:
        locked = set()
        for lock_path in sorted({export_lock_path(path) for path in paths}):
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = locks.enter_context(lock_path.open("a"))
            stat = os.fstat(lock.fileno())
            inode = (stat.st_dev, stat.st_ino)
            if inode in locked:
                raise ValueError("pair and outcomes paths must use distinct files")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"another writer is using export lock {lock_path}") from error
            locked.add(inode)
        yield


def _reciprocal_reservation(path, reservation):
    counterpart = logical_artifact_path(path.parent / reservation["counterpart"])
    reverse = export_reservation_path(counterpart)
    if not reverse.exists():
        return False
    actual = json.loads(reverse.read_text(encoding="utf-8"))
    required_role = {"pair": "outcomes", "outcomes": "pair"}[reservation["role"]]
    if actual["role"] != required_role:
        return False
    back = export_reservation_path(logical_artifact_path(counterpart.parent / actual["counterpart"]))
    marker = export_reservation_path(path)
    return back.exists() and back.samefile(marker)


def _reservation(path):
    marker = export_reservation_path(path)
    return json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}


def _has_paired_binding(path, reservation):
    return (collection_manifest_path(path).exists() or export_record_path(path).exists()
            or export_reference_path(path).exists()
            or (reservation.get("role") in {"pair", "outcomes"}
                and _reciprocal_reservation(path, reservation)))


def validate_journal_destination(path):
    """Check the journal role while the caller holds the shared path lock."""
    validate_data_path(path)
    current = _reservation(path)
    if current.get("role") == "standalone" or is_version_view(path) or _has_paired_binding(path, current):
        raise ValueError(f"journal path belongs to an export: {path}")


def is_claimed_journal(path):
    return _reservation(path).get("role") == "journal"


def claim_journal_destination(path):
    validate_journal_destination(path)
    if not is_claimed_journal(path):
        atomic_write_json(export_reservation_path(path), {"role": "journal"})


def _reject_journal_export(path, reservation):
    if reservation.get("role") == "journal":
        raise ValueError(f"export path belongs to a journal: {path}")
    # Older journals have no reservation marker. Recognize their first saved
    # request without loading the whole journal on every export.
    if path.is_file():
        with path.open("rb") as handle:
            first = next((line for line in handle if line.strip()), b"")
        try:
            attempt = json.loads(first)
        except (ValueError, UnicodeError):
            return
        if (isinstance(attempt, dict) and isinstance(attempt.get("request"), dict)
                and isinstance(attempt.get("request_id"), str) and attempt.get("status") in ("ok", "error")):
            raise ValueError(f"export path belongs to a journal: {path}")


def validate_run_artifact_destination(path):
    """Protect collection and journal paths from other versioned publishers."""
    validate_data_path(path)
    current = _reservation(path)
    _reject_journal_export(path, current)
    if current.get("role") == "standalone" or _has_paired_binding(path, current):
        raise ValueError(f"run artifact path belongs to a collection: {path}")


def _reserve_collection_exports(pairs_path, outcomes_path, *, manifest_hash=None, publish_manifest=None):
    record_path = export_record_path(pairs_path)
    reference_path = export_reference_path(outcomes_path)
    files = {role: os.path.relpath(path, record_path.parent)
             for role, path in (("pairs", pairs_path), ("outcomes", outcomes_path))}
    reference = {"completion": os.path.relpath(record_path, reference_path.parent)}

    if (export_reference_path(pairs_path).exists() or export_record_path(outcomes_path).exists()
            or collection_manifest_path(outcomes_path).exists()):
        raise ValueError("pair and outcomes paths cannot switch export roles")
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != EXPORT_SCHEMA or record.get("files") != files:
            raise ValueError(f"pair path belongs to another collection: {pairs_path}")
        if "manifest_sha256" in record and record["manifest_sha256"] != manifest_hash:
            raise ValueError(f"collection manifest does not match the export record: {pairs_path}")
    if reference_path.exists() and json.loads(reference_path.read_text(encoding="utf-8")) != reference:
        raise ValueError(f"outcomes path belongs to another collection: {outcomes_path}")

    reservations = (
        ("pair", pairs_path, outcomes_path),
        ("outcomes", outcomes_path, pairs_path),
    )
    pending = []
    for role, path, other in reservations:
        marker = export_reservation_path(path)
        expected = {"role": role, "counterpart": os.path.relpath(other, path.parent)}
        if manifest_hash is not None:
            expected.update(owner="cli", manifest_sha256=manifest_hash)
        current = _reservation(path)
        _reject_journal_export(path, current)
        member = "pairs.jsonl" if role == "pair" else "outcomes.jsonl"
        if is_version_view(path) and os.readlink(path) != version_view_target(pairs_path, path, member):
            raise ValueError(f"export path belongs to another publication: {path}")
        if current:
            if current == expected:
                continue
            if current.get("role") == "standalone":
                raise ValueError(f"paired export would overwrite a standalone export: {path}")
            if _reciprocal_reservation(path, current):
                legacy = (manifest_hash is not None and "manifest_sha256" not in current
                          and current.get("owner", "cli") == "cli"
                          and all(current.get(key) == expected[key] for key in ("role", "counterpart")))
                if not legacy:
                    raise ValueError(f"{role} path belongs to another collection: {path}")
        pending.append((marker, expected))

    # A lone marker is provisional. The reservation commits when both endpoints
    # name each other, before either public sidecar or data file is published.
    for marker, expected in pending:
        atomic_write_json(marker, expected)
    if publish_manifest is not None:
        publish_manifest()
    if not record_path.exists():
        atomic_write_json(record_path, _completion_record(files, None, manifest_hash))
    if not reference_path.exists():
        atomic_write_json(reference_path, reference)
    return files


def _completion_record(files, hashes, manifest_hash):
    record = {"schema_version": EXPORT_SCHEMA, "files": files, "sha256": hashes}
    if manifest_hash is not None:
        record["manifest_sha256"] = manifest_hash
    return record


def _canonical_export_paths(pairs_path, outcomes_path):
    validate_data_path(pairs_path)
    validate_data_path(outcomes_path)
    pairs_path, outcomes_path = map(logical_artifact_path, (pairs_path, outcomes_path))
    if pairs_path == outcomes_path:
        raise ValueError("pair and outcomes paths must use distinct files")
    return pairs_path, outcomes_path


def reserve_collection_exports(pairs_path, outcomes_path, *, prepare_manifest):
    """Check both destinations before letting the CLI publish its manifest."""
    pairs_path, outcomes_path = _canonical_export_paths(pairs_path, outcomes_path)
    with lock_data_paths(pairs_path, outcomes_path):
        manifest, publish_manifest = prepare_manifest()
        _reserve_collection_exports(pairs_path, outcomes_path, manifest_hash=digest(manifest),
                                    publish_manifest=publish_manifest)
        return manifest


def write_standalone_export(rows, path, *, verify_source=None):
    validate_data_path(path)
    path = logical_artifact_path(path)
    with lock_data_paths(path):
        current = _reservation(path)
        _reject_journal_export(path, current)
        if is_version_view(path) or _has_paired_binding(path, current):
            raise ValueError(f"standalone export would overwrite a paired collection: {path}")
        if current.get("role") != "standalone":
            atomic_write_json(export_reservation_path(path), {"role": "standalone"})
        atomic_write_jsonl(rows, path, before_replace=verify_source)


def write_collection_exports(pairs, outcomes, pairs_path, outcomes_path, *, manifest=None, verify_source=None):
    pairs_path, outcomes_path = _canonical_export_paths(pairs_path, outcomes_path)
    record_path = export_record_path(pairs_path)
    with lock_data_paths(pairs_path, outcomes_path):
        manifest_file = collection_manifest_path(pairs_path)
        if manifest is None and manifest_file.exists():
            raise ValueError(f"pair path belongs to a CLI collection: {pairs_path}")
        if manifest is not None and (not manifest_file.exists()
                or json.loads(manifest_file.read_text(encoding="utf-8")) != manifest):
            raise ValueError(f"collection manifest changed before publishing: {manifest_file}")
        manifest_hash = digest(manifest) if manifest is not None else None
        files = _reserve_collection_exports(pairs_path, outcomes_path, manifest_hash=manifest_hash)
        views = {pairs_path: "pairs.jsonl", outcomes_path: "outcomes.jsonl", record_path: "record.json"}

        def write_files(stage):
            atomic_write_jsonl(pairs, stage / "pairs.jsonl")
            atomic_write_jsonl(outcomes, stage / "outcomes.jsonl")
            hashes = {role: sha256(stage / f"{role}.jsonl") for role in ("pairs", "outcomes")}
            atomic_write_json(stage / "record.json", _completion_record(files, hashes, manifest_hash))

        def verify_legacy(stage):
            legacy = json.loads((stage / "record.json").read_text(encoding="utf-8"))
            hashes = {role: sha256(stage / f"{role}.jsonl") for role in ("pairs", "outcomes")}
            if (legacy.get("schema_version") != EXPORT_SCHEMA or legacy.get("files") != files
                    or legacy.get("sha256") != hashes
                    or legacy.get("manifest_sha256", manifest_hash) != manifest_hash):
                raise ValueError("legacy collection changed or failed verification during migration")

        def check():
            if verify_source is not None:
                verify_source()
            if manifest is not None and json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
                raise ValueError(f"collection manifest changed before publishing: {manifest_file}")

        publish_version(pairs_path, views, write_files, legacy=verify_legacy, verify_source=check)


def read_collection_export(path, role):
    """Return bytes verified against the shared record, or read a standalone input."""
    path = logical_artifact_path(path)
    # Writers claim paths before replacing their data. Open the payload first:
    # bytes from a managed publish can never pass a prior standalone check.
    payload, missing = None, None
    version_view = is_version_view(path)
    if not version_view:
        try:
            payload = path.read_bytes()
        except OSError as error:
            missing = error
    record_path = export_record_path(path)
    reference_path = export_reference_path(path)
    if not record_path.exists() and not reference_path.exists():
        reservation = _reservation(path)
        if reservation.get("role") == "journal":
            raise ValueError(f"Path belongs to a journal: {path}")
        if version_view or reservation.get("role") in {"pair", "outcomes"} or collection_manifest_path(path).exists():
            raise ValueError(f"Collection export at {path} is incomplete. Resume collection.")
        if missing is not None:
            raise missing
        return payload

    try:
        if role == "pairs" and not record_path.exists():
            raise ValueError("path is reserved for outcomes")
        if role == "outcomes" and not reference_path.exists():
            raise ValueError("path is reserved for pairs")
        if role == "outcomes":
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            record_path = reference_path.parent / reference["completion"]
        record_path = record_path.parent.resolve() / record_path.name
        suffix = ".exports.json"
        if not record_path.name.endswith(suffix):
            raise ValueError("invalid collection export record path")
        anchor = record_path.with_name(record_path.name.removesuffix(suffix))
        version = committed_version(anchor)
        record_file = version / "record.json" if version is not None else record_path
        record = json.loads(record_file.read_text(encoding="utf-8"))
        files = {name: logical_artifact_path(record_path.parent / record["files"][name])
                 for name in ("pairs", "outcomes")}
        if (record["schema_version"] != EXPORT_SCHEMA or
                record_path != export_record_path(files["pairs"]) or files[role] != path):
            raise ValueError("invalid collection export binding")
        counterpart = export_reference_path(files["outcomes"])
        expected = os.path.relpath(record_path, counterpart.parent)
        if json.loads(counterpart.read_text(encoding="utf-8")).get("completion") != expected:
            raise ValueError("invalid outcomes export reference")
        if "manifest_sha256" in record:
            manifest_file = collection_manifest_path(files["pairs"])
            if digest(json.loads(manifest_file.read_text(encoding="utf-8"))) != record["manifest_sha256"]:
                raise ValueError("collection manifest does not match the export record")
        hashes = record["sha256"]
        if hashes is None:
            raise ValueError("collection exports have not completed")
        other_role = "outcomes" if role == "pairs" else "pairs"
        if version is not None:
            payload = (version / f"{role}.jsonl").read_bytes()
            other_path = version / f"{other_role}.jsonl"
        else:
            if missing is not None:
                raise missing
            if payload is None:
                raise ValueError("collection export has no committed generation")
            other_path = files[other_role]
        if (hashlib.sha256(payload).hexdigest() != hashes[role] or
                sha256(other_path) != hashes[other_role]):
            raise ValueError("collection exports do not match their completion record")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot use collection {role} at {path}: {error}. Resume collection.") from error
    return payload
