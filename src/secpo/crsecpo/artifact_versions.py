"""Publish related artifacts through one atomic pointer to immutable files."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile


VERSION_SUFFIX = ".crsecpo-versions"


def version_root(anchor):
    return anchor.with_name(anchor.name + VERSION_SUFFIX)


def is_version_view(path):
    if not path.is_symlink():
        return False
    target = Path(os.readlink(path))
    return (len(target.parts) >= 3 and target.parts[-2] == "current"
            and target.parts[-3].endswith(VERSION_SUFFIX))


def version_view_target(anchor, path, name):
    return os.path.relpath(version_root(anchor) / "current" / name, path.parent)


def logical_artifact_path(path):
    """Follow user aliases while keeping our public artifact links as logical names."""
    path = Path(path).absolute()
    seen = set()
    while True:
        path = path.parent.resolve() / path.name
        if not path.is_symlink():
            return path
        if path in seen:
            raise ValueError(f"artifact path contains a symlink cycle: {path}")
        seen.add(path)
        if is_version_view(path):
            return path
        path = path.parent / os.readlink(path)


def committed_version(anchor):
    root = version_root(anchor)
    pointer = root / "current"
    if root.is_symlink():
        raise ValueError(f"artifact version directory cannot be a symlink: {root}")
    try:
        version = os.readlink(pointer)
    except FileNotFoundError:
        return None
    if (not re.fullmatch(r"[0-9a-f]{64}(?:-[0-9a-f]{32})?", version) or (root / version).is_symlink()
            or not (root / version).is_dir()):
        raise ValueError(f"invalid artifact version pointer: {pointer}")
    return root / version


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_link(path, target):
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.symlink_to(target)
        os.replace(staged, path)
    _sync_directory(path.parent)


def _fingerprint(directory, members, *, sync=False, payloads=None):
    fingerprint = hashlib.sha256()
    for name in sorted(members):
        fingerprint.update(name.encode() + b"\0")
        if payloads is not None and name in payloads:
            fingerprint.update(hashlib.sha256(payloads[name]).digest())
            continue
        with (directory / name).open("rb") as handle:
            fingerprint.update(hashlib.file_digest(handle, "sha256").digest())
            if sync:
                os.fsync(handle.fileno())
    return fingerprint.hexdigest()


def _store_version(stage, root, members):
    fingerprint = _fingerprint(stage, members, sync=True)
    version = root / fingerprint
    if version.exists():
        try:
            if _fingerprint(version, members) == fingerprint:
                return version
        except FileNotFoundError:
            pass
        # Keep the old generation available for diagnosis and stage fresh bytes.
        version = root / f"{fingerprint}-{os.urandom(16).hex()}"
    _sync_directory(stage)
    stage.rename(version)
    _sync_directory(root)
    return version


def read_version_file(anchor, name, members):
    """Verify a pinned version using the exact bytes returned to the caller."""
    version = committed_version(anchor)
    if version is None:
        return None
    path = version / name
    try:
        payload = path.read_bytes()
        fingerprint = _fingerprint(version, members, payloads={name: payload})
    except OSError as error:
        raise ValueError(f"cannot verify committed artifact version: {version}") from error
    if fingerprint != version.name.split("-", 1)[0]:
        raise ValueError(f"corrupt committed artifact version: {version}")
    return payload


def publish_version(anchor, views, write_files, *, legacy=None, verify_source=None):
    """Stage a complete generation and atomically switch all public views.

    The caller holds the writer lock. Old generations remain available to
    readers that already resolved the pointer.
    """
    for path, name in views.items():
        if is_version_view(path) and os.readlink(path) != version_view_target(anchor, path, name):
            raise ValueError(f"artifact view belongs to another publication: {path}")
    root = version_root(anchor)
    if root.is_symlink():
        raise ValueError(f"artifact version directory cannot be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    _sync_directory(root.parent)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        stage = Path(temporary)
        write_files(stage)
        version = _store_version(stage, root, views.values())
    if verify_source is not None:
        verify_source()

    # Convert a valid pre-versioned export without changing its visible bytes.
    if committed_version(anchor) is None and legacy is not None and all(path.is_file() for path in views):
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            stage = Path(temporary)
            for path, name in views.items():
                shutil.copyfile(path, stage / name)
            legacy(stage)
            previous = _store_version(stage, root, views.values())
        _replace_link(root / "current", previous.name)

    for path, name in views.items():
        target = version_view_target(anchor, path, name)
        if not path.is_symlink() or os.readlink(path) != target:
            _replace_link(path, target)
    if verify_source is not None:
        verify_source()
    _replace_link(root / "current", version.name)
