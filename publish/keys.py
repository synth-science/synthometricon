"""Fernet key resolution for the release artifact; the only module that touches the key.

Order: ``$publish.key_env`` -> key file -> generate (only when ``allow_create``).
Never print the key, only :func:`fingerprint`. See docs/architecture-publish.md.
"""
from __future__ import annotations

import hashlib
import os
import stat as stat_mod
import sys
from pathlib import Path

from cryptography.fernet import Fernet

from . import paths

DEFAULT_KEY_ENV = "SYNTH_NET_RELEASE_KEY"


def fingerprint(key: bytes) -> str:
    """``sha256(key)[:12]`` — the only key property allowed in reports, logs, manifest or card."""
    return hashlib.sha256(key).hexdigest()[:12]


def validate_key(key: bytes, origin: str) -> bytes:
    """Return ``key`` if Fernet accepts it, else exit naming ``origin`` (never the key)."""
    try:
        Fernet(key)
    except (ValueError, TypeError) as err:
        sys.exit(f"{origin} does not hold a valid Fernet key "
                 f"(44-char url-safe base64 of 32 bytes): {err}")
    return key


def read_key_file(path: Path, report: list[str]) -> bytes:
    """Read and validate the key at ``path``; warn if others can read it."""
    try:
        raw = path.read_bytes().strip()
    except OSError as err:
        sys.exit(f"cannot read key file {path}: {err}")
    if not raw:
        sys.exit(f"key file is empty: {path}")
    mode = path.stat().st_mode
    if mode & (stat_mod.S_IRWXG | stat_mod.S_IRWXO):
        report.append(f"WARNING: {path} is group/world-accessible "
                      f"(mode {mode & 0o777:04o}) — chmod 600 it")
    return validate_key(raw, str(path))


def write_key_file(path: Path, key: bytes) -> None:
    """Write ``key`` with mode 0600; ``O_EXCL`` never clobbers an existing key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        sys.exit(f"key file appeared concurrently at {path}; re-run")
    except OSError as err:
        sys.exit(f"cannot create key file {path}: {err}")
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)


def key_env_name(cfg: dict) -> str:
    """Environment variable this config reads the key from."""
    return (cfg.get("publish", {}) or {}).get("key_env") or DEFAULT_KEY_ENV


def configured_key_path(cfg: dict):
    """The key file beside ``data.publish``, or None when unset."""
    artifact = paths.release_path(cfg)
    return paths.key_path(artifact) if artifact else None


def resolve_key(cfg: dict, report: list[str], *, allow_create: bool,
                key_path=None) -> tuple[bytes, str]:
    """Resolve the release key; returns ``(key, source_label)``.

    ``allow_create`` must be True only for a real ``publish`` run."""
    env_name = key_env_name(cfg)
    file_path = key_path or configured_key_path(cfg)
    file_path = Path(file_path) if file_path else None

    env_val = (os.environ.get(env_name) or "").strip()
    if env_val:
        key = validate_key(env_val.encode(), f"${env_name}")
        if file_path is not None and file_path.exists():
            disk = read_key_file(file_path, report)
            if disk != key:
                sys.exit(
                    f"${env_name} (fingerprint {fingerprint(key)}) and "
                    f"{file_path} (fingerprint {fingerprint(disk)}) hold "
                    f"different keys — refusing to guess which one consumers "
                    f"have. Unset one, or pass --key-path explicitly.")
        report.append(f"key source: ${env_name} "
                      f"(fingerprint {fingerprint(key)})")
        return key, f"env:{env_name}"

    if file_path is not None and file_path.exists():
        key = read_key_file(file_path, report)
        report.append(f"key source: {file_path} "
                      f"(fingerprint {fingerprint(key)})")
        return key, f"file:{file_path}"

    if not allow_create:
        sys.exit(f"no release key found: set ${env_name} or place one at "
                 f"{file_path or '<data.publish unset>'}. A real "
                 f"`python -m publish --step publish` run generates one.")

    if file_path is None:
        sys.exit("data.publish must be set before a key can be generated")

    key = Fernet.generate_key()
    write_key_file(file_path, key)
    report.append(
        f"generated new key: {file_path} (mode 0600, fingerprint "
        f"{fingerprint(key)}) — do not commit it, back it up, and note that "
        f"consumers of any previous release now need this file")
    return key, f"new:{file_path}"


def describe_key(cfg: dict, report: list[str], key_path=None) -> str:
    """One report line about the key for dry runs — never generates one."""
    env_name = key_env_name(cfg)
    file_path = key_path or configured_key_path(cfg)
    if (os.environ.get(env_name) or "").strip():
        key = validate_key(os.environ[env_name].strip().encode(), f"${env_name}")
        return f"key: ${env_name} is set (fingerprint {fingerprint(key)})"
    if file_path and Path(file_path).exists():
        key = read_key_file(Path(file_path), report)
        return f"key: {file_path} (exists, fingerprint {fingerprint(key)})"
    return (f"key: not found at {file_path or '<data.publish unset>'} and "
            f"${env_name} unset — a real run would generate one")
