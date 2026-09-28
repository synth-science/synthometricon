"""Release directory layout: everything is derived from ``data.publish``.

See docs/architecture-publish.md for the file list.
"""
from __future__ import annotations

from pathlib import Path


def release_path(cfg: dict):
    """``data.publish`` — the encrypted release artifact, or None when unset."""
    return (cfg.get("data", {}) or {}).get("publish")


def key_path(artifact) -> Path:
    """The Fernet key beside the artifact: ``x.parquet`` -> ``x.key``."""
    return Path(artifact).with_suffix(".key")


def report_path(artifact) -> Path:
    """The descriptives report beside the artifact: ``x.parquet`` -> ``x.md``."""
    return Path(artifact).with_suffix(".md")


def readme_path(artifact) -> Path:
    """The release directory's ``README.md``."""
    return Path(artifact).parent / "README.md"
