"""``<artifact>.manifest.json`` sidecar: describes the release without row values,
local paths or key material. ``plaintext_sha256`` is the upload idempotency key.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .keys import fingerprint

#: Bump whenever the payload shape changes.
MANIFEST_SCHEMA = 1

_CHUNK = 8 << 20


def manifest_path(artifact_path) -> Path:
    """Sidecar path: ``x.enc.bin`` -> ``x.enc.bin.manifest.json`` (appends; ``with_suffix`` would eat ``.bin``)."""
    return Path(str(artifact_path) + ".manifest.json")


def sha256_file(path, chunk: int = _CHUNK) -> str:
    """Streaming SHA-256 of a file, hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_provenance(repo_root=None) -> dict:
    """``{commit, dirty}`` for the pipeline checkout; ``"unknown"`` without git."""
    root = str(repo_root or Path(__file__).resolve().parent.parent)
    def _git(*args):
        return subprocess.run(["git", "-C", root, *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout
    try:
        commit = _git("rev-parse", "--short", "HEAD").strip()
        dirty = bool(_git("status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _first_value(series: pd.Series):
    """First non-null value, or None."""
    hits = series.dropna()
    return hits.iloc[0] if len(hits) else None


def _is_vector_column(series: pd.Series) -> bool:
    """True for a list/array-valued column (the embedding columns)."""
    value = _first_value(series)
    return hasattr(value, "__len__") and not isinstance(value, (str, bytes))


def _embedding_dim(series: pd.Series):
    """Length of the first non-null vector in a list-valued column."""
    value = _first_value(series)
    try:
        return int(len(value))
    except TypeError:
        return None


def _counts(series: pd.Series) -> dict:
    """``value -> count`` as plain ints, for a low-cardinality label column."""
    return {str(k): int(v) for k, v in series.value_counts().items()}


def build_manifest(df: pd.DataFrame, cfg: dict, artifact, key: bytes, *,
                   token_bytes: int, token_sha256: str,
                   plaintext_bytes: int, plaintext_sha256: str,
                   source_artifact=None) -> dict:
    """Describe a released frame. Carries no row values and no key material."""
    model_cols = [c for c in df.columns if _is_vector_column(df[c])]
    frame = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "instrument_rows": int(df["is_instrument"].sum()),
        "scale_rows": int(len(df) - df["is_instrument"].sum()),
        "column_types": {c: str(df[c].dtype) for c in df.columns},
        "embedding_dims": {c: _embedding_dim(df[c]) for c in model_cols},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns
                        if int(df[c].isna().sum())},
    }
    if "corpus_source" in df.columns:
        frame["corpus_source_counts"] = _counts(df["corpus_source"])
    if "meta_language" in df.columns:
        frame["language_counts"] = _counts(df["meta_language"])
    if "public_year" in df.columns:
        years = pd.to_numeric(df["public_year"], errors="coerce").dropna()
        frame["public_year"] = {
            "min": int(years.min()) if len(years) else None,
            "max": int(years.max()) if len(years) else None,
            "null": int(len(df) - len(years)),
        }
    frame["flag_true_counts"] = {
        c: int(df[c].fillna(False).astype(bool).sum())
        for c in df.columns if c.startswith("flag_")
    }

    encode_cfg = cfg.get("encode", {}) or {}
    models = []
    for role in ("item_models", "scale_models"):
        for entry in encode_cfg.get(role) or []:
            # `name` only — `path` is a local model directory and stays local.
            name = str(entry.get("name", ""))
            models.append({"role": role.removesuffix("_models"), "name": name})

    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": {"repo": "synth-net-pipline", **git_provenance()},
        "artifact": {
            "name": Path(artifact).name,
            "bytes": int(token_bytes),
            "sha256": token_sha256,
            "encryption": ("fernet-v1 (AES-128-CBC + HMAC-SHA256, random IV "
                           "per encrypt — the ciphertext differs every run)"),
            "key_fingerprint": f"sha256:{fingerprint(key)}",
            "plaintext_format": "parquet",
            "plaintext_bytes": int(plaintext_bytes),
            "plaintext_sha256": plaintext_sha256,
        },
        "source_artifact": Path(source_artifact).name if source_artifact else None,
        "frame": frame,
        "models": models,
    }


def write_manifest(artifact, payload: dict) -> Path:
    """Write the sidecar atomically next to ``artifact``; returns its path."""
    path = manifest_path(artifact)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read_manifest(artifact) -> Optional[dict]:
    """Read an artifact's sidecar; None when missing or unparseable."""
    path = manifest_path(artifact)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def manifest_stale(artifact) -> bool:
    """True when the artifact is newer than its sidecar, or either is missing."""
    art, side = Path(artifact), manifest_path(artifact)
    if not art.exists() or not side.exists():
        return True
    return art.stat().st_mtime > side.stat().st_mtime
