"""``<artifact>.stats.json`` sidecars: per-step counters + report lines of the run that wrote the artifact.

Overwritten on every real run (history lives in ``logs/assemble-*.log``); read back by ``assemble.report``.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def stats_path(artifact_path) -> Path:
    """Sidecar path for an artifact: ``x.parquet`` -> ``x.stats.json``."""
    return Path(artifact_path).with_suffix(".stats.json")


def write_stats(artifact_path, stage: str, stats: dict,
                report_lines: Optional[list] = None) -> Path:
    """Write the sidecar next to ``artifact_path``; returns its path."""
    path = stats_path(artifact_path)
    payload = {
        "stage": stage,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "artifact": str(artifact_path),
        "stats": stats,
        "report": list(report_lines or []),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
    tmp.replace(path)
    return path


def read_stats(artifact_path) -> Optional[dict]:
    """Read an artifact's sidecar; None when missing or unparseable."""
    path = stats_path(artifact_path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def stats_stale(artifact_path) -> bool:
    """True when the sidecar is missing or older than its artifact."""
    artifact = Path(artifact_path)
    sidecar = stats_path(artifact_path)
    if not artifact.exists() or not sidecar.exists():
        return True
    return artifact.stat().st_mtime > sidecar.stat().st_mtime
