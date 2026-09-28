"""Configuration loading and per-extractor option resolution."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import yaml


_GLOB_CHARS = set("*?[")


def expand_input_files(entries: list[str]) -> list[str]:
    """Expand glob / directory / file entries into deduped absolute PDF paths.

    Uses ``os.path.abspath`` (no symlink resolution) so existing parquet keys stay stable.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)

    for entry in entries:
        if any(c in entry for c in _GLOB_CHARS):
            matches = sorted(glob.glob(entry, recursive=True))
            for m in matches:
                if m.lower().endswith(".pdf"):
                    _add(m)
        elif os.path.isdir(entry):
            for m in sorted(glob.glob(os.path.join(entry, "**", "*.pdf"), recursive=True)):
                _add(m)
        else:
            _add(entry)
    return out


def load_config(path: str = "config.yaml") -> dict:
    """Load the project config; normalises legacy ``input_file`` and expands ``input_files``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "input_file" in cfg and "input_files" not in cfg:
        cfg["input_files"] = [cfg.pop("input_file")]
    elif "input_files" not in cfg:
        cfg["input_files"] = []
    cfg["input_files"] = expand_input_files(cfg["input_files"])
    return cfg


def load_test_cases(
    path: str = "tests/test_files.yaml",
    fixtures_dir: str = "tests/fixtures",
) -> list[dict]:
    """Load enabled test cases as ``{"path", "enabled", "fixtures"}`` dicts.

    Fixtures come from ``<fixtures_dir>/<pdf-stem>.yaml`` (validity tags preserved);
    a missing file yields ``{}``.
    """
    with open(path) as f:
        items = yaml.safe_load(f) or []
    cases = [{"path": i} if isinstance(i, str) else i for i in items]
    cases = [c for c in cases if c.get("enabled", True)]

    # Lazy import to avoid a config ↔ validity circular dependency.
    from .validity import load_full_fixture_with_tags

    fixtures_root = Path(fixtures_dir)
    for case in cases:
        stem = Path(case["path"]).stem
        case["fixtures"] = load_full_fixture_with_tags(fixtures_root / f"{stem}.yaml")
    return cases


def write_fixture(fixtures_dir: str, stem: str, fixtures: dict) -> None:
    """Overwrite ``<fixtures_dir>/<stem>.yaml`` with the given top-level keys."""
    fixtures_root = Path(fixtures_dir)
    fixtures_root.mkdir(parents=True, exist_ok=True)
    target = fixtures_root / f"{stem}.yaml"
    with open(target, "w") as f:
        yaml.safe_dump(
            fixtures,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=10**6,
        )


def build_options(raw: dict) -> dict:
    """Strip null values from an options dict."""
    return {k: v for k, v in raw.items() if v is not None}


def resolve_options(config: dict, extractor_name: str) -> dict:
    """Merge global ``options`` keys with the ``options[extractor_name]`` override (nulls kept)."""
    raw = config.get("options") or {}
    base = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    override = raw.get(extractor_name)
    if isinstance(override, dict):
        base.update(override)
    return base


def resolve_think(config: dict, extractor_name: str) -> bool:
    """Resolve ``config["think"]`` (bool or per-extractor dict with ``default``) to a bool."""
    raw = config.get("think", False)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, dict):
        if extractor_name in raw:
            return bool(raw[extractor_name])
        return bool(raw.get("default", False))
    return False
