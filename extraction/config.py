"""Configuration loading and per-extractor option resolution."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import yaml


_GLOB_CHARS = set("*?[")

# OS copy of a file, e.g. "doc (2).pdf" next to "doc.pdf" (Finder/Explorer duplicates).
_OS_COPY = re.compile(r"^(?P<base>.+) \((?P<n>\d+)\)(?P<ext>\.pdf)$", re.IGNORECASE)

DEFAULT_FIXTURES_DIR = "tests/validation-fixtures"


def _drop_os_copies(paths: list[str]) -> list[str]:
    """Drop ``<name> (N).pdf`` when ``<name>.pdf`` (any extension case) is in the list; log what was dropped."""
    present = {p.lower() for p in paths}
    kept, dropped = [], []
    for p in paths:
        m = _OS_COPY.match(os.path.basename(p))
        original = (os.path.join(os.path.dirname(p), m["base"] + m["ext"]).lower() if m else None)
        (dropped if original in present else kept).append(p)
    if dropped:
        print(f"input_files: skipped {len(dropped)} OS duplicate cop{'y' if len(dropped) == 1 else 'ies'} "
              f"of PDFs already listed: {', '.join(os.path.basename(p) for p in dropped)}")
    return kept


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
    return _drop_os_copies(out)


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


def resolve_fixtures_dir(config: dict, root: str | Path | None = None) -> str:
    """``testing.fixtures_dir`` (default: the human-coded ``tests/validation-fixtures``), resolved against ``root``."""
    fixtures_dir = Path((config.get("testing") or {}).get("fixtures_dir") or DEFAULT_FIXTURES_DIR)
    if root is not None and not fixtures_dir.is_absolute():
        fixtures_dir = Path(root) / fixtures_dir
    return str(fixtures_dir)


def load_test_cases(
    path: str = "tests/test_files.yaml",
    fixtures_dir: str = DEFAULT_FIXTURES_DIR,
) -> list[dict]:
    """Load enabled test cases as ``{"path", "enabled", "fixtures"}`` dicts.

    Fixtures come from ``<fixtures_dir>/<pdf-stem>.yaml`` (validity tags preserved);
    a missing file yields ``{}``. Pass ``resolve_fixtures_dir(config)`` so pytest and the
    CLI read the same (human-coded) fixtures.
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
