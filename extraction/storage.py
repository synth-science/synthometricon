"""Parquet-backed extraction store: one row per PDF, JSON result + usage columns per extractor.

Schema: docs/architecture-extraction.md#storage.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


PATH_COL = "path"

# Per-extractor usage columns (``{name}_extractor_{suffix}``); shared with :mod:`orchestrator`.
USAGE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("timestamp", "string"),
    ("error", "string"),
    ("total_duration", "Float64"),
    ("prompt_eval_count", "Int64"),
    ("eval_count", "Int64"),
    ("image_tokens", "Int64"),
    ("system_tokens", "Int64"),
    ("user_text_tokens", "Int64"),
    ("ttft", "Float64"),
    # Server-side timings (llama-server) — decompose TTFT into prefill vs reuse.
    ("prefill_ms", "Float64"),          # prompt-processing / prefill wall-time
    ("predicted_ms", "Float64"),        # generation wall-time
    ("prompt_per_second", "Float64"),   # prefill throughput
    ("predicted_per_second", "Float64"),  # generation throughput
    ("cached_tokens", "Int64"),         # prompt tokens served from slot KV cache
    ("draft_acceptance", "Float64"),    # speculative-decoding accept rate (or null)
    ("thinking_chars", "Int64"),        # reasoning chars streamed (0 when off)
)

_DOC_SUMMARY_COLS: tuple[tuple[str, str], ...] = (
    ("has_errors", "boolean"),
    ("total_duration", "Float64"),
)

# Kept separate from USAGE_SUFFIXES: usage cells are nulled on failure, attempts must survive.
RETRY_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("attempts", "Int64"),
)


def _ensure_column(df: pd.DataFrame, name: str, dtype: str) -> None:
    if name in df.columns:
        df[name] = df[name].astype(dtype)
    else:
        df[name] = pd.Series([pd.NA] * len(df), dtype=dtype)


def load_or_init_df(path: Path, extractor_names: list[str]) -> pd.DataFrame:
    """Load the parquet at ``path`` (or an empty frame), adding any missing typed columns."""
    if path.exists():
        df = pd.read_parquet(path)
        if PATH_COL not in df.columns:
            df.insert(0, PATH_COL, pd.Series([pd.NA] * len(df), dtype="string"))
        else:
            df[PATH_COL] = df[PATH_COL].astype("string")
    else:
        df = pd.DataFrame({PATH_COL: pd.Series([], dtype="string")})

    for col, dtype in _DOC_SUMMARY_COLS:
        _ensure_column(df, col, dtype)

    for name in extractor_names:
        _ensure_column(df, f"{name}_extractor_content", "string")
        for suffix, dtype in USAGE_SUFFIXES:
            _ensure_column(df, f"{name}_extractor_{suffix}", dtype)
        for suffix, dtype in RETRY_SUFFIXES:
            _ensure_column(df, f"{name}_extractor_{suffix}", dtype)
    return df


def read_paths(path: Path) -> list[str]:
    """Return the document paths already recorded in the parquet (``[]`` if absent)."""
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=[PATH_COL])
    return [str(p) for p in df[PATH_COL].tolist() if not pd.isna(p)]


def upsert_row(
    df: pd.DataFrame, pdf_path: str, updates: dict[str, object]
) -> pd.DataFrame:
    """Apply ``updates`` to the row keyed by ``pdf_path``, creating it if absent."""
    for col in updates:
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype="string")

    mask = df[PATH_COL] == pdf_path
    if mask.any():
        idx = df.index[mask][0]
        for col, val in updates.items():
            df.at[idx, col] = pd.NA if val is None else val
        return df

    new_row = {PATH_COL: pdf_path}
    for col in df.columns:
        if col == PATH_COL:
            continue
        val = updates.get(col)
        new_row[col] = pd.NA if val is None else val
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` via a temp file + ``os.replace`` so a partial write never corrupts it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def get_cell(df: pd.DataFrame, pdf_path: str, column: str) -> Optional[str]:
    """Return the JSON string for ``(pdf_path, column)``, or None if missing/null."""
    if column not in df.columns:
        return None
    mask = df[PATH_COL] == pdf_path
    if not mask.any():
        return None
    val = df.loc[mask, column].iloc[0]
    return None if pd.isna(val) else str(val)


def load_cell(
    df: pd.DataFrame,
    pdf_path: str,
    column: str,
    model_cls: type[BaseModel],
) -> Optional[BaseModel]:
    """Parse the JSON cell at ``(pdf_path, column)`` into ``model_cls``, or ``None``."""
    raw = get_cell(df, pdf_path, column)
    return None if raw is None else model_cls.model_validate_json(raw)


def cell_state(df: pd.DataFrame, pdf_path: str, name: str) -> str:
    """Classify a cell as ``"done"`` (content), ``"failed"`` (error only) or ``"pending"``."""
    if get_cell(df, pdf_path, f"{name}_extractor_content") is not None:
        return "done"
    if get_cell(df, pdf_path, f"{name}_extractor_error") is not None:
        return "failed"
    return "pending"


def get_attempts(df: pd.DataFrame, pdf_path: str, name: str) -> int:
    """Return how many times ``name`` has been attempted on ``pdf_path`` (0 if never)."""
    raw = get_cell(df, pdf_path, f"{name}_extractor_attempts")
    return 0 if raw is None else int(float(raw))
