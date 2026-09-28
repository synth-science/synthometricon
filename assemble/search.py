"""Embedding-search sanity checks against the pooled corpus (import-only, not a stage).

Usage:
    from assemble.search import search_items, search_scales
    search_items(["I am the life of the party.", "I keep in the background."],
                 reverse=[False, True])
    search_scales("Extraversion", top_k=5)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .encode import DEFAULT_BATCH_SIZE, _model_entries
from .pool import _is_missing, keyed_centroid

RESULT_COLS = ("path", "corpus_source", "public_doi", "doi_psyctests",
               "meta_title_raw", "is_instrument", "scale_id", "scale_name",
               "scale_depth", "n_items", "n_scales")

_model_cache: dict[str, object] = {}
_frame_cache: dict[str, pd.DataFrame] = {}


def _load_config(config_path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def _get_model(cfg: dict, key: str, index: int):
    """(SentenceTransformer, sanitized_name) for ``encode.{key}[index]``."""
    entries = _model_entries(cfg.get("encode", {}).get(key), key)
    if not 0 <= index < len(entries):
        raise IndexError(f"model_index {index} out of range: encode.{key} "
                         f"lists {len(entries)} model(s)")
    path, name = entries[index]
    if path not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[path] = SentenceTransformer(str(path))
    return _model_cache[path], name


def _load_pooled(cfg: dict, data) -> pd.DataFrame:
    if data is not None:
        return data
    path = str(cfg.get("data", {}).get("assemble", {}).get("pooled", ""))
    if path in ("", "."):
        raise ValueError("data.assemble.pooled must be set in config")
    if path not in _frame_cache:
        if not Path(path).exists():
            raise FileNotFoundError(f"pooled corpus not found: {path}")
        _frame_cache[path] = pd.read_parquet(path)
    return _frame_cache[path]


def pool_query(vectors: np.ndarray, reverse=None) -> np.ndarray:
    """One query vector from ``vectors`` (n, d) via the pool stage's ``keyed_centroid``."""
    if reverse is None:
        reverse = np.zeros(len(vectors), dtype=bool)
    rev = np.asarray(reverse, dtype=bool)
    if len(rev) != len(vectors):
        raise ValueError(f"reverse has {len(rev)} entries for "
                         f"{len(vectors)} query strings")
    return keyed_centroid(vectors, rev)


def _rank(df: pd.DataFrame, column: str, query_vec: np.ndarray,
          top_k: int) -> pd.DataFrame:
    """Top-``top_k`` rows by cosine similarity of ``column`` to ``query_vec``; null cells skipped."""
    if column not in df.columns:
        raise ValueError(f"column {column} not in the pooled frame — was the "
                         "pool stage run with this model configured?")
    cells = df[column].to_numpy(dtype=object)
    valid = np.array([not _is_missing(c) for c in cells])
    if not valid.any():
        raise ValueError(f"column {column} has no non-null vectors")
    mat = np.stack(cells[valid]).astype(np.float32)
    denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(query_vec)
    sims = np.divide(mat @ query_vec, denom,
                     out=np.zeros(len(mat)), where=denom > 0)
    cols = [c for c in RESULT_COLS if c in df.columns]
    result = df.loc[valid, cols].copy()
    result.insert(0, "similarity", sims)
    return (result.sort_values("similarity", ascending=False)
            .head(top_k).reset_index(drop=True))


def search_items(query, *, reverse=None, model_index: int = 0,
                 top_k: int = 10, data: pd.DataFrame | None = None,
                 config_path="config.yaml") -> pd.DataFrame:
    """Rank pooled rows by similarity to ``item_pooled_{model}``.

    A list ``query`` is pooled into one vector; ``reverse`` marks its reverse-keyed entries.
    Results mix scale and instrument rows (filter on ``is_instrument``).
    """
    texts = [query] if isinstance(query, str) else list(query)
    if isinstance(query, str) and reverse is not None:
        raise ValueError("reverse only applies to a list of query strings")
    cfg = _load_config(config_path)
    model, name = _get_model(cfg, "item_models", model_index)
    vectors = np.asarray(
        model.encode(texts, batch_size=DEFAULT_BATCH_SIZE,
                     convert_to_numpy=True), dtype=np.float32)
    query_vec = pool_query(vectors, reverse)
    return _rank(_load_pooled(cfg, data), f"item_pooled_{name}",
                 query_vec, top_k)


def search_scales(query, *, model_index: int = 0, top_k: int = 10,
                  data: pd.DataFrame | None = None,
                  config_path="config.yaml") -> pd.DataFrame:
    """Rank pooled rows by similarity to ``scale_pooled_{model}``; a list ``query`` is plain-mean pooled."""
    texts = [query] if isinstance(query, str) else list(query)
    cfg = _load_config(config_path)
    model, name = _get_model(cfg, "scale_models", model_index)
    vectors = np.asarray(
        model.encode(texts, batch_size=DEFAULT_BATCH_SIZE,
                     convert_to_numpy=True), dtype=np.float32)
    return _rank(_load_pooled(cfg, data), f"scale_pooled_{name}",
                 pool_query(vectors), top_k)
