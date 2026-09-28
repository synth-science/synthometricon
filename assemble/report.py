"""Paper-facing descriptives report over the assembled artifacts and ``.stats.json`` sidecars.

Not a stage: the publish step writes it beside the release; ``python -m assemble.report`` previews it.
Counting conventions and table formats: docs/assemble-report.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from extraction.storage import PATH_COL

from .combine import source_from_path
from .patch import SOURCE_COL, _doc_keys
from .stats import read_stats, stats_stale

Section = tuple[str, list[str]]  # (title, lines)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _read(path, columns=None) -> pd.DataFrame:
    """Column-pruned parquet read that skips columns the file lacks (older partials)."""
    if columns is not None:
        present = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [c for c in columns if c in present]
    return pd.read_parquet(path, columns=columns)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.3f}"
    return str(v)


def _table(headers: list[str], rows: list[list]) -> list[str]:
    """Markdown pipe table; numbers right-aligned by the renderer."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return lines


def _flatten(stats: dict, prefix: str = "") -> list[tuple[str, object]]:
    """Nested stats dict -> flat (dotted key, scalar) pairs."""
    out: list[tuple[str, object]] = []
    for key, value in stats.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.extend(_flatten(value, dotted))
        else:
            out.append((dotted, value))
    return out


def _keys(df: pd.DataFrame) -> pd.Series:
    """Document keys with the null-path fallback, tolerating missing fallback columns."""
    df = df.copy(deep=False)
    for col in (SOURCE_COL, "meta_title_raw"):
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index)
    return _doc_keys(df)


def _n_docs(df: pd.DataFrame) -> int:
    return int(_keys(df).nunique())


def _n_distinct(df: pd.DataFrame, id_col: str) -> int:
    """Distinct (document, id) pairs among rows where ``id_col`` is set."""
    if id_col not in df.columns:
        return 0
    mask = df[id_col].notna()
    if not mask.any():
        return 0
    keys = _keys(df)
    pairs = pd.DataFrame({"doc": keys[mask], "id": df.loc[mask, id_col]})
    return int(len(pairs.drop_duplicates()))


def _n_scale_nodes(df: pd.DataFrame) -> int:
    """Distinct (document, node) over ``scale_id_path``; item-bearing count if absent."""
    if "scale_id_path" not in df.columns:
        return _n_distinct(df, "scale_id")
    mask = df["scale_id_path"].notna()
    if not mask.any():
        return 0
    keys = _keys(df)
    pairs = pd.DataFrame({"doc": keys[mask],
                          "node": df.loc[mask, "scale_id_path"]})
    return int(len(pairs.explode("node").drop_duplicates()))


def _n_placements(df: pd.DataFrame) -> int:
    """Rows carrying an item, i.e. ``(item, scale)`` occurrences."""
    if "item_item_id" not in df.columns:
        return len(df)
    return int(df["item_item_id"].notna().sum())


def _counts_frame(df: pd.DataFrame) -> dict[str, int]:
    return {
        "placements": _n_placements(df),
        "documents": _n_docs(df),
        "scales": _n_scale_nodes(df),
        "item-bearing scales": _n_distinct(df, "scale_id"),
        "items": _n_distinct(df, "item_item_id"),
    }


COUNT_COLS = ("placements", "documents", "scales", "item-bearing scales",
              "items")


def _shell_row_note(df: pd.DataFrame) -> list[str]:
    """Footnote reconciling placements with physical rows when shell rows exist."""
    shells = len(df) - _n_placements(df)
    if shells <= 0:
        return []
    return ["", f"(the parquet holds {len(df):,} rows: the {shells:,} extra "
                "are the item-less shell rows a failed extraction leaves "
                "behind, which are no item's placement)"]


def _per_source(df: pd.DataFrame) -> list[list]:
    """One row per source plus a total row, columns per ``COUNT_COLS``."""
    rows = []
    for source, sub in df.groupby(SOURCE_COL, dropna=False):
        c = _counts_frame(sub)
        rows.append([source] + [c[k] for k in COUNT_COLS])
    total = _counts_frame(df)
    rows.append(["**total**"] + [total[k] for k in COUNT_COLS])
    return rows


# ---------------------------------------------------------------------------
# Distributions (M / SD and friends)
# ---------------------------------------------------------------------------

DIST_HEADERS = ["n", "M", "SD", "min", "Mdn", "IQR (Q1–Q3)", "p95", "max"]

# Caption suffix for the compact comparison tables; keep it next to ``_ms``.
MS_LEGEND = "M (SD) · Mdn [IQR Q1–Q3]"


def _dist_row(series: pd.Series | None) -> list:
    """``[n, M, SD, min, Mdn, IQR, p95, max]`` for a numeric series (SD ddof=1; IQR as ``Q1–Q3``)."""
    if series is None:
        return ["(not available)"] + [""] * (len(DIST_HEADERS) - 1)
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return ["(no values)"] + [""] * (len(DIST_HEADERS) - 1)
    sd = _fmt(float(s.std(ddof=1))) if len(s) > 1 else "n/a"
    q1, med, q3 = (float(v) for v in s.quantile([0.25, 0.5, 0.75]))
    return [len(s), _fmt(float(s.mean())), sd, _fmt(float(s.min())),
            _fmt(med), f"{_fmt(q1)}–{_fmt(q3)}",
            _fmt(float(s.quantile(0.95))), _fmt(float(s.max()))]


def _dist_table(series: pd.Series | None) -> list[str]:
    """One-row distribution table for a single series."""
    return _table(DIST_HEADERS, [_dist_row(series)])


def _ms(series: pd.Series | None) -> str:
    """Compact ``M (SD) · Mdn [Q1–Q3]`` cell for the comparison tables."""
    if series is None:
        return "n/a"
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return "n/a"
    sd = f"{s.std(ddof=1):,.2f}" if len(s) > 1 else "n/a"
    q1, med, q3 = (float(v) for v in s.quantile([0.25, 0.5, 0.75]))
    return f"{s.mean():,.2f} ({sd}) · {med:,.2f} [{q1:,.2f}–{q3:,.2f}]"


# Reporting order; see docs/assemble-report.md#structural-distributions.
STRUCTURE_UNITS = (
    "items per document",
    "placements per document",
    "scales per document",
    "item-bearing scales per document",
    "items per scale (subtree)",
    "items per item-bearing scale (direct)",
    "placements per item (scales an item sits on)",
)


def _structure_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-unit count series behind ``STRUCTURE_UNITS``; empty units count as 0.

    Units whose source column is absent are omitted.
    """
    out: dict[str, pd.Series] = {}
    if df.empty:
        return out
    keys = _keys(df)
    docs = pd.Index(pd.unique(keys), name="doc")
    has_items = "item_item_id" in df.columns
    has_paths = "scale_id_path" in df.columns

    if has_items:
        mask = df["item_item_id"].notna()
        pairs = pd.DataFrame({"doc": keys[mask],
                              "id": df.loc[mask, "item_item_id"]})
        # Distinct pairs = items, rows = placements.
        per_item = pairs.groupby(["doc", "id"]).size()
        out["items per document"] = (pairs.drop_duplicates().groupby("doc")
                                     .size().reindex(docs, fill_value=0))
        out["placements per document"] = (pairs.groupby("doc").size()
                                          .reindex(docs, fill_value=0))
        out["placements per item (scales an item sits on)"] = per_item

    nodes = None
    if has_paths:
        mask = df["scale_id_path"].notna()
        nodes = (pd.DataFrame({"doc": keys[mask],
                               "node": df.loc[mask, "scale_id_path"]})
                 .explode("node").dropna(subset=["node"]).drop_duplicates())
        out["scales per document"] = (nodes.groupby("doc").size()
                                      .reindex(docs, fill_value=0))

    if "scale_id" in df.columns:
        mask = df["scale_id"].notna()
        bearing = pd.DataFrame({"doc": keys[mask],
                                "scale": df.loc[mask, "scale_id"]})
        bearing = bearing.drop_duplicates()
        out["item-bearing scales per document"] = (
            bearing.groupby("doc").size().reindex(docs, fill_value=0))
        if has_items:
            mask = mask & df["item_item_id"].notna()
            trios = pd.DataFrame({"doc": keys[mask],
                                  "scale": df.loc[mask, "scale_id"],
                                  "id": df.loc[mask, "item_item_id"]})
            counts = trios.drop_duplicates().groupby(["doc", "scale"]).size()
            index = pd.MultiIndex.from_frame(bearing, names=["doc", "scale"])
            out["items per item-bearing scale (direct)"] = counts.reindex(
                index, fill_value=0)

    # Subtree counts mirror pool's n_items.
    if has_paths and has_items and nodes is not None:
        mask = df["scale_id_path"].notna() & df["item_item_id"].notna()
        sub = (pd.DataFrame({"doc": keys[mask],
                             "node": df.loc[mask, "scale_id_path"],
                             "id": df.loc[mask, "item_item_id"]})
               .explode("node").dropna(subset=["node"]).drop_duplicates())
        counts = sub.groupby(["doc", "node"]).size()
        index = pd.MultiIndex.from_frame(nodes, names=["doc", "node"])
        out["items per scale (subtree)"] = counts.reindex(index, fill_value=0)

    return {unit: out[unit] for unit in STRUCTURE_UNITS if unit in out}


def _structure_table(df: pd.DataFrame) -> list[str]:
    """Full distribution table (one row per structural unit)."""
    series = _structure_series(df)
    if not series:
        return ["(no structural columns in this artifact)"]
    return _table(["unit", *DIST_HEADERS],
                  [[unit, *_dist_row(s)] for unit, s in series.items()])


def _structure_ms_table(frames: list[tuple[str, pd.DataFrame]]) -> list[str]:
    """Compact comparison table: structural units (rows) × named frames (columns)."""
    cols = [(label, _structure_series(df)) for label, df in frames]
    units = [u for u in STRUCTURE_UNITS
             if any(u in series for _, series in cols)]
    if not units:
        return ["(no structural columns in these artifacts)"]
    return _table(["unit"] + [label for label, _ in cols],
                  [[unit] + [_ms(series.get(unit)) for _, series in cols]
                   for unit in units])


# ---------------------------------------------------------------------------
# Scale-hierarchy nesting
# ---------------------------------------------------------------------------

def _row_depth(df: pd.DataFrame) -> pd.Series | None:
    """Depth of each row's immediate scale (1 = top-level); ``None`` without path/depth columns."""
    if "scale_id_path" in df.columns:
        return df["scale_id_path"].map(
            lambda v: len(v) if v is not None and hasattr(v, "__len__")
            else pd.NA).astype("Float64")
    if "scale_depth" in df.columns:
        return pd.to_numeric(df["scale_depth"], errors="coerce").astype(
            "Float64")
    return None


def _doc_nesting(df: pd.DataFrame) -> pd.Series | None:
    """Levels of nesting per document (deepest scale node; scale-less documents = 0)."""
    depth = _row_depth(df)
    if depth is None or df.empty:
        return None
    keys = _keys(df)
    docs = pd.Index(pd.unique(keys), name="doc")
    per_doc = depth.groupby(keys.to_numpy()).max()
    return (per_doc.reindex(docs).fillna(0).astype("int64")
            .rename("levels of nesting"))


def _node_depths(df: pd.DataFrame) -> pd.Series | None:
    """Depth of every distinct ``(document, node)``; needs ``scale_id_path``."""
    if "scale_id_path" not in df.columns or df.empty:
        return None
    mask = df["scale_id_path"].notna()
    if not mask.any():
        return None
    keys = _keys(df)
    paths = df.loc[mask, "scale_id_path"]
    nodes = pd.DataFrame({
        "doc": keys[mask].repeat(paths.map(len).to_numpy()),
        "node": [n for path in paths for n in path],
        "depth": [i + 1 for path in paths for i, _ in enumerate(path)],
    }).dropna(subset=["node"])
    return (nodes.drop_duplicates(subset=["doc", "node"])["depth"]
            .reset_index(drop=True).rename("scale node depth"))


def _counts_by_level(series: pd.Series, label: str) -> list[str]:
    """``level | count | share | cumulative`` table, filling gaps between observed levels."""
    vc = series.value_counts().sort_index()
    levels = range(int(series.min()), int(series.max()) + 1)
    total = int(len(series))
    rows, running = [], 0
    for level in levels:
        n = int(vc.get(level, 0))
        running += n
        rows.append([level, n, f"{n / total:.1%}", f"{running / total:.1%}"])
    rows.append(["**total**", total, "100.0%", ""])
    return _table([label, "count", "share", "cumulative"], rows)


def _nesting_lines(df: pd.DataFrame) -> list[str]:
    """Nesting-depth descriptives per instrument and per scale node."""
    per_doc = _doc_nesting(df)
    if per_doc is None or per_doc.empty:
        return ["(no scale-hierarchy columns in this artifact)"]
    lines = ["Levels of nesting per instrument (depth of its deepest scale "
             "node; 0 = no scales, 1 = flat list of scales, 2 = scales with "
             "subscales, ...):", ""]
    lines += _dist_table(per_doc)
    lines.append("")
    lines.append("Instruments per level of nesting:")
    lines.append("")
    lines += _counts_by_level(per_doc, "levels of nesting")

    if SOURCE_COL in df.columns:
        sources = [str(s) for s, _ in df.groupby(SOURCE_COL, dropna=False)]
        per_source = {
            str(source): _doc_nesting(sub)
            for source, sub in df.groupby(SOURCE_COL, dropna=False)}
        levels = range(int(per_doc.min()), int(per_doc.max()) + 1)
        rows = []
        for level in levels:
            cells = [int((per_source[s] == level).sum())
                     if per_source[s] is not None else 0 for s in sources]
            rows.append([level, *cells, int((per_doc == level).sum())])
        rows.append(["**total**",
                     *[int(len(per_source[s])) if per_source[s] is not None
                       else 0 for s in sources], int(len(per_doc))])
        lines.append("")
        lines.append("Instruments per level of nesting and `corpus_source`:")
        lines.append("")
        lines += _table(["levels of nesting", *sources, "**total**"], rows)
        lines.append("")
        lines.append(f"Levels of nesting per `corpus_source` — {MS_LEGEND}:")
        lines.append("")
        lines += _table(["unit", *sources, "**total**"],
                        [["levels of nesting per instrument",
                          *[_ms(per_source[s]) for s in sources],
                          _ms(per_doc)]])

    depths = _node_depths(df)
    if depths is not None and not depths.empty:
        lines.append("")
        lines.append("Scale nodes per hierarchy level (every distinct "
                     "`(document, node)`, parents included):")
        lines.append("")
        lines += _counts_by_level(depths, "hierarchy level")
        lines.append("")
        lines.append("Depth of the individual scale node:")
        lines.append("")
        lines += _dist_table(depths)
    return lines


def _by_source_frames(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(label, frame) pairs per ``corpus_source``, plus the total."""
    if SOURCE_COL not in df.columns:
        return [("all", df)]
    frames = [(str(source), sub)
              for source, sub in df.groupby(SOURCE_COL, dropna=False)]
    return frames + [("**total**", df)]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_extraction(cfg: dict) -> Section:
    """Documents considered in extraction (the per-PDF store)."""
    lines: list[str] = []
    store = (cfg.get("data") or {}).get("extractions")
    if not store or not Path(store).exists():
        return ("Extraction (before assembly)",
                [f"WARNING: extraction store not found: {store}"])
    schema = pq.ParquetFile(store).schema_arrow.names
    names = sorted(c[: -len("_extractor_content")] for c in schema
                   if c.endswith("_extractor_content"))
    cols = [PATH_COL, "has_errors"] + [f"{n}_extractor_{s}"
                                       for n in names
                                       for s in ("content", "error")]
    df = _read(store, cols)
    lines.append(f"documents considered: **{len(df):,}** "
                 f"({df[PATH_COL].nunique():,} distinct paths)")
    if df[PATH_COL].nunique() != len(df):
        lines.append("WARNING: store paths are not unique")
    if "has_errors" in df.columns:
        n = int(df["has_errors"].fillna(False).astype(bool).sum())
        lines.append(f"documents with extraction errors (`has_errors`): {n:,}")
    rows = []
    for name in names:
        content = df[f"{name}_extractor_content"]
        error = df[f"{name}_extractor_error"]
        done = int(content.notna().sum())
        failed = int((content.isna() & error.notna()).sum())
        rows.append([name, done, failed, len(df) - done - failed])
    lines.append("")
    lines += _table(["extractor", "done", "failed", "pending"], rows)

    exploded = (cfg.get("data") or {}).get("assemble", {}).get("exploded")
    if exploded and Path(exploded).exists():
        n_exp = _read(exploded, [PATH_COL])[PATH_COL].nunique()
        lines.append("")
        lines.append(f"documents in exploded output: {n_exp:,}")
        if n_exp != len(df):
            stale = Path(exploded).stat().st_mtime < Path(store).stat().st_mtime
            if stale:
                lines.append(f"WARNING: exploded parquet covers {n_exp:,} of "
                             f"{len(df):,} store documents and is older than "
                             "the store — re-run "
                             "`python -m assemble --step explode`")
            else:
                lines.append(
                    f"note: {len(df) - n_exp:,} store documents contribute no "
                    "exploded rows — successful extractions whose instrument "
                    "tree holds zero items (explode emits one row per item; "
                    "only null-output documents get a carry-only shell row)")
    return ("Extraction (before assembly)", lines)


def section_partials(cfg: dict) -> Section:
    """Documents/instruments, scales, and items per input partial."""
    data_cfg = cfg.get("data") or {}
    paths = ([data_cfg.get("assemble", {}).get("exploded")]
             + list(data_cfg.get("partials", [])))
    rows, lines = [], []
    frames: list[tuple[str, pd.DataFrame]] = []
    for path in paths:
        if not path or not Path(path).exists():
            lines.append(f"WARNING: partial not found: {path}")
            continue
        df = _read(path, [PATH_COL, "item_item_id", "scale_id",
                          "scale_id_path", "meta_title_raw"])
        null_paths = int(df[PATH_COL].isna().sum()) \
            if PATH_COL in df.columns else len(df)
        if SOURCE_COL not in df.columns:
            df[SOURCE_COL] = source_from_path(Path(path))
        c = _counts_frame(df)
        rows.append([source_from_path(Path(path))]
                    + [c[k] for k in COUNT_COLS] + [null_paths])
        frames.append((source_from_path(Path(path)), df))
    lines += _table(["partial", *COUNT_COLS, "null-path rows"], rows)
    if frames:
        lines.append("")
        lines.append(f"Structure per partial — {MS_LEGEND}:")
        lines.append("")
        lines += _structure_ms_table(frames)
    return ("Partial datasets (before assembly)", lines)


def section_combined(cfg: dict) -> Section:
    """Per-source counts after explode+combine, incl. bucket composition."""
    combined = (cfg.get("data") or {}).get("assemble", {}).get("combined")
    if not combined or not Path(combined).exists():
        return ("Combined corpus (after explosion)",
                [f"WARNING: combined parquet not found: {combined}"])
    df = _read(combined, [SOURCE_COL, PATH_COL, "item_item_id", "scale_id",
                          "scale_id_path", "bucket", "meta_title_raw"])
    lines = _table(["corpus_source", *COUNT_COLS], _per_source(df))
    lines += _shell_row_note(df)
    if "bucket" in df.columns:
        lines.append("")
        lines.append("Placement composition by `bucket` (null = shell row "
                     "from a failed extraction, counted as no placement):")
        lines.append("")
        pivot = (df.assign(bucket=df["bucket"].fillna("(null)"))
                 .groupby([SOURCE_COL, "bucket"]).size().unstack(fill_value=0))
        lines += _table([SOURCE_COL] + list(pivot.columns),
                        [[idx] + [int(v) for v in row]
                         for idx, row in pivot.iterrows()])
    lines.append("")
    lines.append("Structure of the combined corpus:")
    lines.append("")
    lines += _structure_table(df)
    lines.append("")
    lines.append(f"Structure per `corpus_source` — {MS_LEGEND}:")
    lines.append("")
    lines += _structure_ms_table(_by_source_frames(df))
    return ("Combined corpus (after explosion)", lines)


def _sidecar_section(cfg: dict, stage: str, artifact_key: str,
                     rerun_hint: str) -> list[str]:
    artifact = (cfg.get("data") or {}).get("assemble", {}).get(artifact_key)
    lines: list[str] = []
    payload = read_stats(artifact) if artifact else None
    if payload is None:
        lines.append(f"{stage} step figures unavailable (no stats sidecar) — "
                     f"re-run `{rerun_hint}` to generate them.")
        return lines
    if stats_stale(artifact):
        lines.append(f"WARNING: {stage} stats sidecar is older than the "
                     f"artifact — figures may describe a previous run; "
                     f"re-run `{rerun_hint}`.")
        lines.append("")
    lines.append(f"run of {payload.get('generated', '?')} "
                 f"(`{Path(artifact).name}`):")
    lines.append("")
    lines += _table(["step counter", "value"],
                    [[k, v] for k, v in _flatten(payload.get("stats", {}))])
    return lines


def section_patch(cfg: dict) -> Section:
    lines = _sidecar_section(cfg, "patch", "patched",
                             "python -m assemble --step patch")
    patched = (cfg.get("data") or {}).get("assemble", {}).get("patched")
    if patched and Path(patched).exists():
        df = _read(patched, [PATH_COL, "is_patched", "meta_title_raw",
                             SOURCE_COL, "item_item_id", "scale_id",
                             "scale_id_path"])
        if "is_patched" in df.columns:
            flagged = df["is_patched"].fillna(False).astype(bool)
            lines.append("")
            lines.append(
                f"`is_patched` (from the artifact): {int(flagged.sum()):,} of "
                f"{len(df):,} parquet rows ({flagged.mean():.1%}) across "
                f"{_n_docs(df.loc[flagged]):,} documents")
        lines.append("")
        lines.append("Structure of the patched corpus (patch never drops "
                     "rows, so these match the combined corpus unless a "
                     "step changed an id):")
        lines.append("")
        lines += _structure_table(df)
        lines += _shell_row_note(df)
    return ("Patch stage — what was done", lines)


def section_postprocess(cfg: dict) -> Section:
    lines = _sidecar_section(cfg, "postprocess", "postprocessed",
                             "python -m assemble --step postprocess")
    asm = (cfg.get("data") or {}).get("assemble", {})
    combined, post = asm.get("combined"), asm.get("postprocessed")
    if (combined and post and Path(combined).exists()
            and Path(post).exists()):
        cdf = _read(combined, [SOURCE_COL, PATH_COL, "item_item_id",
                               "scale_id", "scale_id_path", "meta_title_raw"])
        pdf_ = _read(post, [SOURCE_COL, PATH_COL, "item_item_id", "scale_id",
                            "scale_id_path", "meta_title_raw"])
        cc, pc = _counts_frame(cdf), _counts_frame(pdf_)
        lines.append("")
        lines.append("Attrition combined -> postprocessed "
                     "(from the artifacts):")
        lines.append("")
        lines += _table(
            ["", "combined", "postprocessed", "removed"],
            [[k, cc[k], pc[k], cc[k] - pc[k]] for k in COUNT_COLS])
        # Sidecar filter_rows.rows.* counters are physical rows, not placements.
        lines += _shell_row_note(cdf)
        lines.append("")
        lines.append(f"Structure before and after pruning — {MS_LEGEND}:")
        lines.append("")
        lines += _structure_ms_table([("combined", cdf),
                                      ("postprocessed", pdf_)])
        lines.append("")
        lines.append("(the full postprocessed distribution is in *Final "
                     "datasets* — encode adds columns only, so the published "
                     "corpus has the same structure)")
    return ("Postprocess stage — what was done", lines)


def section_final(cfg: dict) -> Section:
    asm = (cfg.get("data") or {}).get("assemble", {})
    lines: list[str] = []
    embedded = asm.get("embedded")
    n_embedded_docs = n_embedded_nodes = None
    if embedded and Path(embedded).exists():
        df = _read(embedded, [SOURCE_COL, PATH_COL, "item_item_id",
                              "scale_id", "scale_id_path", "scale_depth",
                              "meta_title_raw"])
        n_embedded_docs = _n_docs(df)
        n_embedded_nodes = _n_scale_nodes(df)
        lines.append("### `embedded` (item-level dataset)")
        lines.append("")
        lines += _table(["corpus_source", *COUNT_COLS], _per_source(df))
        lines.append("")
        lines.append("Structure of the published corpus:")
        lines.append("")
        lines += _structure_table(df)
        lines.append("")
        lines.append(f"Structure per `corpus_source` — {MS_LEGEND}:")
        lines.append("")
        lines += _structure_ms_table(_by_source_frames(df))
        lines.append("")
        lines.append("#### Nesting of the scale hierarchy")
        lines.append("")
        lines += _nesting_lines(df)
    else:
        lines.append(f"WARNING: embedded parquet not found: {embedded}")

    pooled = asm.get("pooled")
    if pooled and Path(pooled).exists():
        df = _read(pooled, [SOURCE_COL, "is_instrument"])
        pivot = (df.groupby(SOURCE_COL)["is_instrument"]
                 .agg(instruments="sum", rows="size"))
        rows = [[idx, int(r["instruments"]), int(r["rows"] - r["instruments"])]
                for idx, r in pivot.iterrows()]
        rows.append(["**total**", int(df["is_instrument"].sum()),
                     int((~df["is_instrument"]).sum())])
        lines.append("")
        lines.append("### `pooled` (centroid dataset)")
        lines.append("")
        lines += _table(["corpus_source", "instruments", "scales"], rows)
        if n_embedded_docs is not None:
            n_inst = int(df["is_instrument"].sum())
            check = "consistent" if n_inst == n_embedded_docs else \
                f"MISMATCH vs {n_embedded_docs:,} embedded documents"
            lines.append("")
            lines.append(f"sanity: {n_inst:,} pooled instruments — {check}")
            n_pool_scales = int((~df["is_instrument"]).sum())
            check = "consistent" if n_pool_scales == n_embedded_nodes else \
                f"MISMATCH vs {n_embedded_nodes:,} embedded scale nodes"
            lines.append(f"sanity: {n_pool_scales:,} pooled scales — {check}")
    else:
        lines.append(f"WARNING: pooled parquet not found: {pooled}")
    return ("Final datasets", lines)


# ---------------------------------------------------------------------------
# Verbatim fidelity (item text vs source PDF)
# ---------------------------------------------------------------------------

FIDELITY_HEADERS = ["unit", "n", "M", "SD", "Mdn", "IQR (Q1–Q3)", "p5",
                    "min", "= 100%", "≥ threshold"]

# (label, lo, hi) as [lo, hi); the first is the exact-match point mass.
FIDELITY_BANDS = [("100% (verbatim)", 1.0, 1.0), ("95% to <100%", 0.95, 1.0),
                  ("80% to <95%", 0.80, 0.95), ("50% to <80%", 0.50, 0.80),
                  ("<50%", 0.0, 0.50)]


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _fidelity_row(label: str, s: pd.Series, threshold: float) -> list:
    """One similarity distribution row in percent (p5, since the low tail is informative)."""
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if s.empty:
        return [label, "(no values)"] + [""] * (len(FIDELITY_HEADERS) - 2)
    q1, med, q3, p5 = (float(v) for v in s.quantile([0.25, 0.5, 0.75, 0.05]))
    sd = _pct(float(s.std(ddof=1))) if len(s) > 1 else "n/a"
    return [label, len(s), _pct(float(s.mean())), sd, _pct(med),
            f"{_pct(q1)}–{_pct(q3)}", _pct(p5), _pct(float(s.min())),
            _pct(float(s.eq(1.0).mean())), _pct(float(s.ge(threshold).mean()))]


def _fidelity_lines(df: pd.DataFrame, keys: pd.Series) -> list[str]:
    """Verbatim-fidelity block: method, similarity distributions, threshold counts.

    See docs/assemble-report.md#verbatim-fidelity.
    """
    from importlib.metadata import PackageNotFoundError, version

    from .postprocess import PDF_MATCH_NORM_MAX, PDF_TEXT_CHARS_MIN

    try:
        rf_version = version("rapidfuzz")
    except PackageNotFoundError:
        rf_version = "?"
    threshold = 1.0 - PDF_MATCH_NORM_MAX
    thr = f"{threshold:.0%}"

    sim = 1.0 - pd.to_numeric(df["pdf_match_edit_distance_norm"],
                              errors="coerce").astype(float)
    f = pd.DataFrame({"doc": keys, "sim": sim})
    f["item"] = (df["item_item_id"] if "item_item_id" in df.columns
                 else pd.Series(range(len(df)), index=df.index))
    chars = (pd.to_numeric(df["pdf_text_chars"], errors="coerce")
             if "pdf_text_chars" in df.columns
             else pd.Series(float("nan"), index=df.index))
    f["thin"] = chars.lt(PDF_TEXT_CHARS_MIN)
    f["translated"] = (df["flag_item_translated"].astype("boolean")
                       .fillna(False).astype(bool)
                       if "flag_item_translated" in df.columns else False)
    measured = f[f["sim"].notna()]
    items = measured.drop_duplicates(["doc", "item"])
    checkable = items[~items["thin"]]
    original = checkable[~checkable["translated"]]
    n_docs_total = int(keys.nunique())

    def per_doc(frame: pd.DataFrame, how: str) -> pd.Series:
        return frame.groupby("doc")["sim"].agg(how)

    lines = [
        "### Verbatim fidelity — extracted item text vs. source PDF",
        "",
        "**Method.** For every published item of an `apa-psyctests` "
        "instrument, the item text (`item_item_text` as published, i.e. after "
        "the patch stage's repairs such as numbering-prefix stripping) is "
        "located in the text layer of its source PDF. *Reference text:* the "
        "PDF's embedded text layer as returned by PyMuPDF `page.get_text()` "
        "for pages 2..n joined with newlines (page 1 is the PsycTESTS "
        "database cover page and is excluded); no OCR is applied, so a "
        "scanned PDF contributes only whatever text layer it carries. "
        f"*Algorithm:* (1) rapidfuzz {rf_version} "
        "`fuzz.partial_ratio_alignment(item, pdf_text)` finds the substring "
        "(window) of the PDF text that best aligns with the item (partial "
        "ratio = normalized Indel similarity, maximized over windows); "
        "(2) the Levenshtein distance *d* (unit-cost insertions, deletions "
        "and substitutions) between the item and that window is computed and "
        "normalized as *d* / max(len(item), len(window)); (3) **similarity = "
        "1 − normalized distance**, so 100% means the item occurs verbatim "
        "in the PDF and 0% means no character aligns. *Preprocessing:* none "
        "— the comparison is on the raw strings: case-sensitive, and "
        "whitespace, line breaks, end-of-line hyphenation, punctuation and "
        "Unicode forms are **not** normalized, so an item broken across "
        "lines or pages in the PDF loses a few percentage points even when "
        "it was extracted correctly. *Threshold:* an item counts as a "
        f"near-verbatim match at similarity ≥ {thr} (normalized distance "
        f"≤ {PDF_MATCH_NORM_MAX}; `flag_item_text_deviation` marks the "
        "rest). *Exclusions:* non-APA (partial) sources have no PDF and are "
        "not measured; PDFs whose text layer (pages 2..n) is shorter than "
        f"{PDF_TEXT_CHARS_MIN} characters are treated as *not checkable* "
        "(no real text layer — the distance would reflect the missing text, "
        "not the item); items whose original language is not English "
        "(`flag_item_translated`) are published as English translations and "
        "therefore cannot match the original-language PDF, so they are "
        "shown both included and excluded.",
        "",
        f"Similarity distribution by unit of aggregation (in percent; `p5` = "
        f"5th percentile; `= 100%` / `≥ threshold` = share of units at "
        f"exactly 100% / at or above {thr}):",
        "",
    ]
    rows = [
        _fidelity_row("items — all measured", items["sim"], threshold),
        _fidelity_row(f"items — checkable (text layer ≥ "
                      f"{PDF_TEXT_CHARS_MIN} chars)", checkable["sim"],
                      threshold),
        _fidelity_row("items — checkable, English originals only",
                      original["sim"], threshold),
        _fidelity_row("instruments — mean item similarity, all measured",
                      per_doc(items, "mean"), threshold),
        _fidelity_row("instruments — mean item similarity, checkable",
                      per_doc(checkable, "mean"), threshold),
        _fidelity_row("instruments — mean item similarity, checkable, "
                      "English originals only",
                      per_doc(original, "mean"), threshold),
        _fidelity_row("instruments — minimum item similarity, checkable",
                      per_doc(checkable, "min"), threshold),
        _fidelity_row("placements — all measured (stored grain, reference)",
                      measured["sim"], threshold),
    ]
    lines += _table(FIDELITY_HEADERS, rows)
    lines.append("")
    lines.append("Instrument-level mean = the mean of each instrument's item "
                 "similarities (items weighted equally within an instrument, "
                 "instruments weighted equally across the corpus); its "
                 "`≥ threshold` share is the share of instruments whose "
                 "*average* item reaches the threshold, which is **not** the "
                 "share whose every item does — that is the minimum row.")

    lines.append("")
    lines.append("Item similarity bands (checkable items; English originals "
                 "only in the last column):")
    lines.append("")
    band_rows = []
    for label, lo, hi in FIDELITY_BANDS:
        def band(s: pd.Series) -> pd.Series:
            if lo == hi:
                return s.eq(hi)
            return s.ge(lo) & s.lt(hi) if lo > 0 else s.lt(hi)
        a, b = band(checkable["sim"]), band(original["sim"])
        band_rows.append([label, int(a.sum()), _pct(float(a.mean())),
                          int(b.sum()), _pct(float(b.mean()))])
    lines += _table(["similarity", "items", "share",
                     "items (English originals)", "share"], band_rows)

    def below_counts(frame: pd.DataFrame) -> tuple[int, int]:
        by_doc = frame.groupby("doc")["sim"]
        return int(by_doc.min().lt(threshold).sum()), int(by_doc.ngroups)

    n_any, n_chk = below_counts(checkable)
    n_any_en, n_chk_en = below_counts(original)
    n_any_all, n_meas = below_counts(items)
    doc_mean = per_doc(items, "mean")
    n_mean_below = int(doc_mean.lt(threshold).sum())
    n_thin_docs = int(measured.loc[measured["thin"], "doc"].nunique())
    lines.append("")
    lines.append(f"Instruments below the {thr} threshold:")
    lines.append("")
    lines += _table(
        ["criterion", "instruments", "of", "share"],
        [[f"≥ 1 checkable item below {thr}", n_any, n_chk,
          _pct(n_any / n_chk) if n_chk else "n/a"],
         [f"≥ 1 checkable English-original item below {thr}", n_any_en,
          n_chk_en, _pct(n_any_en / n_chk_en) if n_chk_en else "n/a"],
         [f"≥ 1 measured item below {thr} (thin text layers included)",
          n_any_all, n_meas, _pct(n_any_all / n_meas) if n_meas else "n/a"],
         [f"mean item similarity below {thr} (all measured)", n_mean_below,
          n_meas, _pct(n_mean_below / n_meas) if n_meas else "n/a"]])
    lines.append("")
    lines.append(
        f"Denominators: {n_meas:,} instruments have ≥ 1 measured item, "
        f"{n_chk:,} ≥ 1 checkable item ({n_thin_docs:,} measured instruments "
        f"have a text layer under {PDF_TEXT_CHARS_MIN} characters); the "
        f"final dataset holds {n_docs_total:,} instruments in total. Measured "
        f"items: {len(items):,} distinct ({len(measured):,} placements); "
        f"checkable: {len(checkable):,}; of those English originals: "
        f"{len(original):,}.")
    return lines


def section_extras(cfg: dict) -> Section:
    """Supplementary distributions for the paper."""
    asm = (cfg.get("data") or {}).get("assemble", {})
    lines: list[str] = []

    pooled = asm.get("pooled")
    if pooled and Path(pooled).exists():
        df = _read(pooled, [SOURCE_COL, "is_instrument", "n_items",
                            "n_scales", "scale_depth"])
        inst, scale = df["is_instrument"].astype(bool), ~df["is_instrument"]
        groups = [
            ("items per scale (pooled `n_items`, subtree)", scale, "n_items"),
            ("items per instrument (pooled `n_items`)", inst, "n_items"),
            ("scales per instrument (pooled `n_scales`, every hierarchy "
             "node)", inst, "n_scales"),
        ]
        lines.append("Pooled group sizes:")
        lines.append("")
        lines += _table(["unit", *DIST_HEADERS],
                        [[label, *_dist_row(df.loc[mask, col])]
                         for label, mask, col in groups])
        if "scale_depth" in df.columns:
            lines.append("")
            lines.append("Scale hierarchy depth (pooled scale rows, "
                         "`scale_depth`; 1 = top-level scale) — the per-node "
                         "view; how deep each *instrument* nests is in "
                         "*Final datasets*:")
            lines.append("")
            lines += _dist_table(df.loc[scale, "scale_depth"])
            lines.append("")
            lines.append("Pooled scale rows per hierarchy level:")
            lines.append("")
            lines += _counts_by_level(
                pd.to_numeric(df.loc[scale, "scale_depth"],
                              errors="coerce").dropna().astype(int),
                "hierarchy level")
        if SOURCE_COL in df.columns:
            sources = list(df.groupby(SOURCE_COL, dropna=False).groups)
            rows = []
            for label, mask, col in groups:
                cells = [_ms(df.loc[mask & df[SOURCE_COL].eq(s), col])
                         for s in sources]
                rows.append([label, *cells, _ms(df.loc[mask, col])])
            lines.append("")
            lines.append(f"Pooled group sizes per `corpus_source` — {MS_LEGEND}:")
            lines.append("")
            lines += _table(["unit", *[str(s) for s in sources], "**total**"],
                            rows)

    embedded = asm.get("embedded")
    if embedded and Path(embedded).exists():
        cols = [PATH_COL, SOURCE_COL, "meta_title_raw", "item_item_id",
                "item_text_chars", "item_item_type", "item_options",
                "item_reverse_coded", "meta_language", "permissions_category",
                "pdf_match_edit_distance_norm", "pdf_text_chars",
                "flag_item_count_deviation", "flag_scale_count_deviation",
                "flag_item_text_deviation", "flag_item_translated",
                "record_item_count"]
        df = _read(embedded, cols)
        keys = _keys(df)
        if "item_item_id" in df.columns:
            pairs = pd.DataFrame({"doc": keys, "id": df["item_item_id"]})
            items = df.loc[~pairs.duplicated()]
        else:
            items = df

        if "item_text_chars" in items.columns:
            lines.append("")
            lines.append("Item text length in characters (distinct items):")
            lines.append("")
            lines += _dist_table(items["item_text_chars"])
            if SOURCE_COL in items.columns:
                lines.append("")
                lines.append(f"Item text length per `corpus_source` — {MS_LEGEND}:")
                lines.append("")
                rows = [[str(source), _ms(sub["item_text_chars"])]
                        for source, sub
                        in items.groupby(SOURCE_COL, dropna=False)]
                rows.append(["**total**", _ms(items["item_text_chars"])])
                lines += _table([SOURCE_COL, "characters"], rows)
        if "item_options" in items.columns:
            n_options = items["item_options"].map(
                lambda v: len(v) if v is not None and hasattr(v, "__len__")
                else pd.NA)
            lines.append("")
            lines.append("Response options per item (distinct items carrying "
                         f"an option list — {int(n_options.notna().sum()):,} "
                         f"of {len(items):,}):")
            lines.append("")
            lines += _dist_table(n_options)
        if "item_item_type" in items.columns:
            vc = items["item_item_type"].value_counts(dropna=False)
            lines.append("")
            lines.append("Item type (distinct items):")
            lines.append("")
            lines += _table(["item_item_type", "items", "share"],
                            [[str(k), int(v), f"{v / len(items):.1%}"]
                             for k, v in vc.items()])
        if "item_reverse_coded" in items.columns:
            rev = items["item_reverse_coded"].fillna(False).astype(bool)
            lines.append("")
            lines.append(f"Reverse-coded items: {int(rev.sum()):,} of "
                         f"{len(items):,} ({rev.mean():.1%})")
        if "meta_language" in df.columns:
            docs = df.assign(_key=keys).drop_duplicates("_key")
            vc = docs["meta_language"].value_counts(dropna=False).head(10)
            lines.append("")
            lines.append("Document language (top 10):")
            lines.append("")
            lines += _table(["meta_language", "documents"],
                            [[str(k), int(v)] for k, v in vc.items()])
        if "permissions_category" in df.columns:
            docs = df.assign(_key=keys).drop_duplicates("_key")
            vc = docs["permissions_category"].value_counts(dropna=False)
            lines.append("")
            lines.append("Permissions category (documents):")
            lines.append("")
            lines += _table(["permissions_category", "documents"],
                            [[str(k), int(v)] for k, v in vc.items()])
        if "pdf_match_edit_distance_norm" in df.columns:
            lines.append("")
            lines += _fidelity_lines(df, keys)
        flag_cols = [c for c in df.columns if c.startswith("flag_")]
        if flag_cols:
            rows = []
            for col in flag_cols:
                mask = df[col].astype("boolean")
                flagged = mask.fillna(False)
                docs_flagged = _n_docs(df.loc[flagged])
                rows.append([col, int(flagged.sum()), docs_flagged,
                             int(mask.isna().sum())])
            lines.append("")
            lines.append("Extraction-quality flags (placements / "
                         "documents flagged, NA = not checkable):")
            lines.append("")
            lines += _table(["flag", "placements", "documents",
                             "not checkable"], rows)
        if "record_item_count" in df.columns:
            cov = df["record_item_count"].notna()
            lines.append("")
            lines.append(
                f"PsycTESTS record reference coverage: "
                f"{_n_docs(df.loc[cov]):,} of {_n_docs(df):,} documents "
                "carry a record item count")
    return ("Additional statistics", lines)


def section_appendix(cfg: dict) -> Section:
    """Static glossary; keep in sync with ``patch._doc_keys``, explode, postprocess and pool."""
    lines = [
        "### A.1 Identifiers and counting units",
        "",
        "- **document key** — the `path` column (source PDF path in the "
        "extraction store). Rows without a path (a handful of partial rows) "
        "fall back to the synthetic key `doc::<corpus_source>::"
        "<meta_title_raw>` (`assemble.patch._doc_keys`). *documents* and "
        "*instruments* are the same unit.",
        "- **placement** — one `(item, scale)` occurrence: the grain the "
        "corpus is stored at, one row per placement. An item attached to "
        "several scales is placed once under each, carrying that scale's "
        "membership; orphan and unscaled items have exactly one placement. "
        "Placement counts therefore exceed item counts wherever multi-scale "
        "items exist, and *placements per item* (reported with the "
        "structural distributions) is exactly that multiplier.",
        "- **item** — distinct `(document key, item_item_id)`: one physical "
        "question. `item_item_id` is assigned per document by extraction and "
        "stays constant across a multi-scale item's placements, so this pair "
        "de-duplicates them back to items.",
        "- **row** — a physical row of the parquet. Rows and placements "
        "coincide except for the carry-only shell row a failed extraction "
        "leaves behind (null `bucket`, no `item_item_id`), which is a row "
        "but no placement; `postprocess`'s `null_bucket` filter drops those, "
        "so from the postprocessed corpus onwards the two are identical. "
        "The report says *placements* when it means the item-level grain and "
        "*rows* only for physical parquet rows.",
        "- **scale (node)** — distinct `(document key, node)` over the "
        "`scale_id_path` lists: every node of the hierarchy, parents "
        "included. This is the pooled dataset's granularity (one pooled row "
        "per node).",
        "- **item-bearing scale** — distinct `(document key, scale_id)`: "
        "only scales with directly attached items; umbrella scales whose "
        "items all live in subscales, and orphan/unscaled placements (null "
        "`scale_id`), drop out.",
        "- **items per scale, subtree vs direct** — *subtree* counts an item "
        "towards every node of its `scale_id_path` (an item on `[1, 2, 4]` "
        "counts for scales 1, 2 and 4), which is what the pooled dataset's "
        "`n_items` measures; *direct* counts only the items whose immediate "
        "`scale_id` is that node. Subtree means are over every scale node, "
        "direct means over item-bearing scales only.",
        "- **levels of nesting** — an instrument's deepest scale node: 0 "
        "when the document has no scale hierarchy at all (every placement is "
        "orphan or unscaled), 1 for a flat list of scales, 2 once scales have "
        "subscales, and so on. Per document it is the maximum "
        "`len(scale_id_path)` over its placements; per node the position in "
        "that path (the `scale_depth` column) is the **hierarchy level**. "
        "Every document counts, so the scale-less ones sit in level 0.",
        "- **M / SD / Mdn / IQR** — mean, sample standard deviation "
        "(ddof=1), median, and interquartile range over the units named in "
        "the table's first column. The IQR is given as the range `Q1–Q3` "
        "(linear interpolation, pandas' default), so the middle half of the "
        "units lies between those two values; subtract them for its width. "
        "Units with zero members are included in the denominator (see the "
        "header note). Compact cells read `" + MS_LEGEND + "`.",
        "",
        "### A.2 Column glossary",
        "",
    ]
    glossary = [
        ("path", "Source PDF path; the primary document/instrument "
                 "identifier."),
        ("corpus_source", "Input dataset the row came from: `apa-psyctests` "
                          "(exploded extraction output) or an external "
                          "partial (`aligns`, `scale-hunt`, `semanticnet`)."),
        ("bucket", "Kind of placement, from extraction: `scaled` (attached "
                   "to a scale), `orphan` (in a scaled instrument but off "
                   "the scale tree), `unscaled` (instrument has no scales), "
                   "null (shell row of a failed extraction — a row, but no "
                   "placement)."),
        ("item_item_id", "Per-document item id from extraction; shared by "
                         "all rows of a multi-scale item."),
        ("item_item_text / item_text_chars", "The item text and its trimmed "
                                             "length in characters."),
        ("item_item_type", "Response format; postprocess keeps only "
                           "`rating_scale` items."),
        ("item_options", "The item's response-option labels (list); its "
                         "length is the response-option count reported "
                         "above. Null where the extraction recorded none."),
        ("item_reverse_coded", "Item is keyed opposite its scale."),
        ("scale_id / scale_name", "The item's immediate scale (null for "
                                  "orphan/unscaled rows)."),
        ("scale_id_path / scale_name_path / scale_depth",
         "Root->leaf ids/names of the scale hierarchy and the leaf's depth."),
        ("meta_title_raw", "Instrument title as extracted; part of the "
                           "fallback document key."),
        ("meta_language", "Document language (ISO 639-1) after "
                          "normalization in patch."),
        ("permissions_category", "Normalized usage-permissions class from "
                                 "the PsycTESTS record."),
        ("has_errors", "Extraction store: at least one extractor failed for "
                       "the document."),
        ("is_patched", "Row changed by at least one patch step."),
        ("public_doi / public_source / public_year",
         "Public-facing provenance derived in postprocess (PsycTESTS DOI / "
         "APA-style citation, with partial fallbacks / the citation's year, "
         "null when undated)."),
        ("pdf_match_edit_distance_norm",
         "Normalized Levenshtein distance of the item text to its best-"
         "aligning window of the PDF body text (0 = verbatim, 1 = nothing "
         "alike; similarity = 1 − this); raw strings, no normalization. NA "
         "where no PDF exists. Method and aggregation: *Verbatim fidelity* "
         "in Additional statistics."),
        ("pdf_text_chars", "Length of the PDF's text layer; below the "
                           "thin-text threshold the match distance reflects "
                           "the missing layer, not the item."),
        ("record_item_count / record_scale_count",
         "Item/scale counts from the PsycTESTS record, kept as references "
         "for the deviation flags."),
        ("flag_item_count_deviation", "Document's distinct items deviate "
                                      "from `record_item_count`; NA when no "
                                      "record count exists."),
        ("flag_scale_count_deviation", "Document's distinct leaf scales "
                                       "deviate from `record_scale_count`; "
                                       "NA when no record count exists."),
        ("flag_item_text_deviation",
         "Item is not a near-verbatim match of the PDF body "
         "(`pdf_match_edit_distance_norm` above threshold); NA when not "
         "measurable (no PDF, or thin text layer)."),
        ("flag_item_translated",
         "Item's original language (`item_language`) is not English, so the "
         "published text is a translation; NA when no language is known."),
        ("is_instrument (pooled)", "Pooled row is the instrument-level "
                                   "centroid; otherwise it is one scale "
                                   "node."),
        ("n_items (pooled)", "Distinct items pooled into the row's centroid "
                             "(de-duplicated by `item_item_id`)."),
        ("n_scales (pooled)", "Hierarchy nodes in the row's subtree, itself "
                              "included; on instrument rows, every scaled "
                              "node of the document (0 if none)."),
    ]
    lines += _table(["column", "meaning"], list(map(list, glossary)))
    return ("Appendix — identifiers and column reference", lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SECTIONS = [section_extraction, section_partials, section_combined,
            section_patch, section_postprocess, section_final,
            section_extras, section_appendix]


def run(cfg: dict, *, output_path=None) -> list[str]:
    """Build the report lines; write them to ``output_path`` when given."""
    report: list[str] = [
        "# Assembly descriptives",
        "",
        f"generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "**Counting conventions.** *documents* (= instruments): distinct "
        "`path` (source+title fallback for null paths). *scales*: every node "
        "of a document's scale hierarchy, parents included (distinct entries "
        "of the `scale_id_path` lists) — the granularity of the pooled "
        "dataset, one pooled row per node. *item-bearing scales*: only "
        "scales with directly attached items (the items' immediate "
        "`scale_id`); umbrella scales whose items all live in subscales are "
        "excluded. *items*: distinct `(document, item_item_id)` — one "
        "physical question, counted once however many scales it belongs to. "
        "*placements*: the un-de-duplicated companion of that unit — one "
        "`(item, scale)` occurrence, which is the grain the corpus is stored "
        "at, so an item shared by three scales is one item and three "
        "placements. Both are reported side by side; the word *rows* is kept "
        "for the physical parquet row count, which additionally holds the "
        "item-less shell row a failed extraction leaves behind. Full "
        "definitions of these units and of every column referenced below are "
        "in the Appendix (last section).",
        "",
        "**Distributions.** Distribution tables report `n` (the number of "
        "units the distribution is over — documents, scales or items, never "
        "rows), the mean `M`, the sample standard deviation `SD` (ddof=1; "
        "`n/a` for a single unit), the median `Mdn`, the interquartile "
        "range as `Q1–Q3` (the middle half of the units lies in that range), "
        "and min/p95/max. These counts are heavily right-skewed (a handful "
        "of instruments carry hundreds of items), so the median and its IQR "
        "are the more faithful summary and `M (SD)` the one comparable to "
        "the totals. Denominators include "
        "empty units: a document whose extraction failed counts as 0 items, "
        "an umbrella scale whose items all sit in subscales counts with its "
        "subtree total — so `M` is over all documents / all scale nodes of "
        "the stage, and multiplying `M` by the unit count in the adjacent "
        "count table reproduces the total. Compact cells read "
        f"`{MS_LEGEND}`.",
        "",
    ]
    for index, build in enumerate(SECTIONS, start=1):
        title, lines = build(cfg)
        report += [f"## {index}. {title}", ""] + lines + [""]

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out) + ".tmp")
        tmp.write_text("\n".join(report) + "\n", encoding="utf-8")
        tmp.replace(out)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview the descriptive-statistics report over the "
                    "assembled corpus. The release copy is written by "
                    "`python -m publish --step publish`.")
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument("--output", default=None,
                    help="Also write the markdown here (default: print only).")
    args = ap.parse_args()
    try:
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError as err:
        sys.exit(str(err))
    print("\n".join(run(cfg, output_path=args.output)))


if __name__ == "__main__":
    main()
