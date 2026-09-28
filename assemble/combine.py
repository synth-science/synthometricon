"""Load, tag, harmonize, and concatenate exploded extraction partials."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


_STRIP_SUFFIX = re.compile(r"-extractions?-exploded$")


def source_from_path(path: str) -> str:
    """Derive a corpus-source tag from a parquet filename.

    ``aligns-extractions-exploded.parquet`` -> ``aligns``
    ``apa-psyctests-extractions-exploded.parquet`` -> ``apa-psyctests``
    """
    stem = Path(path).stem
    return _STRIP_SUFFIX.sub("", stem)


def load_partials(
    paths: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    """Read each parquet and tag rows with a ``corpus_source`` column."""
    partials: list[tuple[str, pd.DataFrame]] = []
    for path in paths:
        source = source_from_path(path)
        df = pd.read_parquet(path)
        df.insert(0, "corpus_source", source)
        partials.append((source, df))
    return partials


def combine(partials: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate tagged partials with an outer join on columns."""
    frames = [df for _, df in partials]
    return pd.concat(frames, ignore_index=True)


def _canonicalize_type(field: pa.Field) -> pa.Field:
    """Normalize Arrow types for a consistent output schema."""
    t = field.type
    if pa.types.is_large_string(t):
        return field.with_type(pa.string())
    if pa.types.is_large_binary(t):
        return field.with_type(pa.binary())
    if pa.types.is_int32(t):
        return field.with_type(pa.int64())
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        inner = t.value_type
        if pa.types.is_int32(inner):
            return field.with_type(pa.list_(pa.int64()))
    return field


def canonicalize_schema(table: pa.Table) -> pa.Table:
    """Cast an Arrow table to a canonical schema (no large_string, etc.)."""
    target = pa.schema([_canonicalize_type(f) for f in table.schema])
    if target == table.schema:
        return table
    return table.cast(target)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet with a canonicalized schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = canonicalize_schema(table)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    import os
    os.replace(tmp, path)


def report(
    combined: pd.DataFrame,
    partials: list[tuple[str, pd.DataFrame]],
) -> str:
    """Build a human-readable combine report."""
    lines: list[str] = []
    lines.append("corpus combine report")
    lines.append("")
    lines.append("partials:")
    for source, df in partials:
        lines.append(f"  {source:30s} {len(df):>10,} rows   {len(df.columns):>3} cols")
    lines.append("")
    lines.append(
        f"combined: {len(combined):,} rows, {len(combined.columns)} columns"
    )
    sources = combined["corpus_source"].value_counts()
    lines.append("")
    lines.append("corpus_source distribution:")
    for source, count in sources.items():
        lines.append(f"  {source:30s} {count:>10,}")
    return "\n".join(lines)


def run(cfg: dict, *, report_only: bool = False) -> list[str]:
    """Stage entry point: combine partials into ``data.assemble.combined``.

    Concatenates the exploded extraction frame with the **raw** external
    partials (``data.partials``).
    """
    data_cfg = cfg.get("data", {})
    asm_cfg = data_cfg.get("assemble", {})

    extraction_exploded = asm_cfg.get("exploded")
    if not extraction_exploded:
        sys.exit("data.assemble.exploded must be set")

    paths = [extraction_exploded] + list(data_cfg.get("partials", []))
    output_path = Path(asm_cfg.get("combined", "./data/corpus.parquet"))

    for p in paths:
        if not Path(p).exists():
            sys.exit(f"partial not found: {p}")

    partials = load_partials(paths)
    combined = combine(partials)

    lines = [report(combined, partials)]

    if not report_only:
        write_parquet(combined, output_path)
        lines.append(f"\nwrote {output_path}")

    return lines
