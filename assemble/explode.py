"""Explode stage: document-keyed extraction store -> one row per ``(item, scale)`` occurrence.

Column set is derived from the Pydantic models; see docs/architecture-assemble.md#explode.
"""
from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from tqdm import tqdm

from extraction.models import Instrument, Meta, ScaledItem

NESTED_CONTENT_SUFFIX = "_extractor_content"
OUTPUT_COL = "extraction_output"

# Explode columns describing scale membership; null for orphan/unscaled/null rows.
_SCALE_COLS = (
    "scale_id",
    "scale_name",
    "scale_construct_name",
    "scale_id_path",
    "scale_name_path",
    "scale_depth",
)


def _carry_columns(columns) -> list[str]:
    """Scalar columns to carry: all but ``*_extractor_content`` and ``extraction_output``."""
    return [
        c
        for c in columns
        if not c.endswith(NESTED_CONTENT_SUFFIX) and c != OUTPUT_COL
    ]


def _output_columns(carry_cols: list[str]) -> list[str]:
    """Ordered exploded column set (``meta_*`` from ``Meta``, ``item_*`` from ``ScaledItem``).

    Gives every streamed batch the same layout, so the output parquet has one schema.
    """
    meta_cols = [f"meta_{f}" for f in Meta.model_fields]
    item_cols = [f"item_{f}" for f in ScaledItem.model_fields]
    return list(carry_cols) + ["bucket", *_SCALE_COLS] + meta_cols + item_cols


def _item_dict(item: BaseModel) -> dict[str, Any]:
    """Dump ``Item``/``ScaledItem`` to aligned dicts (``id`` -> ``item_id``, ``reverse_coded`` default None)."""
    d = item.model_dump(mode="json")
    if "id" in d and "item_id" not in d:
        d["item_id"] = d.pop("id")
    d.setdefault("reverse_coded", None)
    return d


def _prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {prefix + k: v for k, v in d.items()}


def _explode_row(row: dict[str, Any], carry_cols: list[str]) -> Iterator[dict[str, Any]]:
    """Yield one record per ``(item, scale)`` occurrence and orphan/unscaled item of a document.

    A null ``extraction_output`` yields a single carry-only shell record.
    """
    carry = {c: row.get(c) for c in carry_cols}
    raw = row.get(OUTPUT_COL)

    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        yield {**carry, "bucket": None, **{c: None for c in _SCALE_COLS}}
        return

    inst = Instrument.model_validate_json(raw)
    meta_fields = (
        _prefix(inst.meta.model_dump(mode="json"), "meta_") if inst.meta else {}
    )

    def walk(node, id_path: list, name_path: list) -> Iterator[dict[str, Any]]:
        id_path = id_path + [node.id]
        name_path = name_path + [node.scale_name]
        for si in node.items:
            yield {
                **carry,
                **meta_fields,
                "bucket": "scaled",
                "scale_id": node.id,
                "scale_name": node.scale_name,
                "scale_construct_name": node.construct_name,
                "scale_id_path": id_path,
                "scale_name_path": name_path,
                "scale_depth": len(id_path),
                **_prefix(_item_dict(si), "item_"),
            }
        for sub in node.subscales:
            yield from walk(sub, id_path, name_path)

    for top in inst.scales:
        yield from walk(top, [], [])

    null_scale = {c: None for c in _SCALE_COLS}
    for bucket, items in (("orphan", inst.orphan_items),
                          ("unscaled", inst.unscaled_items)):
        for it in items:
            yield {
                **carry,
                **meta_fields,
                "bucket": bucket,
                **null_scale,
                **_prefix(_item_dict(it), "item_"),
            }


def explode_extractions(df: pd.DataFrame, *, progress: bool = False) -> pd.DataFrame:
    """Explode a document-keyed extraction frame in memory into one row per item occurrence."""
    carry_cols = _carry_columns(df.columns)
    rows = df.to_dict(orient="records")
    if progress:
        rows = tqdm(rows, unit="doc", desc="exploding")
    records: list[dict[str, Any]] = []
    for row in rows:
        records.extend(_explode_row(row, carry_cols))
    return pd.DataFrame(records)


def _read_columns(path: Path | str) -> tuple[pq.ParquetFile, list[str], list[str]]:
    """Return ``(parquet_file, carry_cols, read_cols)``; ``read_cols`` skips the nested content columns."""
    pf = pq.ParquetFile(path)
    carry_cols = _carry_columns(pf.schema_arrow.names)
    read_cols = carry_cols + ([OUTPUT_COL] if OUTPUT_COL in pf.schema_arrow.names else [])
    return pf, carry_cols, read_cols


def _iter_exploded_batches(
    pf: pq.ParquetFile,
    carry_cols: list[str],
    read_cols: list[str],
    master_cols: list[str],
    *,
    batch_size: int,
    progress: bool,
) -> Iterator[pd.DataFrame]:
    """Yield one exploded frame per input batch, reindexed to ``master_cols``; empty batches skipped."""
    bar = tqdm(total=pf.metadata.num_rows, unit="doc", desc="exploding") if progress else None
    try:
        for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            rows = batch.to_pylist()
            records: list[dict[str, Any]] = []
            for row in rows:
                records.extend(_explode_row(row, carry_cols))
            if bar is not None:
                bar.update(len(rows))
            if records:
                yield pd.DataFrame(records).reindex(columns=master_cols)
    finally:
        if bar is not None:
            bar.close()


def explode_extractions_to_parquet(
    in_path: Path | str,
    out_path: Path | str,
    *,
    batch_size: int = 256,
    progress: bool = True,
) -> Path:
    """Stream-explode ``in_path`` into ``out_path`` with memory bounded by one batch; returns ``out_path``."""
    out_path = Path(out_path)
    pf, carry_cols, read_cols = _read_columns(in_path)
    master_cols = _output_columns(carry_cols)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    try:
        for bdf in _iter_exploded_batches(
            pf, carry_cols, read_cols, master_cols,
            batch_size=batch_size, progress=progress,
        ):
            if writer is None:
                table = pa.Table.from_pandas(bdf, preserve_index=False)
                # All-null columns in the first batch infer as null type; widen to string.
                schema = pa.schema([
                    f.with_type(pa.string()) if pa.types.is_null(f.type) else f
                    for f in table.schema
                ])
                table = table.cast(schema)
                writer = pq.ParquetWriter(out_path, schema)
            else:
                table = pa.Table.from_pandas(bdf, schema=schema, preserve_index=False)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:  # no rows at all — still emit an empty, well-formed parquet
        pd.DataFrame(columns=master_cols).to_parquet(out_path)
    return out_path


def explode_extractions_file(
    path: Path | str,
    *,
    out_path: Path | str | None = None,
    batch_size: int = 256,
    progress: bool = True,
) -> pd.DataFrame | Path:
    """Explode the parquet at ``path``: stream to ``out_path`` if given, else return the frame."""
    if out_path is not None:
        return explode_extractions_to_parquet(
            path, out_path, batch_size=batch_size, progress=progress,
        )

    pf, carry_cols, read_cols = _read_columns(path)
    master_cols = _output_columns(carry_cols)
    batches = list(_iter_exploded_batches(
        pf, carry_cols, read_cols, master_cols,
        batch_size=batch_size, progress=progress,
    ))
    if not batches:
        return pd.DataFrame(columns=master_cols)
    return pd.concat(batches, ignore_index=True)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(cfg: dict, *, report_only: bool = False) -> list[str]:
    """Stage entry point: explode ``data.extractions`` -> ``data.assemble.exploded``."""
    data_cfg = cfg.get("data", {})
    in_path = data_cfg.get("extractions")
    out_path = data_cfg.get("assemble", {}).get("exploded")
    if not in_path or not out_path:
        sys.exit("data.extractions and data.assemble.exploded must be set")
    in_path, out_path = Path(in_path), Path(out_path)
    if not in_path.exists():
        sys.exit(f"extraction parquet not found: {in_path}")

    report: list[str] = []
    num_rows = pq.ParquetFile(in_path).metadata.num_rows
    if report_only:
        report += [
            f"explode: would explode {num_rows} documents",
            f"  from {in_path}",
            f"  to   {out_path}",
            "  (skipped in report-only)",
        ]
        return report

    report.append(f"exploding {num_rows} documents from {in_path}")
    explode_extractions_to_parquet(in_path, out_path)
    report.append(f"wrote {out_path}")
    return report
