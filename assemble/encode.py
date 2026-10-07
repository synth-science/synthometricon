"""Encode stage: add sentence-transformer embedding columns (``postprocessed`` → ``embedded``).

Row-preserving; models come from the ``encode:`` config block. See docs/assemble-encode-pool.md.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .combine import write_parquet

DEFAULT_BATCH_SIZE = 256

# Sidecar of the embedded parquet: one row per distinct scale-node name (parents included),
# one vector column per scale model. Pool reads it for nodes without direct rows.
SCALE_NAMES_SUFFIX = ".scale-names.parquet"
SCALE_NAME_COL = "scale_name"
SCALE_NAME_PATH_COL = "scale_name_path"


def scale_names_path(embedded_path) -> Path:
    """Sidecar path next to the embedded parquet (``x.parquet`` -> ``x.scale-names.parquet``)."""
    p = Path(embedded_path)
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    return p.with_name(stem + SCALE_NAMES_SUFFIX)


def _model_name(name: str) -> str:
    """Sanitized column-name segment from a configured model name."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _model_entries(entries, key: str) -> list[tuple[str, str]]:
    """(model_path, sanitized_name) pairs from an ``encode.{key}`` list."""
    out: list[tuple[str, str]] = []
    for entry in entries or []:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("path"):
            sys.exit(f"encode.{key} entries must be mappings with "
                     f"'name' and 'path', got: {entry!r}")
        out.append((str(entry["path"]), _model_name(entry["name"])))
    return out


def _targets(encode_cfg: dict) -> list[tuple[str, str, str]]:
    """(model_path, source_column, output_column) triples, config order."""
    out: list[tuple[str, str, str]] = []
    for path, name in _model_entries(encode_cfg.get("item_models"), "item_models"):
        out.append((path, "item_item_text", f"item_embedding_{name}"))
    for path, name in _model_entries(encode_cfg.get("scale_models"), "scale_models"):
        out.append((path, "scale_name", f"scale_embedding_{name}"))
        out.append((path, "meta_title_raw", f"instrument_embedding_{name}"))
    return out


def _unique_texts(series: pd.Series) -> list[str]:
    """Distinct non-blank strings in ``series``."""
    texts = series.dropna()
    return sorted({t for t in texts if isinstance(t, str) and t.strip()})


def path_names(df: pd.DataFrame) -> list[str]:
    """Distinct non-blank names anywhere in ``scale_name_path`` (parent nodes included)."""
    if SCALE_NAME_PATH_COL not in df.columns:
        return []
    names: set[str] = set()
    for path in df[SCALE_NAME_PATH_COL].dropna():
        names.update(t for t in path if isinstance(t, str) and t.strip())
    return sorted(names)


def encode_texts(texts, model, batch_size: int) -> dict[str, np.ndarray]:
    """{text: float32 vector}, each distinct non-blank text encoded once (one model call)."""
    uniq = sorted({t for t in texts if isinstance(t, str) and t.strip()})
    if not uniq:
        return {}
    vectors = model.encode(uniq, batch_size=batch_size,
                           convert_to_numpy=True, show_progress_bar=True)
    return {t: v.astype(np.float32) for t, v in zip(uniq, vectors)}


def map_texts(series: pd.Series, lookup: dict) -> pd.Series:
    """Vector per row from ``lookup``; null/blank or unknown texts map to None."""
    if not lookup:
        return pd.Series([None] * len(series), index=series.index, dtype=object)
    return series.map(lambda t: lookup.get(t) if isinstance(t, str) else None)


def encode_column(series: pd.Series, model, batch_size: int) -> pd.Series:
    """Encode each unique text once; null/blank texts map to None."""
    return map_texts(series, encode_texts(_unique_texts(series), model, batch_size))


def run(cfg: dict, *, report_only: bool = False,
        input_path=None, output_path=None) -> list[str]:
    """Stage entry point: ``data.assemble.postprocessed`` → ``data.assemble.embedded``."""
    asm_cfg = cfg.get("data", {}).get("assemble", {})
    inp = Path(input_path or asm_cfg.get("postprocessed", ""))
    out = Path(output_path or asm_cfg.get("embedded", ""))
    if str(inp) == "." or str(out) == ".":
        sys.exit("data.assemble.postprocessed and data.assemble.embedded "
                 "must be set")
    if not inp.exists():
        sys.exit(f"input not found: {inp}")

    encode_cfg = cfg.get("encode", {}) or {}
    targets = _targets(encode_cfg)
    if not targets:
        sys.exit("encode.item_models / encode.scale_models must list at least "
                 "one model")
    batch_size = int(encode_cfg.get("batch_size", DEFAULT_BATCH_SIZE))

    report = ["corpus encode report", f"input: {inp}"]
    df = pd.read_parquet(inp)
    report.append(f"{len(df):,} rows, {len(df.columns)} columns")

    missing = {src for _, src, _ in targets if src not in df.columns}
    if missing:
        sys.exit(f"source columns missing from input: {', '.join(sorted(missing))}")

    models: dict[str, object] = {}
    # Parent nodes have no row of their own, so their names live only in
    # scale_name_path; they are encoded with the row names and go to the sidecar.
    extra_names = path_names(df)
    name_lookups: dict[str, dict] = {}  # scale_embedding_{m} -> {name: vector}
    for path, src, col in targets:
        series = df[src]
        texts = _unique_texts(series)
        report.append(f"--- {col} ---")
        report.append(f"model: {path}")
        report.append(f"source: {src}, {len(texts):,} unique texts, "
                      f"{int(series.isna().sum()):,} null rows")
        if src == SCALE_NAME_COL:
            only_path = sorted(set(extra_names) - set(texts))
            texts = sorted(set(texts) | set(extra_names))
            report.append(f"plus {len(only_path):,} names that occur only in "
                          f"{SCALE_NAME_PATH_COL} (parent nodes)")
        if report_only:
            continue
        if path not in models:
            from sentence_transformers import SentenceTransformer
            models.clear()  # free the previous model before loading the next
            models[path] = SentenceTransformer(str(path))
        lookup = encode_texts(texts, models[path], batch_size)
        df[col] = map_texts(series, lookup)
        if src == SCALE_NAME_COL:
            name_lookups[col] = lookup
        dim = next((len(v) for v in df[col] if v is not None), 0)
        report.append(f"dim: {dim}, {int(df[col].isna().sum()):,} null embeddings")
    models.clear()

    report.append("")
    report.append(f"final: {len(df):,} rows, {len(df.columns)} columns")
    if not report_only:
        write_parquet(df, out)
        report.append(f"wrote {out}")
        if name_lookups:
            sidecar = scale_names_path(out)
            write_parquet(scale_names_frame(name_lookups), sidecar)
            report.append(f"wrote {sidecar} "
                          f"({max(len(v) for v in name_lookups.values()):,} names)")
    return report


def scale_names_frame(name_lookups: dict[str, dict]) -> pd.DataFrame:
    """One row per distinct scale-node name, one vector column per scale model."""
    texts = sorted(set().union(*name_lookups.values()))
    frame = {SCALE_NAME_COL: texts}
    for col, lookup in name_lookups.items():
        frame[col] = [lookup.get(t) for t in texts]
    return pd.DataFrame(frame)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone encode stage: add sentence-transformer "
                    "embedding columns to the postprocessed corpus.",
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument("--input", default=None,
                    help="Input parquet (default: data.assemble.postprocessed).")
    ap.add_argument("--output", default=None,
                    help="Output parquet (default: data.assemble.embedded).")
    ap.add_argument("--report-only", action="store_true",
                    help="Print the report without encoding or writing.")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    lines = run(cfg, report_only=args.report_only,
                input_path=args.input, output_path=args.output)
    print("\n".join(lines))
    if args.report_only:
        print("(report-only: nothing written)")


if __name__ == "__main__":
    main()
