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


def encode_column(series: pd.Series, model, batch_size: int) -> pd.Series:
    """Encode each unique text once; null/blank texts map to None."""
    uniq = _unique_texts(series)
    if not uniq:
        return pd.Series([None] * len(series), index=series.index, dtype=object)
    vectors = model.encode(uniq, batch_size=batch_size,
                           convert_to_numpy=True, show_progress_bar=True)
    lookup = {t: v.astype(np.float32) for t, v in zip(uniq, vectors)}
    return series.map(lambda t: lookup.get(t) if isinstance(t, str) else None)


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
    for path, src, col in targets:
        series = df[src]
        n_uniq = len(_unique_texts(series))
        report.append(f"--- {col} ---")
        report.append(f"model: {path}")
        report.append(f"source: {src}, {n_uniq:,} unique texts, "
                      f"{int(series.isna().sum()):,} null rows")
        if report_only:
            continue
        if path not in models:
            from sentence_transformers import SentenceTransformer
            models.clear()  # free the previous model before loading the next
            models[path] = SentenceTransformer(str(path))
        df[col] = encode_column(series, models[path], batch_size)
        dim = next((len(v) for v in df[col] if v is not None), 0)
        report.append(f"dim: {dim}, {int(df[col].isna().sum()):,} null embeddings")
    models.clear()

    report.append("")
    report.append(f"final: {len(df):,} rows, {len(df.columns)} columns")
    if not report_only:
        write_parquet(df, out)
        report.append(f"wrote {out}")
    return report


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
