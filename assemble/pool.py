"""Pool stage: aggregate ``embedded`` item rows into per-scale and per-instrument rows (``pooled``).

See docs/assemble-encode-pool.md for grouping, pooling and carried-column semantics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .combine import write_parquet
from .encode import SCALE_NAME_COL, scale_names_path

# Document-constant; taken from the group's document row.
DOC_COLS = (
    "corpus_source", "public_doi", "doi_psyctests", "meta_title_raw",
    "meta_language", "public_year",
    "flag_item_count_deviation", "flag_scale_count_deviation",
)

# Node-constant but varies across a document's scales; taken from the node's own row
# (document row for parents and instrument rows).
SCALE_COLS = ("version",)

# OR-ed over the group's pooled items by ``_any_flag``.
ITEM_FLAG_COLS = ("flag_item_text_deviation", "flag_item_translated")

REQUIRED_COLS = DOC_COLS + SCALE_COLS + ITEM_FLAG_COLS

# Declared, not copied: parquet read-back flattens nullable Int64/boolean dtypes.
CARRIED_DTYPES = {
    "public_year": "Int64",
    "flag_item_count_deviation": "boolean",
    "flag_scale_count_deviation": "boolean",
    "flag_item_text_deviation": "boolean",
    "flag_item_translated": "boolean",
}


def detect_models(columns, kind: str) -> dict[str, str]:
    """{model_name: column} for ``{kind}_embedding_*`` columns."""
    prefix = f"{kind}_embedding_"
    return {c[len(prefix):]: c for c in columns if c.startswith(prefix)}


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Row-wise unit vectors (float64); all-zero rows stay zero."""
    v = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 0)


def keyed_centroid(vectors: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    """Mean of unit-normalized rows with ``reverse`` rows negated (±1-weighted sum score).

    Not renormalized. See docs/assemble-encode-pool.md#keyed-centroid.
    """
    reverse = np.asarray(reverse, dtype=bool)
    signs = np.where(reverse, -1.0, 1.0)
    return (_unit_rows(vectors) * signs[:, None]).mean(axis=0)


def keying_disagreement(vectors: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    """(n,) bool: keyed item points against the keyed leave-one-out mean of the rest.

    Diagnostic only; see docs/assemble-encode-pool.md#keying-disagreement.
    """
    reverse = np.asarray(reverse, dtype=bool)
    keyed = _unit_rows(vectors) * np.where(reverse, -1.0, 1.0)[:, None]
    n = len(keyed)
    if n < 2:
        return np.zeros(n, dtype=bool)
    loo = (keyed.sum(axis=0) - keyed) / (n - 1)
    return (keyed * loo).sum(axis=1) < 0


def _is_missing(v) -> bool:
    return v is None or v is pd.NA or (np.isscalar(v) and pd.isna(v))


def _any_flag(cells, rows):
    """Three-valued OR: True if any flagged, False if any checked, else None (NA = not checkable)."""
    checked = False
    for i in rows:
        v = cells[i]
        if _is_missing(v):
            continue
        if bool(v):
            return True
        checked = True
    return False if checked else None


def _ms(values) -> str:
    """``M (SD), Mdn [IQR q1-q3]`` summary for the stage report (IQR is the range)."""
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return "n/a"
    sd = f"{s.std(ddof=1):,.2f}" if len(s) > 1 else "n/a"
    q1, med, q3 = (float(v) for v in s.quantile([0.25, 0.5, 0.75]))
    return (f"M {s.mean():,.2f} (SD {sd}), "
            f"Mdn {med:,.2f} [IQR {q1:,.2f}-{q3:,.2f}]")


def load_scale_names(embedded_path, scale_models: dict[str, str]) -> dict[str, dict]:
    """{model: {name: vector}} from encode's scale-name sidecar; empty when it is absent."""
    sidecar = scale_names_path(embedded_path)
    if not scale_models or not sidecar.exists():
        return {}
    tbl = pd.read_parquet(sidecar)
    out = {}
    for m, col in scale_models.items():
        if col in tbl.columns:
            out[m] = {t: v for t, v in zip(tbl[SCALE_NAME_COL], tbl[col])
                      if isinstance(t, str) and v is not None}
    return out


def node_vectors(cells, keys, names, sid_row, lookup) -> dict:
    """Own-name vector per scale node: its direct row's cell, else its name from ``lookup``.

    Parent-only nodes have no direct row; their name comes from ``scale_name_path``.
    """
    out = {}
    for key in keys:
        own = sid_row.get(key)
        v = cells[own] if own is not None else None
        if v is None and lookup:
            name = names[key][0]
            if isinstance(name, str):
                v = lookup.get(name)
        out[key] = v
    return out


def build_groups(df: pd.DataFrame):
    """Single pass building instrument and scale-node groups.

    Returns ``(inst, scales, names, doc_row, sid_row)``: group dicts per ``path`` and
    ``(path, scale_id)``, ``(name, depth)`` per node, and first row per document / direct scale.
    """
    paths = df["path"].to_numpy(dtype=object)
    buckets = df["bucket"].to_numpy(dtype=object)
    id_paths = df["scale_id_path"].to_numpy(dtype=object)
    name_paths = df["scale_name_path"].to_numpy(dtype=object)
    item_ids = df["item_item_id"].to_numpy(dtype=object)
    revs = df["item_reverse_coded"].to_numpy(dtype=object)

    def new_group():
        return {"rows": [], "revs": [], "seen": set(),
                "subtree": set(), "sids": set()}

    inst: dict = {}
    scales: dict = {}
    names: dict = {}
    doc_row: dict = {}
    sid_row: dict = {}
    for i in range(len(df)):
        p = None if _is_missing(paths[i]) else paths[i]
        doc_row.setdefault(p, i)
        rev = (not _is_missing(revs[i])) and bool(revs[i])
        item_key = i if _is_missing(item_ids[i]) else item_ids[i]
        g = inst.setdefault(p, new_group())
        if item_key not in g["seen"]:
            g["seen"].add(item_key)
            g["rows"].append(i)
            g["revs"].append(rev)
        if buckets[i] != "scaled" or _is_missing(id_paths[i]):
            continue
        ip = [int(x) for x in id_paths[i]]
        np_ = name_paths[i]
        g["subtree"].update(ip)
        g["sids"].add(ip[-1])
        sid_row.setdefault((p, ip[-1]), i)
        for pos, node in enumerate(ip):
            names.setdefault(
                (p, node),
                (np_[pos] if np_ is not None and pos < len(np_) else None,
                 pos + 1))
            sg = scales.setdefault((p, node), new_group())
            if item_key not in sg["seen"]:
                sg["seen"].add(item_key)
                sg["rows"].append(i)
                sg["revs"].append(rev)
            sg["subtree"].update(ip[pos:])
            sg["sids"].add(ip[-1])
    return inst, scales, names, doc_row, sid_row


def run(cfg: dict, *, report_only: bool = False,
        input_path=None, output_path=None) -> list[str]:
    """Stage entry point: ``data.assemble.embedded`` → ``data.assemble.pooled``."""
    asm_cfg = cfg.get("data", {}).get("assemble", {})
    inp = Path(input_path or asm_cfg.get("embedded", ""))
    out = Path(output_path or asm_cfg.get("pooled", ""))
    if str(inp) == "." or str(out) == ".":
        sys.exit("data.assemble.embedded and data.assemble.pooled must be set")
    if not inp.exists():
        sys.exit(f"input not found: {inp}")

    report = ["corpus pool report", f"input: {inp}"]
    df = pd.read_parquet(inp)
    report.append(f"{len(df):,} rows, {len(df.columns)} columns")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"input is missing carried columns: {', '.join(missing)}. "
                 "The input was built from an incomplete upstream run — "
                 "rerun patch (all steps), postprocess and encode before "
                 "pooling.")

    item_models = detect_models(df.columns, "item")
    scale_models = detect_models(df.columns, "scale")
    instr_models = detect_models(df.columns, "instrument")
    report.append(f"item models: {', '.join(item_models) or 'none'}; "
                  f"scale models: {', '.join(scale_models) or 'none'}")
    if not item_models and not scale_models:
        sys.exit("no *_embedding_* columns found in input")

    inst, scales, names, doc_row, sid_row = build_groups(df)
    report.append(f"{len(inst):,} instrument groups, "
                  f"{len(scales):,} scale-node groups "
                  f"({len(sid_row):,} with direct item rows)")
    if report_only:
        return report

    name_vecs = load_scale_names(inp, scale_models)
    if scale_models and not name_vecs:
        report.append(f"WARNING: no scale-name sidecar ({scale_names_path(inp)}) — "
                      "parent-only nodes get no name embedding and their names "
                      "are missing from scale_pooled_*; rerun encode")

    # (matrix, row -> matrix position) per item model.
    mats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m, col in item_models.items():
        has = df[col].notna().to_numpy()
        mat = (np.stack(df.loc[has, col].to_numpy())
               if has.any() else np.zeros((0, 0), dtype=np.float32))
        pos = np.full(len(df), -1)
        pos[np.flatnonzero(has)] = np.arange(int(has.sum()))
        mats[m] = (mat, pos)
    scale_cells = {m: df[col].to_numpy(dtype=object)
                   for m, col in scale_models.items()}
    node_vecs = {m: node_vectors(cells, scales, names, sid_row, name_vecs.get(m))
                 for m, cells in scale_cells.items()}
    for m, vecs in node_vecs.items():
        named = [k for k in scales if isinstance(names[k][0], str)]
        parent_only = [k for k in named if k not in sid_row]
        report.append(
            f"scale_embedding_{m}: {sum(vecs[k] is not None for k in scales):,} of "
            f"{len(scales):,} nodes ({len(named):,} named); parent-only named "
            f"nodes with an own-name vector: "
            f"{sum(vecs[k] is not None for k in parent_only):,} of {len(parent_only):,}")
    instr_cells = {m: df[col].to_numpy(dtype=object)
                   for m, col in instr_models.items()}
    doc_meta = {c: df[c].to_numpy(dtype=object) for c in DOC_COLS}
    scale_meta = {c: df[c].to_numpy(dtype=object) for c in SCALE_COLS}
    item_flags = {c: df[c].astype("boolean").to_numpy(dtype=object)
                  for c in ITEM_FLAG_COLS}

    def pool_items(g) -> dict:
        cells = {}
        rows = np.asarray(g["rows"])
        rev = np.asarray(g["revs"], dtype=bool)
        for m, (mat, pos) in mats.items():
            p = pos[rows]
            valid = p >= 0
            if valid.any():
                sub, r = mat[p[valid]], rev[valid]
                cells[f"item_pooled_{m}"] = (
                    keyed_centroid(sub, r).astype(np.float32))
                cells[f"keying_disagreements_{m}"] = int(
                    keying_disagreement(sub, r).sum())
            else:
                cells[f"item_pooled_{m}"] = None
                cells[f"keying_disagreements_{m}"] = None
        return cells

    def pool_scales(path, nodes, di) -> dict:
        """Mean of the own-name vectors of ``nodes`` (each once) and the instrument title's."""
        cells = {}
        for m, vec_of in node_vecs.items():
            vecs = [vec_of[(path, s)] for s in sorted(nodes)
                    if vec_of.get((path, s)) is not None]
            instr_arr = instr_cells.get(m)
            if instr_arr is not None and instr_arr[di] is not None:
                vecs.append(instr_arr[di])
            cells[f"scale_pooled_{m}"] = (
                np.mean(vecs, axis=0).astype(np.float32) if vecs else None)
        return cells

    records: list[dict] = []
    for (path, node), g in scales.items():
        name, depth = names[(path, node)]
        di = doc_row[path]
        own = sid_row.get((path, node))
        rec = {"path": path,
               **{c: doc_meta[c][di] for c in DOC_COLS},
               **{c: scale_meta[c][di if own is None else own]
                  for c in SCALE_COLS},
               **{c: _any_flag(cells, g["rows"])
                  for c, cells in item_flags.items()},
               "is_instrument": False, "scale_id": node,
               "scale_name": name, "scale_depth": depth,
               "n_items": len(g["rows"]), "n_scales": len(g["subtree"])}
        rec.update(pool_items(g))
        # The node itself and every descendant node, intermediate ones included.
        rec.update(pool_scales(path, g["subtree"], di))
        for m, vec_of in node_vecs.items():
            rec[f"scale_embedding_{m}"] = vec_of[(path, node)]
        for m in instr_cells:
            rec[f"instrument_embedding_{m}"] = None
        records.append(rec)
    for path, g in inst.items():
        di = doc_row[path]
        rec = {"path": path,
               **{c: doc_meta[c][di] for c in DOC_COLS},
               **{c: scale_meta[c][di] for c in SCALE_COLS},
               **{c: _any_flag(cells, g["rows"])
                  for c, cells in item_flags.items()},
               "is_instrument": True, "scale_id": None,
               "scale_name": None, "scale_depth": None,
               "n_items": len(g["rows"]), "n_scales": len(g["subtree"])}
        rec.update(pool_items(g))
        # Instrument rows keep their label: the item-bearing scales' names and the title.
        rec.update(pool_scales(path, g["sids"], di))
        for m in scale_cells:
            rec[f"scale_embedding_{m}"] = None
        for m, cell_arr in instr_cells.items():
            rec[f"instrument_embedding_{m}"] = cell_arr[di]
        records.append(rec)

    pooled = pd.DataFrame.from_records(records)
    for col, dtype in CARRIED_DTYPES.items():
        pooled[col] = pooled[col].astype(dtype)
    inst_rows = pooled["is_instrument"].astype(bool)
    report.append("")
    for label, mask, col in (
            ("items per instrument", inst_rows, "n_items"),
            ("scales per instrument", inst_rows, "n_scales"),
            ("items per scale (subtree)", ~inst_rows, "n_items"),
            ("nodes per scale subtree", ~inst_rows, "n_scales")):
        report.append(f"{label}: {_ms(pooled.loc[mask, col])}")
    report.append("")
    for c in REQUIRED_COLS:
        report.append(f"{c}: {int(pooled[c].isna().sum()):,} null")
    for c in ITEM_FLAG_COLS:
        flagged = pooled[c].fillna(False).astype(bool)
        report.append(f"{c}: {int(flagged.sum()):,} of "
                      f"{int(pooled[c].notna().sum()):,} checkable rows "
                      "flagged (any pooled item flagged)")
    report.append("")
    for kind in ("item_pooled", "scale_pooled"):
        for c in [c for c in pooled.columns if c.startswith(kind + "_")]:
            report.append(f"{c}: {int(pooled[c].isna().sum()):,} null")
    for m in mats:
        c = f"keying_disagreements_{m}"
        d = pooled.loc[~inst_rows, c].dropna().astype(int)
        report.append(
            f"{c} (scale rows): {int(d.sum()):,} flagged items in "
            f"{int((d > 0).sum()):,} of {len(d):,} scale rows "
            f"({(d > 0).mean():.1%}) — encoder direction disagrees with "
            "documented keying; diagnostic only, pooling trusts the keying")
    report.append("")
    report.append(f"final: {len(pooled):,} rows "
                  f"({len(scales):,} scale + {len(inst):,} instrument), "
                  f"{len(pooled.columns)} columns")
    write_parquet(pooled, out)
    report.append(f"wrote {out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone pool stage: aggregate the embedded corpus "
                    "into per-scale and per-instrument pooled rows.",
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument("--input", default=None,
                    help="Input parquet (default: data.assemble.embedded).")
    ap.add_argument("--output", default=None,
                    help="Output parquet (default: data.assemble.pooled).")
    ap.add_argument("--report-only", action="store_true",
                    help="Print group counts without pooling or writing.")
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
