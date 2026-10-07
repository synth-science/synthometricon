"""Publish step: reduce the pooled corpus to the release column contract,
Fernet-encrypt it, and write the release directory (see docs/architecture-publish.md).

Consumers without this package decrypt with::

    from cryptography.fernet import Fernet
    import io, pandas as pd
    raw = Fernet(key).decrypt(Path(enc_path).read_bytes())
    df = pd.read_parquet(io.BytesIO(raw))
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cryptography.fernet import Fernet

from assemble import report as descriptives
from assemble.combine import canonicalize_schema
from assemble.stats import mirror_report

from . import keys, manifest, paths
from .paths import release_path

# Scalar release columns; all required. Must stay documented in dataset_card.md (tested).
RELEASE_COLS = (
    "version", "meta_language", "public_year",
    "flag_item_count_deviation", "flag_scale_count_deviation",
    "flag_item_text_deviation", "flag_item_translated",
    "path", "corpus_source", "public_doi", "doi_psyctests", "meta_title_raw",
    "is_instrument", "scale_id", "scale_name", "scale_depth",
    "n_items", "n_scales",
)
# Model-suffixed columns, matched by prefix; order here is the release order.
MODEL_COL_PREFIXES = (
    "item_pooled_", "keying_disagreements_", "scale_pooled_",
    "scale_embedding_", "instrument_embedding_",
)
# At least one must match, else the input is not a pooled corpus.
EMBEDDING_PREFIXES = (
    "item_pooled_", "scale_pooled_", "scale_embedding_",
    "instrument_embedding_",
)


def read_release(path=None, key_path=None, config: str = "config.yaml",
                 columns=None) -> pd.DataFrame:
    """Decrypt the release artifact into a DataFrame (``columns`` projects at the read).

    Defaults come from config; never generates a key."""
    cfg = {}
    if path is None or key_path is None:
        with open(config) as fh:
            cfg = yaml.safe_load(fh) or {}
    path = path or release_path(cfg)
    if not path:
        raise ValueError("release path must be given or set in config "
                         "(data.publish)")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"release artifact not found: {path}")
    key, _ = keys.resolve_key(cfg, [], allow_create=False,
                              key_path=key_path or paths.key_path(path))
    raw = Fernet(key).decrypt(path.read_bytes())
    return pd.read_parquet(io.BytesIO(raw), columns=columns)


def reduce_columns(df: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """Select the release column contract; drop and report anything else."""
    missing = [c for c in RELEASE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"input is missing release columns: {', '.join(missing)}. "
                 "Rerun the pool stage before releasing.")
    model_cols = [c for prefix in MODEL_COL_PREFIXES
                  for c in df.columns if c.startswith(prefix)]
    if not any(c.startswith(EMBEDDING_PREFIXES) for c in model_cols):
        sys.exit("no embedding columns found in input — not a pooled parquet?")
    keep = list(RELEASE_COLS) + model_cols
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        report.append(f"dropped non-release columns: {', '.join(dropped)}")
    return df[keep]


def mutate(df: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """Release-only derived variables (none yet; ``n_items`` comes from pool)."""
    null_counts = int(df["n_items"].isna().sum())
    if null_counts:
        report.append(f"WARNING: {null_counts:,} rows with null n_items")
    return df


def serialize_frame(df: pd.DataFrame) -> bytes:
    """Serialize ``df`` to parquet bytes in memory."""
    table = canonicalize_schema(pa.Table.from_pandas(df, preserve_index=False))
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def encrypt_frame(df: pd.DataFrame, key: bytes) -> bytes:
    """Serialize ``df`` to parquet bytes in memory and Fernet-encrypt them."""
    return Fernet(key).encrypt(serialize_frame(df))


def _check_disk_space(out: Path, needed: int, report: list[str]) -> None:
    """Exit early if there is no room for the tmp copy beside the existing artifact."""
    try:
        free = shutil.disk_usage(out.parent).free
    except OSError:
        return
    if free < needed:
        sys.exit(f"not enough free space in {out.parent}: need ~"
                 f"{needed / 1e9:,.1f} GB for the temporary copy, "
                 f"{free / 1e9:,.1f} GB available")
    report.append(f"disk: {free / 1e9:,.1f} GB free in {out.parent}")


def run(cfg: dict, *, report_only: bool = False, input_path=None,
        output_path=None, key_path=None) -> list[str]:
    """Step entry point: encrypt ``data.assemble.pooled`` into the release directory."""
    asm_cfg = cfg.get("data", {}).get("assemble", {}) or {}
    inp = Path(input_path or asm_cfg.get("pooled", ""))
    out = Path(output_path or release_path(cfg) or "")
    if str(inp) == "." or str(out) == ".":
        sys.exit("data.assemble.pooled and data.publish must be set")
    if not inp.exists():
        sys.exit(f"input not found: {inp}")
    key_path = key_path or paths.key_path(out)
    md_path, readme = paths.report_path(out), paths.readme_path(out)

    report = ["corpus publish report", f"input: {inp}", f"output: {out}"]
    df = pd.read_parquet(inp)
    report.append(f"{len(df):,} rows, {len(df.columns)} columns")

    df = reduce_columns(df, report)
    df = mutate(df, report)

    n_inst = int(df["is_instrument"].sum())
    report.append(f"{n_inst:,} instrument rows, {len(df) - n_inst:,} scale "
                  f"rows, {len(df.columns)} release columns")

    if report_only:
        report.append(keys.describe_key(cfg, report, key_path))
        report.append(f"would write: {md_path.name}, {readme.name} "
                      f"(in {out.parent})")
        return report

    key, _ = keys.resolve_key(cfg, report, allow_create=True,
                              key_path=key_path)

    plaintext = serialize_frame(df)
    plaintext_sha = hashlib.sha256(plaintext).hexdigest()
    token = Fernet(key).encrypt(plaintext)
    del plaintext

    out.parent.mkdir(parents=True, exist_ok=True)
    _check_disk_space(out, len(token) + (1 << 30), report)
    tmp = Path(str(out) + ".tmp")
    tmp.write_bytes(token)
    os.replace(tmp, out)
    report.append(f"wrote {out} ({len(token) / 1e6:,.0f} MB encrypted)")

    # Round-trip check: the only guard against shipping an undecryptable blob.
    raw = Fernet(key).decrypt(out.read_bytes())
    schema = pq.read_schema(io.BytesIO(raw))
    rows = pq.ParquetFile(io.BytesIO(raw)).metadata.num_rows
    if rows != len(df) or len(schema) != len(df.columns):
        sys.exit(f"round-trip mismatch: wrote {len(df):,}x{len(df.columns)}, "
                 f"read back {rows:,}x{len(schema)}")
    if hashlib.sha256(raw).hexdigest() != plaintext_sha:
        sys.exit("round-trip mismatch: decrypted bytes differ from what was "
                 "encrypted")
    report.append(f"round-trip verified: {rows:,} rows, "
                  f"{len(schema)} columns decrypt cleanly")
    plaintext_bytes = len(raw)
    del raw

    payload = manifest.build_manifest(
        df, cfg, out, key,
        token_bytes=len(token), token_sha256=hashlib.sha256(token).hexdigest(),
        plaintext_bytes=plaintext_bytes, plaintext_sha256=plaintext_sha,
        source_artifact=inp)
    side = manifest.write_manifest(out, payload)
    report.append(f"wrote manifest {side.name} "
                  f"(plaintext sha256 {plaintext_sha[:12]})")

    # Built here so the report can never drift from the artifact beside it.
    descriptives.run(cfg, output_path=md_path)
    report.append(f"wrote descriptives {md_path.name}")
    if mirrored := mirror_report(md_path, cfg):
        report.append(f"mirrored descriptives to {mirrored}")
    write_readme(readme, out, payload)
    report.append(f"wrote {readme.name}")
    return report


def write_readme(readme: Path, artifact: Path, payload: dict) -> None:
    """Write the release directory's short ``README.md`` (build time + file guide)."""
    art, frame = payload["artifact"], payload["frame"]
    name = artifact.name
    text = f"""# Release data

Release frame built: **{payload["generated"]}** (UTC), pipeline commit
`{payload["pipeline"].get("commit")}`. {frame["rows"]:,} rows
({frame["instrument_rows"]:,} instrument, {frame["scale_rows"]:,} scale) x
{frame["columns"]} columns.

Everything here is written together by `python -m publish --step publish`;
re-run it rather than editing files by hand.

| file | contents |
|---|---|
| `{name}` | The release frame: parquet bytes, Fernet-encrypted (not readable as parquet). Decrypt with `publish.publish.read_release()`. |
| `{manifest.manifest_path(artifact).name}` | Manifest: shape, column types, counts, hashes, key fingerprint. Uploaded alongside the artifact. |
| `{paths.key_path(artifact).name}` | The Fernet key (fingerprint `{art["key_fingerprint"]}`). **Private** — never commit, print or upload it; losing it locks out every consumer. |
| `{paths.report_path(artifact).name}` | Descriptive statistics of the corpus behind this release (for the paper). |
| `README.md` | This file. |
"""
    tmp = Path(str(readme) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, readme)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone publish step: reduce the pooled corpus to "
                    "the release column contract, Fernet-encrypt it and write "
                    "the release directory.",
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument("--input", default=None,
                    help="Input parquet (default: data.assemble.pooled).")
    ap.add_argument("--output", default=None,
                    help="Output file (default: data.publish).")
    ap.add_argument("--key-path", default=None,
                    help="Fernet key file (default: <output>.key). "
                         "Overridden by $SYNTHOMETRICON_RELEASE_KEY when set.")
    ap.add_argument("--report-only", action="store_true",
                    help="Report counts without encrypting or writing. Never "
                         "generates a key.")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    lines = run(cfg, report_only=args.report_only, input_path=args.input,
                output_path=args.output, key_path=args.key_path)
    print("\n".join(lines))
    if args.report_only:
        print("(report-only: nothing written)")


if __name__ == "__main__":
    main()
