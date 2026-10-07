"""Postprocess stage: prune the patched corpus into the public-facing dataset.

First stage allowed to drop rows. Steps: filter_rows -> drop_columns -> derive ->
pdf_match -> flags; semantics in docs/assemble-postprocess.md.

Usage:
    python -m assemble --step postprocess [--report-only]
    python -m assemble.postprocess --input X --output Y --steps derive
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import yaml

from extraction.storage import PATH_COL, RETRY_SUFFIXES, USAGE_SUFFIXES

from .combine import write_parquet
from .stats import mirror_report, write_stats
from .patch import (DOI_PSYCTESTS_COL, PDF_FULL_TEXT_COL, apa_mask,
                    doi_in_text)

# Telemetry = ends with ``_extractor_{suffix}``; derived so new storage suffixes drop automatically.
TELEMETRY_SUFFIXES: tuple[str, ...] = (
    tuple(s for s, _ in USAGE_SUFFIXES) + tuple(s for s, _ in RETRY_SUFFIXES)
)

# Doc-level telemetry (mirrors storage._DOC_SUMMARY_COLS; sync-tested).
DOC_TELEMETRY_COLS: tuple[str, ...] = ("has_errors", "total_duration")

EXTRA_DROP_COLS: tuple[str, ...] = (
    "meta_doi_raw",  # superseded by the corroborated doi_psyctests
    # PDF-shape internals (page and image counts are dropped by pdf_match, which needs them)
    "meta_char_count_excl_first_page",
    # scraper QA fields (partial-only); ``version`` is deliberately kept
    "source_grade", "extraction_confidence",
    "verify_verdict", "verbatim_claimed", "retrieval_notes",
)


@dataclass
class Ctx:
    """Carries config and collects the report across steps."""

    cfg: dict                      # the ``postprocess:`` config block
    full_cfg: dict                 # the whole config
    report_only: bool = False
    report: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def log(self, line: str) -> None:
        self.report.append(line)

    def stat(self, key: str, value) -> None:
        self.stats[key] = value


# ---------------------------------------------------------------------------
# Step 1: row filters
# ---------------------------------------------------------------------------

Filter = Callable[[pd.DataFrame], pd.Series]

# One definition for filter and published column, so both see the same length.
ITEM_TEXT_CHARS_COL = "item_text_chars"

# Longer "items" are merged blocks, instruction paragraphs or vignettes.
ITEM_TEXT_CHARS_MAX = 250


def _item_text_chars(df: pd.DataFrame) -> pd.Series:
    """Trimmed length of each item stem; NA where the row carries no stem."""
    if "item_item_text" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    return df["item_item_text"].map(
        lambda v: len(v.strip()) if isinstance(v, str) else pd.NA
    ).astype("Int64")


def _filter_extraction_error(df: pd.DataFrame) -> pd.Series:
    if "has_errors" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["has_errors"].fillna(False).astype(bool)


def _filter_null_bucket(df: pd.DataFrame) -> pd.Series:
    # Explode's shell rows; overlaps extraction_error, logged separately to expose drift.
    if "bucket" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["bucket"].isna()


def _filter_non_rating_scale(df: pd.DataFrame) -> pd.Series:
    # Null type is a missing field, not evidence: kept.
    if "item_item_type" not in df.columns:
        return pd.Series(False, index=df.index)
    t = df["item_item_type"]
    return (t.notna() & ~t.eq("rating_scale")).astype(bool)


def _filter_empty_item_text(df: pd.DataFrame) -> pd.Series:
    if "item_item_text" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["item_item_text"].map(
        lambda v: not isinstance(v, str) or not v.strip()).astype(bool)


def _filter_long_item_text(df: pd.DataFrame) -> pd.Series:
    return (_item_text_chars(df).gt(ITEM_TEXT_CHARS_MAX)
            .fillna(False).astype(bool))


def _filter_has_image(df: pd.DataFrame) -> pd.Series:
    if "item_has_image" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["item_has_image"].fillna(False).astype(bool)


def _filter_sample_version(df: pd.DataFrame) -> pd.Series:
    if "is_sample_version" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["is_sample_version"].fillna(False).astype(bool)


def _filter_unscaled_bucket(df: pd.DataFrame) -> pd.Series:
    if "bucket" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["bucket"].eq("unscaled").fillna(False).astype(bool)


def _filter_intake_form(df: pd.DataFrame) -> pd.Series:
    # apa: null drops too (Meta always ran there); partials: only explicit True drops.
    if "meta_intake_form" not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df["meta_intake_form"]
    return raw.fillna(True).where(apa_mask(df), raw.fillna(False)).astype(bool)


def _filter_objective_measure(df: pd.DataFrame) -> pd.Series:
    if "meta_objective_measure" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["meta_objective_measure"].fillna(False).astype(bool)


FILTERS: dict[str, Filter] = {  # each returns an NA-safe *drop* mask
    "extraction_error": _filter_extraction_error,
    "null_bucket": _filter_null_bucket,
    "non_rating_scale": _filter_non_rating_scale,
    "empty_item_text": _filter_empty_item_text,
    "long_item_text": _filter_long_item_text,
    "has_image": _filter_has_image,
    "sample_version": _filter_sample_version,
    "unscaled_bucket": _filter_unscaled_bucket,
    "intake_form": _filter_intake_form,
    "objective_measure": _filter_objective_measure,
}


# Pre-filter per-document counts for the count flags; see docs/assemble-postprocess.md#flags.
OBSERVED_COUNT_COLS: dict[str, str] = {  # column added -> distinct-id column
    "observed_item_count": "item_item_id",
    "observed_scale_count": "scale_id",
}


def _attach_observed_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``OBSERVED_COUNT_COLS``; null-path rows stay NA rather than pooling into one pseudo-document."""
    if PATH_COL not in df.columns:
        return df
    groups = df.groupby(PATH_COL)
    for name, id_col in OBSERVED_COUNT_COLS.items():
        if id_col in df.columns:
            df[name] = groups[id_col].transform("nunique").astype("Int64")
    return df


def step_filter_rows(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Attach observed counts, then drop the union of ``FILTERS`` (isolated/new/cumulative report)."""
    df = _attach_observed_counts(df)
    attached = [c for c in OBSERVED_COUNT_COLS if c in df.columns]
    ctx.log("pre-filter counts attached: "
            + (", ".join(attached) if attached else "none (no document key)"))
    width = max(len(n) for n in FILTERS)
    ctx.log(f"{'filter':<{width}} {'isolated':>9} {'new':>9} {'cumulative':>10}")
    drop = pd.Series(False, index=df.index)
    attrition: dict[str, dict[str, int]] = {}
    for name, fn in FILTERS.items():
        mask = fn(df).fillna(False).astype(bool)
        new = mask & ~drop
        drop |= mask
        attrition[name] = {"isolated": int(mask.sum()), "new": int(new.sum()),
                           "cumulative": int(drop.sum())}
        ctx.log(f"{name:<{width}} {int(mask.sum()):>9,} "
                f"{int(new.sum()):>9,} {int(drop.sum()):>10,}")
    ctx.stat("filter_rows.attrition", attrition)
    ctx.stat("filter_rows.rows", {"in": len(df), "dropped": int(drop.sum()),
                                  "out": len(df) - int(drop.sum())})
    if PATH_COL in df.columns:
        ctx.stat("filter_rows.documents",
                 {"in": int(df[PATH_COL].nunique()),
                  "out": int(df.loc[~drop, PATH_COL].nunique())})
    ctx.log(f"dropping {int(drop.sum()):,} of {len(df):,} rows, "
            f"{len(df) - int(drop.sum()):,} remain")
    return df.loc[~drop].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2: column drops
# ---------------------------------------------------------------------------

def telemetry_columns(columns) -> list[str]:
    """Per-extractor telemetry columns among ``columns``."""
    return [c for c in columns
            if any(c.endswith(f"_extractor_{s}") for s in TELEMETRY_SUFFIXES)]


def step_drop_columns(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    groups = {
        "telemetry": telemetry_columns(df.columns),
        "doc telemetry": [c for c in DOC_TELEMETRY_COLS if c in df.columns],
        "extra": [c for c in EXTRA_DROP_COLS if c in df.columns],
    }
    for name, cols in groups.items():
        ctx.log(f"{name}: {len(cols)} columns")
    absent = [c for c in (*DOC_TELEMETRY_COLS, *EXTRA_DROP_COLS)
              if c not in df.columns]
    if absent:
        ctx.log(f"expected but absent: {', '.join(absent)}")
    drop = [c for cols in groups.values() for c in cols]
    df = df.drop(columns=drop)
    ctx.stat("drop_columns.groups",
             {name: len(cols) for name, cols in groups.items()})
    ctx.stat("drop_columns.dropped", len(drop))
    ctx.stat("drop_columns.remaining", len(df.columns))
    ctx.log(f"dropped {len(drop)}, {len(df.columns)} columns remain")
    return df


# ---------------------------------------------------------------------------
# Step 3: derived columns
# ---------------------------------------------------------------------------

def _segment(value) -> Optional[str]:
    """Trimmed citation segment without its trailing period, else None."""
    if not isinstance(value, str):
        return None
    s = value.strip().rstrip(".").strip()
    return s or None


def _authors_segment(value) -> Optional[str]:
    """Author string as-is (trailing periods belong to initials)."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s or None


def _year(value) -> Optional[str]:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        s = str(value).strip()
        return s or None


def _join(*parts: Optional[str]) -> str:
    return " ".join(p for p in parts if p)


def _record_citation(authors, year, title, doi) -> str:
    """APA-style citation of the PsycTESTS record; degrades gracefully on null fields."""
    url = f"https://doi.org/{doi}" if isinstance(doi, str) and doi else None
    title_s, authors_s = _segment(title), _authors_segment(authors)
    year_s = _year(year) or "n.d."
    if not title_s:
        return _join("APA PsycTests database record.", url)
    head = f"{title_s} [Database record]."
    if authors_s:
        return _join(f"{authors_s} ({year_s}).", head, "APA PsycTests.", url)
    # no authors: title moves to the author slot (APA convention)
    return _join(head, f"({year_s}).", "APA PsycTests.", url)


def _source_citation(authors, year, title, journal, doi) -> Optional[str]:
    """Source-publication citation; None unless there is a title plus authors or a DOI."""
    title_s, authors_s = _segment(title), _authors_segment(authors)
    doi_s = doi if isinstance(doi, str) and doi else None
    if not title_s or not (authors_s or doi_s):
        return None
    year_s = _year(year) or "n.d."
    journal_s = _segment(journal)
    url = f"https://doi.org/{doi_s}" if doi_s else None
    tail = _join(f"{journal_s}." if journal_s else None, url)
    if authors_s:
        return _join(f"{authors_s} ({year_s}).", f"{title_s}.", tail)
    return _join(f"{title_s}.", f"({year_s}).", tail)


# First parenthesised four-digit group = the APA year slot of every public_source form.
PUBLIC_YEAR_COL = "public_year"
PUBLIC_YEAR_RE = re.compile(r"\((\d{4})\)")


def _public_year(source: pd.Series) -> pd.Series:
    """First ``(YYYY)`` of each citation; NA where the citation has no year."""
    def one(value):
        if not isinstance(value, str):
            return pd.NA
        m = PUBLIC_YEAR_RE.search(value)
        return int(m.group(1)) if m else pd.NA
    return source.map(one).astype("Int64")


def step_derive(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    # Ahead of the DOI guard so a degraded run still publishes it.
    df[ITEM_TEXT_CHARS_COL] = _item_text_chars(df)
    lengths = df[ITEM_TEXT_CHARS_COL].dropna()
    if not lengths.empty:
        vals = lengths.astype(int)
        q = vals.quantile([0.25, 0.5, 0.75, 0.95])
        mean, sd = float(vals.mean()), float(vals.std(ddof=1))
        ctx.log(f"{ITEM_TEXT_CHARS_COL}: {len(lengths):,} rows measured, "
                f"M {mean:,.1f} (SD {sd:,.1f}), "
                f"Mdn {q.loc[0.5]:,.1f} "
                f"[IQR {q.loc[0.25]:,.1f}-{q.loc[0.75]:,.1f}], "
                f"p95 {int(q.loc[0.95]):,}, max {int(vals.max()):,} "
                f"(filter_rows dropped anything over {ITEM_TEXT_CHARS_MAX})")
        ctx.stat("derive.item_text_chars",
                 {"rows_measured": len(lengths), "mean": round(mean, 3),
                  "sd": round(sd, 3), "median": float(q.loc[0.5]),
                  "iqr_q1": float(q.loc[0.25]), "iqr_q3": float(q.loc[0.75]),
                  "p95": int(q.loc[0.95]), "max": int(vals.max())})
    else:
        ctx.log(f"{ITEM_TEXT_CHARS_COL}: no item text to measure")

    if DOI_PSYCTESTS_COL not in df.columns:
        ctx.log(f"WARNING: {DOI_PSYCTESTS_COL} missing — skipped, so "
                "public_doi/public_source/public_year are ABSENT from the "
                "output. The input is not a fully patched corpus; rerun "
                "patch with all steps.")
        return df
    apa = apa_mask(df)
    source_doi = (df["meta_source_doi"]
                  if "meta_source_doi" in df.columns
                  else pd.Series(pd.NA, index=df.index))
    df["public_doi"] = df[DOI_PSYCTESTS_COL].where(
        apa, source_doi.fillna(df[DOI_PSYCTESTS_COL]))
    ctx.log(f"public_doi: {int(df['public_doi'].notna().sum()):,} set, "
            f"{int((~apa & source_doi.notna()).sum()):,} from meta_source_doi")
    ctx.stat("derive.public_doi",
             {"set": int(df["public_doi"].notna().sum()),
              "from_meta_source_doi": int((~apa & source_doi.notna()).sum())})

    # apa: meta_authors/year describe the instrument; partials: the source publication (crossref backfill).
    cols = ["meta_authors_raw", "meta_publication_year_raw", "meta_title_raw",
            "meta_journal_venue_raw", "meta_source_raw", "meta_source_doi",
            DOI_PSYCTESTS_COL]
    sub = df.reindex(columns=cols)  # absent columns become all-NA
    counts = {"apa record": 0, "source_raw": 0, "built": 0, "record fallback": 0}
    values: list[str] = []
    for (authors, pub_year, title, journal, source_raw, src_doi,
         record_doi), is_apa in zip(sub.itertuples(index=False), apa.tolist()):
        if is_apa:
            counts["apa record"] += 1
            values.append(_record_citation(authors, pub_year, title, record_doi))
            continue
        src_doi = src_doi if isinstance(src_doi, str) and src_doi else None
        raw = source_raw.strip() if isinstance(source_raw, str) else ""
        if raw:
            counts["source_raw"] += 1
            if src_doi and doi_in_text(raw) is None:
                raw = f"{raw} https://doi.org/{src_doi}"
            values.append(raw)
            continue
        built = _source_citation(authors, pub_year, title, journal, src_doi)
        counts["built" if built else "record fallback"] += 1
        values.append(built or _record_citation(authors, pub_year, title,
                                                record_doi))
    df["public_source"] = pd.Series(values, index=df.index, dtype=object)
    ctx.stat("derive.public_source", dict(counts))
    ctx.log("public_source: " + ", ".join(f"{v:,} {k}"
                                          for k, v in counts.items()))

    df[PUBLIC_YEAR_COL] = _public_year(df["public_source"])
    years = df[PUBLIC_YEAR_COL].dropna()
    if not years.empty:
        vals = years.astype(int)
        q = vals.quantile([0.25, 0.5, 0.75])
        ctx.log(f"{PUBLIC_YEAR_COL}: {len(years):,} parsed, "
                f"{int(df[PUBLIC_YEAR_COL].isna().sum()):,} undated "
                f"(no year in the citation), "
                f"Mdn {q.loc[0.5]:.0f} "
                f"[IQR {q.loc[0.25]:.0f}-{q.loc[0.75]:.0f}], "
                f"range {int(vals.min())}-{int(vals.max())}")
        ctx.stat("derive.public_year",
                 {"parsed": len(years),
                  "undated": int(df[PUBLIC_YEAR_COL].isna().sum()),
                  "median": float(q.loc[0.5]),
                  "iqr_q1": float(q.loc[0.25]), "iqr_q3": float(q.loc[0.75]),
                  "min": int(vals.min()), "max": int(vals.max())})
    else:
        ctx.log(f"{PUBLIC_YEAR_COL}: no year parsed from any citation")
        ctx.stat("derive.public_year", {"parsed": 0,
                                        "undated": int(len(df))})
    return df


# ---------------------------------------------------------------------------
# Step 4: item text vs PDF body text
# ---------------------------------------------------------------------------

PDF_MATCH_DIST_COL = "pdf_match_edit_distance"
PDF_MATCH_NORM_COL = "pdf_match_edit_distance_norm"
# Kept so flag_item_text_deviation stays auditable after pdf_full_text is dropped.
PDF_TEXT_CHARS_COL = "pdf_text_chars"
# Whether the text layer can be checked at all (False: too thin, or items printed as images).
PDF_CHECKABLE_COL = "pdf_text_checkable"
# PDF shape from extraction; needed for the image-only rule, dropped after pdf_match.
PDF_SHAPE_COLS: tuple[str, ...] = ("meta_page_count", "meta_image_count")


def _text_checkable(df: pd.DataFrame) -> pd.Series:
    """False when the text layer cannot show the items; NA when there is no text layer to measure.

    Two cases: the layer is under ``PDF_TEXT_CHARS_MIN`` characters, or the items are printed as images,
    i.e. at least one image per body page beyond the per-page PsycTESTS logo and fewer than
    ``PDF_TEXT_CHARS_PER_PAGE_MIN`` characters per body page (pages 2..n). Calibration in
    docs/assemble-postprocess.md#pdf_match.
    """
    def num(name: str) -> pd.Series:
        s = df[name] if name in df.columns else pd.Series(float("nan"), index=df.index)
        return pd.to_numeric(s, errors="coerce").astype(float)

    chars, pages, images = num(PDF_TEXT_CHARS_COL), num("meta_page_count"), num("meta_image_count")
    body = pages - 1
    image_only = ((body > 0) & (images - pages >= body)
                  & (chars / body.where(body > 0) < PDF_TEXT_CHARS_PER_PAGE_MIN))
    checkable = chars.ge(PDF_TEXT_CHARS_MIN) & ~image_only.fillna(False)
    return checkable.astype("boolean").mask(chars.isna())


def step_pdf_match(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Fuzzy-match each item against the PDF body text, record ``pdf_text_chars``, then drop ``pdf_full_text``.

    The drop lives here (not drop_columns) because the text is copyrighted and needed until now.
    """
    from extraction.text_utils import best_partial_distance

    dist = pd.Series(pd.NA, index=df.index, dtype="Int64")
    norm = pd.Series(pd.NA, index=df.index, dtype="Float64")
    if PDF_FULL_TEXT_COL not in df.columns:
        ctx.log(f"WARNING: {PDF_FULL_TEXT_COL} missing — both distance "
                "columns are all-NA. The input is not a fully patched "
                "corpus; rerun patch with all steps.")
        df[PDF_MATCH_DIST_COL], df[PDF_MATCH_NORM_COL] = dist, norm
        df[PDF_TEXT_CHARS_COL] = pd.Series(pd.NA, index=df.index,
                                           dtype="Int64")
        df[PDF_CHECKABLE_COL] = pd.Series(pd.NA, index=df.index,
                                          dtype="boolean")
        return df.drop(columns=[c for c in PDF_SHAPE_COLS if c in df.columns])

    def _nonempty(s: pd.Series) -> pd.Series:
        return s.map(lambda t: isinstance(t, str) and bool(t))

    item_text = (df["item_item_text"] if "item_item_text" in df.columns
                 else pd.Series(pd.NA, index=df.index))
    computable = _nonempty(df[PDF_FULL_TEXT_COL]) & _nonempty(item_text)
    pairs = df.loc[computable, PDF_FULL_TEXT_COL].to_frame("haystack")
    pairs["needle"] = item_text[computable]
    rows = pairs.itertuples()
    try:
        from tqdm import tqdm
        rows = tqdm(rows, desc="pdf_match: fuzzy item lookups", unit="item",
                    total=len(pairs))
    except ImportError:
        pass
    for idx, haystack, needle in rows:
        res = best_partial_distance(needle, haystack)
        if res is not None:
            dist.at[idx], norm.at[idx] = res

    df[PDF_MATCH_DIST_COL], df[PDF_MATCH_NORM_COL] = dist, norm
    df[PDF_TEXT_CHARS_COL] = df[PDF_FULL_TEXT_COL].map(
        lambda v: len(v) if isinstance(v, str) else pd.NA).astype("Int64")
    df = df.drop(columns=[PDF_FULL_TEXT_COL])
    measured = df[PDF_MATCH_DIST_COL].notna()
    ctx.log(f"{PDF_MATCH_DIST_COL}: {int(measured.sum()):,} rows measured "
            f"({_doc_count(df, measured):,} documents), "
            f"{int((~computable).sum()):,} rows not measurable")
    ctx.stat("pdf_match.measured",
             {"rows": int(measured.sum()),
              "documents": _doc_count(df, measured),
              "rows_not_measurable": int((~computable).sum())})
    if measured.any():
        vals = df[PDF_MATCH_NORM_COL].dropna().astype(float)
        q = vals.quantile([0.25, 0.5, 0.75, 0.95])
        mean, sd = float(vals.mean()), float(vals.std(ddof=1))
        ctx.log(f"{PDF_MATCH_NORM_COL}: M {mean:.3f} (SD {sd:.3f}), "
                f"Mdn {q.loc[0.5]:.3f} "
                f"[IQR {q.loc[0.25]:.3f}-{q.loc[0.75]:.3f}], "
                f"p95 {q.loc[0.95]:.3f}")
        ctx.stat("pdf_match.edit_distance_norm",
                 {"mean": round(mean, 4), "sd": round(sd, 4),
                  "median": float(q.loc[0.5]),
                  "iqr_q1": round(float(q.loc[0.25]), 4),
                  "iqr_q3": round(float(q.loc[0.75]), 4),
                  "p95": float(q.loc[0.95])})
    thin = df[PDF_TEXT_CHARS_COL].lt(PDF_TEXT_CHARS_MIN).fillna(False)
    ctx.log(f"{PDF_TEXT_CHARS_COL}: {int(thin.sum()):,} rows "
            f"({_doc_count(df, thin):,} documents) below "
            f"{PDF_TEXT_CHARS_MIN} chars — text layer too thin to check "
            "an item against")
    ctx.stat("pdf_match.thin_text_layer",
             {"rows": int(thin.sum()), "documents": _doc_count(df, thin)})
    df[PDF_CHECKABLE_COL] = _text_checkable(df)
    image_only = df[PDF_CHECKABLE_COL].eq(False).fillna(False) & ~thin
    ctx.log(f"{PDF_CHECKABLE_COL}: {int(image_only.sum()):,} further rows "
            f"({_doc_count(df, image_only):,} documents) not checkable — "
            "items printed as images (>= 1 extra image per body page, < "
            f"{PDF_TEXT_CHARS_PER_PAGE_MIN} chars per body page)")
    ctx.stat("pdf_match.items_as_images",
             {"rows": int(image_only.sum()),
              "documents": _doc_count(df, image_only)})
    df = df.drop(columns=[c for c in PDF_SHAPE_COLS if c in df.columns])
    ctx.log(f"dropped {PDF_FULL_TEXT_COL} (copyrighted PDF body text)")
    return df


# ---------------------------------------------------------------------------
# Step 5: warning flags
# ---------------------------------------------------------------------------

# PsycTESTS record field -> ``record_*`` column (the record's claims, vs extracted ``meta_*``).
META_REFERENCE_FIELDS: dict[str, str] = {
    "number_of_test_items_best_guess": "record_item_count",
    "number_of_factors_subscales": "record_scale_count",
}


def load_meta_reference(path) -> pd.DataFrame:
    """``data.meta`` JSON records -> reference frame indexed by lowercase DOI; non-int values become null."""
    empty = pd.DataFrame(columns=list(META_REFERENCE_FIELDS.values()),
                         dtype="Int64")
    if not path or not Path(path).exists():
        return empty
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    rows = []
    for record in records:
        doi = record.get("DOI")
        if not isinstance(doi, str) or not doi:
            continue
        row = {"doi": doi.strip().lower()}
        for field_name, col in META_REFERENCE_FIELDS.items():
            value = record.get(field_name)
            row[col] = value if isinstance(value, int) and not isinstance(
                value, bool) else None
        rows.append(row)
    if not rows:
        return empty
    ref = pd.DataFrame(rows).drop_duplicates("doi", keep="first").set_index("doi")
    return ref.astype("Int64")


def _observations(df: pd.DataFrame) -> pd.DataFrame:
    """Per-document distinct item/scale counts; prefers the pre-filter ``observed_*`` columns."""
    obs = pd.DataFrame(index=df.index)
    groups = df.groupby(PATH_COL) if PATH_COL in df.columns else None
    for pre_col, id_col in OBSERVED_COUNT_COLS.items():
        name = pre_col.removeprefix("observed_")
        if pre_col in df.columns:
            obs[name] = df[pre_col].astype("Int64")
        elif groups is not None and id_col in df.columns:
            obs[name] = groups[id_col].transform("nunique").astype("Int64")
    return obs


def _deviates(index, observed: Optional[pd.Series],
              reference: Optional[pd.Series]) -> pd.Series:
    """True where the two differ, NA where either side is missing."""
    if observed is None or reference is None:
        return pd.Series(pd.NA, index=index, dtype="boolean")
    return (observed.ne(reference).astype("boolean")
            .mask(observed.isna() | reference.isna()))


def _column(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    return df[name] if name in df.columns else None


def _flag_item_count_deviation(df: pd.DataFrame,
                               obs: pd.DataFrame) -> pd.Series:
    """Distinct items differ from ``record_item_count``; NA for partial sources."""
    return _deviates(df.index, _column(obs, "item_count"),
                     _column(df, "record_item_count")).mask(~apa_mask(df))


def _flag_scale_count_deviation(df: pd.DataFrame,
                                obs: pd.DataFrame) -> pd.Series:
    """Distinct scales differ from ``record_scale_count`` (a recorded 0 is not checkable); NA for partial sources."""
    reference = _column(df, "record_scale_count")
    if reference is not None:
        reference = reference.mask(reference.eq(0))
    return _deviates(df.index, _column(obs, "scale_count"), reference).mask(~apa_mask(df))


# Normalised distance, i.e. a 0.95-similarity cutoff.
PDF_MATCH_NORM_MAX = 0.05

# Below this the PDF has no usable text layer; calibration in docs/assemble-postprocess.md#pdf_match.
PDF_TEXT_CHARS_MIN = 200
# Image-heavy PDFs with less body text per page than this print their items as images (same calibration).
PDF_TEXT_CHARS_PER_PAGE_MIN = 500


def _flag_item_text_deviation(df: pd.DataFrame,
                              obs: pd.DataFrame) -> pd.Series:
    """Item is not a near-verbatim match of its PDF body text; NA if unmeasured or the text layer is not checkable."""
    norm = _column(df, PDF_MATCH_NORM_COL)
    if norm is None:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    # Explicit isna mask: after a parquet round-trip the column is float64 and gt(NaN) is False.
    values = pd.to_numeric(norm, errors="coerce")
    flag = values.gt(PDF_MATCH_NORM_MAX).astype("boolean").mask(values.isna())
    checkable = _column(df, PDF_CHECKABLE_COL)
    if checkable is not None:  # pdf_match ran: thin and image-only text layers
        flag = flag.mask(checkable.astype("boolean").eq(False).fillna(False).astype(bool))
        return flag
    chars = _column(df, PDF_TEXT_CHARS_COL)  # standalone flags run on older output
    if chars is not None:
        thin = pd.to_numeric(chars, errors="coerce").lt(PDF_TEXT_CHARS_MIN)
        flag = flag.mask(thin.fillna(False).astype(bool))
    return flag


ITEM_LANGUAGE_COL = "item_language"
ITEM_LANGUAGE_EN = "en"


def _flag_item_translated(df: pd.DataFrame, obs: pd.DataFrame) -> pd.Series:
    """Original ``item_language`` is not ``en`` (published text is a translation); NA if unknown."""
    lang = _column(df, ITEM_LANGUAGE_COL)
    if lang is None:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    # Normalise for standalone runs on unpatched input; empty string -> NA.
    codes = lang.astype(object).map(
        lambda v: str(v).strip().lower() or None, na_action="ignore")
    return codes.ne(ITEM_LANGUAGE_EN).astype("boolean").mask(codes.isna())


# name -> nullable-boolean ``flag_{name}``: True warning, False consistent, NA not checkable.
Flag = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]
FLAGS: dict[str, Flag] = {
    "item_count_deviation": _flag_item_count_deviation,
    "scale_count_deviation": _flag_scale_count_deviation,
    "item_text_deviation": _flag_item_text_deviation,
    "item_translated": _flag_item_translated,
}


def step_flags(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Join the record reference values and add the ``flag_*`` columns."""
    meta_path = (ctx.full_cfg.get("data") or {}).get("meta")
    reference = load_meta_reference(meta_path)
    if reference.empty:
        ctx.log(f"WARNING: no PsycTESTS records at data.meta ({meta_path}) — "
                "reference columns are empty and every record-based flag is NA")
    doi = (df[DOI_PSYCTESTS_COL].astype(object).str.strip().str.lower()
           if DOI_PSYCTESTS_COL in df.columns
           else pd.Series(None, index=df.index, dtype=object))
    if DOI_PSYCTESTS_COL not in df.columns:
        ctx.log(f"WARNING: {DOI_PSYCTESTS_COL} missing — nothing to join the "
                "record reference on; record-based flags are NA")
    for col in META_REFERENCE_FIELDS.values():
        values = (doi.map(reference[col]) if col in reference.columns
                  else pd.Series(pd.NA, index=df.index))
        df[col] = values.astype("Int64")
        ctx.log(f"{col}: {int(df[col].notna().sum()):,} rows "
                f"({_doc_count(df, df[col].notna()):,} documents) referenced")
        ctx.stat(f"flags.reference.{col}",
                 {"rows": int(df[col].notna().sum()),
                  "documents": _doc_count(df, df[col].notna())})

    obs = _observations(df)
    width = max(len(n) for n in FLAGS) + len("flag_")
    ctx.log(f"{'flag':<{width}} {'rows':>10} {'documents':>10} "
            f"{'not checkable':>14}")
    for name, fn in FLAGS.items():
        col = f"flag_{name}"
        mask = fn(df, obs).astype("boolean")
        df[col] = mask
        flagged = mask.fillna(False)
        ctx.log(f"{col:<{width}} {int(flagged.sum()):>10,} "
                f"{_doc_count(df, flagged):>10,} "
                f"{int(mask.isna().sum()):>14,}")
        ctx.stat(f"flags.{col}",
                 {"rows": int(flagged.sum()),
                  "documents": _doc_count(df, flagged),
                  "not_checkable": int(mask.isna().sum())})
    return df


def _doc_count(df: pd.DataFrame, mask: pd.Series) -> int:
    """Distinct documents among the masked rows (row count when untracked)."""
    if PATH_COL not in df.columns:
        return int(mask.sum())
    return int(df.loc[mask, PATH_COL].nunique())


# ---------------------------------------------------------------------------
# Registry and runners
# ---------------------------------------------------------------------------

Step = Callable[[pd.DataFrame, Ctx], pd.DataFrame]
STEPS: dict[str, Step] = {  # insertion order = execution order
    "filter_rows": step_filter_rows,
    "drop_columns": step_drop_columns,
    "derive": step_derive,
    "pdf_match": step_pdf_match,
    "flags": step_flags,
}
STEP_SCOPE: dict[str, str] = {
    "filter_rows": "all", "drop_columns": "all", "derive": "all",
    "pdf_match": "apa", "flags": "all",
}


def _run_steps(df: pd.DataFrame, selected: list[str], ctx: Ctx) -> pd.DataFrame:
    for name in STEPS:  # registry order, regardless of CLI order
        if name not in selected:
            continue
        ctx.log(f"--- {name} [{STEP_SCOPE[name]}] ---")
        df = STEPS[name](df, ctx)
    return df


def _select_steps(steps_arg: Optional[str]) -> list[str]:
    if not steps_arg:
        return list(STEPS)
    selected = [s.strip() for s in steps_arg.split(",") if s.strip()]
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        sys.exit(f"Unknown steps: {', '.join(unknown)}. "
                 f"Available: {', '.join(STEPS)}")
    return selected


def run(cfg: dict, *, report_only: bool = False,
        input_path=None, output_path=None, steps: Optional[str] = None) -> list[str]:
    """Stage entry point: ``data.assemble.patched`` -> ``data.assemble.postprocessed``."""
    asm_cfg = cfg.get("data", {}).get("assemble", {})
    inp = Path(input_path or asm_cfg.get("patched", ""))
    out = Path(output_path or asm_cfg.get("postprocessed", ""))
    if not str(inp) or not str(out) or str(inp) == "." or str(out) == ".":
        sys.exit("data.assemble.patched and data.assemble.postprocessed "
                 "must be set")
    if not inp.exists():
        sys.exit(f"input not found: {inp}")

    ctx = Ctx(cfg=cfg.get("postprocess", {}) or {}, full_cfg=cfg,
              report_only=report_only)
    ctx.log("corpus postprocess report")
    ctx.log(f"input: {inp}")
    df = pd.read_parquet(inp)
    ctx.log(f"{len(df):,} rows, {len(df.columns)} columns")

    selected = _select_steps(steps)
    if not report_only and len(selected) < len(STEPS):
        skipped = [s for s in STEPS if s not in selected]
        ctx.log(f"WARNING: running {len(selected)} of {len(STEPS)} steps — "
                f"the output is a PARTIAL postprocess (skipped: "
                f"{', '.join(skipped)}) and overwrites {out}. Use --output "
                "to write elsewhere when trying a step subset.")

    df = _run_steps(df, selected, ctx)
    ctx.log("")
    ctx.log(f"final: {len(df):,} rows, {len(df.columns)} columns")

    if not report_only:
        write_parquet(df, out)
        ctx.log(f"wrote {out}")
        mirror_report(write_stats(out, "postprocess", ctx.stats, ctx.report), cfg)
    return ctx.report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone postprocess stage: filter, prune, and derive "
                    "the public-facing corpus columns.",
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument("--input", default=None,
                    help="Input parquet (default: data.assemble.patched).")
    ap.add_argument("--output", default=None,
                    help="Output parquet (default: data.assemble.postprocessed).")
    ap.add_argument("--steps", default=None,
                    help=f"Comma-separated subset of: {', '.join(STEPS)}. "
                         "Always executed in registry order.")
    ap.add_argument("--report-only", action="store_true",
                    help="Print the report without writing anything.")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    lines = run(cfg, report_only=args.report_only,
                input_path=args.input, output_path=args.output,
                steps=args.steps)
    print("\n".join(lines))
    if args.report_only:
        print("(report-only: nothing written)")


if __name__ == "__main__":
    main()
