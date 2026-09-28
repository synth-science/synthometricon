"""Postprocess stage: pure helpers and step invariants on synthetic frames (no I/O)."""
from __future__ import annotations

import pandas as pd
import pytest

import json

from assemble.postprocess import (
    Ctx,
    ITEM_TEXT_CHARS_MAX,
    DOC_TELEMETRY_COLS,
    EXTRA_DROP_COLS,
    FLAGS,
    PDF_MATCH_NORM_MAX,
    PDF_TEXT_CHARS_MIN,
    TELEMETRY_SUFFIXES,
    _observations,
    _public_year,
    _record_citation,
    _run_steps,
    _select_steps,
    _source_citation,
    load_meta_reference,
    step_derive,
    step_drop_columns,
    step_filter_rows,
    step_flags,
    step_pdf_match,
    telemetry_columns,
)


def ctx(**cfg) -> Ctx:
    return Ctx(cfg=cfg, full_cfg={})


# --- Schema sync with extraction/storage.py ---

def test_telemetry_suffixes_in_sync():
    from extraction import storage

    assert len(TELEMETRY_SUFFIXES) == (
        len(storage.USAGE_SUFFIXES) + len(storage.RETRY_SUFFIXES))
    assert "timestamp" in TELEMETRY_SUFFIXES
    assert "attempts" in TELEMETRY_SUFFIXES
    assert DOC_TELEMETRY_COLS == tuple(n for n, _ in storage._DOC_SUMMARY_COLS)


def test_telemetry_columns():
    cols = ["items_extractor_ttft", "meta_extractor_attempts",
            "total_duration", "item_item_text", "meta_extractor_content"]
    assert telemetry_columns(cols) == ["items_extractor_ttft",
                                       "meta_extractor_attempts"]


# --- filter_rows ---

def test_filter_rows_drops_shell_rows():
    df = pd.DataFrame({
        "has_errors": [True, False, pd.NA],
        "bucket": ["scaled", None, "orphan"],
        "item_item_text": ["a", "b", "c"],
    })
    c = ctx()
    out = step_filter_rows(df, c)
    # row 0 (has_errors) and row 1 (null bucket) dropped, NA has_errors kept
    assert out["item_item_text"].tolist() == ["c"]
    assert list(out.index) == [0]


def test_filter_rows_item_filters():
    df = pd.DataFrame({
        "item_item_type": ["rating_scale", "choice", "open", None,
                           "rating_scale", "rating_scale", "rating_scale"],
        "item_item_text": ["a", "b", "c", "d", None, "  ", "keep"],
        "item_has_image": [True, False, False, False, False, False, None],
    })
    out = step_filter_rows(df, ctx())
    # 0: image; 1, 2: known non-rating type; 3: null type kept;
    # 4, 5: null/blank text; 6: NA has_image kept
    assert out["item_item_text"].tolist() == ["d", "keep"]


def test_filter_rows_long_item_text():
    at = "x" * ITEM_TEXT_CHARS_MAX
    df = pd.DataFrame({
        "item_item_text": [at, at + "x", f"  {at}  ", None, "short"],
    })
    out = step_filter_rows(df, ctx())
    # at max passes, one over drops; length is judged after trimming; null stems aren't overruns
    assert out["item_item_text"].tolist() == [at, f"  {at}  ", "short"]


def test_filter_rows_flag_filters():
    df = pd.DataFrame({
        "bucket": ["scaled", "orphan", "unscaled", "scaled", "scaled",
                   "orphan", "scaled"],
        "item_item_text": ["a", "b", "c", "d", "e", "f", "keep"],
        "is_sample_version": [True, False, False, None, False, False, None],
        "meta_intake_form": [False, False, False, False, True, None, False],
    })
    out = step_filter_rows(df, ctx())
    # 0: sample version; 2: unscaled bucket; 4: intake form True;
    # 5: intake form null (not an explicit False); 1, 3, 6: kept
    assert out["item_item_text"].tolist() == ["b", "d", "keep"]


def test_filter_rows_objective_measure():
    df = pd.DataFrame({
        "item_item_text": ["a", "b", "keep"],
        "meta_objective_measure": [True, None, False],
    })
    out = step_filter_rows(df, ctx())
    # 0: objective measure True; 1: null kept (not flagged); 2: kept
    assert out["item_item_text"].tolist() == ["b", "keep"]


def test_filter_rows_report_counts():
    df = pd.DataFrame({
        "has_errors": [True, False, False],
        "bucket": [None, "scaled", "scaled"],
        "item_item_text": [None, None, "a"],
    })
    c = ctx()
    step_filter_rows(df, c)
    counts = {l.split()[0]: l.split()[1:] for l in c.report[1:-1]}
    # shell row: isolated by all three, but only new for the first filter
    assert counts["extraction_error"] == ["1", "1", "1"]
    assert counts["null_bucket"] == ["1", "0", "1"]
    assert counts["empty_item_text"] == ["2", "1", "2"]
    assert "dropping 2 of 3 rows, 1 remain" in c.report[-1]


def test_filter_rows_missing_columns_noop():
    df = pd.DataFrame({"other": ["a", "b"]})
    out = step_filter_rows(df, ctx())
    assert len(out) == 2


# --- drop_columns ---

def test_drop_columns():
    df = pd.DataFrame({
        "items_extractor_ttft": [1.0],
        "scales_extractor_timestamp": ["t"],
        "has_errors": [False],
        "total_duration": [2.0],
        "meta_doi_raw": ["10.1037/t00001-000"],
        "verify_verdict": ["pass"],
        "item_item_text": ["a"],
        "doi_psyctests": ["10.1037/t00001-000"],
    })
    out = step_drop_columns(df, ctx())
    assert list(out.columns) == ["item_item_text", "doi_psyctests"]


def test_drop_columns_absent_expected_ok():
    df = pd.DataFrame({"item_item_text": ["a"]})
    c = ctx()
    out = step_drop_columns(df, c)
    assert list(out.columns) == ["item_item_text"]
    assert any("expected but absent" in l for l in c.report)


# --- derive: public_doi ---

def _derive_frame(**overrides) -> pd.DataFrame:
    base = {
        "corpus_source": ["apa-psyctests", "semanticnet", "scale-hunt"],
        "doi_psyctests": ["10.1037/t00001-000", "10.1037/t00002-000",
                          "10.1037/t00003-000"],
        "meta_source_doi": ["10.1000/apa.1", "10.1000/src.2", None],
        "meta_authors_raw": ["Raskin, R. N., & Hall, C. S.", None, None],
        "meta_publication_year_raw": ["1979", None, None],
        "meta_title_raw": ["Narcissistic Personality Inventory", None, None],
        "meta_journal_venue_raw": [None, None, None],
        "meta_source_raw": [None, None, None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_public_doi_precedence():
    out = step_derive(_derive_frame(), ctx())
    # apa ignores meta_source_doi; partial prefers it; null falls back
    assert out["public_doi"].tolist() == [
        "10.1037/t00001-000", "10.1000/src.2", "10.1037/t00003-000"]
    assert out["public_doi"].notna().all()


def test_derive_skips_without_doi_column():
    df = pd.DataFrame({"item_item_text": ["a"]})
    out = step_derive(df, ctx())
    assert "public_doi" not in out.columns
    # item_text_chars needs no DOI, so a degraded run still publishes it
    assert out["item_text_chars"].tolist() == [1]


def test_derive_item_text_chars():
    df = _derive_frame(item_item_text=["I act on impulse.", "  padded  ",
                                       None])
    c = ctx()
    out = step_derive(df, c)
    assert out["item_text_chars"].tolist()[:2] == [17, 6]  # trimmed
    assert pd.isna(out["item_text_chars"].iloc[2])         # no stem
    assert out["item_text_chars"].dtype.name == "Int64"
    assert any("item_text_chars: 2 rows measured" in l for l in c.report)


def test_derive_item_text_chars_without_item_column():
    c = ctx()
    out = step_derive(_derive_frame(), c)  # no item_item_text at all
    assert out["item_text_chars"].isna().all()
    assert any("no item text to measure" in l for l in c.report)


# --- derive: public_source ---

def test_public_source_apa_record_citation():
    out = step_derive(_derive_frame(), ctx())
    assert out["public_source"].iloc[0] == (
        "Raskin, R. N., & Hall, C. S. (1979). "
        "Narcissistic Personality Inventory [Database record]. "
        "APA PsycTests. https://doi.org/10.1037/t00001-000")


def test_record_citation_degrades_gracefully():
    # no authors: title moves to the author slot
    assert _record_citation(None, 1979.0, "Some Scale", "10.1037/t1-000") == (
        "Some Scale [Database record]. (1979). APA PsycTests. "
        "https://doi.org/10.1037/t1-000")
    # no year -> (n.d.)
    assert "(n.d.)" in _record_citation("A. B.", None, "Some Scale", "10.1037/t1-000")
    # no title -> minimal form
    assert _record_citation("A. B.", 1979, None, "10.1037/t1-000") == (
        "APA PsycTests database record. https://doi.org/10.1037/t1-000")


def test_public_source_partial_raw_passthrough():
    df = _derive_frame(
        meta_source_raw=[None,
                         "Gross, J. (1995). Emotion elicitation using films.",
                         None])
    out = step_derive(df, ctx())
    # DOI appended when the citation lacks one
    assert out["public_source"].iloc[1] == (
        "Gross, J. (1995). Emotion elicitation using films. "
        "https://doi.org/10.1000/src.2")


def test_public_source_no_double_doi_append():
    cite = "Gross, J. (1995). Films. https://doi.org/10.1000/src.2"
    df = _derive_frame(meta_source_raw=[None, cite, None])
    out = step_derive(df, ctx())
    assert out["public_source"].iloc[1] == cite


def test_public_source_built_citation():
    df = _derive_frame(
        meta_authors_raw=[None, "Gross, J. J., & Levenson, R. W.", None],
        meta_publication_year_raw=[None, 1995.0, None],
        meta_title_raw=[None, "Emotion elicitation using films", None],
        meta_journal_venue_raw=[None, "Cognition and Emotion", None],
    )
    out = step_derive(df, ctx())
    assert out["public_source"].iloc[1] == (
        "Gross, J. J., & Levenson, R. W. (1995). "
        "Emotion elicitation using films. Cognition and Emotion. "
        "https://doi.org/10.1000/src.2")


def test_source_citation_requires_title_and_anchor():
    assert _source_citation("A. B.", 1995, None, None, "10.1/x") is None
    assert _source_citation(None, 1995, "Title", None, None) is None
    # title + doi suffices even without authors
    assert _source_citation(None, 1995, "Title", None, "10.1/x") == (
        "Title. (1995). https://doi.org/10.1/x")


def test_public_source_record_fallback_never_null():
    # partial with no source info at all -> PsycTests record citation
    out = step_derive(_derive_frame(), ctx())
    assert out["public_source"].iloc[2] == (
        "APA PsycTests database record. https://doi.org/10.1037/t00003-000")
    assert out["public_source"].notna().all()


# --- derive: public_year ---

def test_public_year_from_citations():
    c = ctx()
    out = step_derive(_derive_frame(), c)
    # apa record citation carries (1979); the two partials fall back to a
    # record citation with no year at all
    assert out["public_year"].tolist()[0] == 1979
    assert out["public_year"].isna().tolist()[1:] == [True, True]
    assert out["public_year"].dtype.name == "Int64"
    assert any("public_year: 1 parsed" in l for l in c.report)


def test_public_year_reads_the_apa_year_slot_only():
    # volume(issue), DOI suffixes and page ranges must not be mistaken for it
    s = pd.Series([
        "Nevid, J. S. (1980). Attitudes toward mental illness. Journal of "
        "Humanistic Psychology, 20(2), 71-85. "
        "https://doi.org/10.1177/002216788002000207",
        "Messick, S. (1985). Three Factor Eating Questionnaire. Journal of "
        "Psychosomatic Research. https://doi.org/10.1016/0022-3999(85)90010-8",
        "Rating Scale [Database record]. (n.d.). APA PsycTests. "
        "https://doi.org/10.1037/t00004-000",
        "APA PsycTests database record. https://doi.org/10.1037/t65408-000",
        None,
    ])
    out = _public_year(s)
    assert out.tolist()[:2] == [1980, 1985]
    assert out.isna().tolist()[2:] == [True, True, True]


# --- flags ---

RECORDS = [
    {"DOI": "10.1037/T00001-000",  # upper-cased on the record, lower on join
     "number_of_test_items_best_guess": 3,
     "number_of_factors_subscales": 2},
    {"DOI": "10.1037/t00002-000",  # no factor/subscale count on this record
     "number_of_test_items_best_guess": 2},
    {"DOI": "10.1037/t00003-000",  # 0 factors = none reported, not "zero"
     "number_of_test_items_best_guess": None,
     "number_of_factors_subscales": 0},
    {"DOI": None, "number_of_test_items_best_guess": 9},  # unusable
]


def _meta_file(tmp_path, records=RECORDS):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _flag_ctx(tmp_path, records=RECORDS) -> Ctx:
    return Ctx(cfg={}, full_cfg={"data": {"meta": str(_meta_file(tmp_path,
                                                                 records))}})


def _flag_frame() -> pd.DataFrame:
    # doc a: 3 distinct items over 2 leaf scales (item 2 sits in both, so it
    # occupies two rows); doc b: 2 items, no scales; doc c: unknown DOI.
    return pd.DataFrame({
        "path": ["a.pdf"] * 4 + ["b.pdf"] * 2 + ["c.pdf"],
        "doi_psyctests": ["10.1037/t00001-000 "] * 4
                         + ["10.1037/t00002-000"] * 2 + ["10.1037/t99999-000"],
        "item_item_id": [1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 1.0],
        "scale_id": [1.0, 1.0, 2.0, 2.0, None, None, 7.0],
    })


def test_load_meta_reference(tmp_path):
    ref = load_meta_reference(_meta_file(tmp_path))
    assert list(ref.columns) == ["record_item_count", "record_scale_count"]
    assert ref.index.tolist() == ["10.1037/t00001-000", "10.1037/t00002-000",
                                  "10.1037/t00003-000"]
    assert ref.loc["10.1037/t00001-000", "record_scale_count"] == 2
    assert pd.isna(ref.loc["10.1037/t00002-000", "record_scale_count"])
    assert str(ref.dtypes.iloc[0]) == "Int64"


def test_load_meta_reference_missing_file(tmp_path):
    ref = load_meta_reference(tmp_path / "nope.json")
    assert ref.empty and list(ref.columns) == ["record_item_count",
                                               "record_scale_count"]


def test_observations_count_distinct_per_document():
    obs = _observations(_flag_frame())
    # item 2 of doc a occupies two rows but counts once; orphan rows (null
    # scale_id) contribute no scale
    assert obs["item_count"].tolist() == [3, 3, 3, 3, 2, 2, 1]
    assert obs["scale_count"].tolist() == [2, 2, 2, 2, 0, 0, 1]


def test_observations_prefer_prefilter_counts():
    # observed_* columns from filter_rows win over a recount
    df = _flag_frame()
    df["observed_item_count"] = pd.array([9] * 7, dtype="Int64")
    df["observed_scale_count"] = pd.array([8] * 7, dtype="Int64")
    obs = _observations(df)
    assert obs["item_count"].tolist() == [9] * 7
    assert obs["scale_count"].tolist() == [8] * 7


def test_filter_rows_attaches_prefilter_counts():
    # doc a: one of 3 items dropped as unscaled; observed counts reflect the extraction
    df = pd.DataFrame({
        "path": ["a.pdf"] * 3 + ["b.pdf"],
        "bucket": ["scaled", "scaled", "unscaled", "scaled"],
        "item_item_text": ["a", "b", "c", "d"],
        "item_item_id": [1.0, 2.0, 3.0, 1.0],
        "scale_id": [1.0, 2.0, None, 1.0],
    })
    out = step_filter_rows(df, ctx())
    assert out["item_item_text"].tolist() == ["a", "b", "d"]
    assert out["observed_item_count"].tolist() == [3, 3, 1]
    assert out["observed_scale_count"].tolist() == [2, 2, 1]


def test_prefilter_counts_null_document_key_reads_na():
    # rows with no document key must not be lumped into one pseudo-document
    df = pd.DataFrame({
        "path": ["a.pdf", None, None],
        "item_item_text": ["a", "b", "c"],
        "item_item_id": [1.0, 1.0, 2.0],
        "scale_id": [1.0, 1.0, 2.0],
    })
    out = step_filter_rows(df, ctx())
    assert out["observed_item_count"].tolist() == [1, pd.NA, pd.NA]
    assert out["observed_scale_count"].tolist() == [1, pd.NA, pd.NA]


def test_flags_judge_prefilter_counts(tmp_path):
    # doc a matches its record pre-filter but would deviate after the unscaled row is dropped.
    df = _flag_frame()
    df["item_item_text"] = list("abcdefg")
    df["bucket"] = ["scaled", "scaled", "scaled", "unscaled",
                    "scaled", "scaled", "scaled"]
    out = step_flags(step_filter_rows(df, ctx()), _flag_ctx(tmp_path))
    assert len(out) == 6  # doc a's unscaled row is gone
    assert out["observed_item_count"].tolist()[:3] == [3, 3, 3]
    flag = out["flag_item_count_deviation"]
    assert flag.tolist()[:5] == [False] * 5
    assert pd.isna(flag.iloc[5])  # doc c: unknown DOI, not checkable
    assert out["flag_scale_count_deviation"].tolist()[:3] == [False] * 3


def test_flags_deviation_semantics(tmp_path):
    c = _flag_ctx(tmp_path)
    out = step_flags(_flag_frame(), c)
    # doc a matches the record on both counts; doc b has no scales against a
    # record that never reported any; doc c has no record at all
    assert out["flag_item_count_deviation"].tolist() == [
        False, False, False, False, False, False, pd.NA]
    assert out["flag_scale_count_deviation"].tolist() == [
        False, False, False, False, pd.NA, pd.NA, pd.NA]
    assert str(out["flag_item_count_deviation"].dtype) == "boolean"
    assert out["record_item_count"].tolist()[:6] == [3, 3, 3, 3, 2, 2]


def test_flags_fire_on_mismatch(tmp_path):
    df = _flag_frame().drop(index=[3]).reset_index(drop=True)  # doc a: 2 items
    out = step_flags(df, _flag_ctx(tmp_path))
    assert out["flag_item_count_deviation"].tolist() == [
        True, True, True, False, False, pd.NA]
    # scale structure is untouched, so that flag stays clear
    assert out["flag_scale_count_deviation"].tolist()[:3] == [False] * 3


def test_flags_zero_scale_count_not_checkable(tmp_path):
    df = pd.DataFrame({
        "path": ["d.pdf"],
        "doi_psyctests": ["10.1037/t00003-000"],
        "item_item_id": [1.0],
        "scale_id": [1.0],
    })
    out = step_flags(df, _flag_ctx(tmp_path))
    # record_scale_count 0 means "no structure reported": nothing to compare
    assert out["record_scale_count"].tolist() == [0]
    assert out["flag_scale_count_deviation"].isna().all()
    # record_item_count is null on that record too
    assert out["flag_item_count_deviation"].isna().all()


def test_flags_without_meta_are_all_na():
    df = _flag_frame()
    c = ctx()  # full_cfg has no data.meta
    out = step_flags(df, c)
    for name in ("item_count_deviation", "scale_count_deviation"):
        assert out[f"flag_{name}"].isna().all()
    assert any("no PsycTESTS records" in l for l in c.report)


def test_flag_item_text_deviation():
    df = _flag_frame().iloc[:5].copy()
    df["pdf_match_edit_distance_norm"] = pd.array(
        [0.0, PDF_MATCH_NORM_MAX, 0.051, 1.0, None], dtype="Float64")
    out = step_flags(df, ctx())
    # verbatim and at-threshold clear; beyond warns; no distance -> NA
    flag = out["flag_item_text_deviation"]
    assert flag.tolist()[:4] == [False, False, True, True]
    assert flag.isna().tolist() == [False] * 4 + [True]
    assert str(flag.dtype) == "boolean"


def test_flag_item_text_deviation_tolerates_nan_and_missing_column():
    df = _flag_frame().iloc[:3].copy()
    # plain float64 as read from parquet: gt(NaN) is False, so the NA mask must be explicit
    df["pdf_match_edit_distance_norm"] = [float("nan"), None, 0.9]
    out = step_flags(df, ctx())
    flag = out["flag_item_text_deviation"]
    assert flag.isna().tolist() == [True, True, False]
    assert flag.iloc[2] is True or bool(flag.iloc[2])
    # pdf_match skipped entirely: nothing measured, so nothing checkable
    out = step_flags(_flag_frame().iloc[:3].copy(), ctx())
    assert out["flag_item_text_deviation"].isna().all()


def test_flag_item_text_deviation_thin_text_layer_is_not_checkable():
    """Distances against a near-empty PDF text layer are NA, not evidence."""
    df = _flag_frame().iloc[:4].copy()
    df["pdf_match_edit_distance_norm"] = pd.array(
        [0.9, 0.9, 0.0, 0.9], dtype="Float64")
    df["pdf_text_chars"] = pd.array(
        [PDF_TEXT_CHARS_MIN - 1, PDF_TEXT_CHARS_MIN, PDF_TEXT_CHARS_MIN,
         None], dtype="Int64")
    flag = step_flags(df, ctx())["flag_item_text_deviation"]
    # below min -> NA; at min the distance counts; null length does not gate
    assert flag.isna().tolist() == [True, False, False, False]
    assert flag.iloc[1] and flag.iloc[3]
    assert not flag.iloc[2]


def test_flag_item_translated():
    df = _flag_frame().iloc[:6].copy()
    df["item_language"] = ["en", "EN ", "de", "pt-br", None, ""]
    flag = step_flags(df, ctx())["flag_item_translated"]
    # english (any case/padding) = original; other = translated; missing -> NA
    assert flag.tolist()[:4] == [False, False, True, True]
    assert flag.isna().tolist() == [False] * 4 + [True, True]
    assert str(flag.dtype) == "boolean"


def test_flag_item_translated_without_language_column():
    out = step_flags(_flag_frame().iloc[:3].copy(), ctx())
    assert out["flag_item_translated"].isna().all()


def test_flags_report_counts(tmp_path):
    c = _flag_ctx(tmp_path)
    step_flags(_flag_frame().drop(index=[3]).reset_index(drop=True), c)
    row = next(l for l in c.report if l.startswith("flag_item_count_deviation"))
    # 3 rows, 1 document flagged, 1 row not checkable
    assert row.split()[1:] == ["3", "1", "1"]


# --- pdf_match ---

def test_best_partial_distance():
    from extraction.text_utils import best_partial_distance

    body = "Instructions.\nI plan ahead carefully.\nI act on impulse.\n"
    assert best_partial_distance("I plan ahead carefully.", body) == (0, 0.0)
    d, nd = best_partial_distance("I plan ahed carefully.", body)
    assert d >= 1 and 0 < nd < 0.3
    assert best_partial_distance("", body) is None
    assert best_partial_distance("item", "") is None


def test_step_pdf_match():
    body = "Filler page.\nI plan ahead carefully.\nI act on impulse.\n"
    df = pd.DataFrame({
        "path": ["a.pdf", "a.pdf", "a.pdf", "b.url"],
        "item_item_text": ["I plan ahead carefully.",
                           "I plan ahed carefully.", None, "partial item"],
        "pdf_full_text": [body, body, body, None],
    })
    c = ctx()
    out = step_pdf_match(df, c)
    assert "pdf_full_text" not in out.columns  # copyrighted text dropped
    assert out["pdf_match_edit_distance"].iloc[0] == 0
    assert out["pdf_match_edit_distance_norm"].iloc[0] == 0.0
    assert out["pdf_match_edit_distance"].iloc[1] >= 1
    assert 0 < out["pdf_match_edit_distance_norm"].iloc[1] < 0.5
    assert out["pdf_match_edit_distance"].iloc[2:].isna().all()
    assert out["pdf_match_edit_distance_norm"].iloc[2:].isna().all()
    assert out["pdf_match_edit_distance"].dtype.name == "Int64"
    assert out["pdf_match_edit_distance_norm"].dtype.name == "Float64"
    # haystack size travels with the distance so the flag stays auditable
    assert out["pdf_text_chars"].tolist()[:3] == [len(body)] * 3
    assert pd.isna(out["pdf_text_chars"].iloc[3])
    assert out["pdf_text_chars"].dtype.name == "Int64"


def test_step_pdf_match_missing_column():
    df = pd.DataFrame({"item_item_text": ["a"]})
    c = ctx()
    out = step_pdf_match(df, c)
    assert out["pdf_match_edit_distance"].isna().all()
    assert out["pdf_match_edit_distance_norm"].isna().all()
    assert out["pdf_text_chars"].isna().all()
    assert any("WARNING" in l for l in c.report)


# --- Runner ---

def test_select_steps_unknown_exits():
    with pytest.raises(SystemExit):
        _select_steps("bogus")
    assert _select_steps(None) == ["filter_rows", "drop_columns", "derive",
                                   "pdf_match", "flags"]
    assert _select_steps("derive") == ["derive"]


def test_run_steps_end_to_end():
    df = _derive_frame(
        has_errors=[False, False, True],
        bucket=["scaled", "scaled", None],
        items_extractor_ttft=[0.1, 0.2, 0.3],
    )
    c = ctx()
    out = _run_steps(df, _select_steps(None), c)
    assert len(out) == 2  # error/shell row dropped
    assert "items_extractor_ttft" not in out.columns
    assert "has_errors" not in out.columns
    assert {"public_doi", "public_source", "item_text_chars"} <= set(out.columns)
    assert {"pdf_match_edit_distance", "pdf_match_edit_distance_norm",
            "pdf_text_chars"} <= set(out.columns)
    assert {f"flag_{n}" for n in FLAGS} <= set(out.columns)
    assert any(l.startswith("--- filter_rows") for l in c.report)


# --- stats sidecar ---

def test_run_writes_stats_sidecar(tmp_path):
    from assemble.postprocess import run
    from assemble.stats import read_stats, stats_path

    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    pd.DataFrame({"path": ["a.pdf"], "corpus_source": ["apa-psyctests"],
                  "bucket": ["scaled"]}).to_parquet(inp)
    run({}, input_path=inp, output_path=out, steps="drop_columns")
    payload = read_stats(out)
    assert payload["stage"] == "postprocess"
    assert "drop_columns.dropped" in payload["stats"]

    out2 = tmp_path / "out2.parquet"
    run({}, input_path=inp, output_path=out2, steps="drop_columns",
        report_only=True)
    assert not out2.exists() and not stats_path(out2).exists()
