"""Report stage and stats sidecars: counting conventions, staleness, and graceful degradation."""
from __future__ import annotations

import os

import pandas as pd

from assemble import report
from assemble.stats import (mirror_report, read_stats, stats_path, stats_stale,
                            write_stats)


# --- stats sidecar ---

def test_stats_roundtrip(tmp_path):
    artifact = tmp_path / "x.parquet"
    artifact.write_bytes(b"")
    path = write_stats(artifact, "patch", {"a.b": 1, "t": {"n": 2}}, ["line"])
    assert path == stats_path(artifact) == tmp_path / "x.stats.json"
    payload = read_stats(artifact)
    assert payload["stage"] == "patch"
    assert payload["stats"] == {"a.b": 1, "t": {"n": 2}}
    assert payload["report"] == ["line"]


def test_read_stats_missing_and_corrupt(tmp_path):
    assert read_stats(tmp_path / "nope.parquet") is None
    artifact = tmp_path / "y.parquet"
    stats_path(artifact).write_text("{not json")
    assert read_stats(artifact) is None


def test_stats_stale(tmp_path):
    artifact = tmp_path / "z.parquet"
    artifact.write_bytes(b"")
    assert stats_stale(artifact)  # no sidecar yet
    write_stats(artifact, "patch", {})
    assert not stats_stale(artifact)
    later = artifact.stat().st_mtime + 60
    os.utime(artifact, (later, later))
    assert stats_stale(artifact)


def test_mirror_report(tmp_path):
    src = write_stats(tmp_path / "x.parquet", "patch", {"a": 1})
    assert mirror_report(src, {}) is None
    assert mirror_report(src, {"data": {"reports_dir": None}}) is None
    dest = mirror_report(src, {"data": {"reports_dir": str(tmp_path / "reports")}})
    assert dest == tmp_path / "reports" / "x.stats.json"
    assert dest.read_bytes() == src.read_bytes()


# --- counting conventions ---

def _frame() -> pd.DataFrame:
    # doc a: item 1 on two subscales of parent 1 (item must count once;
    # parent 1 counts as a scale node but not as item-bearing), item 2 orphan
    # (null scale). doc with null path: fallback key from source+title.
    return pd.DataFrame({
        "path": ["a.pdf", "a.pdf", "a.pdf", None, None],
        "corpus_source": ["s1", "s1", "s1", "s2", "s2"],
        "meta_title_raw": ["A", "A", "A", "B", "B"],
        "item_item_id": [1, 1, 2, 1, 2],
        "scale_id": [10, 11, None, 10, 10],
        "scale_id_path": [[1, 10], [1, 11], None, [10], [10]],
    })


def test_counts_frame_dedups_items_and_scales():
    c = report._counts_frame(_frame())
    assert c["placements"] == 5  # one per (item, scale) occurrence
    assert c["documents"] == 2  # a.pdf + the null-path fallback doc
    assert c["items"] == 4      # (a,1), (a,2), (B,1), (B,2)
    assert c["scales"] == 4     # nodes (a,1), (a,10), (a,11), (B,10)
    assert c["item-bearing scales"] == 3  # (a,10), (a,11), (B,10); orphan out


def test_scale_nodes_falls_back_without_path_column():
    df = _frame().drop(columns=["scale_id_path"])
    assert report._n_scale_nodes(df) == 3  # falls back to item-bearing count


def test_per_source_totals():
    rows = report._per_source(_frame())
    by_source = {r[0]: r for r in rows}
    # columns: placements, documents, scales, item-bearing scales, items
    assert by_source["s1"][1:] == [3, 1, 3, 2, 2]
    assert by_source["s2"][1:] == [2, 1, 1, 1, 2]
    assert by_source["**total**"][1:] == [5, 2, 4, 3, 4]


# --- distributions (M / SD) ---

def test_dist_row_reports_mean_sd_median_and_iqr_range():
    row = report._dist_row(pd.Series([1, 2, 3, 4]))
    assert report.DIST_HEADERS == ["n", "M", "SD", "min", "Mdn",
                                   "IQR (Q1–Q3)", "p95", "max"]
    assert row[0] == 4
    assert row[1] == "2.500"          # M
    assert row[2] == "1.291"          # sample SD (ddof=1)
    assert row[3] == "1.000"          # min
    assert row[4] == "2.500"          # Mdn
    assert row[5] == "1.750–3.250"    # IQR as the range Q1-Q3, not its width
    assert (row[6], row[7]) == ("3.850", "4.000")  # p95, max


def test_dist_row_degenerate_inputs():
    assert report._dist_row(pd.Series([7]))[2] == "n/a"   # SD needs n > 1
    assert report._dist_row(pd.Series([], dtype=float))[0] == "(no values)"
    assert report._dist_row(None)[0] == "(not available)"


def test_ms_cell():
    # M (SD) · Mdn [Q1–Q3]
    assert report._ms(pd.Series([2, 4])) == "3.00 (1.41) · 3.00 [2.50–3.50]"
    assert report._ms(pd.Series([2])) == "2.00 (n/a) · 2.00 [2.00–2.00]"
    assert report._ms(None) == "n/a"


def _structure_frame() -> pd.DataFrame:
    # doc a: items 1+2 under scale 10, item 1 also under 11 (both children of
    # parent 1), item 3 orphan. doc c: a shell row, no item and no scale.
    return pd.DataFrame({
        "path": ["a.pdf", "a.pdf", "a.pdf", "a.pdf", "c.pdf"],
        "corpus_source": ["s1"] * 5,
        "meta_title_raw": ["A", "A", "A", "A", "C"],
        "item_item_id": [1, 1, 2, 3, None],
        "scale_id": [10, 11, 10, None, None],
        "scale_id_path": [[1, 10], [1, 11], [1, 10], None, None],
    })


def test_placements_are_item_bearing_rows_not_parquet_rows():
    df = _structure_frame()          # 5 rows, one of them an item-less shell
    assert len(df) == 5
    assert report._n_placements(df) == 4
    assert report._counts_frame(df)["placements"] == 4
    note = " ".join(report._shell_row_note(df))
    assert "5 rows" in note and "1 " in note
    # once the shell rows are filtered out, rows and placements coincide
    real = df.dropna(subset=["item_item_id"])
    assert report._shell_row_note(real) == []


def test_structure_series_separates_items_from_placements():
    s = report._structure_series(_structure_frame())
    # item 1 sits on two scales: 3 items but 4 placements in a.pdf
    assert dict(s["items per document"]) == {"a.pdf": 3, "c.pdf": 0}
    assert dict(s["placements per document"]) == {"a.pdf": 4, "c.pdf": 0}
    assert sorted(s["placements per item (scales an item sits on)"]) == \
        [1, 1, 2]


def test_structure_series_counts_empty_units_too():
    s = report._structure_series(_structure_frame())
    # the shell document counts as a zero, so means are over both documents
    assert dict(s["items per document"]) == {"a.pdf": 3, "c.pdf": 0}
    assert dict(s["scales per document"]) == {"a.pdf": 3, "c.pdf": 0}
    assert dict(s["item-bearing scales per document"]) == {"a.pdf": 2,
                                                           "c.pdf": 0}
    # subtree: parent 1 inherits items 1+2, leaf 10 has 1+2, leaf 11 has 1
    assert sorted(s["items per scale (subtree)"]) == [1, 2, 2]
    # direct: only the items whose immediate scale_id is the node
    assert sorted(s["items per item-bearing scale (direct)"]) == [1, 2]


def test_structure_series_skips_absent_columns():
    df = _structure_frame().drop(columns=["scale_id_path", "scale_id"])
    s = report._structure_series(df)
    assert list(s) == ["items per document", "placements per document",
                       "placements per item (scales an item sits on)"]
    assert report._structure_series(pd.DataFrame()) == {}
    assert "no structural columns" in report._structure_table(
        pd.DataFrame({"path": []}))[0]


def test_structure_tables_render():
    df = _structure_frame()
    table = report._structure_table(df)
    assert table[0].startswith("| unit | n | M | SD |")
    assert any(ln.startswith("| items per document |") for ln in table)
    ms = report._structure_ms_table(report._by_source_frames(df))
    assert ms[0] == "| unit | s1 | **total** |"
    # items per document: M (SD) · Mdn [Q1–Q3] over a.pdf (3) and c.pdf (0)
    assert any("1.50 (2.12) · 1.50 [0.75–2.25]" in ln for ln in ms)


# --- nesting of the scale hierarchy ---

def _nesting_frame() -> pd.DataFrame:
    # doc a: two levels (parent 1 -> leaves 10/11) plus an orphan item.
    # doc b: a flat top-level scale only. doc c: shell row, no scale at all.
    return pd.DataFrame({
        "path": ["a.pdf", "a.pdf", "a.pdf", "b.pdf", "c.pdf"],
        "corpus_source": ["s1", "s1", "s1", "s2", "s2"],
        "meta_title_raw": ["A", "A", "A", "B", "C"],
        "item_item_id": [1, 1, 3, 1, None],
        "scale_id": [10, 11, None, 20, None],
        "scale_id_path": [[1, 10], [1, 11], None, [20], None],
    })


def test_doc_nesting_is_the_deepest_node_per_document():
    depth = report._doc_nesting(_nesting_frame())
    # scale-less documents count as 0 rather than dropping out
    assert dict(depth) == {"a.pdf": 2, "b.pdf": 1, "c.pdf": 0}


def test_doc_nesting_falls_back_to_scale_depth_column():
    df = _nesting_frame().drop(columns=["scale_id_path"])
    df["scale_depth"] = [2, 2, None, 1, None]
    assert dict(report._doc_nesting(df)) == {"a.pdf": 2, "b.pdf": 1,
                                             "c.pdf": 0}
    assert report._doc_nesting(df.drop(columns=["scale_depth"])) is None


def test_node_depths_cover_parents_once():
    depths = report._node_depths(_nesting_frame())
    # nodes (a,1) at level 1, (a,10) and (a,11) at level 2, (b,20) at level 1
    assert sorted(depths) == [1, 1, 2, 2]


def test_counts_by_level_fills_gaps_and_totals():
    table = report._counts_by_level(pd.Series([0, 2, 2]), "levels of nesting")
    assert table[0] == "| levels of nesting | count | share | cumulative |"
    body = [ln for ln in table[2:]]
    assert body[0] == "| 0 | 1 | 33.3% | 33.3% |"
    assert body[1] == "| 1 | 0 | 0.0% | 33.3% |"   # unobserved level kept
    assert body[2] == "| 2 | 2 | 66.7% | 100.0% |"
    assert body[3] == "| **total** | 3 | 100.0% |  |"


def test_nesting_lines_render_all_blocks():
    lines = report._nesting_lines(_nesting_frame())
    text = "\n".join(lines)
    assert "Levels of nesting per instrument" in text
    assert "Instruments per level of nesting:" in text
    assert "| levels of nesting | s1 | s2 | **total** |" in text
    assert "Scale nodes per hierarchy level" in text
    # one instrument at level 2 (s1), one each at 0 and 1 (s2)
    assert "| 2 | 1 | 0 | 1 |" in text
    assert report._nesting_lines(pd.DataFrame({"path": []}))[0].startswith(
        "(no scale-hierarchy columns")


def test_keys_without_fallback_columns():
    df = pd.DataFrame({"path": ["a", "b", "b"]})
    assert report._n_docs(df) == 2


def test_flatten_nested():
    flat = dict(report._flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}}))
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}


# --- sections degrade gracefully ---

def test_sections_with_missing_artifacts(tmp_path):
    cfg = {"data": {
        "extractions": str(tmp_path / "nope.parquet"),
        "partials": [str(tmp_path / "gone.parquet")],
        "assemble": {k: str(tmp_path / f"{k}.parquet")
                     for k in ("exploded", "combined", "patched",
                               "postprocessed", "embedded", "pooled")},
    }}
    for build in report.SECTIONS:
        title, lines = build(cfg)
        assert isinstance(title, str) and isinstance(lines, list)
    lines = report.run(cfg)
    assert any("WARNING" in ln or "unavailable" in ln for ln in lines)


def test_sidecar_section_missing_and_stale(tmp_path):
    patched = tmp_path / "patched.parquet"
    cfg = {"data": {"assemble": {"patched": str(patched)}}}
    lines = report._sidecar_section(cfg, "patch", "patched", "rerun-cmd")
    assert any("unavailable" in ln for ln in lines)

    patched.write_bytes(b"")
    write_stats(patched, "patch", {"schema.blank_cells_nulled": 3})
    later = patched.stat().st_mtime + 60
    os.utime(patched, (later, later))
    lines = report._sidecar_section(cfg, "patch", "patched", "rerun-cmd")
    assert any("WARNING" in ln for ln in lines)
    assert any("schema.blank_cells_nulled" in ln for ln in lines)


# --- end-to-end on synthetic parquets ---

def _write_store(tmp_path):
    path = tmp_path / "store.parquet"
    pd.DataFrame({
        "path": ["a.pdf", "b.pdf", "c.pdf"],
        "has_errors": [False, True, False],
        "items_extractor_content": ["{}", None, "{}"],
        "items_extractor_error": [None, "boom", None],
    }).to_parquet(path)
    return path


def _write_exploded(tmp_path, name, paths):
    path = tmp_path / name
    pd.DataFrame({
        "path": paths,
        "meta_title_raw": ["T"] * len(paths),
        "item_item_id": list(range(1, len(paths) + 1)),
        "scale_id": [1] * len(paths),
        "scale_id_path": [[1]] * len(paths),
    }).to_parquet(path)
    return path


def test_run_end_to_end(tmp_path):
    store = _write_store(tmp_path)
    exploded = _write_exploded(tmp_path, "apa-extractions-exploded.parquet",
                               ["a.pdf", "c.pdf"])  # b.pdf missing -> stale
    partial = _write_exploded(tmp_path, "web-extractions-exploded.parquet",
                              ["x.url"])
    combined = tmp_path / "combined.parquet"
    pd.DataFrame({
        "corpus_source": ["apa", "apa", "web"],
        "path": ["a.pdf", "c.pdf", "x.url"],
        "meta_title_raw": ["T", "T", "T"],
        "item_item_id": [1, 1, 1],
        "scale_id": [1, 1, 1],
        "scale_id_path": [[1], [1], [1]],
        "bucket": ["scaled", "scaled", "scaled"],
    }).to_parquet(combined)
    cfg = {"data": {
        "extractions": str(store),
        "partials": [str(partial)],
        "assemble": {"exploded": str(exploded), "combined": str(combined),
                     "patched": str(tmp_path / "patched.parquet"),
                     "postprocessed": str(tmp_path / "post.parquet"),
                     "embedded": str(tmp_path / "embedded.parquet"),
                     "pooled": str(tmp_path / "pooled.parquet")},
    }}
    lines = report.run(cfg)
    text = "\n".join(lines)
    assert "documents considered: **3**" in text
    assert "1 store documents contribute no exploded rows" in text
    assert not list(tmp_path.glob("*.md"))  # no output_path, nothing written

    out = tmp_path / "published" / "corpus.md"
    lines = report.run(cfg, output_path=out)
    assert out.read_text(encoding="utf-8") == "\n".join(lines) + "\n"


def test_run_fidelity_reads_checkable_column(tmp_path):
    # Doc b has a long text layer but prints its items as images; run() must read pdf_text_checkable.
    embedded = tmp_path / "embedded.parquet"
    pd.DataFrame({
        "corpus_source": ["apa", "apa"],
        "path": ["a.pdf", "b.pdf"],
        "meta_title_raw": ["T", "T"],
        "item_item_id": [1, 1],
        "pdf_match_edit_distance_norm": [0.0, 0.9],
        "pdf_text_chars": [5000, 400],
        "pdf_text_checkable": pd.array([True, False], dtype="boolean"),
    }).to_parquet(embedded)
    cfg = {"data": {"assemble": {"embedded": str(embedded)}}}
    text = "\n".join(report.run(cfg))
    assert "| items — checkable (usable text layer) | 1 |" in text
    assert "| ≥ 1 checkable item below 95% | 0 | 1 |" in text


# --- appendix ---

def test_appendix_is_static_and_covers_key_columns():
    title, lines = report.section_appendix({})  # data-independent: empty cfg
    assert "Appendix" in title
    text = "\n".join(lines)
    # identity rules
    assert "doc::<corpus_source>::<meta_title_raw>" in text
    assert "(document key, item_item_id)" in text
    # every column the report's tables/paragraphs reference has an entry
    for col in ("path", "corpus_source", "bucket", "item_item_id",
                "scale_id", "scale_id_path", "is_patched", "meta_language",
                "permissions_category", "pdf_match_edit_distance_norm",
                "flag_item_count_deviation", "flag_scale_count_deviation",
                "flag_item_text_deviation", "record_item_count",
                "is_instrument", "n_items", "n_scales"):
        assert col in text, col


def test_appendix_is_last_numbered_section(tmp_path):
    cfg = {"data": {"extractions": str(tmp_path / "nope.parquet"),
                    "partials": [], "assemble": {}}}
    text = "\n".join(report.run(cfg))
    n = len(report.SECTIONS)
    assert f"## {n}. Appendix" in text
    assert "Appendix (last section)" in text  # header pointer present


def test_fidelity_lines_units_and_threshold_counts():
    # doc a: one verbatim item placed twice + one at 94% -> mean 97%, min 94%
    # doc b: one item at 50%, translated; doc c: thin text layer (not checkable)
    df = pd.DataFrame({
        "path": ["a", "a", "a", "b", "c"],
        "item_item_id": [1, 1, 2, 1, 1],
        "pdf_match_edit_distance_norm": [0.0, 0.0, 0.06, 0.5, 0.9],
        "pdf_text_chars": [5000, 5000, 5000, 5000, 10],
        "flag_item_translated": [False, False, False, True, False],
    })
    text = "\n".join(report._fidelity_lines(df, df["path"]))
    assert "similarity = 1 − normalized distance" in text
    assert "| items — all measured | 4 |" in text  # placements de-duplicated
    assert "| items — checkable (usable text layer) | 3 |" in text
    assert "| items — checkable, English originals only | 2 |" in text
    assert "| placements — all measured (stored grain, reference) | 5 |" in text
    # "≥ 1 item below" counts a and b; a's mean is taken over items, not placements
    assert "| ≥ 1 checkable item below 95% | 2 | 2 | 100.0% |" in text
    assert "| mean item similarity below 95% (all measured) | 2 | 3 |" in text
    # items printed as images: pdf_match marks doc b not checkable despite its long text layer
    df["pdf_text_checkable"] = pd.array([True, True, True, False, False], dtype="boolean")
    text = "\n".join(report._fidelity_lines(df, df["path"]))
    assert "| items — checkable (usable text layer) | 2 |" in text
    assert "| ≥ 1 checkable item below 95% | 1 | 1 | 100.0% |" in text
