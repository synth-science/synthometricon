"""Patch stage: pure helpers and step invariants; ``CrossrefClient._get`` is monkeypatched (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from assemble import patch
from assemble.patch import (
    Citation,
    Ctx,
    Lookup,
    classify_source,
    doi_from_stem,
    doi_from_url,
    doi_in_text,
    fix_csv_corruption,
    format_authors_apa,
    normalize_language,
    page1_dois,
    parse_citation,
    parse_pipe_authors,
    permissions_category,
    strip_item_number,
    title_language,
    verify_candidate,
    _unescape_html_fixpoint,
)


def ctx(**cfg) -> Ctx:
    return Ctx(cfg=cfg, full_cfg={})


# --- Pure helpers ---

def test_psyctests_doi_re_in_sync():
    """The page-1 record-DOI regex must match the meta extractor's."""
    from extraction.extractors.meta import _DOI_RE

    assert patch._PSYCTESTS_DOI_RE.pattern == _DOI_RE.pattern


def test_page1_dois():
    text = (
        "Narcissistic Personality Inventory\n"
        "doi: https://dx.doi.org/10.1037/t12345-000\n"
        "Original Publication:\n"
        "Raskin, R. (1988). doi: https://dx.doi.org/10.1037/0022-3514.54.5.890.\n"
    )
    assert page1_dois(text) == ("10.1037/t12345-000", "10.1037/0022-3514.54.5.890")
    # record DOI only — the record must never double as the source DOI
    record_only = "doi: http://dx.doi.org/10.1037/t99999-000\nno more dois"
    assert page1_dois(record_only) == ("10.1037/t99999-000", None)


def test_doi_from_stem():
    assert doi_from_stem("/x/999901528_full_001.pdf") == "10.1037/t01528-000"
    assert doi_from_stem("aligns/999901528_audit.url") == "10.1037/t01528-000"
    assert doi_from_stem("web-retrieved/999900741_bdi.url") == "10.1037/t00741-000"
    assert doi_from_stem("no_id_here.pdf") is None
    assert doi_from_stem(None) is None


def test_doi_in_text():
    cite = "Beck, A. T. (1961). An inventory. doi: https://dx.doi.org/10.1037/H0041234."
    assert doi_in_text(cite) == "10.1037/h0041234"
    assert doi_in_text("record doi 10.1037/t12345-000 only") is None
    assert doi_in_text("Supplied by author.") is None


def test_doi_from_url():
    assert doi_from_url("https://doi.org/10.1234/ABC.5") == "10.1234/abc.5"
    assert doi_from_url("https://doi.org/10.1002/j%2E123/x") == "10.1002/j.123/x"
    assert doi_from_url("https://example.com/scale.pdf") is None


def test_classify_source():
    assert classify_source("https://example.com/a?b=1") == ("url", "https://example.com/a?b=1")
    assert classify_source("scalesandmeasures.net (clinical)") == (
        "domain", "https://scalesandmeasures.net")
    assert classify_source("www.psychologytools.com") == (
        "domain", "https://www.psychologytools.com")
    kind, value = classify_source(
        "Berger, B. E. (2001). Measuring stigma. Research in Nursing, 24, 518-529.")
    assert kind == "citation" and value.startswith("Berger")
    assert classify_source("yorku")[0] == "junk"
    assert classify_source("newNEOKey.htm")[0] == "junk"


def test_fix_csv_corruption():
    assert fix_csv_corruption(',"Travis, R. (1993). The MOS alienation scale.') == \
        "Travis, R. (1993). The MOS alienation scale."
    assert fix_csv_corruption('measuring ""rejection"" among youths') == \
        'measuring "rejection" among youths'


def test_parse_citation():
    cit = parse_citation(
        "Beck, A. T., & Ward, C. H. (1961). An inventory for measuring "
        "depression. Archives of General Psychiatry, 4, 561-571.")
    assert cit.family == "Beck"
    assert cit.year == 1961
    assert cit.title == "An inventory for measuring depression"
    assert parse_citation("no year anywhere").family is None


def _work(doi="10.1000/x", title="An inventory for measuring depression",
          family="Beck", year=1961, venue="Archives of General Psychiatry"):
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"family": family, "given": "A. T."}],
        "issued": {"date-parts": [[year]]},
        "container-title": [venue],
    }


def test_verify_candidate():
    cit = Citation(family="Beck", year=1961,
                   title="An inventory for measuring depression")
    ok = dict(title_threshold=0.90, year_tolerance=1)
    assert verify_candidate(cit, _work(), **ok)
    assert verify_candidate(cit, _work(year=1962), **ok)  # within tolerance
    assert not verify_candidate(cit, _work(year=1965), **ok)
    assert not verify_candidate(cit, _work(family="Smith"), **ok)
    assert not verify_candidate(cit, _work(title="A totally different paper"), **ok)


def test_lookup_modes():
    cite = Lookup(citation="Beck, A. T. (1961). An inventory. Archives, 4.")
    assert cite.mode == "citation"
    assert cite.key().startswith("cite::")
    struct = Lookup(title="Beck Depression Inventory", family="Beck", year=1961)
    assert struct.mode == "structured"
    assert struct.key().startswith("struct::")
    assert Lookup(title="Some Scale").mode == "insufficient"


def test_parse_pipe_authors():
    assert parse_pipe_authors("Beck | A. T. | Ward | C. H.") == \
        "Beck, A. T., & Ward, C. H."
    assert parse_pipe_authors("Beck, A. T.|Ward, C. H.|Mendelson, M.") == \
        "Beck, A. T., Ward, C. H., & Mendelson, M."
    # affiliation and email tokens are dropped
    assert parse_pipe_authors(
        "Beck | A. T. | University of Pennsylvania | beck@upenn.edu | Ward | C. H."
    ) == "Beck, A. T., & Ward, C. H."
    assert parse_pipe_authors("dept@uni.edu | Department of Psychology") is None


def test_format_authors_apa():
    assert format_authors_apa([{"family": "Beck", "given": "Aaron T."}]) == \
        "Beck, A. T."
    assert format_authors_apa([{"family": "A", "given": "B"},
                               {"family": "C", "given": "D"}]) == "A, B., & C, D."
    assert format_authors_apa([]) is None


def test_normalize_language():
    assert normalize_language("en") == "en"
    assert normalize_language("EN") == "en"
    assert normalize_language("English") == "en"
    assert normalize_language("English (French title variant)") == "en"
    assert normalize_language("en-US") == "en"
    assert normalize_language("en-GB-AU-US") == "en"
    assert normalize_language("en', 'type': 'choice', 'options': ['Yes']}") == "en"
    assert normalize_language("Persian") == "fa"
    assert normalize_language("open") is None
    assert normalize_language("choice") is None
    assert normalize_language(None) is None


def test_title_language():
    assert title_language("Coping Scale — Kurdish Version") == ("ku", "Kurdish Version")
    assert title_language("some scale--spanish version") == ("es", "Spanish Version")
    assert title_language("Scale — English Version") is None  # en is the default
    assert title_language("Adapted Version of a scale") is None


def test_strip_item_number():
    assert strip_item_number("1. Many opportunities await me.") == \
        "Many opportunities await me."
    assert strip_item_number("2) Best gaze observed.") == "Best gaze observed."
    assert strip_item_number("a. I don't feel self-conscious.") == \
        "I don't feel self-conscious."
    # arithmetic content and bare fragments must not be mangled
    assert strip_item_number("17 - [ 5 - (7 - 9) ] =") == "17 - [ 5 - (7 - 9) ] ="
    assert strip_item_number("3)") == "3)"


def test_permissions_category():
    assert permissions_category(
        "Test content may be reproduced and used for non-commercial research "
        "and educational purposes without seeking written permission."
    ) == "free_reproduction"
    assert permissions_category(
        "Test content may be reproduced and used for non-commercial research "
        "and educational purposes. APA believes that this content is in the "
        "public domain."
    ) == "public_domain"
    assert permissions_category("Contact Publisher and Corresponding Author.") == \
        "contact_publisher_and_author"
    assert permissions_category("Contact Publisher.") == "contact_publisher"
    assert permissions_category("Contact Corresponding Author") == "contact_author"
    assert permissions_category("Not Specified") == "not_specified"
    assert permissions_category("May use for Research/Teaching.") == "research_teaching"
    assert permissions_category("something else") is None
    assert permissions_category(None) is None


def test_unescape_html_fixpoint():
    assert _unescape_html_fixpoint("Cohen &amp;amp; Hoberman") == "Cohen & Hoberman"
    assert _unescape_html_fixpoint("&amp;amp;quot;x&amp;amp;quot;") == '"x"'


# --- Step invariants ---

def test_step_schema_blanks_and_ints():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "semanticnet"],
        "item_item_text": ["  ", ""],
        "item_item_id": [1.0, 2.0],
    })
    out = patch.step_schema(df, ctx())
    assert out["item_item_text"].isna().all()  # apa blanks null too
    assert str(out["item_item_id"].dtype) == "Int64"


def test_step_text_repair():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "apa-psyctests", "scale-hunt",
                          "semanticnet"],
        "item_item_text": ["None", "Real item.", "Item 3 statement group", "ok"],
        "version_attached": ["none", "Full Test", None, None],
        "meta_source_raw": ["Supplied by Author.", "Author supplied", None,
                            "cross?validation of the scale¶"],
        "meta_title_raw": ["-", "Fine Title", "fine", "fine"],
        "retrieval_notes": [None, None, "Cohen &amp;amp; Hoberman", None],
        "item_admin_note": ["line1\\nline2", None, None, None],
    })
    out = patch.step_text_repair(df.copy(), ctx())
    assert pd.isna(out.loc[0, "item_item_text"])          # sentinel
    assert out.loc[1, "item_item_text"] == "Real item."
    assert pd.isna(out.loc[2, "item_item_text"])          # placeholder
    assert pd.isna(out.loc[0, "version_attached"])        # 'none' sentinel
    assert out.loc[1, "version_attached"] == "Full Test"
    assert out.loc[0, "meta_source_raw"] == "Supplied by author."   # canonical
    assert out.loc[1, "meta_source_raw"] == "Supplied by author."
    assert out.loc[3, "meta_source_raw"] == "cross-validation of the scale"
    assert pd.isna(out.loc[0, "meta_title_raw"])          # '-' sentinel
    assert out.loc[2, "retrieval_notes"] == "Cohen & Hoberman"
    assert out.loc[0, "item_admin_note"] == "line1\nline2"


def test_step_doi_psyctests():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "apa-psyctests", "aligns", "scale-hunt"],
        "path": ["/p/999900001_full_001.pdf", "/p/999900002_full_001.pdf",
                 "aligns/999901528_audit.url", None],
        "meta_doi_raw": ["10.1037/t00001-000", "10.1037/t99999-000", None, None],
        "doi": ["10.1037/t00001-000", "10.1037/t99999-000", None, None],
    })
    out = patch.step_doi_psyctests(df.copy(), ctx())
    assert out.loc[0, "doi_psyctests"] == "10.1037/t00001-000"     # stem agrees
    assert pd.isna(out.loc[1, "doi_psyctests"])  # stem says t00002 -> disagreement
    assert out.loc[2, "doi_psyctests"] == "10.1037/t01528-000"     # partial stem fill
    assert pd.isna(out.loc[3, "doi_psyctests"])                    # no path
    assert "doi" not in out.columns


def test_step_source_fields():
    url = "https://example.com/scale"
    df = pd.DataFrame({
        "corpus_source": ["aligns", "semanticnet", "semanticnet", "semanticnet"],
        "meta_source_raw": [url, "Berger, B. E. (2001). Measuring stigma in "
                            "people with HIV. Research in Nursing, 24, 518-529.",
                            "yorku", "real citation text (1999) kept intact ok"],
        "source_url": [url, "Berger, B. E. (2001). Measuring stigma in people "
                       "with HIV. Research in Nursing, 24, 518-529.",
                       "yorku", "https://elsewhere.org"],
        "retrieval_notes": [None] * 4,
    })
    out = patch.step_source_fields(df.copy(), ctx())
    assert pd.isna(out.loc[0, "meta_source_raw"])       # url migrated
    assert out.loc[0, "source_url"] == url
    assert out.loc[1, "meta_source_raw"].startswith("Berger")   # citation kept
    assert pd.isna(out.loc[1, "source_url"])            # byte-copy nulled
    assert pd.isna(out.loc[2, "meta_source_raw"]) and pd.isna(out.loc[2, "source_url"])
    # source_url differs from raw -> untouched
    assert out.loc[3, "source_url"] == "https://elsewhere.org"
    assert out.loc[3, "meta_source_raw"] == "real citation text (1999) kept intact ok"


def test_step_source_doi_fill_only_null():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "apa-psyctests", "scale-hunt"],
        "meta_source_raw": [
            "Beck (1961). Inventory. doi: https://dx.doi.org/10.1037/h0001",
            "Smith (1990). Another. doi: https://dx.doi.org/10.1037/h0002",
            None],
        "source_url": [None, None, "https://doi.org/10.1002/da.123"],
        "meta_source_doi": [None, "10.9999/preset", None],
    })
    out = patch.step_source_doi(df.copy(), ctx())
    assert out.loc[0, "meta_source_doi"] == "10.1037/h0001"
    assert out.loc[1, "meta_source_doi"] == "10.9999/preset"   # never overwritten
    assert out.loc[2, "meta_source_doi"] == "10.1002/da.123"


def test_step_crossref(monkeypatch, tmp_path):
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "apa-psyctests", "aligns", "aligns",
                          "scale-hunt"],
        "path": ["/p/a.pdf", "/p/a.pdf", "al/b.url", "al/b.url", "wr/c.url"],
        "meta_title_raw": ["T", "T", "Beck Depression Inventory"] * 1 + ["Beck Depression Inventory", "Other Scale"],
        "meta_source_raw": [None, None,
                            "Beck, A. T., & Ward, C. H. (1961). An inventory "
                            "for measuring depression. Archives of General "
                            "Psychiatry, 4, 561-571."] * 1 + [
                            "Beck, A. T., & Ward, C. H. (1961). An inventory "
                            "for measuring depression. Archives of General "
                            "Psychiatry, 4, 561-571.", None],
        "meta_source_doi": ["10.1/known", None, None, None, None],
        "meta_authors_raw": [None] * 5,
        "meta_publication_year_raw": [None] * 5,
        "meta_journal_venue_raw": [None] * 5,
    })
    monkeypatch.setattr(patch.CrossrefClient, "_get",
                        lambda self, url, params: {"message": {"items": [_work()]}})
    cfg = ctx(crossref={"enabled": True, "mailto": "t@example.org",
                        "cache": str(tmp_path / "cache.json")})
    out = patch.step_crossref(df.copy(), cfg)
    # intra-doc propagation
    assert out.loc[1, "meta_source_doi"] == "10.1/known"
    # citation-mode accept fills the whole doc, fill-only-null
    assert out.loc[2, "meta_source_doi"] == "10.1000/x"
    assert out.loc[3, "meta_source_doi"] == "10.1000/x"
    # bibliographic backfill lands on partial rows
    assert out.loc[2, "meta_authors_raw"] == "Beck, A. T."
    assert out.loc[2, "meta_publication_year_raw"] == 1961
    # doc without citation or full anchors -> untouched
    assert pd.isna(out.loc[4, "meta_source_doi"])
    assert (tmp_path / "cache.json").exists()


def test_step_crossref_offline_uses_cache_only(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(patch.CrossrefClient, "_get",
                        lambda self, url, params: calls.append(url))
    df = pd.DataFrame({
        "corpus_source": ["aligns"],
        "path": ["al/b.url"],
        "meta_title_raw": ["X"],
        "meta_source_raw": ["Beck, A. T. (1961). An inventory for measuring "
                            "depression. Archives, 4, 561."],
        "meta_source_doi": [None],
        "meta_authors_raw": [None],
        "meta_publication_year_raw": [None],
        "meta_journal_venue_raw": [None],
    })
    cfg = Ctx(cfg={"crossref": {"enabled": True, "mailto": "t@example.org"}},
              full_cfg={}, report_only=True)
    out = patch.step_crossref(df.copy(), cfg)
    assert not calls                       # report-only never queries
    assert pd.isna(out.loc[0, "meta_source_doi"])


def test_step_language():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "scale-hunt", "scale-hunt", "aligns"],
        "path": ["/p/a.pdf", "wr/b.url", "wr/c.url", "al/d.url"],
        "meta_language": ["en", "English", "Spanish", None],
        "meta_language_raw": ["French", "English", "Spanish", None],
        "item_language": ["en', 'type': 'choice', 'options': ['Yes', 'No']}",
                          "English", "Spanish", None],
        "meta_title_raw": ["T", "t", "scale--kurdish version", "t"],
        "item_item_text": ["ok", "ok", "ok", None],
    })
    out = patch.step_language(df.copy(), ctx())
    assert out.loc[0, "item_language"] == "en"           # leakage sanitized
    assert out.loc[0, "meta_language_raw"] == "French"   # apa note kept
    assert out.loc[1, "meta_language"] == "en"           # name normalized
    assert pd.isna(out.loc[1, "meta_language_raw"])      # copy-fill reset
    assert out.loc[2, "meta_language"] == "ku"           # title outranks (partial)
    assert out.loc[2, "meta_language_raw"] == "Kurdish Version"
    assert out.loc[3, "meta_language"] == "en"           # partial default


def test_step_permissions():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "aligns", "semanticnet"],
        "meta_permissions_raw": ["Contact Publisher.", "Not Specified",
                                 "May use for Research/Teaching"],
        "permissions_category": ["wrong_legacy", None, None],
    })
    out = patch.step_permissions(df.copy(), ctx())
    assert out.loc[1, "meta_permissions_raw"] == "Not Specified."   # period added
    assert list(out["permissions_category"]) == [
        "contact_publisher", "not_specified", "research_teaching"]


def test_step_fabricated_guarded():
    df = pd.DataFrame({
        "corpus_source": ["aligns", "aligns", "semanticnet", "apa-psyctests"],
        "item_item_type": ["rating_scale", "rating_scale", "open", "choice"],
        "source_grade": [None, None, "semanticnet", None],
    })
    out = patch.step_fabricated(df.copy(), ctx())
    # item_item_type's constant fill is the corpus default, not fabrication.
    assert list(out["item_item_type"]) == df["item_item_type"].tolist()
    assert pd.isna(out.loc[2, "source_grade"])


def test_step_item_type():
    df = pd.DataFrame({
        "corpus_source": ["aligns", "semanticnet", "scale-hunt",
                          "apa-psyctests", "aligns"],
        "item_item_type": [None, None, None, None, "open"],
    })
    out = patch.step_item_type(df.copy(), ctx())
    # fill-only-null, aligns/semanticnet only; existing values untouched
    assert out.loc[[0, 1], "item_item_type"].tolist() == ["rating_scale"] * 2
    assert out.loc[[2, 3], "item_item_type"].isna().all()
    assert out.loc[4, "item_item_type"] == "open"


def test_step_items():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "scale-hunt", "scale-hunt", "aligns"],
        "path": ["/p/a.pdf", "wr/b.url", "wr/b.url", "al/c.url"],
        "meta_title_raw": ["T"] * 4,
        "item_item_id": [1, 0, 1, 5],
        "item_options": [[], ["a"], None, []],
        "item_item_text": ["1. apa text stays", "1. stripped here",
                           "plain", "ok item"],
    })
    out = patch.step_items(df.copy(), ctx())
    assert pd.isna(out.loc[0, "item_options"])          # [] -> null on apa too
    assert out.loc[1, "item_options"] == ["a"]
    assert list(out.loc[[1, 2], "item_item_id"]) == [1, 2]   # 0-based shifted
    assert out.loc[3, "item_item_id"] == 5                   # not 0-based
    assert out.loc[0, "item_item_text"] == "1. apa text stays"  # apa untouched
    assert out.loc[1, "item_item_text"] == "stripped here"


def test_step_authors():
    df = pd.DataFrame({
        "corpus_source": ["scale-hunt", "scale-hunt", "apa-psyctests"],
        "meta_authors_raw": ["Beck | A. T. | Ward | C. H.",
                             "@@@|###", "Raskin, R., & Hall, C. S."],
    })
    out = patch.step_authors(df.copy(), ctx())
    assert out.loc[0, "meta_authors_raw"] == "Beck, A. T., & Ward, C. H."
    assert out.loc[1, "meta_authors_raw"] == "@@@|###"        # guarded
    assert out.loc[2, "meta_authors_raw"] == "Raskin, R., & Hall, C. S."


def test_split_version_qualifier():
    from assemble.patch import split_version_qualifier as split
    assert split("servqual instrument--adapted version") == \
        ("servqual instrument", "adapted version")
    assert split("Geriatric Depression Scale—Short Form") == \
        ("Geriatric Depression Scale", "Short Form")
    assert split("Disgust Scale--Version 2 (Short Form)") == \
        ("Disgust Scale", "Version 2; Short Form")
    assert split("Sensation Seeking Scale, Form V") == \
        ("Sensation Seeking Scale", "Form V")
    assert split("HEXACO Personality Inventory-Revised") == \
        ("HEXACO Personality Inventory", "Revised")
    assert split("pittsburgh sleep quality index-short version") == \
        ("pittsburgh sleep quality index", "short version")
    # false positives that must NOT split
    assert split("Screening for Somatoform Symptoms—7")[1] is None  # numeric
    assert split("Maladaptive and Adaptive Coping Style Questionnaire")[1] is None
    assert split("Behavior Identification Form--Gamer")[1] is None
    assert split("Self-Esteem Short Form")[1] is None  # mid-compound hyphen
    # en dash inside compound names is not a qualifier separator
    assert split("Structured Clinical Interview for DSM–IV Screen")[1] is None
    # a loose tail naming a sub-measure is not a version
    assert split("Youth Survey--Alcohol Self-Control Behavior Scale Revised")[1] \
        is None
    assert split(None) == (None, None)


def test_title_case_name():
    from assemble.patch import title_case_name as tc
    assert tc("aberrant behavior checklist") == "Aberrant Behavior Checklist"
    assert tc("need for touch scale") == "Need for Touch Scale"
    assert tc("Anticipation Of Relief From Withdrawal Or Dysphoria") == \
        "Anticipation of Relief From Withdrawal or Dysphoria"
    assert tc("UCLA Loneliness scale") == "UCLA Loneliness Scale"       # acronym kept
    assert tc("NEO-PI") == "NEO-PI"                                     # untouched
    assert tc("self-esteem needs") == "Self-Esteem Needs"               # hyphen parts
    assert tc("Left Arm Finger-to-Nose") == "Left Arm Finger-to-Nose"   # minor part
    assert tc("the big five inventory") == "The Big Five Inventory"     # first word
    assert tc("fear of the dark") == "Fear of the Dark"                 # last kept
    assert tc(None) is None


def test_step_version_split():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "semanticnet", "scale-hunt", "aligns"],
        "scale_name": ["servqual instrument--adapted version",
                       "Coping Scale--Short Form",
                       "Some Scale--Short Form",
                       "Plain Scale"],
        "meta_title_raw": ["servqual instrument--adapted version",
                           "Coping Scale--Short Form", None, None],
        "version": [None, None, "1985 original long-form", None],
    })
    out = patch.step_version_split(df.copy(), ctx())
    # relocated: identical qualifiers from both name columns dedupe
    assert out.loc[0, "scale_name"] == "servqual instrument"
    assert out.loc[0, "meta_title_raw"] == "servqual instrument"
    assert out.loc[0, "version"] == "Adapted Version"
    assert out.loc[1, "scale_name"] == "Coping Scale"
    assert out.loc[1, "version"] == "Short Form"
    # occupied version with different text: name left intact
    assert out.loc[2, "scale_name"] == "Some Scale--Short Form"
    assert out.loc[2, "version"] == "1985 original long-form"
    # no qualifier: untouched
    assert out.loc[3, "scale_name"] == "Plain Scale"
    assert pd.isna(out.loc[3, "version"])


def test_step_title_case():
    df = pd.DataFrame({
        "corpus_source": ["semanticnet", "apa-psyctests"],
        "scale_name": ["flow state scale", "UCLA Loneliness Scale"],
        "scale_construct_name": ["fear of failure", None],
        "meta_title_raw": ["flow state scale", "Fine Title"],
        "scale_name_path": [["flow state scale", "sub scale"], None],
    })
    out = patch.step_title_case(df.copy(), ctx())
    assert out.loc[0, "scale_name"] == "Flow State Scale"
    assert out.loc[0, "scale_construct_name"] == "Fear of Failure"
    assert out.loc[0, "meta_title_raw"] == "Flow State Scale"
    assert list(out.loc[0, "scale_name_path"]) == ["Flow State Scale", "Sub Scale"]
    assert out.loc[1, "scale_name"] == "UCLA Loneliness Scale"


def test_step_anomalies_mutates_nothing():
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "aligns"],
        "path": ["/p/999900001_full_001.pdf", None],
        "has_errors": [True, False],
        "scale_name": ["S", "S"],
        "item_item_id": [1, 1],
        "item_item_text": ["ab", "a proper item"],
        "item_item_type": ["choice", None],
        "item_options": [None, None],
        "saved_files": [None, "/Users/foreign/path.pdf"],
    })
    before = df.copy(deep=True)
    c = ctx()
    out = patch.step_anomalies(df, c)
    pd.testing.assert_frame_equal(before, out)
    assert any("shell rows" in line for line in c.report)


def test_run_steps_is_patched():
    df = pd.DataFrame({
        "corpus_source": ["semanticnet", "semanticnet", "semanticnet"],
        "item_item_text": ["fine", "None", "also fine"],
        "is_patched": [False, False, True],
    })
    c = ctx()
    out = patch._run_steps(df, ["text_repair"], c)
    assert list(out["is_patched"]) == [False, True, True]  # change | prior


def test_run_steps_noop_frame():
    df = pd.DataFrame({
        "corpus_source": ["semanticnet"],
        "item_item_text": ["fine"],
    })
    out = patch._run_steps(df, ["text_repair"], ctx())
    assert not out["is_patched"].any()


def test_step_reverse_coded():
    df = pd.DataFrame({
        "bucket": ["scaled", "scaled", "scaled", "orphan", "unscaled", None],
        "item_item_id": [1, 2, 3, 4, 5, None],
        # row 4 is an image-only item: no text, still an item
        "item_item_text": ["a", "b", "c", "d", None, None],
        "item_reverse_coded": pd.array(
            [None, True, False, None, None, None], dtype="boolean"),
    })
    out = patch.step_reverse_coded(df, ctx())
    assert list(out["item_reverse_coded"][:5]) == [
        False, True, False, False, False]
    assert out["item_reverse_coded"][5:].isna().all()  # itemless row kept null


def test_step_reverse_coded_missing_columns():
    df = pd.DataFrame({"bucket": ["scaled"]})
    out = patch.step_reverse_coded(df, ctx())
    assert "item_reverse_coded" not in out.columns


def test_step_pdf_full_text(monkeypatch, tmp_path):
    import extraction.pdf_io as pdf_io

    pdf = tmp_path / "999900001_full_001.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(pdf_io, "pdf_text_excl_first_page",
                        lambda p: "I plan ahead.\nI act on impulse.\n")
    df = pd.DataFrame({
        "corpus_source": ["apa-psyctests", "apa-psyctests", "semanticnet",
                          "apa-psyctests"],
        "path": [str(pdf), str(pdf), "semanticnet/x.url",
                 str(tmp_path / "gone.pdf")],
    })
    c = ctx()
    out = patch.step_pdf_full_text(df, c)
    assert out["pdf_full_text"].iloc[0] == "I plan ahead.\nI act on impulse.\n"
    assert out["pdf_full_text"].iloc[1] == out["pdf_full_text"].iloc[0]
    assert pd.isna(out["pdf_full_text"].iloc[2])  # partial row stays null
    assert pd.isna(out["pdf_full_text"].iloc[3])  # missing file
    assert any("1 missing" in l for l in c.report)


def test_changed_mask_ignores_pdf_full_text():
    """The carried PDF text must not flag every apa row as is_patched."""
    df = pd.DataFrame({"corpus_source": ["apa-psyctests"], "a": [1]})
    after = df.copy()
    after["pdf_full_text"] = ["full body text"]
    assert not patch._changed_mask(df, after).any()


# --- stats sidecar ---

def test_run_writes_stats_sidecar(tmp_path):
    from assemble.patch import run
    from assemble.stats import read_stats, stats_path

    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    pd.DataFrame({"path": ["a.pdf"], "corpus_source": ["apa-psyctests"],
                  "meta_title_raw": ["  "]}).to_parquet(inp)
    run({}, input_path=inp, output_path=out, steps="schema")
    payload = read_stats(out)
    assert payload["stage"] == "patch"
    assert payload["stats"]["schema.blank_cells_nulled"] == 1
    assert "is_patched.rows_changed_this_run" in payload["stats"]
    assert any("schema" in line for line in payload["report"])

    # report-only runs must write nothing
    stats_path(out).unlink()
    out2 = tmp_path / "out2.parquet"
    run({}, input_path=inp, output_path=out2, steps="schema",
        report_only=True)
    assert not out2.exists() and not stats_path(out2).exists()
