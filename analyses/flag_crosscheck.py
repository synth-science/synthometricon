"""Are the quality flags mostly false alarms? Cross-check the production flags of the 100 audited documents
against whether their production extraction matches the human coding.

Each audited document (tests/validation-fixtures) is rebuilt from the exploded extraction parquet, i.e. the
extraction the flags were computed on, and validated against its human coding with the repository's own
comparator (extraction.validity.validate_extraction, string similarity >= .80, as in the validity run).
Extraction samples at temperature 1, so the production extraction is a different draw than the one audited in
the validity run (logs/test-validity-20260815_172054.md) and can differ from it; the validity-run labels would
describe an extraction that was never flagged.

Outcome per document:
  pass           the production extraction matches the human coding
  fixture_error  it fails only where the validity run failed because of errors in the human coding (the 11
                 'FP' documents of the validity summary, plus 022, whose human coding merged multi-part
                 questions into single items). Any issue beyond those of the validity-run extraction, put
                 through the same rebuild, counts as a discrepancy.
  discrepancy    any other failure (a genuine extraction error)

A document counts as flagged if any of its rows in the postprocessed corpus carries a flag. Only audited
documents in the searchable (postprocessed) corpus enter the cross-tabulation. Construct names of scales
without items of their own are not stored in the exploded rows, so they are not compared.

With --control, the validity-run extractions are put through the same row explosion and rebuild; their
status should reproduce the logged validity status.

Usage (from the repository root):
    poetry run python analyses/flag_crosscheck.py <postprocessed.parquet> [--extraction <exploded.parquet>]
        [--records <csv or parquet with DOI, TestItemsAvailable>] [--control]

Output: analyses/output/flag_crosscheck.csv (no item text).
"""
import argparse
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from extraction.models.instrument import Instrument  # noqa: E402
from extraction.models.meta import Meta  # noqa: E402
from extraction.validity import load_full_fixture_with_tags, validate_extraction  # noqa: E402

VAL = REPO / "logs" / "test-validity-20260815_172054.md"
FIX = REPO / "tests" / "validation-fixtures"
OUT = REPO / "analyses" / "output" / "flag_crosscheck.csv"
EXTRACTION = Path("/Users/rubenarslan/research/synth-net-repo/analyses/corpus_coverage/data/final/"
                  "apa-psyctests-extractions-exploded.parquet")
FLAGS = ["flag_item_count_deviation", "flag_scale_count_deviation", "flag_item_text_deviation", "flag_item_translated"]
THRESHOLDS = {"string_similarity": 0.80}
# validity failures traced to the human coding ('FP' in the validity summary), plus 022 (merged multi-part questions)
FIXTURE_ERRORS = set("022 024 045 046 109 110 118 120 123 127 136 145".split())
TRANSIENT = {"018"}  # truncated output in the validity run only
ROW_COLS = ["path", "bucket", "scale_id_path", "scale_name_path", "scale_construct_name", "item_item_id",
            "item_item_text", "item_has_image", "item_item_type", "item_options", "item_admin_note",
            "item_language", "item_reverse_coded", "meta_language"]


def isna(v):
    return v is None or (np.isscalar(v) and pd.isna(v))


def doi_from_path(path):
    m = re.search(r"([0-9]{9})", Path(str(path)).name)
    return f"10.1037/t{m.group(1)[4:9]}-000" if m else None


# ---------------------------------------------------------------- validity log
def yaml_block(lines, start):
    block = []
    for ln in lines[start + 1:]:
        if ln.startswith("    ") or not ln.strip():
            block.append(ln)
        else:
            break
    return yaml.safe_load(textwrap.dedent("\n".join(block))) or {}


def parse_validity_log():
    """Per document: DOI, logged status and the resolved validity-run instrument (None if it failed)."""
    text = VAL.read_text()
    status = {doc: st for doc, st in re.findall(
        r"^\s+(\d{3})_101037t\d{5}000\.pdf\s+(PASS|FAIL)", text[text.index("SUITE VALIDITY SUMMARY"):], flags=re.M)}
    sections = re.split(r"\n=+\nFile \d+ / 100: \S*/(\d{3})_101037(t\d{5})000\.pdf\n=+\n", text)
    out = {}
    for i in range(1, len(sections) - 1, 3):
        doc, doi, body = sections[i], sections[i + 1], sections[i + 2]
        lines = body.split("\n")
        inst = next((yaml_block(lines, j) for j, ln in enumerate(lines) if ln.startswith("  Resolved instrument")), None)
        out[doc] = dict(doi=f"10.1037/{doi}-000", status=status[doc], inst=inst)
    return out


def validity_run_label(doc, status):
    if status == "PASS":
        return "pass"
    return "transient" if doc in TRANSIENT else "fixture_error" if doc in FIXTURE_ERRORS else "discrepancy"


# ---------------------------------------------------------------- rows <-> instrument
def rows_from_inst(inst):
    """Explode a resolved instrument into rows shaped like the extraction parquet."""
    rows = []

    def walk(scales, ids, names):
        for s in scales or []:
            ip, npth = ids + [s.get("id")], names + [s.get("scale_name")]
            for d in s.get("items") or []:
                rows.append(dict(bucket="scaled", scale_id_path=ip, scale_name_path=npth,
                                 scale_construct_name=s.get("construct_name"), item_item_id=d.get("item_id"),
                                 item_item_text=d.get("item_text"), item_has_image=d.get("has_image"),
                                 item_item_type=d.get("item_type"), item_options=d.get("options"),
                                 item_admin_note=d.get("admin_note"), item_language=d.get("language"),
                                 item_reverse_coded=d.get("reverse_coded")))
            walk(s.get("subscales"), ip, npth)

    walk(inst.get("scales"), [], [])
    for bucket, key in (("orphan", "orphan_items"), ("unscaled", "unscaled_items")):
        for d in inst.get(key) or []:
            rows.append(dict(bucket=bucket, item_item_id=d.get("id"), item_item_text=d.get("item_text"),
                             item_has_image=d.get("has_image"), item_item_type=d.get("item_type"),
                             item_options=d.get("options"), item_admin_note=d.get("admin_note"),
                             item_language=d.get("language"), item_reverse_coded=None))
    rows = pd.DataFrame(rows, columns=ROW_COLS[1:-1])
    rows["meta_language"] = (inst.get("meta") or {}).get("language")
    return rows


def _item(r, scaled):
    opts = None if isna(r.item_options) else list(r.item_options)
    d = dict(item_text=None if isna(r.item_item_text) else r.item_item_text,
             has_image=False if isna(r.item_has_image) else bool(r.item_has_image),
             item_type="rating_scale" if isna(r.item_item_type) else r.item_item_type, options=opts,
             admin_note=None if isna(r.item_admin_note) else r.item_admin_note,
             language="en" if isna(r.item_language) else r.item_language)
    if scaled:
        d.update(item_id=int(r.item_item_id),
                 reverse_coded=False if isna(r.item_reverse_coded) else bool(r.item_reverse_coded))
    else:
        d.update(id=None if isna(r.item_item_id) else int(r.item_item_id))
    return d


def inst_from_rows(g):
    """Rebuild the instrument tree of one document from its rows (scale paths, items, buckets)."""
    g = g[g.item_item_id.notna()]
    nodes, roots = {}, []
    for r in g[g.bucket == "scaled"].itertuples():
        ids, names, parent = list(r.scale_id_path), list(r.scale_name_path), None
        for k in range(len(ids)):
            key = tuple(ids[:k + 1])
            if key not in nodes:
                nodes[key] = dict(id=0 if isna(ids[k]) else int(ids[k]), scale_name=names[k], construct_name=None,
                                  items=[], subscales=[])
                (parent["subscales"] if parent else roots).append(nodes[key])
            parent = nodes[key]
        if not isna(r.scale_construct_name):
            parent["construct_name"] = r.scale_construct_name
        parent["items"].append(_item(r, scaled=True))
    inst = Instrument.model_validate(dict(
        scales=roots, unscaled_items=[_item(r, False) for r in g[g.bucket == "unscaled"].itertuples()],
        orphan_items=[_item(r, False) for r in g[g.bucket == "orphan"].itertuples()]))
    lang = g.meta_language.dropna()
    meta = Meta.model_construct(language=lang.iloc[0] if len(lang) else "en", intake_form=False, objective_measure=False)
    return inst.model_copy(update={"meta": meta})


# ---------------------------------------------------------------- validation
def expected_for(fixture):
    """Human coding; construct names of item-less scales are dropped (the exploded rows do not store them)."""
    exp = load_full_fixture_with_tags(fixture).get("extraction_output")

    def walk(scales):
        for s in scales or []:
            if isinstance(s, dict):
                if not s.get("items") and "construct_name" in s:
                    del s["construct_name"]
                walk(s.get("subscales"))

    walk(exp.get("scales"))
    return exp


def issue_category(msg):
    if msg.startswith("missing in actual"):
        return "missing"
    if msg.startswith("extra in actual"):
        return "extra"
    if "similarity" in msg or "sim=" in msg:
        return "text"
    if msg.startswith("list length") or msg.startswith("expected object") or msg.startswith("expected list"):
        return "structure"
    return "value"


def issue_key(issue):
    """Comparable across extractions of one document: expected-side path and issue category, no text."""
    return re.sub(r"\[\+\d+\]", "[+]", issue.path), issue_category(issue.message)


def validate(expected, rows):
    return validate_extraction(expected, inst_from_rows(rows), THRESHOLDS)


# ---------------------------------------------------------------- main
def read_extraction(path, dois):
    dset = ds.dataset(path)
    paths = dset.to_table(columns=["path"]).column("path").unique().to_pylist()
    keep = [p for p in paths if doi_from_path(p) in dois]
    df = dset.to_table(columns=ROW_COLS, filter=pc.field("path").isin(keep)).to_pandas()
    df["doi"] = df.path.map(doi_from_path)
    return df


def read_flags(path):
    cols = ds.dataset(path).schema.names
    use = [c for c in ["path", "doi_psyctests"] + FLAGS if c in cols]
    df = pd.read_parquet(path, columns=use)
    df["doi"] = df["doi_psyctests"] if "doi_psyctests" in df else df["path"].map(doi_from_path)
    flags = df.groupby("doi")[[f for f in FLAGS if f in df]].agg(lambda s: s.astype("boolean").any())
    flags["any_flag"] = flags.any(axis=1)
    return flags


def read_records(path):
    rec = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    return rec.set_index("DOI")["TestItemsAvailable"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("postprocessed")
    ap.add_argument("--extraction", default=str(EXTRACTION))
    ap.add_argument("--records", help="csv/parquet with DOI and TestItemsAvailable (PsycTests snapshot)")
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()

    log = parse_validity_log()
    fixtures = {f.name[:3]: f for f in sorted(FIX.glob("*.yaml"))}
    dois = {log[d]["doi"] for d in fixtures}
    ex = read_extraction(args.extraction, dois)
    flags = read_flags(args.postprocessed)

    res, control = [], []
    for doc, fixture in fixtures.items():
        v = log[doc]
        expected = expected_for(fixture)
        r = dict(doc=doc, doi=v["doi"], validity_run=validity_run_label(doc, v["status"]))
        baseline = None
        if v["inst"]:
            rep = validate(expected, rows_from_inst(v["inst"]))
            baseline = Counter(issue_key(i) for i in rep.issues)
            control.append(dict(doc=doc, logged=v["status"], rebuilt=rep.status))
        g = ex[ex.doi == v["doi"]]
        r["in_extraction"] = bool(g.item_item_id.notna().any())
        r["in_corpus"] = v["doi"] in flags.index
        if r["in_extraction"]:
            rep = validate(expected, g)
            keys = Counter(issue_key(i) for i in rep.issues)
            beyond = keys - baseline if (doc in FIXTURE_ERRORS and baseline is not None) else keys
            cats = Counter(issue_category(i.message) for i in rep.issues)
            r.update(production=rep.status, n_issues=len(rep.issues),
                     issue_categories=";".join(f"{k}:{n}" for k, n in sorted(cats.items())),
                     n_issues_beyond_fixture_errors=sum(beyond.values()) if doc in FIXTURE_ERRORS else np.nan)
            if rep.status == "PASS":
                r["outcome"] = "pass"
            elif doc in FIXTURE_ERRORS and (baseline is None or not beyond):
                r["outcome"] = "fixture_error"
            else:
                r["outcome"] = "discrepancy"
        res.append(r)
    R = pd.DataFrame(res).set_index("doi").join(flags, how="left")
    R["genuine_problem"] = R["outcome"].eq("discrepancy")

    if args.control:
        C = pd.DataFrame(control)
        agree = (C.logged == C.rebuilt).sum()
        print(f"control: validity-run extractions rebuilt from rows reproduce the logged status for {agree} of "
              f"{len(C)} documents with output")
        print(pd.crosstab(C.logged, C.rebuilt).to_string())
        if agree < len(C):
            print("differs:", ", ".join(f"{c.doc} ({c.logged} -> {c.rebuilt})" for c in C[C.logged != C.rebuilt].itertuples()))
        print()

    absent = R[~R.in_extraction]
    removed = R[R.in_extraction & ~R.in_corpus]
    msg = (f"audited documents in the searchable corpus: {int(R.in_corpus.sum())} of {len(R)}; "
           f"{len(absent)} are not in the production extraction at all")
    if args.records:
        avail = read_records(args.records)
        msg += " (PsycTests TestItemsAvailable: " + ", ".join(
            f"{k} {n}" for k, n in absent.index.map(avail).value_counts(dropna=False).items()) + ")"
    msg += f", {len(removed)} were extracted but removed in postprocessing"
    print(msg)

    C73 = R[R.in_corpus]
    print(f"\nproduction outcome of the audited documents in the corpus: "
          + ", ".join(f"{k} {n}" for k, n in C73.outcome.value_counts().items()))
    for f in [c for c in FLAGS if c in C73] + ["any_flag"]:
        flagged = C73[C73[f] == True]  # noqa: E712
        share = flagged.genuine_problem.mean() if len(flagged) else float("nan")
        caught = int((C73[f] == True)[C73.genuine_problem].sum())  # noqa: E712
        print(f"\n{f}: flagged {len(flagged)}, of which {int(flagged.genuine_problem.sum())} with a genuine "
              f"discrepancy ({share:.2f}); caught {caught} of {int(C73.genuine_problem.sum())} genuine discrepancies")
        print(pd.crosstab(C73[f], C73.genuine_problem).to_string())
    changed = C73[C73.outcome != C73.validity_run]
    if len(changed):
        print("\nlabel differs from the validity-run outcome:")
        for d, r in changed.iterrows():
            print(f"  {r.doc} {d}: {r.validity_run} -> {r.outcome} ({r.issue_categories or 'no issues'})")

    cols = (["doc", "validity_run", "in_extraction", "in_corpus", "production", "n_issues", "issue_categories",
             "n_issues_beyond_fixture_errors"] + [c for c in FLAGS if c in R] + ["any_flag", "outcome", "genuine_problem"])
    R[cols].to_csv(OUT)


if __name__ == "__main__":
    main()
