#!/usr/bin/env python
"""Deterministic verbatim check for scale-hunt extractions.

For every extracted item, verify its text/options/admin_note appear in the
archived source files (data/restricted/pdfs/). Replaces agent-based verifiers.

Usage: scale_hunt_check.py <workflow_output.json> [<more.json> ...]
Writes data/processed/scale_hunt_check_report.csv (no item text) and prints a summary.
"""
import sys, os, re, glob, json, csv

import pdf_inspector as pi

BASE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.environ.get("SCALE_HUNT_PDF_DIR", os.path.join(BASE, "data/restricted/pdfs"))
REPORT = os.path.join(BASE, "data/processed/scale_hunt_check_report.csv")


def norm(s):
    s = re.sub(r"<[^>]+>", " ", s)          # strip HTML tags
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def source_text(path):
    try:
        if path.endswith(".pdf"):
            return pi.process_pdf(path).markdown
        if path.endswith((".md", ".txt", ".html")):
            return open(path, errors="replace").read()
    except Exception:
        return None
    return None


def corpus_for(accession, saved_files):
    texts = []
    candidates = list(saved_files or [])
    candidates += glob.glob(os.path.join(PDF_DIR, accession + "_*"))
    seen = set()
    for f in candidates:
        f = f if os.path.isabs(f) else os.path.join(PDF_DIR, f)
        if f in seen or not os.path.exists(f):
            continue
        seen.add(f)
        t = source_text(f)
        if t:
            texts.append(norm(t))
    return " ".join(texts), len(seen)


def probe_variants(p):
    """Progressively forgiving forms of an item string."""
    yield norm(p)
    stripped = re.sub(r"\s*\(\d+\)\s*$", "", p)           # trailing score "(2)"
    stripped = re.sub(r"^\s*Item \d+\s*(\([A-Z]\d+\))?[:.]\s*", "", stripped)  # agent prefixes
    stripped = stripped.rstrip(":.")
    yield norm(stripped)
    words = norm(stripped).split()
    if len(words) >= 8:
        yield " ".join(words[:8])                          # prefix (layout splits)


def check_extraction(e, target_doi=None):
    m = re.search(r"t(\d{5})-000", target_doi or "") or re.search(r"t(\d{5})-000", e.get("doi") or "")
    if not m:
        return dict(accession="?", items=len(e.get("items") or []), matched=None, files=0,
                    status="no_psyctests_doi")
    acc = "9999" + m.group(1)
    corpus, n_files = corpus_for(acc, e.get("saved_files"))
    n_items = len(e.get("items") or [])
    if not corpus:
        return dict(accession=acc, items=n_items, matched=None, files=n_files,
                    status="no_source_text (scanned or nothing archived)")
    matched = 0
    for it in e.get("items") or []:
        probes = []
        t = it.get("text") or ""
        if t and "statement group" not in t:
            probes.append(t)
        opts = it.get("options")
        if isinstance(opts, str):
            probes += opts.split(" || ")
        elif isinstance(opts, list):
            probes += opts
        if it.get("admin_note"):
            probes.append(it["admin_note"][:120])
        probes = [p for p in probes if p and len(norm(p)) > 12]
        ok = all(any(v in corpus for v in probe_variants(p)) for p in probes) if probes else False
        matched += ok
    frac = matched / n_items if n_items else 0
    status = "ok" if frac >= 0.9 else "REVIEW"
    return dict(accession=acc, items=n_items, matched=matched, files=n_files, status=status)


def main(paths):
    rows = []
    for p in paths:
        data = json.load(open(p))
        results = data.get("result", data) or []
        for r in results:
            if not r:
                continue
            e = r.get("extraction") or r
            if not e or not e.get("found"):
                rows.append(dict(doi=(e or {}).get("doi", r.get("target", "?")),
                                 name=(e or {}).get("name", ""), accession="", items=0,
                                 matched=None, files=0, status="not_found"))
                continue
            res = check_extraction(e, r.get("target") if isinstance(r, dict) else None)
            rows.append(dict(doi=r.get("target") or e.get("doi"), name=e.get("name", ""), **res))
    with open(REPORT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"{n_ok}/{len(rows)} scales fully confirmed; report: {REPORT}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['status']:>10s}  {r['name'] or r['doi']} ({r['matched']}/{r['items']} matched, {r['files']} source files)")


if __name__ == "__main__":
    main(sys.argv[1:])
