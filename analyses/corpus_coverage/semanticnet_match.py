#!/usr/bin/env python3
"""Refresh data/semanticnet/scale_matches.csv against the corrected items_clean.csv.

scale_matches.csv maps each SemanticNet scale name to candidate PsycTests DOIs
and records how many items the scale has; `semanticnet_ingest.R` gates on both
(the item count must be within 25% of the PsycTests metadata count).

The original mapping was built against the malformed items_clean.csv, so its
scale names and item counts are partly wrong. It also stored scale names with
punctuation stripped ("action control hesitation") while items_clean.csv keeps
it ("action control - preoccupation"), so `semanticnet_ingest.R`'s join on that
column silently dropped 126 of the 705 DOI-matched scales.

Rebuilding the match tiers from scratch would lose the 77 'fuzzy' matches, whose
rule is not recorded anywhere. So this script is deliberately conservative:

  * output is keyed by the scale name EXACTLY as it appears in items_clean.csv,
    so the downstream join is lossless;
  * existing match rows are carried over -- looked up on a punctuation-insensitive
    key -- with n_items refreshed from the corrected file;
  * scale names with no prior row get exact / core_exact matches computed here
    (no fuzzy tier);
  * rows whose scale no longer exists are dropped.

Usage: semanticnet_match.py [--check]
"""
import csv, os, re, sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "data/semanticnet")
ITEMS = os.path.join(D, "items_clean.csv")
VARIANTS = os.path.join(D, "psyc_name_variants.csv")
MATCHES = os.path.join(D, "scale_matches.csv")

GENERIC = r"\b(scale|scales|questionnaire|inventory|test|index|checklist|survey|measure|form|short|revised|version)\b"


def norm(s):
    """Punctuation-insensitive key, used only to line the two files up."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def core(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(GENERIC, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load():
    counts = defaultdict(int)
    for r in csv.DictReader(open(ITEMS, encoding="utf-8")):
        counts[r["scale"].strip().lower()] += 1

    by_variant, by_core = defaultdict(set), defaultdict(set)
    for v in csv.DictReader(open(VARIANTS, encoding="utf-8")):
        name = v["variant"].strip().lower()
        by_variant[name].add(v["DOI"])
        c = core(name)
        if c:
            by_core[c].add(v["DOI"])
    old = {}
    for r in csv.DictReader(open(MATCHES, encoding="utf-8")):
        old.setdefault(norm(r["semanticnet_scale"]), r)
    return counts, by_variant, by_core, old


def build():
    counts, by_variant, by_core, old = load()
    rows, kept, fresh = [], 0, 0
    for scale in sorted(counts):
        n = counts[scale]
        prev = old.get(norm(scale))
        if prev:                                  # keep the original tier + DOIs
            kept += 1
            rows.append(dict(semanticnet_scale=scale, n_items=n,
                             match_type=prev["match_type"],
                             n_dois=prev["n_dois"], dois=prev["dois"]))
            continue
        fresh += 1
        if scale in by_variant:
            dois, mt = by_variant[scale], "exact"
        elif core(scale) and core(scale) in by_core:
            dois, mt = by_core[core(scale)], "core_exact"
        else:
            dois, mt = set(), "none"
        rows.append(dict(semanticnet_scale=scale, n_items=n, match_type=mt,
                         n_dois=len(dois), dois=";".join(sorted(dois))))
    return rows, kept, fresh, old


def main():
    rows, kept, fresh, old = build()
    gone = set(old) - set(norm(r["semanticnet_scale"]) for r in rows)
    matched = sum(1 for r in rows if r["match_type"] != "none")
    # hyphen/space variants of one name share a normalised key, so compare
    # n_items only where the stored name was identical -- otherwise a no-op
    # re-run reports spurious "corrections" against a sibling's row
    exact_old = {r["semanticnet_scale"].strip().lower(): r for r in
                 csv.DictReader(open(MATCHES, encoding="utf-8"))}
    changed = sum(1 for r in rows
                  if r["semanticnet_scale"] in exact_old
                  and str(exact_old[r["semanticnet_scale"]]["n_items"]) != str(r["n_items"]))
    collisions = len(rows) - len(set(norm(r["semanticnet_scale"]) for r in rows))
    print(f"scales: {len(rows)} ({kept} carried over, {fresh} newly matched); "
          f"dropped names no longer present: {len(gone)}")
    print(f"n_items corrected for {changed} scales; "
          f"{collisions} hyphen/space name variants share a match row")
    print(f"scales with at least one DOI match: {matched}")
    if "--check" in sys.argv:
        return
    with open(MATCHES, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["semanticnet_scale", "n_items",
                                           "match_type", "n_dois", "dois"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MATCHES}")


if __name__ == "__main__":
    main()
