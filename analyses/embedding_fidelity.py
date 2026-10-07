"""Do extraction discrepancies change what the search engine indexes?

Embeds the human-coded fixtures (tests/validation-fixtures) and the machine extractions of the same 100
PsycTests documents with SurveyBot3000, pools items the way the engine does (unit-normalise, negate
reverse-keyed items, average) and reports two numbers. The machine side is either

- the validity run (default; logs/test-validity-20260815_172054.md), an independent extraction of the audited
  documents with the production model and prompts, or
- the production corpus (--postprocessed): the postprocessed rows the search index pools, for the audited
  documents the index holds. The script then also reports the validity run restricted to the same documents and,
  with --exploded, the production extraction before postprocessing (sensitivity check).

The two numbers:

1. Coverage: the share of human-coded scales with no item in the index. A scale is in the index if at least
   one of its items appears in any row the index holds for its document (a machine scale node, or the
   instrument row = all scaled + orphan items). Items correspond if their texts are >= 80% similar
   (normalised Levenshtein) or one contains the other (partial ratio >= 90), matched one-to-one.
2. Agreement: each recovered scale is paired with the index row sharing the most of its items (item-set
   Jaccard; several human scales may share one row, partial overlap is allowed), so merged, split and
   partially extracted scales are included. We correlate the cosines among all pairs of human-coded scale
   vectors (including scale-subscale pairs) with the cosines among their machine counterparts.

Excluded: documents whose validity failures were traced to errors in the human coding ('FP' in the validity
summary, plus 022, see FIXTURE_ERROR), and documents without a machine side. In the validity run, 018 and 022
produced no output (truncated JSON); 018 is a transient failure (its production extraction matches the fixture),
022 a fixture error. Fixture conventions are respected: `!contains` texts are replaced by the matched machine
text, `!extras=N` untranscribed items are filled from the matched machine scale, and unrecorded keying takes the
machine's value.

Run from the repository root (SurveyBot3000 is gated on Hugging Face; set HF_TOKEN):
    HF_TOKEN=... poetry run python analyses/embedding_fidelity.py
    HF_TOKEN=... poetry run python analyses/embedding_fidelity.py \\
        --postprocessed <synthometricon-corpus-postprocessed.parquet> \\
        [--exploded <apa-psyctests-extractions-exploded.parquet>]
Uncertainty: 95% confidence intervals from a document-level cluster bootstrap (BOOT_REPS resamples of the
documents with recovered scales, percentile intervals, fixed seed). Each resample keeps the recovered scales of the
drawn documents and forms all pairs among them, skipping pairs of a scale with its own copy. The share of
human-coded scales recovered gets a Wilson score interval.

Outputs (no item text): analyses/output/embedding_fidelity/{summary.txt, recovered_pairs.png,
recovered_scales.csv, per_document.csv}; with --postprocessed the same files in .../embedding_fidelity/production/.
"""
import argparse
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analyses" / "output" / "embedding_fidelity"
MODEL = "magnolia-psychometrics/surveybot3000"
FIX = REPO / "tests" / "validation-fixtures"
VAL = REPO / "logs" / "test-validity-20260815_172054.md"

# adjudication notes in the summary table of the validity log
FP = set("024 045 046 109 110 118 120 123 127 136 145".split())
TP = set("014 018 022 023 025 031 039 050 103 104 119 122 130 135 138 140".split())
# 022 is TP in the log, but its fixture merges multi-part questions into single 330-750 character items, which no
# extraction can match (its production extraction fails the check for that reason): counted as a fixture error.
FIXTURE_ERROR = FP | {"022"}
# truncated JSON in the validity run, so no machine side there; 018 is a transient failure
NO_VALIDITY_OUTPUT = {"018", "022"}

BOOT_REPS = 1000
BOOT_SEED = 20261002


def group_of(doc):
    return ("fixture_error" if doc in FIXTURE_ERROR else "no_validity_output" if doc in NO_VALIDITY_OUTPUT
            else "genuine_discrepancy" if doc in TP else "pass")


def doi_of_fixture(f):
    return f"10.1037/{re.search(r'101037(t\d{5})000', f.name).group(1)}-000"


def doi_from_path(path):
    m = re.search(r"([0-9]{9})", Path(str(path)).name)
    return f"10.1037/t{m.group(1)[4:9]}-000" if m else None


def _missing(v):
    return v is None or v is pd.NA or (np.isscalar(v) and pd.isna(v))


# ---------------------------------------------------------------- fixture loader (as extraction/validity.py)
@dataclass(frozen=True)
class MatchSpec:
    mode: str
    value: str


class AnywhereList(list):
    pass


class AnywhereDict(dict):
    pass


class ExtrasList(list):
    allow_extras: int = 0


class Loader(yaml.SafeLoader):
    pass


def _match_spec(mode):
    return lambda loader, node: MatchSpec(mode, loader.construct_scalar(node))


def _anywhere(loader, node):
    if isinstance(node, yaml.SequenceNode):
        return AnywhereList(loader.construct_sequence(node, deep=True))
    return AnywhereDict(loader.construct_mapping(node, deep=True))


def _extras(loader, suffix, node):
    lst = ExtrasList(loader.construct_sequence(node, deep=True))
    lst.allow_extras = int(suffix.lstrip("=_-"))
    return lst


Loader.add_constructor("!contains", _match_spec("contains"))
Loader.add_constructor("!exact", _match_spec("exact"))
Loader.add_constructor("!anywhere", _anywhere)
Loader.add_multi_constructor("!extras", _extras)


# ---------------------------------------------------------------- instrument trees
@dataclass
class Item:
    key: str            # identity within document and side
    text: str
    rev: object         # True / False / None (not recorded)
    partial: bool = False


@dataclass
class Node:
    path: tuple         # position in the scale hierarchy
    items: list         # items attached directly
    extras: int = 0     # fixture tolerance for untranscribed items


def human_tree(fx):
    items, nodes = {}, []

    def item_of(d):
        v = d.get("item_text")
        text, partial = (v.value, v.mode == "contains") if isinstance(v, MatchSpec) else (v, False)
        if not isinstance(text, str) or not text:
            return None
        key = "h:" + text.strip().lower()
        if key not in items:
            items[key] = Item(key, text, d.get("reverse_coded"), partial)
        return items[key]

    def walk(scales, prefix):
        for i, s in enumerate(scales or []):
            its = [x for x in (item_of(d) for d in (s.get("items") or [])) if x]
            nodes.append(Node(prefix + (i,), its, getattr(s.get("items"), "allow_extras", 0) or 0))
            walk(s.get("subscales"), prefix + (i,))

    walk(fx.get("scales"), ())
    orphans = [x for x in (item_of(d) for d in (fx.get("orphan_items") or [])) if x]
    return nodes, orphans, items


def machine_tree(inst):
    """Scaled and orphan items of a resolved instrument (validity run); unscaled items are dropped in
    postprocessing and therefore ignored."""
    items, nodes = {}, []

    def item_of(d, orphan=False):
        text = d.get("item_text")
        if not text:
            return None
        # scaled items carry `item_id`, orphan items `id`
        key = f"m:o{d.get('id')}" if orphan else f"m:{d.get('item_id')}"
        if key not in items:
            items[key] = Item(key, text, d.get("reverse_coded"))
        return items[key]

    def walk(scales, prefix):
        for i, s in enumerate(scales or []):
            nodes.append(Node(prefix + (i,), [x for x in (item_of(d) for d in (s.get("items") or [])) if x]))
            walk(s.get("subscales"), prefix + (i,))

    walk(inst.get("scales"), ())
    orphans = [x for x in (item_of(d, orphan=True) for d in (inst.get("orphan_items") or [])) if x]
    return nodes, orphans, items


def rows_tree(g):
    """Tree from corpus rows (one row per scaled item placement or orphan item), keyed like assemble/pool.py:
    items by `item_item_id`, scale nodes by their id path, each node pooling its own and its descendants' items."""
    items, direct, orphans = {}, {}, []
    for k, r in enumerate(g.itertuples(index=False)):
        text = r.item_item_text
        if not isinstance(text, str) or not text:
            continue
        key = f"m:row{k}" if _missing(r.item_item_id) else f"m:{r.item_item_id}"
        it = items.setdefault(key, Item(key, text, None if _missing(r.item_reverse_coded) else bool(r.item_reverse_coded)))
        ids = r.scale_id_path
        if r.bucket == "scaled" and ids is not None and not (np.isscalar(ids) and pd.isna(ids)) and len(ids):
            ip = tuple(int(x) for x in ids)
            for n in range(1, len(ip)):
                direct.setdefault(ip[:n], [])
            node = direct.setdefault(ip, [])
            if all(x.key != key for x in node):
                node.append(it)
        elif r.bucket == "orphan" and all(x.key != key for x in orphans):
            orphans.append(it)
    nodes = [Node(p, its) for p, its in sorted(direct.items())]
    return nodes, orphans, items


def corpus_trees(parquet, dois):
    """{doi: tree} for the audited documents present in a corpus parquet (scaled and orphan rows)."""
    cols = ["path", "bucket", "scale_id_path", "item_item_id", "item_item_text", "item_reverse_coded"]
    df = pd.read_parquet(parquet, columns=cols)
    df["doi"] = df["path"].map(doi_from_path)
    df = df[df["doi"].isin(set(dois)) & df["bucket"].isin(["scaled", "orphan"])]
    return {doi: rows_tree(g) for doi, g in df.groupby("doi", sort=False)}


def subtree(node, nodes):
    """Items of a node and all its descendants (as pooled for a scale row), deduplicated."""
    out = {}
    for n in nodes:
        if n.path[:len(node.path)] == node.path:
            for it in n.items:
                out.setdefault(it.key, it)
    return list(out.values())


# ---------------------------------------------------------------- validity-log parsing
def yaml_block(lines, start):
    block = []
    for ln in lines[start + 1:]:
        if ln.startswith("    ") or not ln.strip():
            block.append(ln)
        else:
            break
    return yaml.safe_load(textwrap.dedent("\n".join(block))) or {}


def parse_val():
    """Per document: the resolved machine instrument tree (None if the extraction failed)."""
    sections = re.split(r"\n=+\nFile \d+ / 100: \S*/(\d{3})_101037(t\d{5})000\.pdf\n=+\n", VAL.read_text())
    out = {}
    for i in range(1, len(sections) - 1, 3):
        doc, doi, body = sections[i], sections[i + 1], sections[i + 2]
        lines = body.split("\n")
        inst = next((yaml_block(lines, j) for j, ln in enumerate(lines) if ln.startswith("  Resolved instrument")), None)
        out[doc] = dict(doi=f"10.1037/{doi}-000", inst=inst)
    return out


# ---------------------------------------------------------------- matching
def match_items(h_items, m_items):
    """One-to-one item correspondence: edit similarity or containment (e.g., an omitted shared stem)."""
    if not h_items or not m_items:
        return {}
    S = np.zeros((len(h_items), len(m_items)))
    for a, h in enumerate(h_items):
        hl = h.text.strip().lower()
        for b, m in enumerate(m_items):
            ml = m.text.strip().lower()
            if h.partial:
                S[a, b] = 1.0 if hl in ml else fuzz.partial_ratio(hl, ml) / 100
            else:
                lev = 1 - Levenshtein.normalized_distance(hl, ml)
                pr = fuzz.partial_ratio(hl, ml) / 100 if min(len(hl), len(ml)) >= 5 else 0
                S[a, b] = max(lev, pr if pr >= 0.9 else 0)
    r, c = linear_sum_assignment(-S)
    return {h_items[a].key: (m_items[b], S[a, b]) for a, b in zip(r, c) if S[a, b] >= 0.8}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


def match_nodes(h_sets, m_sets):
    """One-to-one scale correspondence (Jaccard >= .5); used only to fill `!extras` items."""
    if not h_sets or not m_sets:
        return {}
    J = np.array([[jaccard(h, m) for m in m_sets] for h in h_sets])
    r, c = linear_sum_assignment(-J)
    return {a: b for a, b in zip(r, c) if J[a, b] >= 0.5}


# ---------------------------------------------------------------- pooling (as assemble/pool.py keyed_centroid)
def keyed_centroid(vecs, rev):
    v = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    v = (v * np.where(rev, -1.0, 1.0)[:, None]).mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---------------------------------------------------------------- analysis
def validity_trees():
    """{doc: tree} from the validity run (None where the extraction produced no output)."""
    return {doc: machine_tree(v["inst"]) if v["inst"] else None for doc, v in parse_val().items()}


def build(machine, only=None):
    """Human and machine trees per audited document; `machine` maps doc -> tree (absent/None: no machine side).
    Human trees are rebuilt per call, because fixture conventions borrow text and keying from the machine side."""
    docs = []
    for f in sorted(FIX.glob("*.yaml")):
        doc = f.name[:3]
        if only is not None and doc not in only:
            continue
        fx = (yaml.load(f.read_text(), Loader=Loader) or {}).get("extraction_output") or {}
        hn, ho, hi = human_tree(fx)
        tree = machine.get(doc)
        d = dict(doc=doc, doi=doi_of_fixture(f), group=group_of(doc), h_nodes=hn, h_orph=ho, h_items=hi,
                 ok=tree is not None)
        if d["ok"]:
            mn, mo, mi = tree
            imatch = match_items(list(hi.values()), list(mi.values()))
            for h in hi.values():
                if h.key in imatch:
                    if h.partial:           # `!contains`: the fixture asserts only a substring
                        h.text = imatch[h.key][0].text
                    if h.rev is None:       # unrecorded keying is not asserted
                        h.rev = imatch[h.key][0].rev
            inv = {m.key: hk for hk, (m, _) in imatch.items()}
            h_sets = [{it.key for it in subtree(n, hn)} for n in hn]
            m_sets = [{inv.get(it.key, it.key) for it in subtree(n, mn)} for n in mn]
            for a, b in match_nodes(h_sets, m_sets).items():   # `!extras=N`
                if hn[a].extras:
                    hn[a].items = hn[a].items + [it for it in mn[b].items if it.key not in inv][: hn[a].extras]
            d.update(m_nodes=mn, m_orph=mo, m_items=mi, n_corresponding=len(imatch), m_sets=m_sets,
                     m_inst_set={inv.get(it.key, it.key) for it in mi.values()})
        docs.append(d)
    return docs


def embed(*doc_lists):
    """Encode every item text once and attach pooled human/machine vectors to each document."""
    from sentence_transformers import SentenceTransformer

    texts = set()
    for d in (d for docs in doc_lists for d in docs):
        texts.update(it.text for n in d["h_nodes"] for it in n.items)
        texts.update(it.text for it in d["h_orph"])
        if d["ok"]:
            texts.update(it.text for it in d["m_items"].values())
    texts = sorted(texts)
    model = SentenceTransformer(MODEL, device="cpu")
    emb = dict(zip(texts, model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)))

    def pooled(items):
        items = list({it.key: it for it in items}.values())
        if not items:
            return None
        return keyed_centroid(np.stack([emb[it.text] for it in items]), np.array([bool(it.rev) for it in items]))

    for d in (d for docs in doc_lists for d in docs):
        hn = d["h_nodes"]
        d["vH"] = pooled([it for n in hn for it in n.items] + d["h_orph"])
        d["vH_nodes"] = [pooled(subtree(n, hn)) for n in hn]
        if d["ok"]:
            mn = d["m_nodes"]
            d["vM"] = pooled(list(d["m_items"].values()))       # instrument row: all scaled and orphan items
            d["vM_nodes"] = [pooled(subtree(n, mn)) for n in mn]
    return len(texts)


def assessable(d):
    """Human-coded scales with item text (others cannot be assessed)."""
    hn = d["h_nodes"]
    return [a for a in range(len(hn)) if subtree(hn[a], hn)]


def recovered(docs):
    """Recovered scales and their counterparts; documents with fixture errors or without a machine side are skipped."""
    units, lost, n_total = [], [], 0
    for d in docs:
        if d["group"] == "fixture_error" or not d["ok"]:
            continue
        hn = d["h_nodes"]
        hs = [{it.key for it in subtree(n, hn)} for n in hn]
        for a in range(len(hn)):
            if not hs[a]:
                continue                     # no item text on the human side: cannot be assessed
            n_total += 1
            cands = [(jaccard(hs[a], ms), 1, b) for b, ms in enumerate(d["m_sets"])
                     if ms and d["vM_nodes"][b] is not None]
            if d["m_inst_set"] and d["vM"] is not None:
                cands.append((jaccard(hs[a], d["m_inst_set"]), 0, -1))   # instrument row (ties: scale first)
            best = max(cands) if cands else (0.0, 0, -1)
            if best[0] == 0:
                lost.append((d["doc"], "items not in index"))
                continue
            how = "instrument row" if best[2] < 0 else "clear (J >= .5)" if best[0] >= .5 else "partial (J < .5)"
            units.append(dict(doc=d["doc"], group=d["group"], h_node=a, counterpart=best[2], jaccard=best[0],
                              how=how, vh=d["vH_nodes"][a], vm=d["vM"] if best[2] < 0 else d["vM_nodes"][best[2]]))
    return units, lost, n_total


def wilson(k, n, z=1.959964):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def r_rmse(h, m):
    return np.corrcoef(h, m)[0, 1], np.sqrt(np.mean((m - h) ** 2))


def bootstrap(U, CH, CM, selectors, reps=BOOT_REPS, seed=BOOT_SEED):
    """Document-level cluster bootstrap of r and RMSE for each pair subset; returns {label: (r_ci, rmse_ci)}."""
    docs = U.doc.unique()
    by_doc = {d: np.flatnonzero(U.doc.to_numpy() == d) for d in docs}
    rng = np.random.default_rng(seed)
    draws = {label: [] for label in selectors}
    for _ in range(reps):
        idx = np.concatenate([by_doc[d] for d in rng.choice(docs, size=len(docs), replace=True)])
        i, j = np.triu_indices(len(idx), 1)
        a, b = idx[i], idx[j]
        keep = a != b                        # a scale drawn twice is not paired with its own copy
        a, b = a[keep], b[keep]
        h, m = CH[a, b], CM[a, b]
        for label, sel in selectors.items():
            mask = sel(a, b, h)
            if mask.sum() > 2:
                draws[label].append(r_rmse(h[mask], m[mask]))
    return {label: tuple(np.percentile(np.array(v), [2.5, 97.5], axis=0).T) for label, v in draws.items()}


def analyse(docs):
    """Coverage and agreement for one machine side; returns summary lines and the data for outputs and plot."""
    units, lost, n_total = recovered(docs)
    U = pd.DataFrame([{k: v for k, v in u.items() if k not in ("vh", "vm")} for u in units])
    VH, VM = np.stack([u["vh"] for u in units]), np.stack([u["vm"] for u in units])
    CH, CM = VH @ VH.T, VM @ VM.T
    i, j = np.triu_indices(len(units), 1)
    ch, cm = CH[i, j], CM[i, j]
    clear = (U.how == "clear (J >= .5)").to_numpy()
    passed = (U.group == "pass").to_numpy()
    tp = (U.group == "genuine_discrepancy").to_numpy()
    selectors = {
        "all pairs (incl. scale-subscale)": lambda a, b, h: np.ones_like(h, bool),
        "both counterparts clear (J >= .5)": lambda a, b, h: clear[a] & clear[b],
        "at least one partial / instrument-row counterpart": lambda a, b, h: ~(clear[a] & clear[b]),
        "both from documents that passed the validity check": lambda a, b, h: passed[a] & passed[b],
        "at least one from a genuine-discrepancy document": lambda a, b, h: tp[a] | tp[b],
        "search-relevant: |cos_h| >= .30": lambda a, b, h: np.abs(h) >= .30,
    }
    cis = bootstrap(U, CH, CM, selectors)

    def stat(label):
        mask = selectors[label](i, j, ch)
        h, m = ch[mask], cm[mask]
        e = np.abs(m - h)
        r, rmse = r_rmse(h, m)
        (rlo, rhi), (elo, ehi) = cis[label]
        return (f"  {label:<52} pairs {mask.sum():>7,} | r = {r:.3f} [{rlo:.3f}, {rhi:.3f}] | "
                f"RMSE = {rmse:.3f} [{elo:.3f}, {ehi:.3f}] | MAE = {e.mean():.3f} | max |diff| = {e.max():.3f}")

    n_docs = len({u["doc"] for u in units} | {r[0] for r in lost})
    lo, hi = wilson(n_total - len(lost), n_total)
    lines = [f"Human-coded scales with item text: {n_total} in {n_docs} documents",
             f"(1) COVERAGE: not in the index {len(lost)}/{n_total} = {len(lost) / n_total:.1%} "
             f"{pd.Series([r[1] for r in lost], dtype=object).value_counts().to_dict()}; documents {sorted({r[0] for r in lost})}; "
             f"recovered {n_total - len(lost)}/{n_total} = {(n_total - len(lost)) / n_total:.1%} "
             f"[Wilson 95% CI {lo:.1%}, {hi:.1%}]",
             f"(2) AGREEMENT among the {len(units)} recovered scales (counterparts: {U.how.value_counts().to_dict()}; "
             f"{int((U.groupby(['doc', 'counterpart']).size() > 1).sum())} machine rows serve >1 human scale):",
             f"    [95% CIs: document-level cluster bootstrap, {BOOT_REPS} resamples of {U.doc.nunique()} documents, "
             f"percentile, seed {BOOT_SEED}]"] + [stat(label) for label in selectors]
    return dict(lines=lines, U=U, ch=ch, cm=cm, clear_pair=clear[i] & clear[j], n_units=len(units))


def write_outputs(out, docs, res, header):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    res["U"].to_csv(out / "recovered_scales.csv", index=False)
    pd.DataFrame([dict(doc=d["doc"], doi=d["doi"], group=d["group"], machine_side=d["ok"], n_h_items=len(d["h_items"]),
                       n_m_items=len(d["m_items"]) if d["ok"] else 0, n_corresponding_items=d.get("n_corresponding", 0),
                       n_h_scales=len(d["h_nodes"]), n_m_scales=len(d["m_nodes"]) if d["ok"] else 0,
                       cos_instrument=float(d["vH"] @ d["vM"]) if d["ok"] and d["vH"] is not None and d["vM"] is not None else None)
                  for d in docs]).to_csv(out / "per_document.csv", index=False)
    (out / "summary.txt").write_text("\n".join(header) + "\n")
    print("\n".join(header))

    ch, cm, k = res["ch"], res["cm"], ~res["clear_pair"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(ch[~k], cm[~k], s=2, alpha=.25, color="#0072B2", rasterized=True, label="both scales clearly recovered")
    ax.scatter(ch[k], cm[k], s=2, alpha=.4, color="#E69F00", rasterized=True, label="≥1 merged/split/partial scale")
    ax.plot([-1, 1], [-1, 1], color="grey", lw=.8)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="cosine, human-coded text", ylabel="cosine, machine-extracted text",
           title=f"{res['n_units']} recovered scales, {len(ch):,} pairs, r = {np.corrcoef(ch, cm)[0, 1]:.3f}")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(out / "recovered_pairs.png", dpi=150)
    plt.close(fig)


def main_validity():
    docs = build(validity_trees())
    n_texts = embed(docs)
    res = analyse(docs)
    header = [f"Model: {MODEL}; texts encoded: {n_texts}",
              f"Machine side: validity run, {VAL.relative_to(REPO)}; fixtures: {FIX.relative_to(REPO)}",
              f"Excluded: {len(FIXTURE_ERROR)} documents whose validity failures were due to errors in the human coding "
              f"(incl. 022), and documents without output in the validity run ({', '.join(sorted(NO_VALIDITY_OUTPUT))})",
              ""] + res["lines"]
    write_outputs(OUT, docs, res, header)


def main_production(postprocessed, exploded=None):
    fixtures = {f.name[:3]: doi_of_fixture(f) for f in sorted(FIX.glob("*.yaml"))}
    by_doi = {v: k for k, v in fixtures.items()}
    index = {by_doi[doi]: t for doi, t in corpus_trees(postprocessed, fixtures.values()).items()}
    val = validity_trees()
    in_index = set(index) - FIXTURE_ERROR
    with_val = {doc for doc in in_index if val.get(doc) is not None}
    docs_index = build(index, only=in_index)
    docs_index_v = build(index, only=with_val)
    docs_val = build(val, only=with_val)
    blocks = [("INDEX (postprocessed rows), audited documents the index holds", docs_index),
              ("INDEX, the same documents minus those without validity-run output", docs_index_v),
              ("VALIDITY RUN, the same documents", docs_val)]
    pre = None
    if exploded:
        pre = {by_doi[doi]: t for doi, t in corpus_trees(exploded, fixtures.values()).items()}
        blocks.append(("PRODUCTION EXTRACTION BEFORE POSTPROCESSING (exploded rows), the same documents",
                       build(pre, only=with_val)))
    n_texts = embed(*[b[1] for b in blocks])

    # context: where are the assessable human-coded scales of all non-fixture-error documents?
    everything = build({}, only=set(fixtures) - FIXTURE_ERROR)
    where = {}
    for d in everything:
        status = ("in the index" if d["doc"] in index else "removed in postprocessing" if pre is not None and d["doc"] in pre
                  else "never extracted" if pre is not None else "not in the index")
        n, docs_ = where.get(status, (0, set()))
        where[status] = (n + len(assessable(d)), docs_ | ({d["doc"]} if assessable(d) else set()))
    total = sum(n for n, _ in where.values())

    header = [f"Model: {MODEL}; texts encoded: {n_texts}",
              f"Index: {Path(postprocessed).name} (the rows the search index pools); fixtures: {FIX.relative_to(REPO)}; "
              f"validity run: {VAL.relative_to(REPO)}",
              f"Excluded: {len(FIXTURE_ERROR)} documents whose validity failures were due to errors in the human coding (incl. 022)",
              f"Audited documents in the index: {len(set(index))} of {len(fixtures)}; without fixture errors: {len(in_index)}; "
              f"of these with validity-run output: {len(with_val)} (without: {sorted(in_index - with_val)})",
              f"Human-coded scales with item text, documents without fixture errors: {total}; "
              + "; ".join(f"{s}: {n} scales in {len(ds)} documents" for s, (n, ds) in where.items()),
              "Groups (pass / genuine discrepancy) are the validity-run outcomes of the documents.", ""]
    res_index = None
    for k, (title, docs) in enumerate(blocks):
        res = analyse(docs)
        header += [f"[{'ABCD'[k]}] {title}"] + res["lines"] + [""]
        if k == 0:
            res_index = res
    write_outputs(OUT / "production", docs_index, res_index, header)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--postprocessed", help="postprocessed corpus parquet: compare against what the index holds")
    ap.add_argument("--exploded", help="exploded extraction parquet (with --postprocessed): sensitivity check")
    a = ap.parse_args()
    if a.postprocessed:
        main_production(a.postprocessed, a.exploded)
    else:
        main_validity()


if __name__ == "__main__":
    sys.exit(main())
