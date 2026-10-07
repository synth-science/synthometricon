"""Pool stage: keyed-centroid math, keying-disagreement diagnostic, grouping, and ``run`` round-trips."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from assemble.pool import (build_groups, detect_models, keyed_centroid,
                           keying_disagreement)

F32 = np.float32


# --- detect_models ---

def test_detect_models():
    cols = ["path", "item_embedding_surveybot3000",
            "scale_embedding_all_minilm_l6_v2",
            "instrument_embedding_all_minilm_l6_v2"]
    assert detect_models(cols, "item") == {
        "surveybot3000": "item_embedding_surveybot3000"}
    assert detect_models(cols, "scale") == {
        "all_minilm_l6_v2": "scale_embedding_all_minilm_l6_v2"}
    assert detect_models(cols, "instrument") == {
        "all_minilm_l6_v2": "instrument_embedding_all_minilm_l6_v2"}


# --- keyed_centroid ---

def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_no_reverse_is_mean_of_unit_vectors():
    # Norms are encoder artifacts: the (3, 0) item must not outweigh (0, 1).
    v = np.array([[3, 0], [0, 1]], dtype=F32)
    rev = np.array([False, False])
    assert np.allclose(keyed_centroid(v, rev), [0.5, 0.5])


def test_reverse_items_are_negated():
    # Reverse item (-1, 0) contributes (1, 0), like reverse-scoring.
    v = np.array([[1, 0], [1, 0], [-1, 0]], dtype=F32)
    rev = np.array([False, False, True])
    assert np.allclose(keyed_centroid(v, rev), [1, 0])


def test_negation_trusts_documented_keying_over_encoder():
    # A reverse item the encoder aligns WITH the positive item is still negated.
    v = np.array([[1, 0], [0.9, 0.1]], dtype=F32)
    rev = np.array([False, True])
    expect = (unit(v) * np.array([[1.0], [-1.0]])).mean(axis=0)
    assert np.allclose(keyed_centroid(v, rev), expect)
    assert np.linalg.norm(keyed_centroid(v, rev)) < 0.1


def test_all_reverse_scale_points_at_named_pole():
    # e.g. "Emotional Stability" measured only by neuroticism items.
    v = np.array([[-1, 0], [-1, 0.2]], dtype=F32)
    rev = np.array([True, True])
    out = keyed_centroid(v, rev)
    assert out[0] > 0 and np.allclose(out, -unit(v).mean(axis=0))


def test_cosine_of_pooled_equals_composite_correlation():
    # With unit items and cos(item_i, item_j) = r_ij, cos(pooled A, pooled B)
    # is the correlation of two +/-1-weighted composites from R.
    rng = np.random.default_rng(0)
    V = unit(rng.normal(size=(9, 6)))
    R = V @ V.T
    sA = np.array([1, -1, 1, 1, 0, 0, 0, 0, 0.0])
    sB = np.array([0, 0, 0, 0, 1, -1, 1, 1, -1.0])
    a = keyed_centroid(V[sA != 0], sA[sA != 0] < 0)
    b = keyed_centroid(V[sB != 0], sB[sB != 0] < 0)
    cos_ab = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    corr = sA @ R @ sB / np.sqrt((sA @ R @ sA) * (sB @ R @ sB))
    assert np.isclose(cos_ab, corr)


def test_zero_vector_does_not_propagate_nan():
    v = np.array([[0, 0], [1, 0]], dtype=F32)
    rev = np.array([False, False])
    assert np.allclose(keyed_centroid(v, rev), [0.5, 0])


# --- keying_disagreement ---

def test_disagreement_flags_reverse_item_pointing_with_positives():
    v = np.array([[1, 0], [1, 0.1], [1, -0.1], [0.9, 0.1]], dtype=F32)
    rev = np.array([False, False, False, True])
    assert keying_disagreement(v, rev).tolist() == [False, False, False, True]


def test_disagreement_flags_positive_item_pointing_against():
    v = np.array([[1, 0], [1, 0.1], [1, -0.1], [-1, 0]], dtype=F32)
    rev = np.array([False, False, False, False])
    assert keying_disagreement(v, rev).tolist() == [False, False, False, True]


def test_consistent_keying_has_no_disagreement():
    v = np.array([[1, 0], [1, 0.1], [-0.9, 0.1]], dtype=F32)
    rev = np.array([False, False, True])
    assert not keying_disagreement(v, rev).any()


def test_disagreement_is_leave_one_out():
    # Neither could flag if each item were in its own centroid.
    v = np.array([[1, 0], [-1, 0.01]], dtype=F32)
    rev = np.array([False, False])
    assert keying_disagreement(v, rev).tolist() == [True, True]


def test_disagreement_single_item_is_never_flagged():
    v = np.array([[1, 0]], dtype=F32)
    assert keying_disagreement(v, np.array([True])).tolist() == [False]


# --- build_groups ---

def carried(n, **overrides):
    """Neutral filler for the carried columns ``run`` requires; override by name."""
    values = {"meta_language": "en", "public_year": 2001,
              "flag_item_count_deviation": False,
              "flag_scale_count_deviation": None,
              "version": None, "flag_item_text_deviation": False,
              "flag_item_translated": False}
    values.update(overrides)
    return {c: [v] * n if not isinstance(v, list) else v
            for c, v in values.items()}


def frame(rows):
    cols = ["path", "bucket", "scale_id", "scale_id_path", "scale_name_path",
            "item_item_id", "item_reverse_coded"]
    return pd.DataFrame(rows, columns=cols)


def test_subtree_membership_and_parent_names():
    df = frame([
        ("a.pdf", "scaled", 2, [1, 2], ["Parent", "Child A"], 1, False),
        ("a.pdf", "scaled", 3, [1, 3], ["Parent", "Child B"], 2, True),
    ])
    inst, scales, names, doc_row, sid_row = build_groups(df)
    # Parent scale 1 has no direct rows but pools both items.
    assert set(scales) == {("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 3)}
    assert scales[("a.pdf", 1)]["rows"] == [0, 1]
    assert scales[("a.pdf", 1)]["subtree"] == {1, 2, 3}
    assert scales[("a.pdf", 2)]["rows"] == [0]
    assert names[("a.pdf", 1)] == ("Parent", 1)
    assert names[("a.pdf", 2)] == ("Child A", 2)
    # Own-embedding rows exist only for immediate scales.
    assert ("a.pdf", 1) not in sid_row
    assert sid_row[("a.pdf", 2)] == 0


def test_instrument_dedupes_multiscale_item_and_keeps_orphans():
    df = frame([
        ("a.pdf", "scaled", 1, [1], ["S1"], 1, False),
        ("a.pdf", "scaled", 2, [1, 2], ["S1", "S2"], 1, True),  # same item
        ("a.pdf", "orphan", None, None, None, 9, None),
    ])
    inst, scales, names, doc_row, sid_row = build_groups(df)
    g = inst["a.pdf"]
    assert g["rows"] == [0, 2]          # item 1 once (first), orphan kept
    assert g["revs"] == [False, False]  # null keying -> positive
    assert g["subtree"] == {1, 2}
    # scale 1's subtree pool also counts the item once.
    assert scales[("a.pdf", 1)]["rows"] == [0]


def test_scale_name_collision_yields_separate_groups():
    df = frame([
        ("a.pdf", "scaled", 1, [1], ["Same Name"], 1, False),
        ("a.pdf", "scaled", 2, [2], ["Same Name"], 2, False),
    ])
    _, scales, names, _, _ = build_groups(df)
    assert set(scales) == {("a.pdf", 1), ("a.pdf", 2)}
    assert names[("a.pdf", 1)][0] == names[("a.pdf", 2)][0] == "Same Name"


def test_null_path_document_kept():
    df = frame([
        (None, "scaled", 1, [1], ["S"], 1, False),
        ("b.pdf", "orphan", None, None, None, 1, None),
    ])
    inst, scales, *_ = build_groups(df)
    assert set(inst) == {None, "b.pdf"}
    assert (None, 1) in scales


# --- run: scale_pooled averages scale + instrument embeddings, model-matched ---

def test_scale_pooled_includes_same_model_instrument_embedding(tmp_path):
    from assemble.pool import run

    def vec(*xs):
        return np.array(xs, dtype=F32)

    df = pd.DataFrame({
        "path": ["a.pdf", "b.pdf"],
        "bucket": ["scaled", "scaled"],
        "scale_id_path": [[1], [1]],
        "scale_name_path": [["S1"], ["S1"]],
        "item_item_id": [1, 1],
        "item_reverse_coded": [False, False],
        "corpus_source": ["t", "t"],
        "public_doi": [None, None],
        "doi_psyctests": [None, None],
        "meta_title_raw": ["Title A", "Title B"],
        **carried(2),
        "scale_embedding_m1": [vec(1, 0), vec(1, 0)],
        "instrument_embedding_m1": [vec(0, 1), None],
        "scale_embedding_m2": [vec(4, 0), vec(4, 0)],
        "instrument_embedding_m2": [vec(0, 4), vec(0, 4)],
    })
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(inp)
    run({}, input_path=inp, output_path=out)
    pooled = pd.read_parquet(out).set_index(["path", "is_instrument"])

    for is_instrument in (False, True):
        row = pooled.loc[("a.pdf", is_instrument)]
        # m1 pools with instrument_embedding_m1, m2 with m2's — never across.
        assert np.allclose(row["scale_pooled_m1"], [0.5, 0.5])
        assert np.allclose(row["scale_pooled_m2"], [2, 2])
        # b.pdf: null m1 instrument embedding contributes nothing.
        row_b = pooled.loc[("b.pdf", is_instrument)]
        assert np.allclose(row_b["scale_pooled_m1"], [1, 0])
        assert np.allclose(row_b["scale_pooled_m2"], [2, 2])


def test_item_pooled_and_disagreement_columns(tmp_path):
    from assemble.pool import run

    def vec(*xs):
        return np.array(xs, dtype=F32)

    # Item 4 is reverse-keyed but aligned with the positives (a disagreement);
    # norms vary to exercise unit-normalization.
    df = pd.DataFrame({
        "path": ["a.pdf"] * 4,
        "bucket": ["scaled"] * 4,
        "scale_id_path": [[1]] * 4,
        "scale_name_path": [["S1"]] * 4,
        "item_item_id": [1, 2, 3, 4],
        "item_reverse_coded": [False, False, False, True],
        "corpus_source": ["t"] * 4,
        "public_doi": [None] * 4,
        "doi_psyctests": [None] * 4,
        "meta_title_raw": ["Title A"] * 4,
        **carried(4),
        "item_embedding_m1": [vec(2, 0), vec(1, 0.1), vec(1, -0.1), vec(3, 0)],
    })
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(inp)
    report = run({}, input_path=inp, output_path=out)
    pooled = pd.read_parquet(out).set_index("is_instrument")
    for is_instrument in (False, True):
        row = pooled.loc[is_instrument]
        # unit items: (1,0), (.995,.0995), (.995,-.0995), and (1,0) negated
        x = 1 / np.sqrt(1.01)
        assert np.allclose(row["item_pooled_m1"], [(1 + 2 * x - 1) / 4, 0])
        assert row["keying_disagreements_m1"] == 1
    assert any("keying_disagreements_m1 (scale rows): 1 flagged" in line
               for line in report)


# --- run: carried columns ---

def _run(tmp_path, df):
    from assemble.pool import run
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(inp)
    run({}, input_path=inp, output_path=out)
    return pd.read_parquet(out)


def _base(n, **overrides):
    """One document, one scale, ``n`` items, plus a scale embedding for model detection."""
    df = pd.DataFrame({
        "path": ["a.pdf"] * n,
        "bucket": ["scaled"] * n,
        "scale_id_path": [[1]] * n,
        "scale_name_path": [["S1"]] * n,
        "item_item_id": list(range(1, n + 1)),
        "item_reverse_coded": [False] * n,
        "corpus_source": ["t"] * n,
        "public_doi": [None] * n,
        "doi_psyctests": [None] * n,
        "meta_title_raw": ["Title A"] * n,
        "scale_embedding_m1": [np.array([1, 0], dtype=F32)] * n,
        **carried(n, **overrides),
    })
    return df


def test_document_columns_carried_onto_scale_and_instrument_rows(tmp_path):
    pooled = _run(tmp_path, _base(
        2, meta_language="de", public_year=1998,
        flag_item_count_deviation=True, flag_scale_count_deviation=None))
    assert len(pooled) == 2  # one scale row, one instrument row
    assert set(pooled["meta_language"]) == {"de"}
    assert set(pooled["public_year"]) == {1998}
    assert pooled["flag_item_count_deviation"].all()
    assert pooled["flag_scale_count_deviation"].isna().all()


def test_version_comes_from_the_scale_node_not_the_document(tmp_path):
    # The parent node has no direct row, so it falls back to the document's version.
    df = pd.DataFrame({
        "path": ["a.pdf"] * 2,
        "bucket": ["scaled"] * 2,
        "scale_id_path": [[1, 2], [1, 3]],
        "scale_name_path": [["Parent", "Child A"], ["Parent", "Child B"]],
        "item_item_id": [1, 2],
        "item_reverse_coded": [False, False],
        "corpus_source": ["t"] * 2,
        "public_doi": [None] * 2,
        "doi_psyctests": [None] * 2,
        "meta_title_raw": ["Title A"] * 2,
        "scale_embedding_m1": [np.array([1, 0], dtype=F32)] * 2,
        **carried(2, version=["Short Form", "Long Form"]),
    })
    pooled = _run(tmp_path, df).set_index(["is_instrument", "scale_id"])
    assert pooled.loc[(False, 2), "version"] == "Short Form"
    assert pooled.loc[(False, 3), "version"] == "Long Form"
    # Parent and instrument row: the document's first row.
    assert pooled.loc[(False, 1), "version"] == "Short Form"
    assert pooled.loc[(True, np.nan), "version"] == "Short Form"


@pytest.mark.parametrize("values, expected", [
    ([False, False, True], True),     # any flagged item flags the group
    ([False, False, False], False),   # all checked, none flagged
    ([None, False, False], False),    # one uncheckable item hides nothing
    ([None, None, True], True),       # a flag survives uncheckable siblings
    ([None, None, None], None),       # nothing checkable -> not checkable
])
def test_item_flags_are_ored_over_the_pooled_items(tmp_path, values, expected):
    pooled = _run(tmp_path, _base(3, flag_item_text_deviation=values))
    got = pooled["flag_item_text_deviation"]
    assert len(got) == 2  # scale row and instrument row pool the same items
    if expected is None:
        assert got.isna().all()
    else:
        assert (got.dropna() == expected).all() and got.notna().all()


def test_missing_carried_column_is_a_hard_error(tmp_path):
    from assemble.pool import run
    df = _base(2).drop(columns=["flag_item_translated"])
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(inp)
    with pytest.raises(SystemExit, match="flag_item_translated"):
        run({}, input_path=inp, output_path=out)


# --- run: parent-only nodes get their own name (scale-name sidecar from encode) ---

def _hierarchy(tmp_path, *, sidecar=True, depth3=False):
    """Instrument -> domain (no direct items) -> two facets; one-hot name vectors."""
    from assemble.encode import scale_names_path
    from assemble.pool import run

    e = np.eye(5, dtype=F32)  # e[0] facet A, e[1] facet B, e[2] domain, e[3] title, e[4] total
    paths = [[1, 2], [1, 3]] if not depth3 else [[9, 1, 2], [9, 1, 3]]
    names = [["Domain", "Facet A"], ["Domain", "Facet B"]]
    if depth3:
        names = [["Total", *n] for n in names]
    df = pd.DataFrame({
        "path": ["a.pdf"] * 2,
        "bucket": ["scaled"] * 2,
        "scale_id_path": paths,
        "scale_name_path": names,
        "item_item_id": [1, 2],
        "item_reverse_coded": [False, False],
        "corpus_source": ["t"] * 2,
        "public_doi": [None] * 2,
        "doi_psyctests": [None] * 2,
        "meta_title_raw": ["Title A"] * 2,
        **carried(2),
        "scale_embedding_m1": [e[0], e[1]],
        "instrument_embedding_m1": [e[3], e[3]],
    })
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(inp)
    if sidecar:
        pd.DataFrame({"scale_name": ["Domain", "Facet A", "Facet B", "Total"],
                      "scale_embedding_m1": [e[2], e[0], e[1], e[4]]}
                     ).to_parquet(scale_names_path(inp))
    report = run({}, input_path=inp, output_path=out)
    return pd.read_parquet(out), report, e


def test_parent_only_node_gets_own_name_embedding_and_label(tmp_path):
    pooled, _, e = _hierarchy(tmp_path)
    scale = pooled[~pooled["is_instrument"]].set_index("scale_id")
    inst = pooled[pooled["is_instrument"]].iloc[0]
    # The domain has no direct rows, but now carries its own name vector ...
    assert np.allclose(scale.loc[1, "scale_embedding_m1"], e[2])
    # ... and its label vector averages its own name, both facets and the title.
    assert np.allclose(scale.loc[1, "scale_pooled_m1"], (e[0] + e[1] + e[2] + e[3]) / 4)
    # Leaves are unchanged: own name and title.
    assert np.allclose(scale.loc[2, "scale_embedding_m1"], e[0])
    assert np.allclose(scale.loc[2, "scale_pooled_m1"], (e[0] + e[3]) / 2)
    assert np.allclose(scale.loc[3, "scale_pooled_m1"], (e[1] + e[3]) / 2)
    # The instrument row is unchanged: the item-bearing scales' names and the title.
    assert np.allclose(inst["scale_pooled_m1"], (e[0] + e[1] + e[3]) / 3)
    assert inst["scale_embedding_m1"] is None


def test_intermediate_node_names_enter_ancestor_labels(tmp_path):
    pooled, _, e = _hierarchy(tmp_path, depth3=True)
    scale = pooled[~pooled["is_instrument"]].set_index("scale_id")
    # Total (9) -> Domain (1) -> facets: the top node includes the intermediate domain's name.
    assert np.allclose(scale.loc[9, "scale_embedding_m1"], e[4])
    assert np.allclose(scale.loc[9, "scale_pooled_m1"], (e[0] + e[1] + e[2] + e[3] + e[4]) / 5)
    assert np.allclose(scale.loc[1, "scale_pooled_m1"], (e[0] + e[1] + e[2] + e[3]) / 4)


def test_without_sidecar_parent_nodes_fall_back_and_warn(tmp_path):
    pooled, report, e = _hierarchy(tmp_path, sidecar=False)
    scale = pooled[~pooled["is_instrument"]].set_index("scale_id")
    assert scale.loc[1, "scale_embedding_m1"] is None
    assert np.allclose(scale.loc[1, "scale_pooled_m1"], (e[0] + e[1] + e[3]) / 3)
    assert any(line.startswith("WARNING: no scale-name sidecar") for line in report)
