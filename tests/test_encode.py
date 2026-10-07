"""Encode-stage helpers and invariants with a stubbed encoder (no I/O, no model loads)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from assemble.encode import _model_name, _targets, _unique_texts, encode_column


class StubModel:
    """Deterministic encoder: vector = [len(text), 1.0]; records calls."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts])


def test_model_name_sanitization():
    assert _model_name("surveybot3000") == "surveybot3000"
    assert _model_name("all-MiniLM-L6-v2") == "all_minilm_l6_v2"
    assert _model_name("Some.Model-v2") == "some_model_v2"


def test_targets_column_mapping():
    cfg = {"item_models": [{"name": "itemer", "path": "/m/itemer/"}],
           "scale_models": [{"name": "scaler-x", "path": "/m/scaler-x"}]}
    assert _targets(cfg) == [
        ("/m/itemer/", "item_item_text", "item_embedding_itemer"),
        ("/m/scaler-x", "scale_name", "scale_embedding_scaler_x"),
        ("/m/scaler-x", "meta_title_raw", "instrument_embedding_scaler_x"),
    ]


def test_targets_empty_config():
    assert _targets({}) == []
    assert _targets({"item_models": None, "scale_models": None}) == []


def test_encode_dedupes_and_maps_back():
    s = pd.Series(["alpha", "beta", "alpha", "alpha"])
    model = StubModel()
    out = encode_column(s, model, batch_size=32)
    assert model.calls == [["alpha", "beta"]]  # unique texts, one call
    assert all(len(v) == 2 for v in out)
    assert out[0][0] == 5.0 and out[1][0] == 4.0
    assert out[0] is out[2] is out[3]  # duplicates share the vector object
    assert out[0].dtype == np.float32


def test_encode_null_and_blank_rows():
    s = pd.Series(["alpha", None, "  ", ""])
    out = encode_column(s, StubModel(), batch_size=32)
    assert out[0] is not None
    assert out[1] is None and out[2] is None and out[3] is None
    assert len(out) == len(s)


def test_encode_all_null_skips_model():
    s = pd.Series([None, None])
    model = StubModel()
    out = encode_column(s, model, batch_size=32)
    assert model.calls == []
    assert list(out) == [None, None]


def test_unique_texts():
    s = pd.Series(["b", "a", "b", None, " ", 3])
    assert _unique_texts(s) == ["a", "b"]


def test_scale_names_path():
    from pathlib import Path

    from assemble.encode import scale_names_path
    assert scale_names_path("/d/corpus-embedded.parquet") == Path("/d/corpus-embedded.scale-names.parquet")


def test_path_names_include_parents_and_skip_blanks():
    from assemble.encode import path_names
    df = pd.DataFrame({"scale_name_path": [["Domain", "Facet A"], ["Domain", "Facet B"],
                                           None, ["", None, "  "]]})
    assert path_names(df) == ["Domain", "Facet A", "Facet B"]
    assert path_names(pd.DataFrame({"x": [1]})) == []


def test_run_encodes_parent_names_once_and_writes_sidecar(tmp_path, monkeypatch):
    import sys
    import types

    from assemble.encode import run, scale_names_path

    models: dict[str, StubModel] = {}

    def factory(path):
        return models.setdefault(path, StubModel())

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=factory))
    df = pd.DataFrame({
        "item_item_text": ["i1", "i2"],
        "scale_name": ["Facet A", "Facet B"],
        "scale_name_path": [["Domain", "Facet A"], ["Domain", "Facet B"]],
        "meta_title_raw": ["Title", "Title"],
    })
    inp, out = tmp_path / "post.parquet", tmp_path / "emb.parquet"
    df.to_parquet(inp)
    cfg = {"encode": {"item_models": [{"name": "it", "path": "/m/it"}],
                      "scale_models": [{"name": "sc", "path": "/m/sc"}]}}
    run(cfg, input_path=inp, output_path=out)
    emb = pd.read_parquet(out)
    # Row columns are unchanged: each row's own scale_name.
    assert [v[0] for v in emb["scale_embedding_sc"]] == [7.0, 7.0]
    # One call for scale names (rows and parents together), one for titles.
    assert models["/m/sc"].calls[0] == ["Domain", "Facet A", "Facet B"]
    side = pd.read_parquet(scale_names_path(out))
    assert side["scale_name"].tolist() == ["Domain", "Facet A", "Facet B"]
    assert side.set_index("scale_name").loc["Domain", "scale_embedding_sc"][0] == 6.0
