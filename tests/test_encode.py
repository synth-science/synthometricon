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
