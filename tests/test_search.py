"""Tests for the ranking in ``assemble.search`` (no model needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from assemble.search import _rank


def _frame() -> pd.DataFrame:
    # cosines with the query (1, 0): a .9, b -.95, c .2, d null (skipped)
    vecs = [np.array([0.9, np.sqrt(1 - 0.81)]), np.array([-0.95, np.sqrt(1 - 0.9025)]),
            np.array([0.2, np.sqrt(1 - 0.04)]), None]
    return pd.DataFrame({"scale_name": ["a", "b", "c", "d"], "item_pooled_m": vecs})


def test_rank_signed_puts_negative_last():
    out = _rank(_frame(), "item_pooled_m", np.array([1.0, 0.0]), top_k=3)
    assert out["scale_name"].tolist() == ["a", "c", "b"]


def test_rank_absolute_keeps_sign_in_output():
    out = _rank(_frame(), "item_pooled_m", np.array([1.0, 0.0]), top_k=2, absolute=True)
    assert out["scale_name"].tolist() == ["b", "a"]
    assert np.allclose(out["similarity"].to_numpy(), [-0.95, 0.9])
