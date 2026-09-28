"""Tests for ``assemble.explode``: document-keyed extractions -> one row per item occurrence."""

import pandas as pd
import pytest

from extraction.models import Instrument
from assemble.explode import explode_extractions, run


def _instrument_json() -> str:
    """Item 2 in two scales, a nested subscale, one orphan, one unscaled item, meta."""
    inst = Instrument.model_validate({
        "scales": [
            {
                "id": 1,
                "scale_name": "Big Five",
                "construct_name": None,
                "items": [
                    {"item_id": 1, "item_text": "Item one",
                     "has_image": False, "reverse_coded": False},
                    {"item_id": 2, "item_text": "Item two",
                     "has_image": False, "reverse_coded": False},
                ],
                "subscales": [
                    {
                        "id": 2,
                        "scale_name": "Extraversion",
                        "construct_name": "Extraversion",
                        "items": [
                            {"item_id": 3, "item_text": "Item three",
                             "has_image": False, "reverse_coded": False},
                        ],
                        "subscales": [],
                    },
                ],
            },
            {
                "id": 3,
                "scale_name": "Other",
                "construct_name": None,
                "items": [
                    {"item_id": 2, "item_text": "Item two",
                     "has_image": False, "reverse_coded": True},
                ],
                "subscales": [],
            },
        ],
        "unscaled_items": [{"id": 4, "item_text": "Age?", "has_image": False}],
        "orphan_items": [{"id": 5, "item_text": "I feel anxious.", "has_image": False}],
        "meta": {"language": "en", "page_count": 3},
    })
    return inst.model_dump_json()


def _frame() -> pd.DataFrame:
    """Two documents: one fully extracted, one with null extraction_output."""
    return pd.DataFrame([
        {
            "path": "/docs/a.pdf",
            "has_errors": False,
            "total_duration": 12.5,
            "items_extractor_ttft": 0.3,
            "items_extractor_content": '{"items": []}',   # nested -> dropped
            "extraction_output": _instrument_json(),
        },
        {
            "path": "/docs/b.pdf",
            "has_errors": True,
            "total_duration": 4.0,
            "items_extractor_ttft": None,
            "items_extractor_content": None,
            "extraction_output": None,
        },
    ])


def test_drops_nested_keeps_carry_columns():
    out = explode_extractions(_frame())
    assert "items_extractor_content" not in out.columns
    assert "extraction_output" not in out.columns
    for col in ("path", "has_errors", "total_duration", "items_extractor_ttft"):
        assert col in out.columns


def test_row_count_is_item_occurrences_plus_null_row():
    out = explode_extractions(_frame())
    # doc a: items 1,2 (scale1) + 3 (subscale) + 2 (scale3) + orphan 5 + unscaled 4 = 6
    # doc b: 1 carry-only row
    assert len(out) == 7
    a = out[out["path"] == "/docs/a.pdf"]
    assert len(a) == 6


def test_multi_scale_item_appears_once_per_scale():
    out = explode_extractions(_frame())
    item2 = out[out["item_item_id"] == 2]
    assert len(item2) == 2
    assert set(item2["scale_id"]) == {1, 3}
    assert set(item2["item_reverse_coded"]) == {False, True}


def test_subscale_membership_path_and_depth():
    out = explode_extractions(_frame())
    item3 = out[out["item_item_id"] == 3].iloc[0]
    assert item3["scale_depth"] == 2
    assert item3["scale_id_path"] == [1, 2]
    assert item3["scale_name_path"] == ["Big Five", "Extraversion"]
    assert item3["bucket"] == "scaled"


def test_orphan_and_unscaled_have_null_scale_and_bucket():
    out = explode_extractions(_frame())
    orphan = out[out["item_item_id"] == 5].iloc[0]
    unscaled = out[out["item_item_id"] == 4].iloc[0]
    assert orphan["bucket"] == "orphan"
    assert unscaled["bucket"] == "unscaled"
    for row in (orphan, unscaled):
        assert pd.isna(row["scale_id"])
        assert row["scale_id_path"] is None


def test_meta_broadcast_to_every_item_row():
    out = explode_extractions(_frame())
    a = out[out["path"] == "/docs/a.pdf"]
    assert (a["meta_language"] == "en").all()
    assert (a["meta_page_count"] == 3).all()


def test_null_extraction_output_yields_carry_only_row():
    out = explode_extractions(_frame())
    b = out[out["path"] == "/docs/b.pdf"]
    assert len(b) == 1
    row = b.iloc[0]
    assert pd.isna(row["bucket"])
    assert pd.isna(row["item_item_id"])
    assert row["has_errors"] is True or row["has_errors"] == True  # noqa: E712
    assert row["total_duration"] == 4.0


# --- run() stage entry point ---

def _run_cfg(in_path, out_path) -> dict:
    return {"data": {"extractions": str(in_path),
                     "assemble": {"exploded": str(out_path)}}}


def test_run_reads_raw_extractions(tmp_path):
    in_path = tmp_path / "extractions.parquet"
    out_path = tmp_path / "exploded.parquet"
    _frame().to_parquet(in_path, index=False)

    report = run(_run_cfg(in_path, out_path))
    assert out_path.exists()
    assert any(str(out_path) in line for line in report)
    exploded = pd.read_parquet(out_path)
    assert len(exploded) == 7  # 6 item rows + 1 carry-only row


def test_run_report_only_writes_nothing(tmp_path):
    in_path = tmp_path / "extractions.parquet"
    out_path = tmp_path / "exploded.parquet"
    _frame().to_parquet(in_path, index=False)

    report = run(_run_cfg(in_path, out_path), report_only=True)
    assert not out_path.exists()
    assert any("would explode 2 documents" in line for line in report)


def test_run_missing_config_keys_exit(tmp_path):
    with pytest.raises(SystemExit, match="data.extractions and data.assemble.exploded"):
        run({"data": {}})


def test_run_missing_input_exits(tmp_path):
    cfg = _run_cfg(tmp_path / "nope.parquet", tmp_path / "out.parquet")
    with pytest.raises(SystemExit, match="extraction parquet not found"):
        run(cfg)


def test_run_carries_legacy_scalar_columns(tmp_path):
    """Arbitrary scalar columns on the raw store (e.g. legacy ``doi``) pass through unchanged."""
    frame = _frame()
    frame["doi"] = ["10.1037/t00001-000", None]
    in_path = tmp_path / "extractions.parquet"
    out_path = tmp_path / "exploded.parquet"
    frame.to_parquet(in_path, index=False)

    run(_run_cfg(in_path, out_path))
    exploded = pd.read_parquet(out_path)
    a = exploded[exploded["path"] == "/docs/a.pdf"]
    assert (a["doi"] == "10.1037/t00001-000").all()
