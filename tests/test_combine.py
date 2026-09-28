"""Hermetic tests for ``assemble/combine.py``: ``run()`` and schema canonicalization."""
import pandas as pd
import pyarrow as pa
import pytest

from assemble.combine import canonicalize_schema, run, source_from_path


def _cfg(tmp_path, partials) -> dict:
    return {
        "data": {
            "partials": [str(p) for p in partials],
            "assemble": {
                "exploded": str(tmp_path / "apa-psyctests-extractions-exploded.parquet"),
                "combined": str(tmp_path / "combined.parquet"),
            },
        },
    }


def _seed(tmp_path):
    """Exploded frame + one raw partial; returns cfg."""
    exploded = tmp_path / "apa-psyctests-extractions-exploded.parquet"
    pd.DataFrame({"path": ["/pdf/a.pdf"], "value": ["apa"]}).to_parquet(
        exploded, index=False)

    raw = tmp_path / "aligns-extractions-exploded.parquet"
    pd.DataFrame({
        "path": ["aligns/x.url"],
        "value": ["raw-value"],
        "items_extractor_attempts": pd.array([1], dtype="int32"),
    }).to_parquet(raw, index=False)
    return _cfg(tmp_path, [raw])


def test_run_reads_raw_partials(tmp_path):
    cfg = _seed(tmp_path)
    lines = run(cfg)
    combined = pd.read_parquet(tmp_path / "combined.parquet")

    assert set(combined["value"]) == {"apa", "raw-value"}
    assert set(combined["corpus_source"]) == {"apa-psyctests", "aligns"}
    assert any("wrote" in line for line in lines)
    # int32 counter survives the outer concat (NaN-promoted to float64 for apa rows).
    assert combined.set_index("corpus_source")["items_extractor_attempts"]["aligns"] == 1


def test_run_missing_partial_exits(tmp_path):
    cfg = _seed(tmp_path)
    cfg["data"]["partials"].append(str(tmp_path / "nope.parquet"))
    with pytest.raises(SystemExit, match="partial not found"):
        run(cfg)


def test_run_no_partials(tmp_path):
    cfg = _seed(tmp_path)
    cfg["data"]["partials"] = []
    lines = run(cfg, report_only=True)
    assert any("apa-psyctests" in line for line in lines)
    assert not (tmp_path / "combined.parquet").exists()


def test_canonicalize_schema_scalar_int32():
    table = pa.table({
        "attempts": pa.array([1, 2], type=pa.int32()),
        "ids": pa.array([[1], [2]], type=pa.list_(pa.int32())),
        "text": pa.array(["a", "b"], type=pa.large_string()),
    })
    out = canonicalize_schema(table)
    assert out.schema.field("attempts").type == pa.int64()
    assert out.schema.field("ids").type == pa.list_(pa.int64())
    assert out.schema.field("text").type == pa.string()


def test_source_from_path():
    assert source_from_path("aligns-extractions-exploded.parquet") == "aligns"
    assert source_from_path(
        "/data/partials/semanticnet-extractions-exploded.parquet") == "semanticnet"
    assert source_from_path(
        "apa-psyctests-extractions-exploded.parquet") == "apa-psyctests"
