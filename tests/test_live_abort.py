"""``run_live`` stops on an unreachable inference server instead of failing every remaining document."""
from __future__ import annotations

import httpx
import openai
from pydantic import BaseModel

import extraction.orchestrator as orch
from extraction.storage import cell_state, get_attempts, load_or_init_df


class _Result(BaseModel):
    ok: bool = True


class _FakeExtractor:
    depends_on: list = []
    needs_first_page_text = False
    needs_doc_stats = False
    result_model = _Result

    def __init__(self, name):
        self.name = name


def _run(tmp_path, monkeypatch, fail_on: dict):
    """Run items + scales over a.pdf, b.pdf, c.pdf; ``fail_on[(path, name)]`` is raised there."""
    exts = [_FakeExtractor("items"), _FakeExtractor("scales")]
    calls = []

    def step(ext, images, config, context, log):
        calls.append((context["_path"], ext.name))
        exc = fail_on.get((context["_path"], ext.name))
        if exc is not None:
            raise exc
        return _Result(), 0.0, {}

    current = {}
    monkeypatch.setattr(orch, "_render_pdf_pages",
                        lambda path, config: current.update(path=path) or [])
    monkeypatch.setattr(orch, "_run_extractor_step",
                        lambda ext, images, config, context, log:
                        step(ext, images, config, {**context, "_path": current["path"]}, log))
    out = tmp_path / "store.parquet"
    result = orch.run_live(exts, exts, ["a.pdf", "b.pdf", "c.pdf"], out, {}, no_logs=True,
                           run_timestamp="test")
    return result, load_or_init_df(out, ["items", "scales"]), calls


def test_unreachable_server_aborts_without_recording_failures(tmp_path, monkeypatch):
    down = openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:8080"))
    result, df, calls = _run(tmp_path, monkeypatch, {("b.pdf", "scales"): down})
    assert result["aborted"] and not result["all_passed"]
    assert calls == [("a.pdf", "items"), ("a.pdf", "scales"), ("b.pdf", "items"), ("b.pdf", "scales")]
    assert cell_state(df, "a.pdf", "scales") == "done"
    assert cell_state(df, "b.pdf", "items") == "done"      # kept what succeeded before the outage
    assert cell_state(df, "b.pdf", "scales") == "pending"  # not a document failure
    assert get_attempts(df, "b.pdf", "scales") == 0
    assert "c.pdf" not in set(df.index) or cell_state(df, "c.pdf", "items") == "pending"


def test_timeout_is_still_a_document_failure(tmp_path, monkeypatch):
    slow = openai.APITimeoutError(request=httpx.Request("POST", "http://localhost:8080"))
    result, df, calls = _run(tmp_path, monkeypatch, {("b.pdf", "items"): slow})
    assert not result["aborted"] and not result["all_passed"]
    assert cell_state(df, "b.pdf", "items") == "failed"
    assert get_attempts(df, "b.pdf", "items") == 1
    assert cell_state(df, "c.pdf", "scales") == "done"  # the run went on
