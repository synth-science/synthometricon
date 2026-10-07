"""Extractor run loops over PDFs: reliability, live, status, validity and fixture seeding.

See docs/architecture-extraction.md#orchestrator.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml
from pydantic import BaseModel, ValidationError

from .config import build_options, resolve_options, resolve_think, write_fixture
from .extractors.base import Extractor
from .llm import format_usage_breakdown, stream_and_collect
from .models import (
    Instrument,
    Meta,
    compose_extraction_output,
    derive_items,
    derive_survey,
)
from .pdf_io import pdf_doc_stats, pdf_first_page_text, pdf_to_images
from .storage import (
    USAGE_SUFFIXES,
    atomic_write_parquet,
    cell_state,
    get_attempts,
    get_cell,
    load_cell,
    load_or_init_df,
    read_paths,
    upsert_row,
)
from .text_utils import Tee, write_run_log


_DERIVERS = {
    "items": derive_items,
    "scales": derive_survey,
}

# Usage suffixes nulled on failure (``error``/``timestamp`` are set explicitly).
_USAGE_NUMERIC_SUFFIXES: tuple[str, ...] = tuple(
    suffix for suffix, _dtype in USAGE_SUFFIXES if suffix not in ("error", "timestamp")
)


def build_dependency_context(
    pdf_path: str,
    deps: list[str],
    registry: dict[str, Extractor],
    config: dict,
    case: dict | None = None,
    fixtures_dir: str | None = None,
    force_live: bool = False,
) -> dict[str, BaseModel]:
    """Return ``{dep_name: model}`` for ``pdf_path`` derived from its ``extraction_output:`` fixture.

    Builds the fixture via a full pipeline run when missing (persisted unless ``force_live``).
    See docs/architecture-extraction.md#dependency-context.
    """
    from .validity import normalize_for_model_validation, strip_match_specs

    fixtures = (case or {}).get("fixtures") or {}
    extraction_output_raw = None if force_live else fixtures.get("extraction_output")

    if extraction_output_raw is None:
        mode = "force-live (no persist)" if force_live else "auto-build (no fixture on disk)"
        print(
            f"  no extraction_output for {pdf_path}; running full pipeline [{mode}] …",
            flush=True,
        )
        instrument_result, meta_result = _run_full_pipeline_in_memory(
            pdf_path, list(registry.values()), config, lambda _s: None
        )
        composed = compose_extraction_output(instrument_result, meta_result)
        if composed is None:
            raise RuntimeError(
                f"Cannot build extraction_output for {pdf_path}: "
                "instrument extractor produced no output"
            )
        extraction_output_raw = composed.model_dump(mode="json")

        if case is not None:
            case.setdefault("fixtures", {})["extraction_output"] = extraction_output_raw
        if not force_live and fixtures_dir and case is not None:
            stem = Path(pdf_path).stem
            write_fixture(fixtures_dir, stem, {"extraction_output": extraction_output_raw})
            print(f"  → wrote auto-built extraction_output to {fixtures_dir}/{stem}.yaml")

    clean = strip_match_specs(extraction_output_raw)
    normalized = normalize_for_model_validation(clean)
    instrument = Instrument.model_validate(normalized)

    context: dict[str, BaseModel] = {}
    for dep in deps:
        if dep not in _DERIVERS:
            raise RuntimeError(
                f"Unknown dependency '{dep}' — derivers only support {list(_DERIVERS)}"
            )
        context[dep] = _DERIVERS[dep](instrument)

    return context


def _assess_criteria(
    scores: dict[str, float], thresholds: dict | None
) -> list[tuple[str, float, float | None, bool]]:
    """One ``(metric, score, threshold, passed)`` per criterion; missing/NaN thresholded metrics fail."""
    if thresholds:
        criteria = []
        for metric, threshold in thresholds.items():
            score = scores.get(metric, float("nan"))
            passed = not math.isnan(score) and score >= threshold
            criteria.append((metric, score, threshold, passed))
        return criteria
    return [(metric, score, None, True) for metric, score in scores.items()]


def _assess_status(scores: dict[str, float], thresholds: dict | None) -> str:
    """Return 'PASS' or 'FAIL' (FAIL if any criterion fails)."""
    return (
        "PASS"
        if all(passed for _, _, _, passed in _assess_criteria(scores, thresholds))
        else "FAIL"
    )


def _format_status_lines(
    scores: dict[str, float], thresholds: dict | None
) -> list[str]:
    """One ``[PASS]`` / ``[FAIL]`` row per criterion."""
    lines: list[str] = []
    for metric, score, threshold, passed in _assess_criteria(scores, thresholds):
        if metric not in scores:
            score_str = "missing"
        elif math.isnan(score):
            score_str = "nan"
        else:
            score_str = f"{score:.4f}"
        tag = "PASS" if passed else "FAIL"
        if threshold is None:
            lines.append(f"[{tag}]  {metric}={score_str}")
        else:
            op = "≥" if passed else "<"
            lines.append(f"[{tag}]  {metric}={score_str} {op} {threshold}")
    return lines or [f"[{_assess_status(scores, thresholds)}]"]


def run_once(
    extractor: Extractor,
    images: list[bytes],
    config: dict,
    context: dict[str, object] | None = None,
) -> tuple[BaseModel, float, dict]:
    """Run one extractor call; return ``(result, elapsed_s, usage)``.

    Retries up to ``config.max_retries`` times on parse failure, then re-raises.
    """
    model = config["model"]
    verbose = bool(config.get("verbose", False))
    max_retries = int(config.get("max_retries", 0))
    timeout: float | None = config.get("timeout")
    options = build_options(resolve_options(config, extractor.name))
    think = resolve_think(config, extractor.name)
    base_url: str = config.get("llama_cpp", {}).get("base_url", "http://127.0.0.1:8181/v1")
    image_format: str = config.get("images", {}).get("format", "png")
    server_cfg = (config.get("llama_cpp") or {}).get("server") or {}
    ctx_size = server_cfg.get("ctx_size")
    image_min_tokens = server_cfg.get("image_min_tokens")

    user_prompt = extractor.build_user_prompt(context)

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"  [retry {attempt}/{max_retries}]", flush=True)
        _, content_text, elapsed, usage = stream_and_collect(
            base_url,
            model,
            extractor.system_prompt,
            user_prompt,
            images,
            image_format,
            extractor.llm_schema(),
            options,
            think,
            verbose,
            timeout,
            ctx_size=ctx_size,
            image_min_tokens=image_min_tokens,
        )
        print("Parsing structured output …")
        try:
            return extractor.parse_response(content_text, context), elapsed, usage
        except (ValidationError, ValueError) as exc:
            print(f"\n[PARSE ERROR] attempt {attempt + 1}: {exc}\n")
            print("Raw content:\n", content_text)
            last_exc = exc
    assert last_exc is not None
    raise last_exc


def _render_pdf_pages(pdf_path: str, config: dict) -> list[bytes]:
    """Render every page of ``pdf_path`` and print a one-line progress note."""
    print("Converting PDF pages to images …", flush=True)
    t0 = time.perf_counter()
    images = pdf_to_images(pdf_path, config)
    print(f"  {len(images)} page(s) rendered in {time.perf_counter() - t0:.2f}s\n")
    return images


def _log_usage(usage: dict, log: Callable[[str], None]) -> None:
    """Print and log the usage breakdown + latency line for one extractor call."""
    if not usage:
        return
    breakdown = format_usage_breakdown(usage)
    print(breakdown, flush=True)
    log(breakdown)
    ttft = usage.get("ttft")
    ttft_str = f"{ttft:.2f}s" if ttft is not None else "n/a"
    # prefill_ms ≈ ttft means latency is prompt processing, not hidden generation.
    prefill = usage.get("prefill_ms")
    prefill_str = f"  prefill={prefill / 1000:.2f}s" if prefill is not None else ""
    cached = usage.get("cached_tokens")
    cached_str = f"  cached={cached:,}tok" if cached is not None else ""
    draft = usage.get("draft_acceptance")
    draft_str = f"  draft_acc={draft:.0%}" if draft is not None else ""
    think_chars = usage.get("thinking_chars")
    think_str = f"  thinking={think_chars}ch" if think_chars else ""
    latency = (
        f"  Latency: ttft={ttft_str}{prefill_str}  "
        f"total={usage.get('total_duration', 0) / 1e9:.2f}s"
        f"{cached_str}{draft_str}{think_str}"
    )
    print(latency, flush=True)
    log(latency)


def _run_extractor_step(
    ext: Extractor,
    images: list[bytes],
    config: dict,
    context: dict[str, object],
    log: Callable[[str], None],
) -> tuple[BaseModel, float, dict]:
    """Run :func:`run_once` with header, result and usage logging; exceptions propagate."""
    print(f"--- Running {ext.name} ---", flush=True)
    log(f"  Extractor: {ext.name}")
    result, elapsed, usage = run_once(ext, images, config, context)
    ext.format_run(result, 0, 1, elapsed, log)
    _log_usage(usage, log)
    return result, elapsed, usage


def _doc_header_lines(file_idx: int, total: int, pdf_path: str) -> list[str]:
    bar = "=" * 60
    return [bar, f"File {file_idx + 1} / {total}: {pdf_path}", bar]


def _print_and_log_doc_header(
    file_idx: int, total: int, pdf_path: str, log: Callable[[str], None]
) -> None:
    """Print + log the document banner (blank-line padding is print-only)."""
    lines = _doc_header_lines(file_idx, total, pdf_path)
    print("\n" + "\n".join(lines) + "\n")
    for line in lines:
        log(line)


def run_extractor(
    extractor: Extractor,
    config: dict,
    run_timestamp: str,
    pdf_paths: list[str],
    num_runs: int = 1,
    no_logs: bool = False,
    reliability_thresholds: dict | None = None,
    contexts: dict[str, dict[str, BaseModel]] | None = None,
) -> dict:
    """Reliability mode: run ``extractor`` ``num_runs`` times per PDF and score agreement.

    ``contexts`` maps ``pdf_path → {dep_name: model}``. Returns
    ``{"doc_results": [{"path", "status", "scores", "status_lines"}], "all_passed"}``.
    """
    model: str = config["model"]
    verbose: bool = bool(config.get("verbose", False))
    options = build_options(resolve_options(config, extractor.name))
    reliability_mode: bool = reliability_thresholds is not None
    think = resolve_think(config, extractor.name)
    run_retries = int((config.get("testing") or {}).get("run_retries", 2))

    log_basename = f"test-reliability-{run_timestamp}-{extractor.name}.log"
    out = Tee()

    out.log(f"=== {extractor.name.capitalize()} Extraction Log ===")
    out.log(f"Date/Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.log(f"Model     : {model}")
    out.log(f"Thinking  : {think}")
    out.log(f"Files     : {len(pdf_paths)}")
    out.log(f"Runs      : {num_runs}  (run retries: {run_retries})")
    out.log(f"Options   : {options}")
    if reliability_mode:
        out.log(f"Mode      : reliability  (thresholds: {reliability_thresholds})")
    out.log("")

    print(f"extractor: {extractor.name}")
    print(f"model    : {model}")
    print(f"thinking : {think}")
    print(f"verbose  : {verbose}")
    print(f"files    : {len(pdf_paths)}")
    print(f"runs     : {num_runs}  (run retries: {run_retries})")
    print(f"options  : {options}\n")

    t_total_start = time.perf_counter()
    doc_results: list[dict] = []
    all_doc_signals: list[list] = []

    for file_idx, pdf_path in enumerate(pdf_paths):
        if len(pdf_paths) > 1:
            _print_and_log_doc_header(file_idx, len(pdf_paths), pdf_path, out.log)
        else:
            for line in _doc_header_lines(file_idx, len(pdf_paths), pdf_path):
                out.log(line)

        try:
            images = _render_pdf_pages(pdf_path, config)
        except Exception as exc:
            msg = (
                f"  [FAIL] {extractor.name}: PDF render error: "
                f"{type(exc).__name__}: {exc}"
            )
            out.write(msg)
            doc_results.append(
                {"path": pdf_path, "status": "FAIL", "scores": {},
                 "status_lines": [msg.strip()]}
            )
            continue

        ctx = contexts.get(pdf_path) if contexts else None
        if extractor.depends_on:
            missing = [d for d in extractor.depends_on if not ctx or d not in ctx]
            if missing:
                msg = (
                    f"  [FAIL] {extractor.name}: missing dependency context "
                    f"for {pdf_path}: {missing}"
                )
                out.write(msg)
                doc_results.append(
                    {"path": pdf_path, "status": "FAIL", "scores": {},
                     "status_lines": [msg.strip()]}
                )
                continue

        if getattr(extractor, "needs_first_page_text", False):
            ctx = dict(ctx or {})
            ctx["_first_page_text"] = pdf_first_page_text(pdf_path)
        if getattr(extractor, "needs_doc_stats", False):
            ctx = dict(ctx or {})
            ctx["_doc_stats"] = pdf_doc_stats(pdf_path)

        signals: list = []
        run_times: list[float] = []
        n_retried_runs = 0
        fail_msg: str | None = None
        for run_idx in range(num_runs):
            if num_runs > 1:
                print(f"--- Run {run_idx + 1} / {num_runs} ---", flush=True)
            # Re-attempt errored runs (testing.run_retries) so transient failures don't sink the doc.
            outcome = None
            for attempt in range(run_retries + 1):
                try:
                    outcome = run_once(extractor, images, config, ctx)
                    break
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    if attempt < run_retries:
                        n_retried_runs += 1
                        out.write(
                            f"  [RETRY] {extractor.name}: run {run_idx + 1}/{num_runs} "
                            f"errored (attempt {attempt + 1}/{run_retries + 1}), "
                            f"re-running: {err}"
                        )
                    else:
                        fail_msg = (
                            f"[FAIL] {extractor.name}: run {run_idx + 1}/{num_runs} "
                            f"errored on all {run_retries + 1} attempt(s): {err}"
                        )
                        out.write(f"  {fail_msg}")
            if outcome is None:
                break
            result, run_elapsed, usage = outcome
            run_times.append(run_elapsed)
            extractor.format_run(result, run_idx, num_runs, run_elapsed, out.log)
            _log_usage(usage, out.log)
            signals.append(extractor.signal(result))

        if fail_msg is not None:
            doc_results.append(
                {"path": pdf_path, "status": "FAIL", "scores": {},
                 "status_lines": [fail_msg]}
            )
            out.log("")
            continue

        if len(run_times) > 1:
            arr = np.array(run_times)
            out.log(
                f"  Timing : mean={arr.mean():.2f}s  std={arr.std():.2f}s  "
                f"min={arr.min():.2f}s  max={arr.max():.2f}s"
            )
        else:
            out.log(f"  Timing : {run_times[0]:.2f}s")
        if n_retried_runs:
            out.log(f"  Retries: {n_retried_runs} errored run attempt(s) re-run")
        out.log("")

        all_doc_signals.append(signals)

        scores: dict[str, float] = {}
        if num_runs >= 2:
            report_str, scores = extractor.reliability(signals)
            out.log(report_str)

        status_lines: list[str] = []
        if reliability_mode:
            if num_runs < 2:
                doc_status = "SKIPPED"
                status_lines = ["[SKIPPED]  num_runs < 2 — reliability not assessed"]
            else:
                doc_status = _assess_status(scores, reliability_thresholds)
                status_lines = _format_status_lines(scores, reliability_thresholds)
            out.write(os.path.basename(pdf_path))
            for line in status_lines:
                out.write(line)
            out.log("")
        else:
            doc_status = "N/A"

        doc_results.append(
            {"path": pdf_path, "status": doc_status, "scores": scores,
             "status_lines": status_lines}
        )

    if reliability_mode:
        n_pass = sum(1 for r in doc_results if r["status"] == "PASS")
        n_fail = sum(1 for r in doc_results if r["status"] == "FAIL")
        n_skip = sum(1 for r in doc_results if r["status"] == "SKIPPED")

        summary_lines = [
            "",
            "=" * 60,
            f"SUITE RELIABILITY SUMMARY  ({len(doc_results)} document(s))",
            "=" * 60,
            f"  PASS: {n_pass}   FAIL: {n_fail}   SKIPPED: {n_skip}",
        ]
        suite_report = extractor.suite_reliability(all_doc_signals)
        if suite_report:
            summary_lines.append("")
            summary_lines.extend(suite_report.splitlines())
        summary_lines.extend(["=" * 60, ""])
        out.writelines(summary_lines)

    total_elapsed = time.perf_counter() - t_total_start
    out.log("=" * 60)
    out.log(f"Total duration : {total_elapsed:.2f}s")
    out.log("=" * 60)

    if not no_logs:
        log_path = write_run_log(
            out.dump(), log_basename, len(pdf_paths), config
        )
        print(f"[log written to {log_path}]")

    all_passed = all(r["status"] in ("PASS", "SKIPPED") for r in doc_results)
    return {"doc_results": doc_results, "all_passed": all_passed}


def _success_updates(
    name: str, result: BaseModel, usage: dict, timestamp: str, attempts: int
) -> dict[str, object]:
    """Build the per-extractor parquet updates for a successful run."""
    updates: dict[str, object] = {
        f"{name}_extractor_content": result.model_dump_json(),
        f"{name}_extractor_timestamp": timestamp,
        f"{name}_extractor_error": None,
        f"{name}_extractor_attempts": attempts,
    }
    if usage:
        total_duration = usage.get("total_duration")
        updates[f"{name}_extractor_total_duration"] = (
            total_duration / 1e9 if total_duration is not None else None
        )
        updates[f"{name}_extractor_prompt_eval_count"] = usage.get("prompt_eval_count")
        updates[f"{name}_extractor_eval_count"] = usage.get("eval_count")
        updates[f"{name}_extractor_image_tokens"] = usage.get("image_tokens")
        updates[f"{name}_extractor_system_tokens"] = usage.get("system_tokens")
        updates[f"{name}_extractor_user_text_tokens"] = usage.get("user_text_tokens")
        updates[f"{name}_extractor_ttft"] = usage.get("ttft")
        # Usage key == column suffix for these.
        for suffix in (
            "prefill_ms", "predicted_ms", "prompt_per_second",
            "predicted_per_second", "cached_tokens", "draft_acceptance",
            "thinking_chars",
        ):
            updates[f"{name}_extractor_{suffix}"] = usage.get(suffix)
    return updates


def _is_server_unreachable(exc: BaseException) -> bool:
    """True when the inference server is down or unreachable, i.e. not a failure of the document.

    Timeouts are excluded (``APITimeoutError`` subclasses ``APIConnectionError``): a request that
    runs past the timeout on a long document is that document's failure.
    """
    import httpx
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return False
    return isinstance(exc, (openai.APIConnectionError, httpx.ConnectError, ConnectionError))


def _failure_updates(
    name: str, exc: Exception, timestamp: str, attempts: int
) -> dict[str, object]:
    """Build parquet updates for a failed run: content/usage null, error and attempts recorded."""
    updates: dict[str, object] = {
        f"{name}_extractor_content": None,
        f"{name}_extractor_timestamp": timestamp,
        f"{name}_extractor_error": f"{type(exc).__name__}: {exc}",
        f"{name}_extractor_attempts": attempts,
    }
    for suffix in _USAGE_NUMERIC_SUFFIXES:
        updates[f"{name}_extractor_{suffix}"] = None
    return updates


def _recompose_extraction_output(df, pdf_path: str):
    """Recompose ``extraction_output`` from the row's instrument + meta cells (may be ``None``)."""
    instrument_raw = get_cell(df, pdf_path, "instrument_extractor_content")
    meta_raw = get_cell(df, pdf_path, "meta_extractor_content")
    instrument = Instrument.model_validate_json(instrument_raw) if instrument_raw else None
    meta = Meta.model_validate_json(meta_raw) if meta_raw else None
    return compose_extraction_output(instrument, meta)


def _retry_decision(
    df,
    pdf_path: str,
    ext: Extractor,
    *,
    force: bool,
    retry_failed: bool,
    retry_error: Optional[str],
    max_attempts: Optional[int],
) -> tuple[bool, str]:
    """Decide whether ``ext`` runs on ``pdf_path``; return ``(run, reason)``.

    Reasons: ``pending``/``retry``/``forced`` (run) or ``done``/``failed``/``exhausted`` (skip).
    See docs/architecture-extraction.md#live-mode-resume-policy.
    """
    state = cell_state(df, pdf_path, ext.name)
    if state == "done":
        return (True, "forced") if force else (False, "done")
    if state == "pending":
        return (True, "pending")
    # state == "failed"
    if force:
        return (True, "forced")
    err = get_cell(df, pdf_path, f"{ext.name}_extractor_error") or ""
    want = retry_failed or (
        retry_error is not None and re.search(retry_error, err) is not None
    )
    if not want:
        return (False, "failed")
    if max_attempts is not None and get_attempts(df, pdf_path, ext.name) >= max_attempts:
        return (False, "exhausted")
    return (True, "retry")


def run_live(
    selected_extractors: list[Extractor],
    all_extractors: list[Extractor],
    pdf_paths: list[str],
    output_path: Path,
    config: dict,
    no_logs: bool,
    run_timestamp: str,
    force: bool = False,
    retry_failed: bool = False,
    retry_error: Optional[str] = None,
    max_attempts: Optional[int] = None,
) -> dict:
    """Live mode: run ``selected_extractors`` over ``pdf_paths``, persisting to ``output_path``.

    ``all_extractors`` (REGISTRY order) fixes the schema and supplies cached dependencies.
    See docs/architecture-extraction.md#live-mode-resume-policy.
    """
    all_names = [e.name for e in all_extractors]
    selected_names = {e.name for e in selected_extractors}

    df = load_or_init_df(output_path, all_names)

    log_basename = f"live-{run_timestamp}.log"
    out = Tee()
    retry_desc = (
        "force-all" if force
        else f"retry-error={retry_error!r}" if retry_error is not None
        else "retry-failed" if retry_failed
        else "skip-done"
    )
    max_attempts_desc = "unlimited" if max_attempts is None else str(max_attempts)

    out.log("=== Live Extraction Log ===")
    out.log(f"Date/Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.log(f"Output    : {output_path}")
    out.log(f"Force     : {force}")
    out.log(f"Retry     : {retry_desc}  (max_attempts: {max_attempts_desc})")
    out.log(f"Files     : {len(pdf_paths)}")
    out.log(f"Extractors: {', '.join(e.name for e in selected_extractors)}")
    out.log("")

    print(f"output    : {output_path}")
    print(f"force     : {force}")
    print(f"retry     : {retry_desc}  (max_attempts: {max_attempts_desc})")
    print(f"files     : {len(pdf_paths)}")
    print(f"extractors: {', '.join(e.name for e in selected_extractors)}\n")

    t_total_start = time.perf_counter()
    doc_results: list[dict] = []
    aborted = False  # server unreachable: stop instead of failing the rest of the worklist

    for file_idx, pdf_path in enumerate(pdf_paths):
        if aborted:
            break
        t_doc_start = time.perf_counter()
        _print_and_log_doc_header(file_idx, len(pdf_paths), pdf_path, out.log)

        to_run: list[Extractor] = []
        already_done: list[str] = []
        skipped_failed: list[str] = []
        skipped_exhausted: list[str] = []
        for ext in [e for e in all_extractors if e.name in selected_names]:
            run, reason = _retry_decision(
                df, pdf_path, ext,
                force=force, retry_failed=retry_failed,
                retry_error=retry_error, max_attempts=max_attempts,
            )
            if run:
                to_run.append(ext)
            elif reason == "done":
                already_done.append(ext.name)
            elif reason == "exhausted":
                skipped_exhausted.append(ext.name)
            else:  # "failed"
                skipped_failed.append(ext.name)

        if already_done:
            out.write(f"  cached: {', '.join(already_done)}")
        if skipped_failed:
            out.write(
                f"  skipped (prior failure; pass --retry-failed/--retry-error): "
                f"{', '.join(skipped_failed)}"
            )
        if skipped_exhausted:
            out.write(
                f"  skipped (max-attempts reached): {', '.join(skipped_exhausted)}"
            )

        if not to_run:
            doc_results.append({
                "path": pdf_path, "ran": [], "failed": [],
                "skipped_failed": skipped_failed,
                "skipped_exhausted": skipped_exhausted,
            })
            continue

        # Seed context from cached cells; values are ``object`` because of ``_``-prefixed raw inputs.
        context: dict[str, object] = {}
        for ext in all_extractors:
            if ext in to_run:
                continue
            cached = load_cell(df, pdf_path, f"{ext.name}_extractor_content", ext.result_model)
            if cached is not None:
                context[ext.name] = cached

        # A render failure is recorded against every scheduled extractor; never crash the run.
        try:
            images = _render_pdf_pages(pdf_path, config)
            if any(getattr(e, "needs_first_page_text", False) for e in to_run):
                context["_first_page_text"] = pdf_first_page_text(pdf_path)
            if any(getattr(e, "needs_doc_stats", False) for e in to_run):
                context["_doc_stats"] = pdf_doc_stats(pdf_path)
        except Exception as exc:
            out.write(f"  [FAIL] PDF render error: {type(exc).__name__}: {exc}")
            doc_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            updates: dict[str, object] = {}
            for ext in to_run:
                attempts = get_attempts(df, pdf_path, ext.name) + 1
                updates.update(_failure_updates(ext.name, exc, doc_ts, attempts))
            updates["has_errors"] = True
            updates["total_duration"] = time.perf_counter() - t_doc_start
            df = upsert_row(df, pdf_path, updates)
            atomic_write_parquet(df, output_path)
            out.write(f"  → wrote {output_path}")
            doc_results.append({
                "path": pdf_path, "ran": [], "failed": [e.name for e in to_run],
                "skipped_failed": skipped_failed,
                "skipped_exhausted": skipped_exhausted,
            })
            continue

        updates: dict[str, object] = {}
        ran: list[str] = []
        failed: list[str] = []

        for ext in to_run:
            missing = [d for d in ext.depends_on if d not in context]
            if missing:
                out.write(f"  [SKIP] {ext.name}: missing dependencies {missing}")
                failed.append(ext.name)
                continue
            start_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            attempts = get_attempts(df, pdf_path, ext.name) + 1
            try:
                result, _elapsed, usage = _run_extractor_step(
                    ext, images, config, context, out.log
                )
            except Exception as exc:
                if _is_server_unreachable(exc):
                    # Infrastructure, not this document: record nothing, keep the attempt count.
                    out.write(f"  [ABORT] {ext.name}: inference server unreachable "
                              f"({type(exc).__name__}: {exc}); not recorded as a "
                              "document failure. Stopping the run.")
                    aborted = True
                    break
                out.write(f"  [FAIL] {ext.name}: {exc}")
                updates.update(_failure_updates(ext.name, exc, start_ts, attempts))
                failed.append(ext.name)
                continue

            updates.update(_success_updates(ext.name, result, usage, start_ts, attempts))
            context[ext.name] = result
            ran.append(ext.name)

        if updates:
            updates["has_errors"] = bool(failed)
            updates["total_duration"] = time.perf_counter() - t_doc_start
            df = upsert_row(df, pdf_path, updates)

            # Null until the instrument extractor has produced a tree.
            composed = _recompose_extraction_output(df, pdf_path)
            df = upsert_row(df, pdf_path, {
                "extraction_output": composed.model_dump_json() if composed else None,
            })

            atomic_write_parquet(df, output_path)
            out.write(f"  → wrote {output_path}")

        doc_results.append({
            "path": pdf_path, "ran": ran, "failed": failed,
            "skipped_failed": skipped_failed,
            "skipped_exhausted": skipped_exhausted,
        })

    total_elapsed = time.perf_counter() - t_total_start
    n_ran = sum(len(r["ran"]) for r in doc_results)
    n_failed = sum(len(r["failed"]) for r in doc_results)
    n_skipped_failed = sum(len(r.get("skipped_failed", [])) for r in doc_results)
    n_skipped_exhausted = sum(len(r.get("skipped_exhausted", [])) for r in doc_results)
    out.write(
        f"\nLive extraction complete: {len(doc_results)} doc(s), "
        f"{n_ran} extractor-run(s) succeeded, {n_failed} failed, "
        f"{n_skipped_failed} skipped (prior failure), "
        f"{n_skipped_exhausted} skipped (max-attempts), "
        f"{total_elapsed:.2f}s total."
    )
    if aborted:
        out.write(
            f"ABORTED: inference server unreachable after {len(doc_results)} of "
            f"{len(pdf_paths)} doc(s); the remaining documents were not attempted. "
            "Restart the server and rerun."
        )

    if not no_logs:
        log_path = write_run_log(
            out.dump(), log_basename, len(pdf_paths), config, allow_scratch=False
        )
        print(f"[log written to {log_path}]")

    return {"doc_results": doc_results, "all_passed": n_failed == 0 and not aborted,
            "aborted": aborted}


def run_status(
    all_extractors: list[Extractor],
    output_path: Path,
    top_errors: int = 15,
) -> dict:
    """Print and return per-extractor done/failed/pending counts plus an error-string histogram."""
    all_names = [e.name for e in all_extractors]

    if not output_path.exists():
        print(f"No parquet at {output_path} — nothing to report.")
        return {"n_docs": 0, "per_extractor": {}, "errors": {}}

    df = load_or_init_df(output_path, all_names)
    paths = read_paths(output_path)
    n_docs = len(paths)

    per_extractor: dict[str, dict[str, int]] = {}
    error_counter: Counter[str] = Counter()
    for name in all_names:
        counts = {"done": 0, "failed": 0, "pending": 0}
        for pdf_path in paths:
            state = cell_state(df, pdf_path, name)
            counts[state] += 1
            if state == "failed":
                err = get_cell(df, pdf_path, f"{name}_extractor_error") or "<unknown>"
                error_counter[err] += 1
        per_extractor[name] = counts

    name_w = max([len(n) for n in all_names] + [len("Extractor")])
    print(f"\nExtraction status — {output_path}  ({n_docs} document(s))")
    print("=" * 60)
    print(f"  {'Extractor':<{name_w}}  {'done':>6}  {'failed':>6}  {'pending':>7}")
    print(f"  {'-' * name_w}  {'-' * 6}  {'-' * 6}  {'-' * 7}")
    for name in all_names:
        c = per_extractor[name]
        print(f"  {name:<{name_w}}  {c['done']:>6}  {c['failed']:>6}  {c['pending']:>7}")
    print("=" * 60)

    if error_counter:
        print(f"\nTop error strings ({min(top_errors, len(error_counter))} of {len(error_counter)}):")
        for err, count in error_counter.most_common(top_errors):
            preview = err if len(err) <= 100 else err[:97] + "..."
            print(f"  {count:>5}×  {preview}")
    else:
        print("\nNo failed cells.")
    print()

    return {
        "n_docs": n_docs,
        "per_extractor": per_extractor,
        "errors": dict(error_counter),
    }


def _run_full_pipeline_in_memory(
    pdf_path: str,
    all_extractors: list[Extractor],
    config: dict,
    log: Callable[[str], None],
) -> tuple[Optional[Instrument], Optional[Meta]]:
    """Run every extractor in memory (no parquet); return ``(instrument, meta)``, either may be ``None``."""
    images = _render_pdf_pages(pdf_path, config)

    context: dict[str, object] = {}
    if any(getattr(e, "needs_first_page_text", False) for e in all_extractors):
        context["_first_page_text"] = pdf_first_page_text(pdf_path)
    if any(getattr(e, "needs_doc_stats", False) for e in all_extractors):
        context["_doc_stats"] = pdf_doc_stats(pdf_path)

    instrument_result: Optional[Instrument] = None
    meta_result: Optional[Meta] = None

    for ext in all_extractors:
        missing = [d for d in ext.depends_on if d not in context]
        if missing:
            msg = f"  [SKIP] {ext.name}: missing dependencies {missing}"
            print(msg)
            log(msg)
            continue
        try:
            result, _elapsed, _usage = _run_extractor_step(
                ext, images, config, context, log
            )
        except Exception as exc:
            msg = f"  [FAIL] {ext.name}: {exc}"
            print(msg)
            log(msg)
            continue

        context[ext.name] = result
        if ext.name == "instrument":
            instrument_result = result  # type: ignore[assignment]
        elif ext.name == "meta":
            meta_result = result  # type: ignore[assignment]

    return instrument_result, meta_result


# Stdout truncation width for match-table strings (the log is untruncated).
_MATCH_TEXT_W = 48


def _format_match_table(doc_results: list[dict], truncate: Optional[int] = None) -> list[str]:
    """Render every matched (expected, actual) pair per document, worst similarity first.

    Differing pairs get a second line with the actual string; ``truncate=None`` for full text.
    """
    def fit(s: str) -> str:
        if truncate is None or len(s) <= truncate:
            return s
        return s[: truncate - 1] + "…"

    text_w = truncate if truncate is not None else 0
    header = [
        f"    {'Kind':<5}  {'Sim':<5}  {'Ed':>3}  Expected / Actual",
        f"    {'-' * 5}  {'-' * 5}  {'-' * 3}  {'-' * max(text_w, 17)}",
    ]

    lines: list[str] = []
    for r in doc_results:
        records = r.get("matches") or []
        if not records:
            continue
        ordered = sorted(records, key=lambda m: (m.kind != "scale", m.similarity))
        lines.append(f"  {os.path.basename(r['path'])}")
        lines.extend(header)
        for m in ordered:
            lines.append(
                f"    {m.kind:<5}  {m.similarity:.3f}  {m.edits:>3}  {fit(m.expected)}"
            )
            if m.expected != m.actual:
                lines.append(f"    {'':<5}  {'':<5}  {'':>3}  {fit(m.actual)}")
        lines.append("")
    if lines:
        lines.pop()
    return lines


def run_validity(
    pdf_paths: list[str],
    all_extractors: list[Extractor],
    config: dict,
    test_cases: list[dict],
    no_logs: bool,
    run_timestamp: str,
) -> dict:
    """Validity mode: compare each PDF's composed ``extraction_output`` against its fixture.

    PDFs without an ``extraction_output:`` fixture block are SKIPPED. Returns the
    :func:`run_extractor` shape.
    """
    from .validity import validate_extraction  # lazy import

    thresholds: dict = (
        config.get("testing", {}).get("validity", {}).get("thresholds", {})
    )

    log_basename = f"test-validity-{run_timestamp}.log"
    out = Tee()
    out.log("=== Validity Testing Log ===")
    out.log(f"Date/Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.log(f"Model     : {config['model']}")
    out.log(f"Files     : {len(pdf_paths)}")
    out.log(f"Thresholds: {thresholds}")
    out.log("")

    print("validity testing")
    print(f"model     : {config['model']}")
    print(f"files     : {len(pdf_paths)}")
    print(f"thresholds: {thresholds}\n")

    cases_by_path = {c["path"]: c for c in test_cases}
    t_total_start = time.perf_counter()
    doc_results: list[dict] = []

    for file_idx, pdf_path in enumerate(pdf_paths):
        _print_and_log_doc_header(file_idx, len(pdf_paths), pdf_path, out.log)

        case = cases_by_path.get(pdf_path, {})
        expected = (case.get("fixtures") or {}).get("extraction_output")
        if expected is None:
            out.write(os.path.basename(pdf_path))
            out.write("  [SKIPPED] no extraction_output fixture (run --seed-validity-fixtures first)")
            doc_results.append({"path": pdf_path, "status": "SKIPPED", "issues": [], "metrics": {}})
            continue

        try:
            instrument, meta = _run_full_pipeline_in_memory(
                pdf_path, all_extractors, config, out.log
            )
        except Exception as exc:
            out.write(os.path.basename(pdf_path))
            out.write(f"  [FAIL] pipeline error: {exc}")
            doc_results.append({"path": pdf_path, "status": "FAIL", "issues": [str(exc)], "metrics": {}})
            continue

        composed = compose_extraction_output(instrument, meta)
        if composed is None:
            out.write(os.path.basename(pdf_path))
            out.write("  [FAIL] instrument extractor produced no output — nothing to validate")
            doc_results.append({
                "path": pdf_path,
                "status": "FAIL",
                "issues": ["no instrument output"],
                "metrics": {},
            })
            continue

        report = validate_extraction(expected, composed, thresholds)
        out.write(os.path.basename(pdf_path))
        status_line = (
            f"  [{report.status}]  issues={int(report.metrics.get('n_issues', 0))}  "
            f"missing={int(report.metrics.get('n_missing', 0))}  "
            f"extra={int(report.metrics.get('n_extra', 0))}"
        )
        out.write(status_line)
        if report.issues:
            out.log(f"  Issues ({len(report.issues)}):")
            for issue in report.issues:
                out.log(issue.format())
            print(f"  {len(report.issues)} issue(s):")
            preview = report.issues[:10]
            for issue in preview:
                print(issue.format())
            if len(report.issues) > len(preview):
                print(f"  ... and {len(report.issues) - len(preview)} more (see log)")
        out.log("")

        doc_results.append({
            "path": pdf_path,
            "status": report.status,
            "issues": [i.format() for i in report.issues],
            "metrics": report.metrics,
            "matches": report.matches,
        })

    if any(r.get("matches") for r in doc_results):
        n_matched = sum(len(r.get("matches") or []) for r in doc_results)
        head = [
            "",
            "=" * 60,
            f"STRING MATCHES  ({n_matched} matched pair(s))",
            "=" * 60,
            "  Sim = normalised Levenshtein similarity (1.000 = identical)",
            "  Ed  = raw Levenshtein edit distance (0 = identical)",
            "-" * 60,
        ]
        out.writelines(head)
        for line in _format_match_table(doc_results, truncate=_MATCH_TEXT_W):
            print(line, flush=True)
        out.loglines(_format_match_table(doc_results))
        out.write("=" * 60)

    n_pass = sum(1 for r in doc_results if r["status"] == "PASS")
    n_fail = sum(1 for r in doc_results if r["status"] == "FAIL")
    n_skip = sum(1 for r in doc_results if r["status"] == "SKIPPED")

    STATUS_ICON = {"PASS": "PASS", "FAIL": "FAIL", "SKIPPED": "SKIP"}
    names = [os.path.basename(r["path"]) for r in doc_results]
    col_w = max((len(n) for n in names), default=20)
    table_lines = [
        f"  {'Document':<{col_w}}  Status  Issues",
        f"  {'-' * col_w}  ------  ------",
    ]
    for r, name in zip(doc_results, names):
        icon = STATUS_ICON.get(r["status"], r["status"])
        n_issues = int(r["metrics"].get("n_issues", len(r["issues"])))
        issues_str = str(n_issues) if r["status"] != "SKIPPED" else "-"
        table_lines.append(f"  {name:<{col_w}}  {icon:<6}  {issues_str}")

    out.writelines([
        "",
        "=" * 60,
        f"SUITE VALIDITY SUMMARY  ({len(doc_results)} document(s))",
        "=" * 60,
        f"  PASS: {n_pass}   FAIL: {n_fail}   SKIPPED: {n_skip}",
        "-" * 60,
        *table_lines,
        "=" * 60,
        "",
    ])

    total_elapsed = time.perf_counter() - t_total_start
    out.log(f"Total duration : {total_elapsed:.2f}s")

    if not no_logs:
        log_path = write_run_log(
            out.dump(), log_basename, len(pdf_paths), config
        )
        print(f"[log written to {log_path}]")

    all_passed = all(r["status"] in ("PASS", "SKIPPED") for r in doc_results)
    return {"doc_results": doc_results, "all_passed": all_passed}


def run_seed_validity_fixtures(
    pdf_paths: list[str],
    all_extractors: list[Extractor],
    config: dict,
    fixtures_dir: str,
    force: bool,
    no_logs: bool = False,
) -> dict:
    """Seed ``<fixtures_dir>/<stem>.yaml`` with the composed ``extraction_output`` (untagged).

    Overwrites the whole file; skips PDFs that already have the block unless ``force``.
    """
    fixtures_root = Path(fixtures_dir)
    fixtures_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_basename = f"seed-validity-{timestamp}.log"
    out = Tee()
    out.log("=== Validity Fixture Seeding Log ===")
    out.log(f"Date/Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out.log(f"Force     : {force}")
    out.log(f"Files     : {len(pdf_paths)}")
    out.log("")

    print(f"seeding validity fixtures (force={force})")
    print(f"files: {len(pdf_paths)}\n")

    written: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for file_idx, pdf_path in enumerate(pdf_paths):
        _print_and_log_doc_header(file_idx, len(pdf_paths), pdf_path, out.log)

        stem = Path(pdf_path).stem
        fixture_path = fixtures_root / f"{stem}.yaml"

        if fixture_path.exists():
            # Validity loader so existing !contains/!exact tags parse.
            from .validity import ValiditySafeLoader
            with open(fixture_path) as f:
                existing = yaml.load(f, Loader=ValiditySafeLoader) or {}
            if "extraction_output" in existing and not force:
                out.write("  [skip] extraction_output already present (use --force to overwrite)")
                skipped.append(pdf_path)
                continue

        try:
            instrument, meta = _run_full_pipeline_in_memory(
                pdf_path, all_extractors, config, out.log
            )
        except Exception as exc:
            out.write(f"  [FAIL] pipeline error: {exc}")
            failed.append(pdf_path)
            continue

        composed = compose_extraction_output(instrument, meta)
        if composed is None:
            out.write("  [FAIL] instrument extractor produced no output — cannot seed")
            failed.append(pdf_path)
            continue

        payload = {"extraction_output": composed.model_dump(mode="json")}
        with open(fixture_path, "w") as f:
            yaml.safe_dump(
                payload,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
                width=10**6,
            )
        out.write(f"  [ok] wrote extraction_output → {fixture_path}")
        written.append(pdf_path)

    out.write("\n" + "=" * 60)
    out.write(
        f"SEEDING SUMMARY: wrote {len(written)}  "
        f"skipped {len(skipped)}  failed {len(failed)}"
    )
    out.write("=" * 60)

    if not no_logs:
        log_path = write_run_log(
            out.dump(), log_basename, len(pdf_paths), config
        )
        print(f"[log written to {log_path}]")

    return {"written": written, "skipped": skipped, "failed": failed}
