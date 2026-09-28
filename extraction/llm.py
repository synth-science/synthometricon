"""llama-server (OpenAI-compatible) client and per-call context-budget reporting.

See docs/architecture-extraction.md#llm-client.
"""

from __future__ import annotations

import base64
import time
from typing import Optional

import httpx
import openai


_DIRECT_OPTIONS = frozenset({"temperature", "top_p", "seed"})
# llama.cpp-specific extensions, sent via ``extra_body``.
_EXTRA_BODY_OPTIONS = frozenset({"top_k", "min_p", "repeat_penalty", "repeat_last_n"})


def _split_options(options: dict) -> tuple[dict, dict]:
    """Split options into ``(direct_kwargs, extra_body)``; ``num_predict`` → ``max_tokens``.

    Server-level keys (``num_ctx``, ``image_max_tokens``) are dropped: fixed at server startup.
    """
    direct: dict = {}
    extra: dict = {}
    for k, v in options.items():
        if k in _DIRECT_OPTIONS:
            direct[k] = v
        elif k == "num_predict":
            direct["max_tokens"] = None if v == -1 else v
        elif k in _EXTRA_BODY_OPTIONS:
            extra[k] = v
    return direct, extra


def _tokenize_count(text: str, base_url: str, timeout: float = 5.0) -> Optional[int]:
    """Count tokens via the server-root ``/tokenize`` endpoint; ``None`` on any failure."""
    if not text:
        return 0
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    try:
        resp = httpx.post(f"{root}/tokenize", json={"content": text}, timeout=timeout)
        resp.raise_for_status()
        tokens = resp.json().get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
    except Exception:
        return None
    return None


def _extract_server_timings(chunk) -> dict:
    """Pull llama-server's non-standard ``timings`` and ``cached_tokens`` off the final chunk."""
    out: dict = {}
    extra = getattr(chunk, "model_extra", None) or {}
    timings = extra.get("timings") if isinstance(extra, dict) else None
    if isinstance(timings, dict):
        out["prefill_ms"] = timings.get("prompt_ms")
        out["predicted_ms"] = timings.get("predicted_ms")
        out["prompt_per_second"] = timings.get("prompt_per_second")
        out["predicted_per_second"] = timings.get("predicted_per_second")
        draft_n = timings.get("draft_n")
        draft_acc = timings.get("draft_n_accepted")
        out["draft_acceptance"] = (
            draft_acc / draft_n if isinstance(draft_n, int) and draft_n > 0
            and isinstance(draft_acc, int) else None
        )
    usage = getattr(chunk, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if cached is not None:
        out["cached_tokens"] = cached
    return out


def format_usage_breakdown(usage: dict) -> str:
    """Render the context-window breakdown table for a :func:`stream_and_collect` usage dict."""
    ctx = usage.get("ctx_size")
    n_imgs = usage.get("num_images") or 0
    img_per = usage.get("image_min_tokens")
    sys_t = usage.get("system_tokens")
    usr_t = usage.get("user_text_tokens")
    img_t = usage.get("image_tokens")
    ovhd_t = usage.get("chat_template_overhead")
    prompt_t = usage.get("prompt_eval_count")
    out_t = usage.get("eval_count")
    total_t = (
        (prompt_t or 0) + (out_t or 0)
        if (prompt_t is not None and out_t is not None) else None
    )

    def fmt_tokens(n: Optional[int]) -> str:
        return f"{n:>9,}" if isinstance(n, int) else "      n/a"

    def fmt_pct(n: Optional[int]) -> str:
        if not isinstance(n, int) or not isinstance(ctx, int) or ctx <= 0:
            return "    n/a"
        return f"{100.0 * n / ctx:6.2f}%"

    ctx_str = f"{ctx:,}" if isinstance(ctx, int) else "unknown"
    img_label = (
        f"images ({n_imgs} × {img_per})"
        if isinstance(img_per, int) else f"images ({n_imgs})"
    )

    rows: list[tuple[str, Optional[int]]] = [
        ("system prompt", sys_t),
        ("user prompt (text)", usr_t),
        (img_label, img_t),
        ("template / residual", ovhd_t),
        ("prompt subtotal", prompt_t),
        ("output", out_t),
    ]
    label_w = max(len(r[0]) for r in rows + [("TOTAL", None)])
    sep = "  ├" + "─" * (label_w + 2) + "┼" + "─" * 11 + "┼" + "─" * 9 + "┤"
    top = "  ┌" + "─" * (label_w + 2) + "┬" + "─" * 11 + "┬" + "─" * 9 + "┐"
    bot = "  └" + "─" * (label_w + 2) + "┴" + "─" * 11 + "┴" + "─" * 9 + "┘"

    def row(label: str, n: Optional[int]) -> str:
        return f"  │ {label:<{label_w}} │ {fmt_tokens(n)} │ {fmt_pct(n)} │"

    lines = [f"  Context budget: {ctx_str} tokens", top]
    for label, n in rows:
        lines.append(row(label, n))
        if label == "output":
            lines.append(sep)
    lines.append(row("TOTAL", total_t))
    lines.append(bot)
    return "\n".join(lines)


def _build_user_content(images: list[bytes], image_format: str, user_prompt: str) -> list[dict]:
    """Build the user message with images before text to keep the KV-cache prefix stable."""
    media_type = "image/png" if image_format.lower() == "png" else "image/jpeg"
    content: list[dict] = []
    for img_bytes in images:
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        })
    content.append({"type": "text", "text": user_prompt})
    return content


def _enrich_usage_with_context_budget(
    usage: dict,
    *,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    n_images: int,
    ctx_size: int | None,
    image_min_tokens: int | None,
) -> None:
    """Add context-window breakdown fields to ``usage`` in place (template overhead may be < 0)."""
    img_total = n_images * image_min_tokens if image_min_tokens is not None else None
    sys_tokens = _tokenize_count(system_prompt, base_url)
    usr_tokens = _tokenize_count(user_prompt, base_url)
    prompt_t = usage.get("prompt_eval_count")
    overhead: Optional[int]
    if (
        isinstance(prompt_t, int)
        and isinstance(img_total, int)
        and isinstance(sys_tokens, int)
        and isinstance(usr_tokens, int)
    ):
        overhead = prompt_t - img_total - sys_tokens - usr_tokens
    else:
        overhead = None
    usage["ctx_size"] = ctx_size
    usage["num_images"] = n_images
    usage["image_min_tokens"] = image_min_tokens
    usage["image_tokens"] = img_total
    usage["system_tokens"] = sys_tokens
    usage["user_text_tokens"] = usr_tokens
    usage["chat_template_overhead"] = overhead


def stream_and_collect(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: list[bytes],
    image_format: str,
    format_schema: dict,
    options: dict,
    think: bool,
    verbose: bool,
    timeout: float | None = None,
    ctx_size: int | None = None,
    image_min_tokens: int | None = None,
) -> tuple[str, str, float, dict]:
    """Stream a structured-output request; return ``(thinking, content, elapsed_s, usage)``."""
    client = openai.OpenAI(base_url=base_url, api_key="dummy")
    user_content = _build_user_content(images, image_format, user_prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    direct_kwargs, extra_body = _split_options(options)
    # Gemma's chat template reasons by default; set enable_thinking explicitly so think=False sticks.
    extra_body["chat_template_kwargs"] = {"enable_thinking": bool(think)}

    create_kwargs: dict = dict(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "extraction_output", "schema": format_schema},
        },
        stream=True,
        stream_options={"include_usage": True},
        **direct_kwargs,
    )
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    accumulated_thinking = ""
    accumulated_content = ""
    shown_thinking_header = False
    shown_answer_header = False
    usage: dict = {}

    t_first_token: float | None = None
    t_start = time.perf_counter()

    for chunk in client.chat.completions.create(**create_kwargs):
        elapsed_so_far = time.perf_counter() - t_start
        if timeout is not None and elapsed_so_far > timeout:
            raise TimeoutError(
                f"LLM generation exceeded {timeout:.0f}s timeout "
                f"after {elapsed_so_far:.1f}s"
            )

        if chunk.usage is not None:
            usage = {
                "total_duration": int(elapsed_so_far * 1e9),
                "prompt_eval_count": chunk.usage.prompt_tokens,
                "eval_count": chunk.usage.completion_tokens,
                "ttft": t_first_token,
                **_extract_server_timings(chunk),
            }

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        # Reasoning arrives on reasoning_content or thinking depending on server build.
        thinking_delta: str = (
            getattr(delta, "thinking", None)
            or getattr(delta, "reasoning_content", None)
            or ""
        )
        content_delta: str = delta.content or ""

        if (thinking_delta or content_delta) and t_first_token is None:
            t_first_token = elapsed_so_far
            if verbose:
                print(f"[time to first token: {t_first_token:.2f}s]\n", flush=True)

        if thinking_delta:
            if verbose:
                if not shown_thinking_header:
                    print("=== thinking ===\n", flush=True)
                    shown_thinking_header = True
                print(thinking_delta, end="", flush=True)
            accumulated_thinking += thinking_delta

        if content_delta:
            if verbose:
                if not shown_answer_header:
                    if shown_thinking_header:
                        print("\n\n=== answer ===\n", flush=True)
                    shown_answer_header = True
                print(content_delta, end="", flush=True)
            accumulated_content += content_delta

    elapsed = time.perf_counter() - t_start
    if not usage:
        usage = {
            "total_duration": int(elapsed * 1e9),
            "prompt_eval_count": None,
            "eval_count": None,
            "ttft": t_first_token,
        }
    else:
        usage["total_duration"] = int(elapsed * 1e9)
        usage["ttft"] = t_first_token

    # Hidden-reasoning guard: should stay 0 when thinking is off.
    usage["thinking_chars"] = len(accumulated_thinking)

    _enrich_usage_with_context_budget(
        usage,
        base_url=base_url,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        n_images=len(images),
        ctx_size=ctx_size,
        image_min_tokens=image_min_tokens,
    )

    if verbose:
        print(f"\n\n[total generation time: {elapsed:.2f}s]\n", flush=True)
    else:
        print(f"[generation time: {elapsed:.2f}s]", flush=True)

    return accumulated_thinking, accumulated_content, elapsed, usage
