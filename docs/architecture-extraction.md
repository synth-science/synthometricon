# Architecture — extraction pipeline (`extraction/`)

All modules below live under `extraction/`. CLI: `docs/commands.md`. Testing semantics: `docs/testing.md`, `docs/reliability-metrics.md`. Exploded per-item columns: `docs/extraction-output.md`.

## Pipeline overview

```
PDF → pdf_io (page images) → items extractor → scales extractor
                                    ↓                  ↓
                              instrument extractor (maps items ↔ scales)
                                    ↓
                              meta extractor (document metadata)
                                    ↓
                              compose → Instrument tree → parquet
```

| Extractor | Output | Dependencies |
| --- | --- | --- |
| `items` | List of item stems, types, options | none |
| `scales` | Scale/subscale tree with constructs | none |
| `instrument` | Item-to-scale mappings + reverse-coding | `items`, `scales` |
| `meta` | Document metadata (language, DOI, permissions, …) | `items`, `scales` + page-1 text |

## Modules

**`config.py`** — `load_config` (normalises legacy `input_file`, expands `input_files` globs/directories to deduped absolute paths via `expand_input_files`; `abspath`, not `realpath`, so parquet keys stay stable), `load_test_cases`, `write_fixture`, `build_options`, `resolve_options`, `resolve_think`.

**`llm.py`** — llama-server client (see [LLM client](#llm-client)): `stream_and_collect`, `format_usage_breakdown`.

**`pdf_io.py`** — PyMuPDF wrappers: `pdf_to_images` (DPI/format from `config.images`), `pdf_first_page_text` (meta extractor's regex step), `pdf_text_excl_first_page` (page 1 of PsycTESTS PDFs is database boilerplate), `pdf_doc_stats`.

**`text_utils.py`** — `lev_dist` / `lev_edits` (lowercased normalised / raw Levenshtein), `fuzzy_contains` (best same-length window; backs `!contains`), `best_partial_distance` (case-sensitive, `partial_ratio_alignment`, fast enough for full documents), `extract_json`, `Tee`, and run-log placement (see [Run logs](#run-logs)).

**`orchestrator.py`** — run loops (see [Orchestrator](#orchestrator)).

**`storage.py`** — parquet store (see [Storage](#storage)).

**`validity.py`** — fixture loading and the validity comparator (see [Validity comparator](#validity-comparator)).

**`extractors/`** — task-specific logic behind an ABC (`Extractor`). Each extractor declares `system_prompt`, `user_prompt`, `result_model`, optional `depends_on: list[str]`, and:
- `signal(result)` — reduces a parsed result to the comparable unit (list of item texts, or list of Scale trees)
- `reliability(signals)` — per-document cross-run agreement; returns `(report_str, {metric: score})`
- `suite_reliability(all_signals)` — suite-level statistics (ICC for items, mean/SD for scales)
- `format_run(...)` — stdout/log output for a single run
- `build_user_prompt(context=None)` — defaults to the static `user_prompt`; dependent extractors override it to inject prior results from `context`.

Shared rendering helpers live in `extractors/_render.py`. `extractors/__init__.py` holds `REGISTRY`, the single source of truth; insertion order is pipeline order. The CLI auto-generates `--<name>` flags from it and the test suite parametrises over it.

**`models/`** — Pydantic result models (`Items`, `Survey`/`Scales`, `Scale`, `Instrument`, `Meta`) plus token-efficient wire models the LLM actually emits, and resolvers that convert wire → result:

| Wire model | Result model | Resolver |
| --- | --- | --- |
| `RawItems` | `Items` | `resolve_items` |
| `RawSurvey` | `Survey` / `Scales` | `resolve_scales` |
| `InstrumentMappings` | `Instrument` | `resolve_instrument` |
| `RawMeta` | `Meta` | `resolve_meta` |

Split by stage: `models/items.py`, `scales.py`, `meta.py`, `instrument.py`, plus `compose.py` (composers/derivers over `Instrument`) and `_schema.py` (`inline_schema`). Public API: `from extraction.models import X`. Key points:
- `RawItems`/`RawSurvey` use short JSON keys and drop auto-assigned `id`s; `InstrumentMappings` keeps full `item_id` / `scale_id` / `reverse_coded` keys — shorter aliases measurably hurt semantic anchoring.
- Rare-true booleans are `Optional[Literal[True]] = None` so the grammar can only emit `true` or omit.
- `inlined_schema()` resolves `$ref`/`$defs` because llama.cpp grammar-constrained decoding does not follow `$ref`.
- `Item.id` (sequential) and `Scale.id` (DFS) are assigned post-parse by `model_validator`s.
- `Meta` blends regex-extracted page-1 fields (via `context["_first_page_text"]`) with LLM fields (`language`, `intake_form` — intake/case-history forms without scorable scales, `objective_measure` — tests with objectively correct answers).
- `apply_meta_to_instrument` attaches `Meta` to an `Instrument` (the instrument extractor never sets it). `compose_extraction_output(instrument, meta)` returns `None` until an instrument tree exists; it populates the derived `extraction_output` column.

Wire-model token-efficiency conventions: `docs/wire-model-conventions.md`.

## LLM client

`stream_and_collect` sends a JSON-schema structured-output request and returns `(thinking, content, elapsed_s, usage)`.

- **Options split:** `temperature`/`top_p`/`seed` are standard params; `num_predict` → `max_tokens` (`-1` = unlimited); `top_k`/`min_p`/`repeat_penalty`/`repeat_last_n` go in `extra_body`; `num_ctx`/`image_max_tokens` are dropped (fixed at server startup).
- **Images before text** in the user message so the `[system + images]` prefix stays KV-cache-stable across the several extractor calls per document; otherwise the whole image segment (~1120 tokens/page) is re-prefilled each call.
- **Thinking:** Gemma's chat template reasons by default, so `chat_template_kwargs.enable_thinking` is always sent explicitly. Reasoning arrives on `delta.reasoning_content` or `delta.thinking` depending on server build. `thinking_chars` should stay 0 when thinking is off — non-zero flags reasoning silently turning back on.
- **Server timings:** llama-server appends a non-standard `timings` object to the final chunk (read from `chunk.model_extra`), giving `prefill_ms`, `predicted_ms`, throughput and `draft_acceptance`; `usage.prompt_tokens_details.cached_tokens` shows KV reuse. `prefill_ms ≈ ttft` means latency is prompt processing, not hidden generation.
- **Context budget:** system/user text tokens come from the server-root `/tokenize` endpoint (`None` → `n/a` on failure); images use a fixed per-image budget (`llama_cpp.server.image_min_tokens`). Template overhead is the signed residual against `prompt_eval_count`; slightly negative values are normal (BPE merges across the system↔user boundary).

## Orchestrator

`run_once()` is the shared per-call primitive (retries parse failures up to `max_retries`). Entry points:
- `run_extractor` — reliability mode: N runs per PDF, scored against `testing.reliability.thresholds`. A thresholded metric missing from the scores (or NaN) fails, so a stale threshold key is loud. Errored runs are re-attempted up to `testing.run_retries` (default 2) so a transient failure does not sink the document.
- `run_live` — live mode, see [below](#live-mode-resume-policy).
- `run_status` — read-only done/failed/pending tally per extractor plus an error-string histogram (backs `--status`).
- `run_validity` — full in-memory pipeline per PDF compared against the `extraction_output:` fixture; prints a STRING MATCHES table (stdout truncated, log full).
- `run_seed_validity_fixtures` — writes the composed, untagged `extraction_output:` block, overwriting the whole fixture file; skips existing blocks unless `--force`.

### Dependency context

`build_dependency_context` supplies `{dep: model}` for dependent extractors in reliability mode:
1. Read `case["fixtures"]["extraction_output"]` (may carry validity tags).
2. If missing: run the full pipeline once in memory and compose. Without `--no-fixtures` the result is written to `tests/fixtures/<stem>.yaml`; with it, the result is cached in `case` only (reused by later dependent extractors in the same invocation).
3. `strip_match_specs` → `normalize_for_model_validation` → `Instrument.model_validate` → `derive_items` / `derive_survey`.

Auto-built fixtures are unreviewed: `--validity` against them passes tautologically until an expert corrects the YAML.

### Live-mode resume policy

`run_live` iterates `pdf_paths × selected extractors` (REGISTRY order). The worklist is `input_files`, or with `--retry` the documents already in the parquet. Per cell (`storage.cell_state`, decided by `_retry_decision`):
- **done** — skipped unless `--force`.
- **pending** — always run.
- **failed** — skipped by default (a deterministically failing document is not reprocessed every run); re-attempted only under `--retry-failed` or when `--retry-error` (regex) matches the stored error, and only while `{name}_extractor_attempts < --max-attempts` (else **exhausted**).

Cached cells seed the context so selected extractors can satisfy dependencies. A PDF render failure is recorded against every scheduled extractor. On failure the content and usage cells are nulled and the error string saved; the parquet is rewritten atomically after every document and `extraction_output` is recomposed.

## Storage

The document parquet (`data.extractions`) has one row per PDF keyed by absolute path. Per extractor `{ext}`:
- `{ext}_extractor_content` — `result.model_dump_json()` (null if not yet successful)
- `{ext}_extractor_{suffix}` — usage/timing columns, one per `storage.USAGE_SUFFIXES` (incl. `timestamp`, `error`, token counts, `ttft`, server timings, `thinking_chars`)
- `{ext}_extractor_attempts` — from `RETRY_SUFFIXES`; kept separate from usage because usage cells are nulled on failure while the attempt counter must survive.

Plus per-document `has_errors` / `total_duration` and the derived `extraction_output` column (instrument + meta composed into one `Instrument` tree, recomputed after every document). Helpers: `load_or_init_df`, `upsert_row`, `atomic_write_parquet` (temp file + `os.replace`), `get_cell`, `load_cell`, `read_paths`, `cell_state`, `get_attempts`.

## Run logs

Runs over fewer than `logs.scratch_below_files` PDFs (default 10; `0` disables) are smoke tests: logs go to `logs/scratch/`, pruned to the newest `logs.scratch_keep` (default 50). Larger runs log to `logs/`. Live mode always writes a permanent top-level log (`allow_scratch=False`).

## Validity comparator

Authoring rules and tags are in `docs/testing.md`. Internals:
- `ValiditySafeLoader` subclasses `yaml.SafeLoader` (not mutating it) so tags stay scoped to validity loads. `!extras=N` encodes N in the tag itself because YAML can't attach a scalar argument to a tagged sequence.
- `validate_extraction` receives the raw fixture dict; Pydantic-validating it would inject defaults and erase the absent-vs-present distinction.
- Named lists (`scales`, `subscales`, `items`, `unscaled_items`, `orphan_items`) are Hungarian-matched on `scale_name` / `item_text`. An omitted key scores 1.0 (don't care — lets `items: !anywhere` work inside an untagged host scale). Stem-less items key on joined options (`_OptionsKey`), which never matches real item text (0.0).
- Strict below-threshold pairs are reported as "matched but failing" (points at the closest entry); loose (`!anywhere`) below-threshold pairs are reported as missing. Loose matches that land on a local entry consume it so it isn't flagged as extra. Only `MatchRecord`s with a genuine Levenshtein score are kept.
- `normalize_for_model_validation` (dep-context path) fills gaps in curated fixtures: scale ids DFS-sequential from 1; item ids sequential over unique dedup keys (`item_text`, else joined options), shared across scales; `has_image`/`reverse_coded` default `False`; `meta` dropped (its fields reference un-synthesised IDs). Existing IDs are preserved.

## Adding a new extractor

1. Create `extraction/extractors/<name>.py` implementing `Extractor`.
2. Add `"<name>": MyExtractor()` to `REGISTRY` in `extraction/extractors/__init__.py`.
3. The `--<name>` flag and test parametrisation appear automatically.
4. Set `needs_first_page_text = True` (or `needs_doc_stats = True`) to have the orchestrator inject `context["_first_page_text"]` (or `context["_doc_stats"]`).
5. Update `docs/extraction-output.md` if the exploded columns change.

Reserved context keys: orchestrator-injected raw inputs use leading-underscore keys and never collide with extractor names. Never store extractor results under a `_`-prefixed key.
