# Configuration

`config.yaml` is the single runtime config (override with `--config`); every key has a default and an empty file is valid. Config files carry **no comments** — document keys here instead.


CLI overrides: `--config`, `--think` (thinking on for all extractors), `--base-url`, `--output-path`.

## Top level

| Key | Default | Description |
| --- | --- | --- |
| `input_files` | `[]` | PDF paths, directories (recursive `*.pdf`) or globs; normalised to deduplicated absolute paths. Legacy `input_file` accepted. |
| `model` | — | Model name sent to the OpenAI-compatible API. |
| `seed` | — | Documentation only; not consumed. The sampling seed is `options.seed`. |
| `verbose` | `false` | Stream tokens and print timing / context-budget breakdowns. |
| `max_retries` | `0` | Retries per LLM call on parse failure (ValidationError / ValueError), not network errors. Attempts = `max_retries + 1`. |
| `timeout` | — | Seconds before a streaming generation is killed (guards against repetition loops). Raise it before retrying `TimeoutError` failures, or they repeat identically. |

## `think`

Bool, or a dict: `default` is the fallback, any key matching an extractor name overrides it (e.g. `think.instrument: false`).

## `options`

LLM sampling parameters. Top-level keys are the base; a sub-dict keyed by an extractor name overrides keys for that extractor only (e.g. `options.items.temperature: 0.2`). Also used by the patch `doi_probe` via `options.doi_probe` / `think.doi_probe`.

| Key | API mapping | Notes |
| --- | --- | --- |
| `seed` | `seed` | `null` = unseeded |
| `temperature`, `top_p` | same | |
| `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n` | `extra_body.*` | llama.cpp extensions |
| `num_predict` | `max_tokens` | `-1` = unlimited |

## `images`

`dpi` (default 150) and `format` (`png`/`jpeg`, default `jpeg`; also sets the MIME type sent to the API) for PyMuPDF page rendering.

## `llama_cpp`

`base_url` must equal `http://{server.host}:{server.port}/v1` (overridable via `--base-url`).

`llama_cpp.server` keys are read by `start_server.sh`; `ctx_size` and `image_min_tokens` are also used by the pipeline for context-budget reporting.

| Key | `llama-server` flag | Notes |
| --- | --- | --- |
| `hf_repo` | `-hf` | Used when `model_path` is unset. |
| `model_path` | `-m` | Local GGUF; takes precedence over `hf_repo`. |
| `mmproj_path` | `--mmproj` | Vision projector; required with `model_path` (local load doesn't auto-fetch it). |
| `spec_draft_path` / `spec_draft_hf_repo` | `--model-draft` / `--spec-draft-hf` | Draft model; local path wins. |
| `spec_type` | `--spec-type` | e.g. `draft-simple`. |
| `host`, `port` | `--host`, `--port` | `LLAMA_PORT` env overrides `port`. |
| `ctx_size`, `batch_size`, `ubatch_size` | same | |
| `image_min_tokens`, `image_max_tokens` | same | Per-image token budget. |
| `flash_attn`, `cache_type_k`, `cache_type_v` | same | `flash_attn` takes bool or `on`/`off`/`auto`. A quantized KV cache (`q8_0` ≈ half the memory) requires flash attention. |
| `n_parallel` | `--parallel` | Server slots (unset = llama.cpp default 4). |
| `cache_ram` | `--cache-ram` | Host-RAM prompt cache in MiB (0 disables; unset = 8192). |
| `reasoning` | — | Set to `"off"` in the cluster configs but not read by `start_server.sh`. |

**Keep `n_parallel: 1` and `cache_ram: 0` locally.** At the local `ctx_size`, four slots' context checkpoints plus multi-GB prompt-cache saves exhaust host RAM and the kernel OOM-kills llama-server mid-run (observed 2026-09-05; systemd's scope teardown then logs a misleading "shutdown requested").

## `data`

| Key | Description |
| --- | --- |
| `data.extractions` | Document-level extraction parquet written by live mode (`--output-path` overrides). |
| `data.meta` | PsycTESTS metadata records JSON (each with `DOI` and `TestFileList`); patch derives a `{pdf basename: DOI}` map from it for `doi_psyctests`. |
| `data.partials` | Raw exploded parquets from external sources (ALIGNS, Scale-Hunt, SemanticNet), read by combine. |
| `data.assemble.*` | Per-stage outputs in pipeline order: `exploded`, `combined`, `patched`, `postprocessed`, `embedded`, `pooled` (final assembly output). See [architecture-assemble.md](architecture-assemble.md). |
| `data.publish` | Encrypted release artifact (Fernet token over parquet bytes, not a readable parquet). Its directory also holds `<artifact>.manifest.json`, `<stem>.key`, `<stem>.md` and `README.md`, all derived from this path (`publish/paths.py`). |
| `data.reports_dir` | Git-tracked copy of the stats sidecars and the descriptives `<stem>.md`, refreshed on every real write (`assemble.stats.mirror_report`). Relative to the working directory; null disables. The public config sets null and `reports/` is never published. |

## `encode`

- `item_models` — list of `{name, path}`; each encodes `item_item_text` into `item_embedding_{name}` (`name` sanitized: lowercase, non-alphanumerics → `_`). Pool negates reverse-keyed items, which is only valid for a signed-correlation encoder like `surveybot3000` — see [reverse-keyed-pooling.md](reverse-keyed-pooling.md).
- `scale_models` — list of `{name, path}`; encodes `scale_name` → `scale_embedding_{name}` and `meta_title_raw` → `instrument_embedding_{name}`.
- `batch_size` — default 256.
- `pair` — optional PAIR model (Siamese item-pair correlation predictor over Qwen3-Embedding-8B vectors). Keys: `name`, `path` (Qwen model dir), `checkpoint` (`dnn_siamese_cor.pt`), `prompt` and `max_seq_length` (**must match training**; PAIR only applies its prompt when given the HF id, so a local path silently drops it), `sim_dims` (512), `batch_size` (32), `device` (default: CUDA device with most free memory — llama-server shares the GPUs). Writes `pair_h_{name}` + `pair_sim_{name}`.

The pool stage has no config block; models are auto-detected from embedding columns.

## `patch`

- `crossref` — `enabled`, `cache`, `mailto` (**required for live queries**; without it the step is cache-only), `title_threshold` (0.90), `year_tolerance` (1), `instrument_threshold` (0.85).
- `doi_probe` — `enabled` (false = cache-only, no llama-server needed), `cache`, `instrument_threshold` (0.60). Reuses `llama_cpp.base_url`, `model`, `timeout`, `max_retries`.

## `publish`

See [architecture-publish.md](architecture-publish.md).

| Key | Default | Description |
| --- | --- | --- |
| `key_env` | `SYNTHOMETRICON_RELEASE_KEY` | Env var holding the Fernet key, checked before `<data.publish stem>.key`. The only `os.environ` read in Python — not a pattern to copy. |
| `repo_id` | — | HuggingFace dataset repo. Nulled in the public config (with `path_in_repo`). |
| `path_in_repo` | — | Blob path in the repo; end it in `.bin`. |
| `revision` / `tag` | `main` / `null` | Branch and optional release tag. |
| `private` | `true` | Required visibility, verified before every upload. |
| `create_repo` | `true` | Whether upload may create the repo. |
| `token_env` | `HF_TOKEN` | Env var with the HF write token (falls back to cached login). Never put a token in config. |

## `logs`

Runs over fewer than `scratch_below_files` PDFs (default 10) are smoke tests: their log goes to `logs/scratch/`, pruned to the newest `scratch_keep` (default 50) after every write. Set `scratch_below_files: 0` to send every run to `logs/`. See [testing.md](testing.md).

## `testing`

| Key | Default | Description |
| --- | --- | --- |
| `num_runs` | `2` | Runs per PDF per extractor in reliability mode. |
| `run_retries` | `2` | Reliability mode: re-attempts of a whole errored run (timeout, server error, exhausted `max_retries`) before the document fails. |
| `fixtures_dir` | `tests/validation-fixtures` | Per-document `<stem>.yaml` fixtures (the human coding the validity evaluation compares against). |
| `reliability.thresholds.<extractor>.<metric>` | — | See [testing.md](testing.md) and [reliability-metrics.md](reliability-metrics.md). |
| `validity.thresholds.string_similarity` | — | Levenshtein similarity threshold for the validity comparator. |
