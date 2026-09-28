# Extraction pipeline

Turns psychometric-test PDFs into structured `Instrument` trees with a vision LM behind a grammar-constrained `llama-server`. Each PDF is rendered to page images and run through four extractors in order — **items**, **scales**, **instrument** (item ↔ scale mapping), **meta** — whose outputs are composed into one tree and stored as a parquet row in `data.extractions`.

## Usage

Needs a running `llama-server` (`./start_server.sh`, configured via `config.yaml` → `llama_cpp.server`).

```bash
poetry run python -m extraction                          # all extractors over input_files
poetry run python -m extraction --items                  # single extractor
poetry run python -m extraction --status                 # done/failed/pending tally
poetry run python -m extraction --retry --retry-failed   # re-attempt failed cells
poetry run python -m extraction --reliability            # cross-run agreement tests
poetry run python -m extraction --validity               # compare against curated fixtures
```

Done cells are skipped on re-run and failed cells are skipped unless `--retry-failed` / `--retry-error` is given.

## Docs

- [Commands](../docs/commands.md) — every flag
- [Architecture](../docs/architecture-extraction.md) — modules, extractors, models, adding an extractor
- [Wire-model conventions](../docs/wire-model-conventions.md)
- [Output columns](../docs/extraction-output.md) — exploded per-item parquet
- [Configuration](../docs/configuration.md)
- [Testing](../docs/testing.md) — reliability/validity modes, fixtures, resume semantics
