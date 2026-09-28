# Command reference

CLI reference for all three pipelines. Extraction assumes a running llama-server (`./start_server.sh`).

## Extraction pipeline (`extraction/`)

```bash
# Live extraction (reads input_files from config.yaml; persists to data/extractions.parquet)
poetry run python -m extraction                  # full pipeline (all extractors)
poetry run python -m extraction --items          # only items
poetry run python -m extraction --scales         # only scales
poetry run python -m extraction --items --force  # re-extract even if cell already populated
poetry run python -m extraction --items --think --no-logs --config custom.yaml

# Resume / retry (failed cells are skipped by default; --retry takes the worklist from the parquet instead of input_files)
poetry run python -m extraction --status                                  # done/failed/pending tally + error histogram
poetry run python -m extraction --retry --retry-failed                    # re-attempt failed docs already in the parquet
poetry run python -m extraction --retry --retry-error TimeoutError        # re-attempt only failures matching a regex
poetry run python -m extraction --retry --retry-failed --max-attempts 3   # cap retries (poison-doc guard)

# Reliability / validity testing (reads tests/test_files.yaml; exits 0=all pass, 1=any fail)
poetry run python -m extraction --reliability                       # all extractors
poetry run python -m extraction --items --reliability               # filter to items extractor
poetry run python -m extraction --instrument --reliability --no-fixtures   # E2E: re-run upstream deps live
poetry run python -m extraction --validity                          # compare against curated fixtures
poetry run python -m extraction --seed-validity-fixtures            # seed fixtures from a fresh run

# Run reliability tests directly via pytest
poetry run pytest tests/ -v
poetry run pytest tests/ -v -k "items"
poetry run pytest tests/ -v -k "items-999915498_full_001"  # single file by stem
```

`--<extractor-name>` flags (`--items`, `--scales`, `--instrument`, `--meta`, ...) are auto-generated from the `REGISTRY` in `extraction/extractors/__init__.py` — see [architecture-extraction.md](architecture-extraction.md).

## Dataset assembly pipeline (`assemble/`)

Extraction store under `data.extractions:`, per-stage paths under `data.assemble:` in config.yaml — see [configuration.md](configuration.md).

```bash
poetry run python -m assemble                            # run all stages (explode, combine, patch, postprocess, encode, pool)
poetry run python -m assemble --step combine             # run a single stage
poetry run python -m assemble --explode                  # alias for --step explode (raw store -> item-level parquet)
poetry run python -m assemble --step patch --refresh     # patch, ignoring the Crossref/doi_probe caches
poetry run python -m assemble --step postprocess         # prune rows/columns, derive public_doi/public_source/public_year, pdf_match distances, flag_* warnings
poetry run python -m assemble --step encode              # add sentence-transformer embedding columns (encode: config block)
poetry run python -m assemble --step pool                # aggregate into per-scale/per-instrument pooled rows
poetry run python -m assemble --report-only              # dry run: report only (DOI lookups run cache-only)
poetry run python -m assemble --no-logs                  # skip writing logs/assemble-<timestamp>.log
```

Real (non-`--report-only`) runs write the combined stage reports to `logs/assemble-<timestamp>.log` unless `--no-logs` is given. Real patch/postprocess runs also write a `<artifact>.stats.json` sidecar (structured per-step counters, see `assemble/stats.py`) next to their output parquet — the `report` stage's source for per-step figures.

The patch, postprocess, encode, and pool stages are also standalone re-runnable on an arbitrary parquet:

```bash
poetry run python -m assemble.patch --report-only                       # dry run on data.assemble.combined
poetry run python -m assemble.patch --steps schema,language             # subset of steps (always registry order)
poetry run python -m assemble.patch --input in.parquet --output out.parquet --refresh

poetry run python -m assemble.postprocess --report-only                 # dry run on data.assemble.patched
poetry run python -m assemble.postprocess --steps derive,flags          # subset of steps (always registry order)
poetry run python -m assemble.postprocess --input in.parquet --output out.parquet

poetry run python -m assemble.encode --report-only                      # counts only; no model loads, no encoding
poetry run python -m assemble.encode --input in.parquet --output out.parquet

poetry run python -m assemble.pool --report-only                        # group counts only, no pooling
poetry run python -m assemble.pool --input in.parquet --output out.parquet

poetry run python -m assemble.report                                    # preview descriptives (prints only; the release copy is written by publish)
poetry run python -m assemble.report --output /tmp/descriptives.md      # also write the preview somewhere
```

**`--steps` writes a partial corpus.** A subset run still writes the full output
parquet, so pointing it at the configured stage output (the default) replaces a
fully patched/postprocessed corpus with one where the skipped steps never ran —
e.g. dropping `doi_psyctests`, which makes postprocess skip `public_doi` /
`public_source` / `public_year` and makes `pool` abort on the missing document columns. The
stage warns when it happens; pass `--output` to write a subset run elsewhere,
and re-run the stage in full before continuing downstream.

See [architecture-assemble.md](architecture-assemble.md) for what each stage does.

## Publication pipeline (`publish/`)

Artifact path under `data.publish:`, settings under `publish:` in config.yaml — see [configuration.md](configuration.md). Run separately from `python -m assemble`, on purpose: a Hub commit is effectively permanent.

```bash
poetry run python -m publish                             # both steps: publish, then upload
poetry run python -m publish --step publish              # build the release dir: artifact, manifest, key if new, <stem>.md, README.md
poetry run python -m publish --step upload               # push blob + dataset card + manifest (prompts first)
poetry run python -m publish --step upload --yes         # skip the interactive confirmation (scripted runs)
poetry run python -m publish --report-only               # dry run: no writes, NO network calls, never generates a key
poetry run python -m publish --report-only --check-remote  # dry run + live repo state (visibility, already-current?)
poetry run python -m publish --force                     # upload despite a matching remote or a dirty build tree
poetry run python -m publish --skip-verify               # skip re-hashing the artifact against its manifest
poetry run python -m publish --key-path my.key           # override <data.publish stem>.key ($SYNTH_NET_RELEASE_KEY still wins)
poetry run python -m publish --no-logs                   # skip writing logs/publish-<timestamp>.log
```

Both steps are standalone re-runnable:

```bash
poetry run python -m publish.publish --report-only                      # counts + key status; no key generation, no write
poetry run python -m publish.publish --input in.parquet --output out.parquet --key-path my.key

poetry run python -m publish.hub --report-only --check-remote           # plan + live repo state, uploads nothing
poetry run python -m publish.hub --yes --force
```

`--yes`, `--check-remote`, `--force` and `--skip-verify` only apply to the upload step and error out on a publish-only run.

Reading the encrypted artifact back in-repo:

```python
from publish.publish import read_release
df = read_release()                      # decrypts data.publish
df = read_release(columns=["path"])      # project at the parquet read
```

See [architecture-publish.md](architecture-publish.md) for what each step does and the HuggingFace gotchas.
