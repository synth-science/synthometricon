# Testing: reliability, validity, live mode, and the pytest suite

## pytest suite

```bash
poetry run pytest tests/ -q -k "not reliability"     # offline suite
poetry run pytest tests/ -v -k "items"                # reliability, one extractor
poetry run pytest tests/ -v -k "items-999915498_full_001"   # one extractor x one PDF
```

- `test_reliability.py` / `test_validity.py` drive the real extractors and need a running llama-server and the PDFs listed in `tests/test_files.yaml`; parametrised as `<extractor>-<pdf-stem>` / `<pdf-stem>` over that file. They read fixtures from `testing.fixtures_dir` in `config.yaml` (`config.resolve_fixtures_dir`), exactly like the CLI.
- `test_harness.py` covers the fixtures-directory resolution, input discovery (OS duplicate copies) and the items suite ICC offline.
- Everything else is hermetic: tiny synthetic frames, `tmp_path` parquets, stubbed encoders (`test_encode`), a monkeypatched `CrossrefClient._get` (`test_patch`), and a stub `HfApi` (`test_publish`). No network, no model loads.
- `test_comparator.py` covers the validity comparator and fixture tags offline.
- `test_publish.py` guards two invariants throughout: the release key never reaches a report line, the manifest or the dataset card; and nothing is uploaded to a non-private repo or a remote that is already current. `test_card_documents_every_released_column` keeps the hand-written card in sync with `RELEASE_COLS`.

## Reliability testing

Runs each extractor `testing.num_runs` times per enabled PDF in `tests/test_files.yaml` and checks cross-run agreement against `testing.reliability.thresholds` in `config.yaml`. Metric definitions and math: [reliability-metrics.md](reliability-metrics.md).

```bash
poetry run python -m extraction --reliability
poetry run python -m extraction --items --reliability
poetry run python -m extraction --instrument --reliability --no-fixtures   # re-run upstream deps live
```

| Extractor | Threshold key | Measures |
| --- | --- | --- |
| items | `count_agreement` | run-pairs with identical item count |
| items | `levenshtein_alpha` | Krippendorff's α on item texts (Levenshtein) |
| scales | `tree_overlap` | mean pairwise tree-structure similarity |
| instrument | `assignment_alpha` | Krippendorff's α (nominal) on per-item scale-id sets |
| instrument | `reverse_agreement` | (item, scale) pairs with consistent reverse-coding |
| meta | `language_agreement` / `intake_form_agreement` / `objective_measure_agreement` | run-pairs with identical value |

- `num_runs < 2` → document SKIPPED. Otherwise one `[PASS]`/`[FAIL]` row per threshold key; any failing row fails the document:
  ```
  018_101037t31974000.pdf
  [FAIL]  count_agreement=0.6000 < 1.0
  [FAIL]  levenshtein_alpha=0.4506 < 0.9
  ```
- A threshold key with no computed score reports `missing` and fails — remove stale keys from `config.yaml` rather than adding a scorer.
- A run that errors (timeout, server error, per-call `max_retries` exhausted) is retried up to `testing.run_retries` times (`[RETRY]`) before the document fails, mirroring live mode's retry of failed cells.
- Suite summary: ICC(2,1) on item counts (needs ≥2 documents) and tree-overlap stats.

### Fixtures and dependency context

Per-document fixtures live in `<testing.fixtures_dir>/<pdf-stem>.yaml`, i.e. `tests/validation-fixtures/` (the human coding, see Supplementary Note 4), each with one top-level `extraction_output:` block (a YAML tree mirroring the composed `Instrument`). The same block is validity ground truth and the dependency context for reliability runs of extractors with `depends_on`. `tests/test_files.yaml` carries only `path` (+ optional `enabled`, `comment`); fixtures are matched by stem and loaded with `ValiditySafeLoader`, so tag leaves survive as `MatchSpec`s.

`orchestrator.build_dependency_context` turns a fixture into extractor inputs: strip `MatchSpec`s → `validity.normalize_for_model_validation` (synthesise missing IDs; curated fixtures author by `item_text` / `scale_name`, not IDs) → `Instrument.model_validate` → `derive_items` / `derive_survey`. The derivers must preserve IDs, so they bypass the sequential-ID validators on `Items` / `Survey`. A missing fixture is auto-built live once and written back for reuse (the session `test_cases` fixture is updated in place); `--no-fixtures` bypasses the cache. Both the CLI and pytest read `testing.fixtures_dir`; `load_test_cases` defaults to `tests/validation-fixtures`. `tests/fixtures/` is a legacy, machine-seeded cache (untagged `extraction_output` dumps of earlier runs); it is not ground truth and nothing reads it unless `testing.fixtures_dir` points there.

## Validity testing

```bash
poetry run python -m extraction --validity
poetry run python -m extraction --seed-validity-fixtures   # seed fixtures with a fresh extraction_output
```

Runs the full pipeline once per enabled PDF and compares the composed `extraction_output` against the fixture via `validity.validate_extraction`. PDFs without an `extraction_output:` block are SKIPPED. `--seed-validity-fixtures` writes into `testing.fixtures_dir`; it skips PDFs that already have the block, and `--force` would overwrite the human coding, so point `testing.fixtures_dir` elsewhere before seeding.

- **Omission = "don't care"; presence = "must hold".** Absent fixture fields are unverified; present ones are asserted.
- Lists are matched by Hungarian assignment (`scipy.optimize.linear_sum_assignment`) over Levenshtein similarity on the key field (`scale_name` / `item_text`); unmatched expected entries are "missing", unmatched actual entries "extra". Strings compare against `testing.validity.thresholds.string_similarity`.
- Expected entries without the key field (e.g. an untranscribed item recorded only by `reverse_coded`) are matched after the keyed ones, to actual entries left over (best fit on their recorded fields), so they cannot take a keyed entry's partner. With nothing left over they are "missing".
- `!anywhere` entries that outnumber the candidates in the global pool are reported "missing (anywhere)", like below-threshold ones.

| Tag | Effect |
| --- | --- |
| `!contains "..."` | case-insensitive substring match (still Levenshtein-tolerant) |
| `!exact "..."` | strict equality |
| `!anywhere` | match against the global candidate pool; no extras check |
| `!extras=N` | strict list tolerating up to N unmatched actual entries |

```yaml
extraction_output:
  scales:
    - scale_name: !contains "Depression"    # matches "Beck Depression Inventory-II"
      construct_name: !exact "depression"
      items: !extras=2
        - item_text: !contains "feel sad"
        - item_text: !contains "lost interest"
    - scale_name: "Anxiety"                 # default Levenshtein check
  orphan_items: !anywhere
    - item_text: !contains "demographics"
```

Every matched pair is kept as a `validity.MatchRecord` (`kind`, `path`, `expected`, `actual`, `similarity`, `edits`) on `ValidityReport.matches` and printed by `orchestrator._format_match_table` as a **STRING MATCHES** block before the suite summary: one row per item/scale, grouped by document, scales first, worst first. `Sim` is normalised similarity; `Ed` is raw edit distance (`text_utils.lev_edits`). stdout truncates to `_MATCH_TEXT_W`; the log has full strings. `!contains`/`!exact` keys and omitted keys are not real distances and are not recorded.

## Log retention

Routed by `text_utils.write_run_log`:

- **Suite runs** (≥ `logs.scratch_below_files` PDFs, default 10) → `logs/<name>.log`, kept.
- **Smoke runs** (fewer PDFs) → `logs/scratch/<name>.log`, pruned to the newest `logs.scratch_keep` (default 50) by `text_utils.prune_scratch_logs`.

Live mode is exempt (`allow_scratch=False`) and always logs to `logs/`. `logs.scratch_below_files: 0` disables routing; `--no-logs` skips writing. Nothing under `logs/` is committed — move a smoke log out of `logs/scratch/` to keep it.

## Live mode

`input_files` in `config.yaml` accepts file paths, directories (recursive `*.pdf`) and globs; `config.expand_input_files` dedupes to absolute paths. Results go to `data.extractions` — one row per document, one JSON-string column per extractor.

Resume/retry is per `(document, extractor)` cell (`storage.cell_state`): **done** (content set) is skipped unless `--force`; **pending** runs; **failed** (null content + error string) is skipped by default. Re-attempt with `--retry-failed` or `--retry-error PATTERN`; `--max-attempts N` caps via `{name}_extractor_attempts`. `--retry` takes the worklist from the parquet (`storage.read_paths`) instead of `input_files`. `--status` prints the done/failed/pending tally and an error histogram.

After each document the orchestrator rebuilds the derived `extraction_output` column from `instrument_content` + `meta_content` (`compose_extraction_output`); it is null until `instrument` has run. Per-extractor cells stay authoritative. Load with `storage.load_cell(df, pdf_path, "extraction_output", Instrument)`.

Extractor dependencies are declared via `Extractor.depends_on`; prior results reach `build_user_prompt(context)` from this run or the parquet. `InstrumentExtractor` and `MetaExtractor` depend on `["items", "scales"]`; `MetaExtractor.needs_first_page_text = True` makes the orchestrator inject page-1 text (`pdf_first_page_text`) as `context["_first_page_text"]` for its regex step.
