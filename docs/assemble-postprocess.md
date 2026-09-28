# Postprocess stage (`assemble/postprocess.py`)

Prunes `data.assemble.patched` into the public-facing `data.assemble.postprocessed`. The **first stage permitted to drop rows** (patch's no-drop invariant is per-stage). Same registry shape as patch (`STEPS`, `--steps` subset in registry order, standalone `python -m assemble.postprocess --input X --output Y`), but no caches, network, or `is_patched` diff. Real runs write `<postprocessed>.stats.json` like patch.

Extension points: a row filter is one `FILTERS` entry (NA-safe drop mask); a column drop one `EXTRA_DROP_COLS` entry; a flag one `FLAGS` entry (plus a `META_REFERENCE_FIELDS` entry and/or an `_observations` column if it needs new inputs); a mutation one `STEPS` entry.

## filter_rows

First attaches the pre-filter counts (see [flags](#flags)), then drops the union of `FILTERS` and reports per filter **isolated** (flagged on its own), **new** (not flagged by an earlier filter) and **cumulative** counts.

| filter | drops |
|---|---|
| `extraction_error` | `has_errors` rows |
| `null_bucket` | explode's per-document shell rows for failed extractions (overlaps `extraction_error`; logged separately so drift shows) |
| `non_rating_scale` | known `item_item_type` other than `rating_scale`. **Null type is kept** — a missing field, not evidence; patch's `item_type` backfills known cases |
| `empty_item_text` | null/blank `item_item_text` |
| `long_item_text` | trimmed stem longer than `ITEM_TEXT_CHARS_MAX` (250): merged item blocks, instruction paragraphs, vignettes |
| `has_image` | `item_has_image` true |
| `sample_version` | `is_sample_version` true (null kept) |
| `unscaled_bucket` | `bucket == "unscaled"` — only `scaled` and `orphan` are public |
| `intake_form` | apa rows: `meta_intake_form` not explicitly False (the Meta extractor always ran, so null is not a clearance). Partials never had the field: only explicit True drops |
| `objective_measure` | `meta_objective_measure` true — ability/knowledge tests with objectively scorable responses (null kept) |

Duplicates are deliberately kept for now.

## drop_columns

Drops per-extractor telemetry (`_extractor_{suffix}` for `extraction.storage.USAGE_SUFFIXES` + `RETRY_SUFFIXES`, so new suffixes propagate), doc-level telemetry (`has_errors`, `total_duration`), and `EXTRA_DROP_COLS` (superseded `meta_doi_raw`, PDF-shape metadata, partial-only scraper QA fields). `saved_files`, `is_patched` and `version` (target of patch's `version_split`) are kept.

## derive

- `item_text_chars` — trimmed stem length, the same function `long_item_text` judged, so every published value is ≤ 250. Computed before the DOI guard so a degraded run still has it.
- `public_doi` — apa: `doi_psyctests`; partials: `meta_source_doi`, falling back to `doi_psyctests`. Never null.
- `public_source` — apa: APA citation of the PsycTESTS record (`Authors (Year). Title [Database record]. APA PsycTests. https://doi.org/…`, degrading on null fields). Partials: the `meta_source_raw` citation (with `meta_source_doi` appended when the text has no DOI), else a citation built from the source-publication fields (needs a title plus authors or a DOI), else the record citation. Never null. On apa rows `meta_authors_raw`/year describe the instrument; on partials crossref made them describe the source publication — each citation form uses them accordingly.
- `public_year` (`Int64`) — the first parenthesised four-digit group of `public_source`, i.e. the APA year slot all forms share (never a `21(2)` issue or `(85)` DOI suffix); null for `(n.d.)`. Read off the citation so year and citation can never disagree.

Without `doi_psyctests` the three `public_*` columns are absent (logged WARNING).

## pdf_match

apa rows only. Fuzzy-locates each `item_item_text` in `pdf_full_text` (rapidfuzz `partial_ratio_alignment` via `extraction.text_utils.best_partial_distance`; raw text, no normalization) and adds:

- `pdf_match_edit_distance` (`Int64`) — Levenshtein distance to the best-matching window.
- `pdf_match_edit_distance_norm` (`Float64`, [0, 1]; 0 = verbatim).
- `pdf_text_chars` — body-text length each distance was measured against, kept so the text flag stays auditable.

NA where item or PDF text is missing (all partials). Then **drops `pdf_full_text`**: copyrighted APA text must not ship, and it can't go in `EXTRA_DROP_COLS` because `drop_columns` runs first. Runs after `filter_rows` so matching only touches published rows.

`PDF_TEXT_CHARS_MIN = 200`: below it the PDF has no usable text layer (a scan read by the vision model), so every item "deviates". Calibration: of the documents it excludes, 99.5 % have no verifiably matching item, and it costs only 15 matching rows corpus-wide; 300 would discard 587 matching rows, 400 would discard 3,550.

## flags

Adds `flag_*` nullable booleans — **True = warning, False = checked and consistent, NA = not checkable** — plus the `record_*` reference values they compare against.

- **References**: `load_meta_reference` reads the PsycTESTS records at `data.meta`; `META_REFERENCE_FIELDS` maps `number_of_test_items_best_guess` → `record_item_count`, `number_of_factors_subscales` → `record_scale_count` (non-integer values → null). Joined on lower-cased `doi_psyctests`, so partials with stem-derived DOIs are checked too. `record_*` not `meta_*`: these are the database's claims, not what the Meta extractor read.
- **Observed counts** are per document (`path`), over distinct ids (rows repeat per `(item, leaf scale)`). They are **pre-filter**: `filter_rows` attaches `observed_item_count` / `observed_scale_count` before dropping anything, and they stay in the output, so a row-level filter never reads as an extraction deviation and the flag stays auditable. Counting the frame at hand is only the fallback for a standalone `flags` run. Null-path rows get NA rather than a merged pseudo-document count.

| flag | True when | NA when |
|---|---|---|
| `flag_item_count_deviation` | distinct `item_item_id` ≠ `record_item_count` | no reference |
| `flag_scale_count_deviation` | distinct `scale_id` ≠ `record_scale_count` (field present on ~14 % of records) | no reference, or reference 0 ("no structure reported") |
| `flag_item_text_deviation` | `pdf_match_edit_distance_norm` > `PDF_MATCH_NORM_MAX` (0.05, i.e. < 95 % similarity) | nothing measured, or `pdf_text_chars` < `PDF_TEXT_CHARS_MIN` |
| `flag_item_translated` | `item_language` is an ISO 639-1 code other than `en` (the published English text is a translation) | no language known |

The last two use no record. Gotcha: after a parquet round-trip the norm column is plain float64, where `gt(NaN)` is False, so the flag masks NA explicitly. With no `data.meta` or no `doi_psyctests` the record-based flags degrade to all-NA with a logged WARNING.
