# Architecture — dataset assembly (`assemble/`)

Ordered stages turn the extraction store plus external partials (ALIGNS, Scale-Hunt, SemanticNet) into the public corpus: **explode → combine → patch → postprocess → encode → pool**. Each stage writes its own path under `data.assemble.*`:

| artifact | contents |
|---|---|
| `exploded` | extraction store, one row per `(item, scale)` occurrence |
| `combined` | raw concatenation with the partials (sources keep their conventions) |
| `patched` | repaired + backfilled corpus, no rows dropped |
| `postprocessed` | pruned public-facing dataset with `public_*`, `pdf_match_*`, `flag_*` columns |
| `embedded` | + sentence-transformer embedding columns |
| `pooled` | per-scale / per-instrument centroid rows — the final output |

Encrypting and releasing `pooled` is the separate `publish/` pipeline ([architecture-publish.md](architecture-publish.md)).

**`__main__.py`** — CLI and stage runner over the ordered `STAGES` registry. Default runs all stages; `--step <name>` runs one (`--explode` is a back-compat alias). Real runs write the combined reports to `logs/assemble-<timestamp>.log` (`--no-logs` skips; `--report-only` never writes). Flags: [commands.md](commands.md).

## explode

`assemble/explode.py` flattens the document-keyed `data.extractions` parquet into one row per `(item, scale)` occurrence (a multi-scale item appears once per scale, with that scale's `reverse_coded`), plus one row per orphan/unscaled item. Nothing is hard-coded: carried scalars are every column except `*_extractor_content` and `extraction_output`; `meta_*` / `item_*` columns come from `Meta` / `ScaledItem` fields (`Item.id` is renamed to `item_id`, missing `reverse_coded` → null); scale columns (`scale_id`, `scale_name`, `scale_construct_name`, root→leaf `scale_id_path` / `scale_name_path`, `scale_depth` with 1 = top level) are null outside the `scaled` bucket. A document with null `extraction_output` yields one carry-only **shell row** (null `bucket`) so its telemetry survives; `.dropna(subset=["item_item_id"])` gives an items-only view.

`explode_extractions_to_parquet` streams row-group batches (nested content columns never read) and appends to a `ParquetWriter`, so memory is bounded by one batch; each batch is reindexed to `_output_columns` for a stable schema, and all-null first-batch columns are widened to string. Column reference: [extraction-output.md](extraction-output.md) — keep it in sync with explode changes.

## combine

`assemble/combine.py` reads `exploded` + the raw `data.partials`, tags each with `corpus_source` (filename minus `-extraction(s)-exploded`), outer-join concatenates, and writes atomically with a canonical Arrow schema (large types → regular, int32 → int64 incl. list elements — load-bearing for the partials' int32 counters).

## patch

Value-level repair and backfill; drops rows only for OS duplicate copies of a PDF (`duplicate_files`), DOI backfill is fill-only-null; sets `is_patched`; carries apa PDF body text in `pdf_full_text` for postprocess. Details: [assemble-patch.md](assemble-patch.md).

## postprocess

First stage allowed to drop rows: filters to publishable single rating-scale items (incl. `ITEM_TEXT_CHARS_MAX` = 250), drops telemetry/internal columns, derives `item_text_chars`, `public_doi`, `public_source`, `public_year`, measures item-vs-PDF fidelity (`pdf_match_*`, then drops the copyrighted `pdf_full_text`), and adds nullable-boolean `flag_*` warnings where NA means "not checkable". Count flags judge the pre-filter `observed_*` counts. Details: [assemble-postprocess.md](assemble-postprocess.md).

## encode, pool

Row-preserving embedding columns (encode, which also writes a `<embedded>.scale-names.parquet` sidecar with the name embeddings of parent-only scale nodes), then per-scale/per-instrument pooling with keyed centroids (pool; run encode first so the sidecar exists). See [assemble-encode-pool.md](assemble-encode-pool.md) and [reverse-keyed-pooling.md](reverse-keyed-pooling.md).

## Supporting modules

- **`stats.py`** — `write_stats` dumps a stage's `Ctx.stats` + report lines to `<artifact>.stats.json` (real runs of patch/postprocess only, overwritten each run; history is in `logs/`); `read_stats` returns None on missing/corrupt; `stats_stale` flags a sidecar older than its parquet; `mirror_report` copies a sidecar or report into `data.reports_dir`.
- **`report.py`** — paper-facing descriptives report; not a stage, written by publish beside the release. See [assemble-report.md](assemble-report.md).
- **`search.py`** — import-only helpers (`search_items`, `search_scales`) for sanity-checking embedding search against the pooled corpus; not a stage. `search_items` ranks by absolute cosine by default, as the paper describes (SurveyBot3000 cosines approximate signed correlations, so strongly negatively related scales are hits too); the `similarity` column keeps the sign, and `absolute=False` ranks by signed cosine. `search_scales` (label search) ranks by signed cosine. Scripts outside the repo root need `PYTHONPATH=<repo>` (the repo is not installed as a package).
