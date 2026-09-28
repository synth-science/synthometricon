# Extraction output — exploded parquet columns

The live pipeline writes one document-keyed parquet (`data.extractions`): one row per PDF, one JSON column per extractor. `assemble.explode.explode_extractions*` flattens it into a per-item parquet at the grain **one row per `(item, scale)` occurrence**:

- a multi-scale item appears once per scale (each with that scale's membership and its own `reverse_coded`);
- `orphan_items` / `unscaled_items` contribute one row each (null scale columns);
- a document with null `extraction_output` contributes one carry-only row (null `item_*` / `scale_*` / `meta_*`) so usage scalars aren't lost. Use `.dropna(subset=["item_item_id"])` for an items-only view.

The column set is derived, not hand-listed (`assemble/explode.py::_output_columns`): carried scalars by naming rule, `meta_*` from `Meta.model_fields`, `item_*` from `ScaledItem.model_fields`, plus the fixed `bucket` / scale-membership columns and the per-extractor usage suffixes (`storage.USAGE_SUFFIXES` + `RETRY_SUFFIXES`). **Keep these tables in sync** whenever `Meta`, `Item`, `ScaledItem`, `Instrument`, the extractor `REGISTRY`, or the explode/usage logic changes.

## Document identity & summary (carried per row)

| Column | Type | Meaning |
| --- | --- | --- |
| `path` | str | Absolute path of the source PDF (the document key). |
| `has_errors` | bool | True if any extractor failed for this document. |
| `total_duration` | float | Document wall-clock seconds (sum of per-extractor durations). |

## Per-extractor usage / diagnostics (carried per row)

One block per extractor `{ext}` in `items`, `scales`, `instrument`, `meta`, as
`{ext}_extractor_{suffix}`. Columns null when that extractor didn't run/failed
(`attempts` always present).

| Suffix | Type | Meaning |
| --- | --- | --- |
| `timestamp` | str | ISO-8601 time the extractor ran. |
| `error` | str | Error string on failure; null on success. |
| `total_duration` | float | Call wall-clock seconds. |
| `prompt_eval_count` | int | Prompt tokens evaluated (prefill). |
| `eval_count` | int | Tokens generated. |
| `image_tokens` | int | Tokens for the rendered page image(s). |
| `system_tokens` | int | System-prompt tokens. |
| `user_text_tokens` | int | User text-prompt tokens. |
| `ttft` | float | Time to first token (s). |
| `prefill_ms` | float | Prompt-processing wall-time (ms). |
| `predicted_ms` | float | Generation wall-time (ms). |
| `prompt_per_second` | float | Prefill throughput (tok/s). |
| `predicted_per_second` | float | Generation throughput (tok/s). |
| `cached_tokens` | int | Prompt tokens served from the slot KV cache. |
| `draft_acceptance` | float | Speculative-decoding accept rate; null when off. |
| `thinking_chars` | int | Reasoning chars streamed (0 when thinking off). |
| `attempts` | int | Times this `(document, extractor)` cell was attempted. |

## Explode bucket & scale membership

| Column | Type | Meaning |
| --- | --- | --- |
| `bucket` | str | `scaled` / `orphan` / `unscaled` (null for a null-output document). |
| `scale_id` | int | ID of the item's immediate scale (null for orphan/unscaled). |
| `scale_name` | str | Name of that scale. |
| `scale_construct_name` | str | Construct the scale measures. |
| `scale_id_path` | list[int] | Root-to-leaf scale IDs (handles nested subscales). |
| `scale_name_path` | list[str] | Root-to-leaf scale names. |
| `scale_depth` | int | Depth of the immediate scale (1 = top-level). |

## Item fields (`item_*`, from `ScaledItem`)

| Column | Type | Meaning |
| --- | --- | --- |
| `item_item_id` | int | Item's sequential ID (`ScaledItem.item_id`; note the doubled prefix). |
| `item_item_text` | str | Verbatim item stem; null for stem-less items (labels carried in `item_options`). |
| `item_has_image` | bool | Item presented with an image. |
| `item_item_type` | str | `rating_scale` / `choice` / `open` / `other`. |
| `item_options` | list[str] | Per-item response/anchor labels in scoring order; null when using scale-shared anchors. |
| `item_admin_note` | str | Note to the administrator (not the respondent); null when absent. |
| `item_language` | str | ISO 639-1 of the item's original language (`en` unless translated). |
| `item_reverse_coded` | bool | Reverse-scored under this scale. Null from extraction for orphan/unscaled items; the patch `reverse_coded` step backfills nulls to `False`. |

## Document metadata (`meta_*`, from `Meta`; repeated per row)

| Column | Type | Meaning |
| --- | --- | --- |
| `meta_language` | str | ISO 639-1 of the document's body language (LLM-emitted). |
| `meta_intake_form` | bool | Medical-history / patient-intake form with no scorable scales. |
| `meta_objective_measure` | bool | Ability/cognitive/knowledge test (objectively correct answers). |
| `meta_title_raw` | str | Title (page-1 regex). |
| `meta_doi_raw` | str | DOI (e.g. `10.1037/t08009-000`). |
| `meta_source_doi` | str | DOI of the original publication from the page-1 "Source:" block (any registrant); distinct from the PsycTESTS record DOI. |
| `meta_instrument_type_raw` | str | Verbatim "Instrument Type:" line. |
| `meta_publication_year_raw` | int | Publication year. |
| `meta_authors_raw` | str | Raw authors string from the PsycTESTS citation. |
| `meta_journal_venue_raw` | str | Raw journal/source citation. |
| `meta_test_format_raw` | str | Verbatim test-format / scoring note. |
| `meta_source_raw` | str | Full source citation block. |
| `meta_permissions_raw` | str | Permissions / reproduction notice. |
| `meta_language_raw` | str | Informational language-version note from the title; never overrides `meta_language`. |
| `meta_page_count` | int | Total PDF pages. |
| `meta_image_count` | int | Total image XObjects across all pages. |
| `meta_char_count_excl_first_page` | int | Character count of the document text excluding page 1. |
