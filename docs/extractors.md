# Extractors (`extraction/extractors/`)

Per-extractor behaviour and design notes. Module map and "adding an extractor": `docs/architecture-extraction.md`. Wire-model patterns: `docs/wire-model-conventions.md`.

Prompt strings and Pydantic class docstrings / `Field` descriptions are functional (they reach the LLM) — edit them only deliberately.

## items

Emits `RawItems`; `resolve_items` fills defaults (`type`→`rating_scale`, `lang`→`en`) and `Items` assigns 1-based ids. Reliability: pair-exact count agreement plus Krippendorff's α with a squared-Levenshtein metric over the first `min(counts)` items of each run (truncation avoids missing values).

## scales

Emits `RawSurvey`; `resolve_scales` drops `subscale_evidence` and `Survey` assigns DFS ids. Reliability: greedy tree similarity — per node `0.5·name_sim + 0.5·child_sim`, children matched by descending score, normalised by `max(n, m)`.

## instrument

Depends on `items` + `scales`. `build_user_prompt` renders a numbered item list and an `[id=N]` scale tree (plus an unscaled/orphan hint when `has_unscaled_items`), so the LLM only emits `{item_id, scale_id, reverse_coded?}` triples (`InstrumentMappings`). `parse_response` resolves them via `resolve_instrument` into the nested `Instrument` — that is what is stored and shown.

Resolver semantics:
- An item mapped to several scales is duplicated under each, with per-occurrence `reverse_coded`.
- Mappings to unknown item/scale ids are silently dropped.
- Unmapped items go to `orphan_items` if listed in `orphan_item_ids` (psychometric, no scale — e.g. stand-alone single items, draft/pilot item pools), otherwise `unscaled_items` (demographic/administrative; not scored).
- `Instrument.meta` is never set here; see `compose_extraction_output` / `apply_meta_to_instrument` in `extraction/models/compose.py`.

Reliability: Krippendorff's α (nominal) on each item's sorted scale-id set, reverse-coding agreement per (item, scale) pair, and a per-run tree-comparison grid (`✓` assigned, `R` reverse-coded, `-` absent, `◄` disagreement).

## meta

Two paths merged by `resolve_meta` into one `Meta` cell:
1. **Page-1 regex** (no LLM) over the APA PsycTests citation page, injected as `context["_first_page_text"]` (`needs_first_page_text = True`): title (first line, skipping a leading `Note:`), PsycTESTS DOI, instrument type, year + authors (from the citation block between the citation header and `Instrument Type:`), raw test-format/source/permissions blocks, journal venue (from `Source:`, else `Original Publication:` when Source is "Supplied by author."), and `language_raw` from an "`<Language> Version`" title qualifier (only known language names, so "Revised Version" is ignored). Every field is best-effort and `None` on no match.
2. **LLM** (`RawMeta`): ISO 639-1 `language`, plus rare-true `intake_form` / `objective_measure`. The prompt includes the items and scale tree for context.

Document stats (`page_count`, `image_count`, …) are added via `context["_doc_stats"]` (`needs_doc_stats = True`). Reliability: pair-exact agreement on the three LLM fields.
