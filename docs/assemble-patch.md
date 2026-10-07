# Patch stage (`assemble/patch.py`)

Value-level repair and backfill of `data.assemble.combined` → `data.assemble.patched`. **Drops rows only in `duplicate_files`** (OS duplicate copies of a source file); every other step keeps all rows. Steps live in the ordered `STEPS` registry (insertion order = execution order; `--steps` selects a subset but never reorders). Each step scopes itself to apa-only / partials-only / all rows with the NA-safe `apa_mask` / `partial_mask` on `corpus_source` ("partial" = every non-apa source); `STEP_SCOPE` only labels the report.

## Steps

| # | step | scope | what it does |
|---|---|---|---|
| 1 | `duplicate_files` | all | drops documents whose path is an OS duplicate copy (`name (2).pdf`) of another document in the corpus — the copy is a second extraction of the same PDF (`999900001_full_001 (2).pdf`: 40 rows in the Oct 2026 corpus). A copy whose original is absent is kept. `_run_steps` diffs `is_patched` on the surviving rows |
| 2 | `schema` | all | all-null float DOI columns → object, int-like floats → `Int64`, blank/whitespace strings → null |
| 3 | `text_repair` | all | sentinel strings (`None`, `N/A`, `-`, …) → null, also element-wise in `scale_name_path`; semanticnet mojibake (`?`→`-`, pilcrows) in `meta_source_raw` and `source_url` together; apa source-stub casing (`Supplied by author.`); placeholder texts → null; HTML unescape of `retrieval_notes` |
| 4 | `meta_dois` | apa | re-read PDF page 1; PDF-observed PsycTESTS and source DOIs are authoritative for `meta_doi_raw` / `meta_source_doi` |
| 5 | `pdf_full_text` | apa | PDF body text (every page except boilerplate page 1, via `extraction.pdf_io.pdf_text_excl_first_page`) → `pdf_full_text`. Uncached; excluded from the `is_patched` diff; consumed then dropped by postprocess's `pdf_match` |
| 6 | `doi_psyctests` | all | scalar PsycTESTS DOI. apa: 3-way corroboration of filename stem (`9999NNNNN_…` → `10.1037/tNNNNN-000`), `data.meta` `TestFileList` mapping, and PDF-verified `meta_doi_raw`; any disagreement leaves null. Partials: stem-derived fill. Drops the legacy `doi` column |
| 7 | `source_fields` | partials | CSV-corruption fix, classify `meta_source_raw` as citation/url/domain/junk and migrate URLs into `source_url` (only when `source_url` is null or a byte-copy of the raw value) |
| 8 | `source_doi` | all | regex-extract DOIs already present in citation text or `source_url` |
| 9 | `crossref` | all | intra-document propagation of a unanimous DOI, then Crossref lookups for documents with no DOI at all (citation mode or structured mode); accepted works also backfill authors/year/venue on partial rows |
| 10 | `doi_probe` | partials | LLM proposes the original publication for docs still missing a DOI; Crossref verifies it under row-anchored acceptance |
| 11 | `language` | all | null partial `meta_language_raw` copy-fills, normalize to ISO 639-1, `<Language> Version` title scan (partials), per-document langdetect fill-only-null (skipped when detection disagrees with existing values), partial null `meta_language` → `en` |
| 12 | `permissions` | partials+all | normalize partial strings into the APA vocabulary; recompute `permissions_category` for all rows |
| 13 | `fabricated` | partials | null scraper constant-fills (`_FABRICATED_CONSTANTS`) — guarded: skipped if the column no longer holds only the constant |
| 14 | `item_type` | partials | fill-only-null `item_item_type = rating_scale` for aligns/semanticnet (rating-scale databases whose scrapers never record a type) |
| 15 | `items` | all/partials | empty option lists → null; strip leading numbering from partial item text; 0-based partial ids → 1-based; renumber partial documents whose scraper restarted item numbering per subscale (one id carrying different items), so every distinct item gets its own id — an item is (original id, text cluster; case/punctuation-insensitive, rapidfuzz ratio ≥ `_SAME_ITEM_RATIO` = 90), ids reassigned 1..K by first appearance, so a multi-scale item keeps one id. Then a guard over all rows: any (document, `item_item_id`) still carrying more than one item logs a WARNING and `items.id_text_conflicts` (apa ids are never renumbered, only reported) |
| 16 | `reverse_coded` | all | fill-only-null `item_reverse_coded = False` (the `ScaledItem` default) on every row with an `item_item_id`, any bucket. Keyed on id so image-only items count; itemless shell rows keep their null |
| 17 | `authors` | partials | pipe-delimited author lists → APA |
| 18 | `version_split` | all | see below |
| 19 | `title_case` | all | APA title case for `scale_name`, `scale_construct_name`, `meta_title_raw`, `scale_name_path` elements; word-level, never downcases acronyms or existing capitals; minor words stay lowercase except first, last, or after a colon |
| 20 | `anomalies` | report-only | census of residual oddities; mutates nothing |

## DOI backfill

- **Fill-only-null**: `source_doi`, `crossref`, `doi_probe` never overwrite an existing `meta_source_doi`.
- **DOI regex** (`_ANY_DOI_RE`): printable ASCII except quotes and `<>`, so legacy DOIs with `&` or `[...]` survive while text-layer debris such as `©` ends the DOI. Before matching, U+2010/U+2011/U+2012/U+2212 become `-` and soft hyphens are removed (`_DOI_HYPHENS`); `_clean_doi` strips trailing `.,;:`, unbalanced closing brackets and landing-page tails (`/full`, `/abstract`, `/pdf`, …).
- **Crossref acceptance.** Citation mode (a classified citation in `meta_source_raw`): parsed title fuzzy-matches the work title ≥ `title_threshold`, year within `year_tolerance`, first-author family matches. Structured mode (title + first author + year): instrument name contained in the work title ≥ `instrument_threshold`, same year/author checks. PsycTESTS record DOIs are never accepted as source DOIs.
- **doi_probe acceptance** is anchored on the row's own author/year where present (the LLM guess only fills gaps), and the title check passes on either the instrument name (`doi_probe.instrument_threshold`) or the guessed title (`title_threshold`). A DOI the LLM supplies is looked up first, then a search. The prompt forbids constructing DOIs.
- **Caches**: Crossref verdicts keyed by lookup, probe verdicts by file stem; JSON at `patch.crossref.cache` / `patch.doi_probe.cache`, checkpointed periodically. `--refresh` ignores them. `--report-only` or a missing `crossref.mailto` makes both steps cache-only; `doi_probe` is also cache-only when no llama-server answers. The stage never hard-fails on missing services.

## version_split

Relocates trailing version/form qualifiers (`--Spanish Version`, `--Short Form`, `, Form B`, `(Revised)`) from `scale_name` / `meta_title_raw` / `scale_name_path` elements into `version`.

- Two grammars. After an explicit separator (2+ dashes, em dash, spaced hyphen) or in a trailing parenthetical, a *loose* tail is accepted when its last word is a version word (`version`, `form`, `edition`, `revised`, …) and it contains no instrument noun (`scale`, `inventory`, … — that would be a subscale name). After a weak separator (comma, unspaced hyphen/en dash) only the *tight* grammar applies (`short form`, `form B`, `2nd edition`, …). A single en dash is not an explicit separator because it occurs inside names (`DSM–IV`).
- Repeated qualifiers are peeled right-to-left and joined with `"; "`.
- Fill-only-null: if `version` already holds text that doesn't contain the qualifier, the row keeps its name intact.
- `scale_name_path` is rebuilt from the node names afterwards (`_sync_name_path`), so it never diverges from `scale_name` (pool releases node names from the path): an element whose node has rows of its own takes that node's final `scale_name` (split, kept, or nulled as a sentinel); an element of an umbrella node (no own rows) loses its qualifier only if every row under it records the qualifier in `version`, otherwise it stays intact. Without `scale_id_path` every element is split as before.

## is_patched

Boolean column from a value-level before/after diff of the whole run (dtype-only casts don't count; new columns count where non-null). `pdf_full_text` is excluded, and `doi_psyctests` diffs against the legacy `doi` column. Prior `True` flags are OR-ed in on re-patching.

## Stats sidecar

Every logged count is also recorded via `ctx.stat("<step>.<metric>", value)`; real runs write `Ctx.stats` plus the report lines to `<patched>.stats.json` (`assemble.stats.write_stats`) for the report. Keep `ctx.stat` calls adjacent to their `ctx.log` lines.

## Standalone runs

`python -m assemble.patch --input X --output Y --steps a,b`. Order caveats: `doi_psyctests` without `meta_dois` corroborates whatever `meta_doi_raw` is already stored; `doi_probe` after a skipped `crossref` probes a larger pool. A step subset without `--output` overwrites the patched artifact with a partial patch (the report warns).
