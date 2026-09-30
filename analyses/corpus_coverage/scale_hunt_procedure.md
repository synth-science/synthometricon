# The web scale hunt: procedure

Reconstructed on 2026-09-25 from the surviving workflow scripts, the workflow
output files (which record the model of every agent), the Claude Code session
transcript of 2026-08-05 to 08-08, and the project notes. Two workflow scripts
are archived verbatim in `scale_hunt_workflows/`:

- `scale_hunt_batch01-08.js`: the shared script for batches 01 to 08 (its
  surviving text is the batch08 state; see "What is not recoverable").
- `scale_hunt_batch09.js`: the final pass with per-scale hints.

## Aim and targets

SynthNet's own extractions covered 37.1 % of PsycInfo usage of PsycTests
instruments. The hunt targeted the 156 most-used PsycTests records without
scale-structured item data (`data/processed/scale_hunt_targets.csv`; 133
questionnaires, 23 other rating instruments; 120 non-commercial, 34
commercial, 2 unknown). The purpose stated to every agent was to retrieve
verbatim item text from public web sources in order to compute averaged item
embeddings for a search index; item text is stored locally and never
redistributed.

## Orchestration

The hunt was run as multi-agent workflows in Claude Code (`Workflow` tool),
orchestrated by Claude Fable 5 in the interactive session. Each batch script
defines two phases and runs one agent per target instrument, concurrently:

1. **Extract**: a search agent per instrument.
2. **Rescue**: a second, stronger or higher-effort agent, run only when the
   first returned nothing usable (`found = false`, zero items, or a crash).
   Rescue never re-ran partial successes.

Per-agent models, as recorded in the workflow outputs (`workflowProgress[].model`):

| Run | Date (local) | Targets | Extract | Rescue | Effort |
|---|---|---|---|---|---|
| Pilot | 2026-08-02 | 10 | Claude Sonnet 5 | Verify phase: Claude Haiku 4.5 | not recorded |
| batch01 (salvaged) | 2026-08-03 | 25 | Haiku 4.5 | Sonnet 5 | low / low |
| batch02 to batch05 | 2026-08-03 | 25 each | Haiku 4.5 | Sonnet 5 | low / low |
| batch06 (salvaged) + batch06b | 2026-08-03 | 21 / 16 | Haiku 4.5 | Sonnet 5 | low / low |
| batch07 | 2026-08-03 | 25 | Haiku 4.5 | Sonnet 5 | low / low |
| batch08 (searchless) | 2026-08-05 | 17 | Haiku 4.5 | Sonnet 5 | low / low |
| batch09 (hints) | 2026-08-05 | 10 | Sonnet 5 | Sonnet 5 | medium / high |

Model identifiers in the outputs: `claude-haiku-4-5-20251001`, `claude-sonnet-5`.
batch01 and batch06 were stopped mid-run (to cap effort, and because agents
over-used the paper-retrieval tools) and their completed results salvaged;
those two files carry no per-agent metadata.

## The search agent's instructions

The full prompt is `extractPrompt()` in the scripts. Its elements:

- **Target block** filled from the batch task list: name, acronym, PsycTests
  DOI and accession, alternate names, authors, original publication, official
  websites, instrument type, expected item count and subscales. (These slots
  were filled locally from PsycTests metadata; the committed task lists
  `data/processed/scale_hunt_batches/*.json` retain only name, acronym,
  alternate names, DOI, slug, websites and usage count.)
- **Allowed sources**: any public web source (official instrument sites,
  journal appendices, author-archived PDFs, university course materials,
  clinical organisations, measure repositories). No bypassing logins, no
  CAPTCHAs.
- **Discovery tools**, in order: WebSearch (batches 01 to 07, 09; up to 5
  searches per scale in batch09), DuckDuckGo HTML results via curl (batch08,
  when the WebSearch quota was exhausted, and as fallback in batch09),
  public bibliographic APIs (OpenAlex, Crossref, Semantic Scholar via the
  VPN-Paper-Search MCP server, up to 4 calls per scale, plus one open-access
  `get_paper` by DOI), then direct fetches of likely hosts. Forbidden: any
  tool that switches the machine's VPN or opens interactive browser windows
  (`fetch_paywalled`, `login_to`, `vpn_switch`, `download_witten_*`). After
  batch06 the instruction was at most one paper-retrieval call per scale and
  only after the open web had failed.
- **Effort budget**: about 12 tool calls and 4 source fetches per scale
  (batch09: 16 and 6); stop after two (batch09: three) consecutive sources
  fail; no cross-checking once one good source is archived.
- **Mandatory archiving**: every source actually used is saved with curl to
  `data/restricted/pdfs/<accession>_<slug>_<source>.{pdf,html}` and listed in
  `saved_files`, so a deterministic script can verify the extraction.
- **PDF reading**: convert with `pdf_inspector` to markdown and read that;
  fall back to reading the PDF pages as images for scans.
- **Extraction rules**: extract every item verbatim with original numbering
  from the archived source only; never reconstruct from memory or paraphrase;
  record the version or form; assign subscales where unambiguous; per item
  record type, reverse coding if stated, item-specific response options and
  administration notes. For licensed clinical instruments with long anchor
  definitions, return item stems plus one-sentence definitions and leave the
  full text in the archived file (bulk verbatim licensed text triggers the
  API content filter and loses the run). Interview schedules: extract the
  itemised probe list if one exists.
- **Structured output** (`ITEMS_SCHEMA`): `doi, name, version, found,
  best_source_url, other_source_urls, saved_files, source_grade (official |
  journal_appendix | university_pdf | clinical_org | other_web), language,
  response_scale, subscales[{name, construct}], items[{n, text, subscale,
  options, reverse, item_type, admin_note, has_image}], extracted_item_count,
  verbatim, confidence (high | medium | low), notes`.

The rescue agent received the same prompt prefixed with a note on what the
first attempt returned (found flag, item count, confidence, first 300 to 400
characters of its notes) and the instruction to do better, or, in batch09,
to try different sources and queries. batch09 additionally injected a
hand-written hint per target (exact citation of the original publication,
known hosting sites, version pitfalls, what counts as a useful partial).

## Verification

The pilot had a third phase in which Haiku 4.5 verifier agents re-fetched the
claimed source and spot-checked three items (first, middle, last). This was
dropped after the pilot: the verifiers added little and one confabulated
(it reported that a Perceived Stress Scale PDF contained the Hospital
Anxiety and Depression Scale). Verification moved to a deterministic script,
`scale_hunt_check.py`:

1. For each extraction, build a text corpus from its archived files (PDFs
   converted with `pdf_inspector`; HTML, markdown and text read raw).
2. Normalise both sides: strip HTML tags, lowercase, drop everything but
   letters, digits and spaces, collapse whitespace.
3. For every item, probe the item text, each response option and the first
   120 characters of the administration note (probes shorter than 13
   characters are skipped), with three progressively lenient variants
   (exact; with score suffixes and item-number prefixes stripped; first 8
   words only, to survive PDF line breaks).
4. An item is matched if all its probes are found. A record is `ok` if at
   least 90 % of its items match, otherwise `REVIEW`; `no_source_text` if
   nothing text-bearing was archived (scans); `not_found` if the agent
   reported no source.

The report (`data/processed/scale_hunt_check_report.csv`, no item text)
currently lists 31 ok, 76 REVIEW, 11 without source text and 74 not found.
`REVIEW` overstates problems: the checker also probes the agents'
administration notes, which often contain meta-commentary absent from the
source, so records whose item text checks out still show REVIEW. Two
scanned sources without a text layer (Conflict Tactics Scales from ERIC
ED297030; State-Trait Anxiety Inventory for Children from ERIC ED400304)
were verified by eye against the page images in the orchestrating session.

## Assembly

`scale_hunt_assemble.R` converts each workflow output into the exploded-item
schema of SynthNet's own extractions (one row per item and scale
membership, `bucket = "scaled"`), adding provenance columns (source URL,
version, archived files, source grade, extraction confidence, verbatim
claim, checker verdict, retrieval notes). The PsycTests DOI is encoded in
the `path` column. Combined output:
`data/restricted/scale-hunt-extractions-exploded.parquet` (local only).

## Outcome

| Batch | Found | Items |
|---|---|---|
| pilot | 10 | 264 |
| batch01 | 22 | 813 |
| batch02 | 17 | 390 |
| batch03 | 20 | 423 |
| batch04 | 18 | 575 |
| batch05 | 13 | 347 |
| batch06 + 06b | 16 | 461 |
| batch07 | 4 | 34 |
| batch08 | 7 | 132 |
| batch09 | 6 | 98 |

Combined: 133 instruments and 3,537 item rows; 132 records after the
language-variant fix, 120 records and 3,177 items after content
deduplication against SynthNet's corpus (`dedupe_fill_tiers.R`). Source
grades of successful extractions were mostly official sites and journal
appendices, then other web pages, clinical organisations and university
PDFs; 128 of 143 successful extractions reported high confidence. The hunt
raised usage-weighted coverage from 37.1 % to 65.6 % on its own, 68.0 %
together with the SemanticNet and aligns tiers.

Twenty-one of the 156 targets were declared unrecoverable from the open web
after three passes: SADS (interview booklet available only on request),
Hamilton Depression Inventory, Kohn Symptom Checklist, a Thai resilience
scale (Elsevier only), and the secure commercial batteries (MMPI-2 and
MMPI-2-RF, CPI, NEO PI 1985, PAI, TCI, PSI, SSAGA, COPM, QOLQ, Wechsler,
Bayley, Vineland, WRAT).

Effort: at least 283 agents and 13 million tokens across the runs with
metadata (batch01 and batch06 are not counted); each batch of 25 took 10 to
60 minutes of wall-clock time.

## What is not recoverable

- The pilot's extraction and verifier prompts survive only as 400-character
  previews in the pilot output; the verifier's output schema is recoverable
  from the stored verdicts.
- Batches 01 to 08 shared one script that was edited between runs. Only its
  batch08 state survives, so the exact earlier wording of the WebSearch
  paragraph and of the paper-retrieval cap introduced after batch06 is lost.
- Task lists for batch07 and batch08 were not kept; their targets can be
  reconstructed from the result files.
- The session transcript of 2026-08-02 to 08-03 (pilot and batches 01 to 07)
  was deleted; the instruction history for those days survives in the
  Claude Code history log.
