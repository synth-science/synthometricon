# Assembly logs, stats sidecars and the descriptives report

Implementation map (helpers, how sections are wired): `architecture-assemble.md` → `assemble/report.py`.

## Run artifacts

- **Stage logs** — every real `python -m assemble` run writes the combined stage reports to `logs/assemble-<timestamp>.log` (`--no-logs` skips; `--report-only` never writes).
- **Stats sidecars** — patch and postprocess write `<artifact>.stats.json` next to their parquet on real runs: per-step counters (`Ctx.stats`) plus the report lines. Latest run wins, so older runs survive only in the logs. `assemble.stats.stats_stale` flags a sidecar older than its parquet.
- **Tracked copies** — each sidecar and the published `<stem>.md` is also copied into `data.reports_dir` (`reports/`, committed to the source repo, never to the public one) when written, so data changes show up as diffs in git.

## The report

`assemble/report.py` reads the artifacts (extraction store, partials, combined, patched, postprocessed, embedded, pooled) and the sidecars and builds a markdown descriptives report. It is **not** an assemble stage. `python -m publish --step publish` writes it beside the release artifact as `<data.publish stem>.md`, so it always describes the data that shipped. `python -m assemble.report [--output PATH]` prints a preview.

Sections: extraction tallies → per-partial counts → combined corpus (incl. `bucket` composition) → patch / postprocess step counters (from the sidecars) and combined→postprocessed attrition → final `embedded` / `pooled` counts plus hierarchy nesting → additional statistics (pooled group sizes, item text length, response options, item types, reverse-coded share, languages, permissions, verbatim fidelity, `flag_*` rates) → appendix (identifier and column glossary).

When something is missing or stale (no sidecar, a sidecar older than its artifact, an exploded parquet covering fewer documents than the store), the report prints a WARNING or a "re-run the stage" hint and carries on. It does not hide the gap.

## Counting conventions

These match `postprocess._observations` and pool's granularity.

| unit | definition |
|---|---|
| document = instrument | distinct `path`; null-path rows fall back to `patch._doc_keys` (source + title) |
| item | distinct `(document, item_item_id)`: one physical question |
| placement | one `(item, scale)` occurrence, i.e. one row carrying an item. A multi-scale item is one item but several placements |
| row | a physical parquet row. Differs from placements only by the item-less shell rows of failed extractions (`_shell_row_note` reconciles them; postprocess drops them) |
| scale | every hierarchy node: distinct `(document, node)` over `scale_id_path`, parents included (= one pooled row each) |
| item-bearing scale | distinct `(document, scale_id)`: only scales with directly attached items |

## Structural distributions

Every stage that holds item-level rows reports items / placements / scales / item-bearing scales per document, items per scale (subtree and direct) and placements per item.

- Full tables: `n / M / SD / min / Mdn / IQR (Q1–Q3) / p95 / max`. `n` counts units, never rows, and SD uses ddof=1.
- Compact cells, used when sources or stages are compared side by side: `M (SD) · Mdn [IQR Q1–Q3]` (`MS_LEGEND`).
- The **IQR is always the range `Q1–Q3`, never its width**. The same holds in the stage logs and the `iqr_q1`/`iqr_q3` sidecar keys.
- Both mean and median are reported because the counts are strongly right-skewed.
- Empty units stay in the denominator: a failed document counts as 0 items, and an umbrella scale counts its subtree. So `M × unit count` equals the total in the adjacent count table.
- *Subtree* counts an item towards every node on its `scale_id_path`, the same way pool's `n_items` does. A mismatch between the item-level and pooled tables is therefore a real inconsistency.

## Nesting of the scale hierarchy

The *Final datasets* section reports levels of nesting per instrument: 0 = no scales, 1 = a flat list, 2 = scales with subscales, and so on. It gives this as a distribution and as instrument counts per level, both overall and per `corpus_source`. It also reports the per-node depth distribution and counts per hierarchy level.

- The per-document depth is the maximum `len(scale_id_path)` over the document's placements. Parents have no rows of their own, but a node at depth d exists only if some item's path runs through it, so the maximum over item rows is still the true depth.
- The pooled `scale_depth` table in *Additional statistics* must agree with the per-node table. If they diverge, pool and embedded disagree about the hierarchy.

## Verbatim fidelity

This block (`_fidelity_lines`) reports how closely the extracted item text matches the source PDF.

- **Similarity** = 1 − `pdf_match_edit_distance_norm`.
- **Method paragraph** — generated from `postprocess.PDF_MATCH_NORM_MAX` / `PDF_TEXT_CHARS_MIN` and the installed rapidfuzz version. Its prose (reference text, algorithm, no preprocessing) must be updated by hand if `best_partial_distance` or `pdf_text_excl_first_page` change.
- **Units** — distinct items (a multi-scale item has one text, so de-duplicating loses nothing), the per-instrument mean and per-instrument minimum, and placements for reference.
- **Subsets** — each unit is reported over all measured items, over checkable items (`pdf_text_checkable`: text layer ≥ `PDF_TEXT_CHARS_MIN` and items not printed as images), and over checkable English originals (translated items cannot match the original-language PDF).
- **p5 instead of p95** — similarity piles up at 100%, so the low tail is the informative one.
- **Gotcha:** "instrument *mean* below threshold" and "≥ 1 item below threshold" are different figures (9,149 vs 8,361 in the Oct 2026 corpus). The report prints both.
