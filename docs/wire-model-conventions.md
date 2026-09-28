# Wire-model conventions (token efficiency)

Wire models are what the LLM emits under grammar-constrained decoding; token cost matters for both throughput and quality. Models live in `extraction/models/`. When adding a wire model, check each field against this list.

1. **Wire vs. result split.** The LLM emits a minimal wire form; a resolver builds the persisted result: `RawItems`→`Items`, `RawSurvey`→`Survey`, `InstrumentMappings`→`Instrument`, `RawMeta`→`Meta`. Extractors override `llm_schema()` (wire `inlined_schema()`) and `parse_response()` (run the resolver).
2. **Short keys for free-text-heavy, repeated objects** — `RawItem` (`text`, `img`, `type`, `note`, `lang`), `RawScale` (`name`, `construct`). `construct` is an alias of `construct_name` because it would shadow `BaseModel.construct`.
3. **Full keys for id references** — `ItemScaleMapping` keeps `item_id` / `scale_id` / `reverse_coded`. Bare `item`/`scale` lose the "id reference" cue and `rev` reads as "revision", biasing the model toward spurious truthy values.
4. **`Optional[Literal[True]] = None` for rare-true booleans** — the grammar can only emit `true` or omit. Used for `RawItem.img`, `ItemScaleMapping.reverse_coded`, `RawMeta.intake_form`, `RawMeta.objective_measure`.
5. **Omit-when-empty optionals** — `Optional[List[X]]` / `Optional[str] = None` with a prompt directive "omit entirely; do not emit `null` or `[]`": `RawItem.options` (only choice items and stem-less semantic-differential anchors), `RawItem.note` (administrator-facing text), `RawItem.lang` (only for translated items; `text` is always English), `InstrumentMappings.orphan_item_ids`.
   - **Exception: `RawScale.subscales`** is required (`default_factory=list`) and leaves emit `[]`. Forcing the grammar to commit parent vs. leaf at every node prevents hierarchy collapse.
6. **Chain-of-thought slot first** — `RawSurvey.subscale_evidence` precedes `scales` so the model commits to observed structural cues (per-item code legends, scoring instructions, section headers) before emitting the tree. Prevents flattening when the subscale signal is a tiny footnote (e.g. the ACQ). Dropped by `resolve_scales`.
7. **No auto-assigned fields on the wire** — ids are set post-parse by validators on `Items` (sequential) and `Survey` (DFS).
8. **`inlined_schema()`** — llama.cpp does not follow `$ref`; `extraction/models/_schema.py` inlines refs, capping recursive expansion (e.g. `Scale.subscales`) at depth 4.
