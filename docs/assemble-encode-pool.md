# Encode, pool and embedding search

Stage overview and carried-column details: `architecture-assemble.md` (`encode.py`, `pool.py`). This page covers the semantics the code docstrings point to.

## Encode

- Each entry in the `encode:` config block is a `{name, path}` mapping. `name` is sanitized into a column segment: lowercased, and every run of non-alphanumeric characters collapsed to `_`.
- `item_models` encode `item_item_text` into `item_embedding_{name}`.
- `scale_models` encode `scale_name` into `scale_embedding_{name}` and `meta_title_raw` into `instrument_embedding_{name}`.
- Each unique non-blank text is encoded once. Null or blank source text gives a null cell.
- Only one model is held in memory at a time.

## Pool

### Grouping

- **Instrument rows** key on `path`, which is unique per document (DOIs are not). Each pools **all** of the document's items (scaled, orphan and unscaled), de-duplicated by `item_item_id`.
- **Scale rows** key on `(path, scale_id)`, because `scale_name` can repeat within a document. Each pools the node's whole **subtree**: an item on `scale_id_path [1, 2, 4]` feeds nodes 1, 2 and 4.
- A parent node with no direct items takes its `scale_name` / `scale_depth` from its descendants' `scale_name_path`.

### Keyed centroid

`item_pooled_{m}` = `keyed_centroid`. Each item vector is unit-normalized, documented reverse-keyed items are negated, and the unweighted mean is taken. This is the embedding form of a ±1-weighted sum score. A null `item_reverse_coded` counts as positive, and there are no special cases (a scale where every item is reverse-keyed simply has every vector negated).

- Negating is reverse-scoring: cos(−v, w) = −cos(v, w).
- With a correlation-calibrated encoder (`surveybot3000`), the cosine between two pooled vectors equals the ±1-composite correlation (Hommel & Arslan, 2025).
- Pooled vectors are **not renormalized**, because their norm carries the mean keyed inter-item correlation. Compare them by cosine.
- **Only valid for encoders whose cosine approximates a signed correlation.** Do not add a generic semantic model to `encode.item_models` without revisiting this.

The full rationale and robustness checks are in `reverse-keyed-pooling.md`.

### Keying disagreement

`keying_disagreements_{m}` (`keying_disagreement`) counts the items whose keyed unit vector has a negative dot product with the keyed **leave-one-out** mean of the other items. This is the embedding analogue of a negative corrected item-total correlation.

- The item is left out of the mean it is compared with. If it were included, a two-item scale could never be flagged.
- Scales with fewer than two items are never flagged.
- It is a diagnostic only and never changes pooling. Most flags come from the encoder's weak negative predictions, not from keying errors, so treat them as review pointers.

### Scale pooling

`scale_pooled_{m}` is the unweighted mean of two kinds of vector, each counted once:

- the embedding of each distinct scale in scope, for scales that have direct rows (parent-only nodes have no embedding of their own);
- the document's `instrument_embedding_{m}` from the same model, if it is not null.

## Embedding search (`assemble.search`)

This is an import-only sanity check, not a stage. It embeds ad-hoc queries with `encode.item_models` / `encode.scale_models`, choosing the model by `model_index`, and ranks the pooled rows by cosine similarity.

```python
from assemble.search import search_items, search_scales
search_items("I am the life of the party.")            # vs item_pooled_*
search_items(["I am the life of the party.", "I keep in the background."],
             reverse=[False, True])                    # keyed-centroid query
search_scales("Extraversion", top_k=5)                 # vs scale_pooled_*, plain mean
```

- Results mix scale and instrument rows; filter on `is_instrument`.
- Rows with a null vector are skipped.
- Models and the pooled frame are cached per process. Pass `data=` to reuse a DataFrame you have already loaded.
- The repo is not installed as a package, so run from the repo root or set `PYTHONPATH=<repo>`.
