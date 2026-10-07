---
pretty_name: Synthometricon Corpus — pooled psychometric embeddings (encrypted)
language:
  - en
  - es
  - fr
  - de
  - pt
  - zh
  - it
  - nl
  - pl
  - ja
  - ar
  - ru
  - hu
  - tr
  - ms
  - fa
  - el
  - sv
  - ko
  - da
  - sr
  - "no"
multilinguality: multilingual
annotations_creators:
  - machine-generated
language_creators:
  - found
source_datasets:
  - original
task_categories:
  - sentence-similarity
  - feature-extraction
size_categories:
  - 100K<n<1M
tags:
  - psychometrics
  - psychological-assessment
  - item-response
  - scale-embeddings
  - encrypted
viewer: false
---

<!-- TODO before the first externally-shared release: add a `license:` (and
     `license_name:`/`license_link:` if custom) to the front matter above. -->

# Synthometricon Corpus — pooled psychometric embeddings

A corpus of **psychological measurement instruments** — questionnaires, scales and
their subscales — extracted from test documentation and aggregated into vector
representations that can be compared by cosine similarity. One row describes one
scale (or one whole instrument); the vector on that row stands in for the scale's
item content.

The intended use is *scale-level* retrieval and comparison: finding measures of a
construct, spotting near-duplicate scales across instruments, screening an item
pool, or estimating how strongly two scales would correlate before anyone collects
data.

**The data file in this repository is encrypted.** You need a decryption key, which
is issued separately (see [Access](#access)). Everything else here — this card, the
build manifest — is plain text.

## Access

This is a private repository, and the payload is additionally encrypted with a
symmetric key that is never stored here, in any public repository, or in the code
that produced it.

To request the key, email **bjorn.hommel@magnolia-psychometrics.com**. The key is a
44-character string. Keep it out of notebooks, version control and shared drives; it
is the same key for every release, so leaking it cannot be undone by re-publishing.

`manifest.json` in this repository records a `key_fingerprint` — the first 12 hex
digits of the key's SHA-256. If decryption fails, compare it against your own key to
tell "wrong key" apart from "corrupted download":

```python
import hashlib
hashlib.sha256(open("synthometricon-release.key", "rb").read().strip()).hexdigest()[:12]
```

## How to read the data

The payload is a single [Fernet](https://cryptography.io/en/latest/fernet/) token
(AES-128-CBC with an HMAC-SHA256 authentication tag) wrapping an Apache Parquet
file. It is *not* a readable Parquet file, which is why the dataset viewer is
switched off and why `datasets.load_dataset` will not work on it. Decrypt first,
then read:

```python
import io
import pandas as pd
from cryptography.fernet import Fernet
from huggingface_hub import hf_hub_download

key = open("synthometricon-release.key", "rb").read().strip()

blob = hf_hub_download(
    "magnolia-psychometrics/synthometricon-corpus",
    "data/synthometricon-corpus-release.enc.bin",
    repo_type="dataset",
)
raw = Fernet(key).decrypt(open(blob, "rb").read())
df = pd.read_parquet(io.BytesIO(raw))
```

`pd.read_parquet(..., columns=[...])` accepts a column projection, which is worth
using: the embedding columns are the overwhelming majority of the bytes.

**Memory.** Decryption is not streaming — the whole ciphertext, the whole decrypted
Parquet buffer and then the expanded DataFrame are resident at once. Budget roughly
8 GB of RAM for the unrestricted read.

**Integrity.** `manifest.json` carries `artifact.sha256` (of the encrypted file, to
check the download) and `artifact.plaintext_sha256` (of the Parquet bytes inside, to
check the decryption). Note that the ciphertext is *not* reproducible: Fernet draws a
fresh random initialisation vector on every encryption, so two releases built from an
identical corpus have different `sha256` but the same `plaintext_sha256`.

## Unit of analysis

Instruments in this corpus are hierarchical. A personality inventory may have five
domain scales, each with six facet subscales, and items attached at the facet level;
a short mood questionnaire may have no hierarchy at all.

The corpus gives **one row to every node of every hierarchy**, plus **one row per
instrument**:

| Row type | `is_instrument` | Identified by | Covers |
| --- | --- | --- | --- |
| Instrument | `True` | `path` | the whole document — every item in it |
| Scale node | `False` | `(path, scale_id)` | that node and its entire subtree |

Scale rows exist at *every* depth, so a domain scale and each of its facets all get
rows, and the domain row's vector pools the items of all its facets. Parents and
children therefore overlap by construction: an item attached to a facet contributes
to the facet row, to its parent domain row, and to the instrument row.

**This matters for any corpus-wide statistic you compute.** Rows are not independent
observations, and `df` is not a list of distinct scales. Filter to what you actually
want — e.g. `df[df.is_instrument]` for one row per instrument, or
`df[~df.is_instrument & (df.scale_depth == 1)]` for top-level scales only — before
averaging, counting or fitting anything.

## Columns

All columns present in the release are documented below. Per-release counts,
dimensions and null counts are in `manifest.json` rather than in this card, so that
this card cannot go stale about them.

### Identity and provenance

| Column | Type | Meaning |
| --- | --- | --- |
| `path` | string | Identifier of the source document the instrument was extracted from. Constant within an instrument; the document-level key. |
| `corpus_source` | string | Which sub-corpus the row came from: `apa-psyctests` (the bulk, extracted from APA PsycTests documentation), `scale-hunt`, `semanticnet`, `aligns`. Sub-corpora differ in how they were produced and in which quality checks are applicable — see [Quality flags](#quality-flags). |
| `public_doi` | string | Best available DOI for the instrument. For APA rows this is the PsycTests DOI; for other sources it is the DOI of the source publication where one could be established. Null where none could be. |
| `doi_psyctests` | string | The APA PsycTests DOI specifically, where the instrument has one. Null otherwise. |
| `meta_title_raw` | string | Instrument title as printed in the source document, before any version/form qualifier was split off into `version`. |
| `meta_language` | string | ISO 639-1 code for the language of the **source document**. Note the distinction from `flag_item_translated` below: item *text* is published in English throughout. |
| `public_year` | Int64 (nullable) | Four-digit publication year parsed from the source citation. Null where the citation carries no year (e.g. "n.d."). Range in this corpus: 1918–2023. |
| `version` | string | Version, form or translation qualifier for this node — e.g. a short form, a parallel form, a language adaptation. Split out of the title and scale names during assembly so that two forms of the same instrument are distinguishable without string surgery. |

### Position in the hierarchy

| Column | Type | Meaning |
| --- | --- | --- |
| `is_instrument` | bool | `True` for the one document-level row per instrument, `False` for scale-node rows. |
| `scale_id` | float (nullable) | Node identifier within the document. **Null on instrument rows.** Unique only in combination with `path`. |
| `scale_name` | string | Name of the scale or subscale as documented. **Null on instrument rows** (use `meta_title_raw` there). |
| `scale_depth` | float (nullable) | Depth of the node: `1` for a top-level scale, `2` for its subscales, and so on (observed maximum in this corpus: 4). **Null on instrument rows.** |
| `n_items` | int64 | Number of items pooled into this row — the node's whole subtree, not only items attached directly to it. On an instrument row, every item in the document. |
| `n_scales` | int64 | Number of scale nodes in this row's subtree, including itself. `1` for a leaf scale. |

### Quality flags

Four boolean columns warning that a row may be unreliable. All four are
**three-valued**, and the distinction is load-bearing:

- `True` — the check ran and found a deviation.
- `False` — the check ran and found none.
- **NA — the check could not be run.** This is *not* a clean bill of health, and
  `.fillna(False)` will silently turn "unknown" into "fine".

| Column | Meaning | When it is NA |
| --- | --- | --- |
| `flag_item_count_deviation` | The number of items extracted from the document differs from the item count stated in the instrument's catalogue record — items may have been missed or duplicated. Document-level: constant across all rows of an instrument. | No catalogue record to compare against (non-APA sub-corpora). |
| `flag_scale_count_deviation` | As above, for the number of scales. Document-level. | As above. |
| `flag_item_text_deviation` | At least one item pooled into this row is not a near-verbatim quote of the source PDF — the extraction model may have paraphrased, merged or invented it. Aggregated over the row's items: `True` if any pooled item is flagged, `False` if none is and at least one was checkable. | No item in the row could be checked: no source PDF (partial sub-corpora), or the PDF's text layer was too thin for the comparison to mean anything. |
| `flag_item_translated` | At least one item pooled into this row was **translated into English** for this corpus — the original item was in another language. Aggregated the same way. | The original item language was not recorded. |

For most analyses the useful filter is "drop rows flagged `True`, and decide
deliberately what to do with NA". Treating NA as `False` inflates your effective
sample with exactly the rows that could not be verified.

### Embeddings

These column names carry the name of the model that produced them, so they change if
the encoding configuration changes. `manifest.json` records the exact names,
dimensions and null counts of the current release.

| Column | Meaning |
| --- | --- |
| `item_pooled_surveybot3000` | **The main vector.** A keying-aware centroid of the embeddings of every item pooled into this row, encoded with `surveybot3000`. See [How items are pooled](#how-items-are-pooled). Do not re-normalise it — its length is informative. |
| `keying_disagreements_surveybot3000` | Count of items in this row whose keyed vector points *against* the keyed average of the row's other items — the embedding analogue of a negative corrected item–total correlation. A review pointer, not a defect count; see the caveat below. |
| `scale_pooled_all_minilm_l6_v2` | Centroid of the *name* embeddings in this row's subtree, each encoded with `all-MiniLM-L6-v2` and averaged unweighted: on scale rows the row's own scale name, every constituent scale name (intermediate levels included) and the instrument title; on instrument rows the names of the scales that hold items and the instrument title. A label-based counterpart to `item_pooled_*`, useful when you care what a scale is *called* rather than what it asks. |
| `scale_embedding_all_minilm_l6_v2` | Embedding of this node's own `scale_name` text alone, also for parent nodes whose items all sit in subscales. **Null on instrument rows, and null on any node that carries no name of its own.** |
| `instrument_embedding_all_minilm_l6_v2` | Embedding of `meta_title_raw`, the instrument title. **Null on all scale rows** — it is populated only where `is_instrument` is `True`. |

**Nullability is not incidental.** `np.stack(df.scale_embedding_all_minilm_l6_v2)`
raises on a mixed column of arrays and nulls. Filter with `.notna()` first, or use
`item_pooled_*`, which is populated wherever the row pools at least one item.

## How items are pooled

`item_pooled_*` reproduces, in embedding space, what a scale score does to items.

1. Every item vector is scaled to length 1. The encoder is trained so that the
   *angle* between two item vectors predicts their correlation; length carries no
   information and would otherwise weight the average.
2. Every item documented as reverse-keyed is multiplied by −1.
3. The vectors are averaged.

Negating a vector flips the sign of all its cosines and leaves their magnitudes
alone — which is exactly what reverse-scoring does to an item's correlations. So
step 2 is reverse-scoring performed in the embedding space, using only the model's
predictions for the item as written and trusting the documented key the way a scoring
key is trusted.

The consequence for interpretation: with unit-length items whose cosines approximate
correlations, **the cosine between two pooled vectors is the predicted correlation
between the two scale scores** (the scale-level estimator of Hommel & Arslan, 2025).
Read a cosine of .62 between two pooled scale vectors as "these two scales would
correlate about .62".

Two practical notes:

- **Do not re-normalise pooled vectors.** Their length reflects the average keyed
  inter-item correlation — that is, the scale's predicted internal consistency.
  Normalising throws that away.
- **Scales with no reverse-keyed items** get the plain mean of unit vectors; scales
  where everything is reverse-keyed have everything flipped, so the vector points at
  the pole the scale's *label* names. There are no special cases.

### The `keying_disagreements_*` caveat

This diagnostic flags roughly 31% of reverse-keyed items and 1% of positively keyed
ones. That asymmetry reflects the encoder's comparatively weak predictions on the
negative side, not a 31% documentation-error rate. Treat a nonzero count as a pointer
for review, and use a stricter threshold than zero if you want likely keying errors
specifically.

## Limitations

- **The content is machine-extracted.** Items, scale names and hierarchies were read
  out of test documentation by a vision–language model. It is accurate enough to be
  useful at corpus scale and not accurate enough to be trusted row by row. The
  `flag_*` columns are the corpus's own account of where it is unsure; use them.
- **The flags are heuristic**, and their NA rate varies systematically by
  `corpus_source` — the non-APA sub-corpora have no catalogue record to check against,
  so the count flags are NA throughout.
- **DOI and year backfill is best-effort**, matched against Crossref and PsycTests
  records. Nulls are common and a populated value is a strong-but-not-certain match.
- **No item text ships in this release.** Rows carry pooled vectors, not the items
  they were pooled from: the source documentation is copyrighted and the item text
  cannot be redistributed. You can compare and retrieve scales; you cannot reconstruct
  their content from this file.
- **Rows are nested, not independent** — see [Unit of analysis](#unit-of-analysis).
- **The embeddings are model-specific.** `item_pooled_*` is interpretable as a
  predicted correlation only because `surveybot3000` is trained to place item cosines
  on a signed correlation scale. A generic semantic encoder would put a reverse-keyed
  item next to its positive siblings, and step 2 above would be actively wrong.

## This release

Per-release facts — row counts, column names and dimensions, null counts, language
and sub-corpus distributions, checksums, key fingerprint, and the pipeline commit
that built it — are in **`manifest.json`** in this repository, rather than restated
here where they would drift.

## Citation

<!-- TODO: replace with the corpus citation once the accompanying paper is out. -->

The scale-level estimator that makes pooled cosines interpretable as predicted
correlations:

```bibtex
@article{hommel2025,
  author  = {Hommel, Bj{\"o}rn E. and Arslan, Ruben C.},
  title   = {Language models accurately emulate human psychometric responses},
  year    = {2025}
}
```

## Contact

bjorn.hommel@magnolia-psychometrics.com

---

*This card is published from `publish/dataset_card.md` in the pipeline repository and
overwrites the copy on the Hub on every release. Edit it there, not in the Hub UI.*
