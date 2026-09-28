# Reliability metrics for the extraction pipeline

A plain-language guide to how we measure whether the extraction pipeline gives the same answer twice.

## The basic idea: runs as raters

Every extractor in the pipeline is driven by a language model that samples its output. Ask it to extract the items from the same PDF twice and you will not necessarily get byte-identical results. That is the same situation psychologists face whenever two coders independently rate the same material, and we treat it the same way.

For each test document we run each extractor several times (typically three runs, always at least two). We then treat **each run as an independent rater** and the extracted content as that rater's coding of the document. The question every metric below answers is a version of: *how much do the raters agree?*

Two building blocks appear again and again.

**Run pairs.** With R runs there are R(R − 1)/2 distinct pairs of runs. Three runs give three pairs (1 vs 2, 1 vs 3, 2 vs 3). Most of the simple metrics ask "in what fraction of pairs did the two runs agree?"

**Text similarity.** Free text is compared with the *normalised Levenshtein distance*: the number of single-character edits (insertions, deletions, substitutions) needed to turn one string into the other, divided by the length of the longer string, after converting both to lower case. It ranges from 0 (identical) to 1 (nothing in common). Similarity is simply 1 minus the distance.

> Example: "I feel calm" versus "I feel clam" needs two edits (swap the a and l), and the longer string has 11 characters, so distance = 2 / 11 ≈ 0.18 and similarity ≈ 0.82.

## Metrics at a glance

| Extractor | Metric | Answers the question | Pass threshold |
|---|---|---|---|
| Items | Count agreement | Did runs find the same number of items? | ≥ 0.90 |
| Items | Levenshtein alpha | Are the item texts the same? (Krippendorff's α) | ≥ 0.80 |
| Scales | Count agreement | Did runs find the same number of scales and subscales? | ≥ 0.90 |
| Scales | Tree overlap | Is the scale/subscale structure the same? | ≥ 0.80 |
| Instrument | Count agreement | Did runs make the same number of item-to-scale assignments? | reported only |
| Instrument | Assignment alpha | Were items assigned to the same scales? (Krippendorff's α) | ≥ 0.80 |
| Instrument | Reverse-coding agreement | Do runs agree on which items are reverse-scored? | ≥ 0.90 |
| Meta-data | Language agreement | Same language code? | ≥ 0.95 |
| Meta-data | Intake-form agreement | Same intake-form classification? | ≥ 0.95 |
| Meta-data | Objective-measure agreement | Same objective-measure classification? | ≥ 0.95 |

A document passes an extractor's reliability check only if **every** thresholded metric reaches its threshold. If a metric cannot be computed, that counts as a fail.

---

## Items

The items extractor returns an ordered list of item texts (for items that are pictures rather than sentences, the response options stand in for the text).

### Count agreement

**Question.** Did every run find the same number of items?

**How it works.** Compare the item counts of every pair of runs. The score is the proportion of pairs whose counts are identical.

```
Count agreement = (number of run pairs with identical item counts)
                  / (total number of run pairs)
```

**Example.** Three runs extract 20, 20 and 21 items. Runs 1 and 2 agree; runs 1 and 3 and runs 2 and 3 do not. Count agreement = 1 / 3 = 0.33.

**Reading it.** This is a strict, all-or-nothing measure. A single run that misses one item already pulls the score down sharply. It is deliberately strict because item count is the first thing that must be right: a missing or invented item corrupts everything downstream.

### Levenshtein alpha

**Question.** Setting aside how many items were found, is the *wording* of the items the same across runs?

**How it works.** This is Krippendorff's alpha, the reliability coefficient familiar from content analysis, with one adaptation: instead of asking "did the raters pick the same category?" it asks "how far apart are the texts the raters produced?", using the squared normalised Levenshtein distance as the measure of disagreement.

The units of analysis are item positions (1st item, 2nd item, ...). The raters are the runs. Because runs may differ in how many items they found, only the first *m* items are used, where *m* is the smallest item count across runs. Count differences are already captured by count agreement, so they are not double-counted here.

```
alpha = 1 − (observed disagreement) / (expected disagreement)
```

*Observed disagreement* is the average squared text distance between runs for the same item position. *Expected disagreement* is the average squared distance you would see if item texts were shuffled at random across positions, that is, the disagreement of raters who were not looking at the document at all.

**Reading it.** Alpha is 1 when every run produced identical wording for every item, 0 when agreement is no better than chance, and can go slightly negative when agreement is worse than chance. The usual content-analysis conventions apply: 0.80 and above is considered reliable, which is why that is the pass threshold. Because the disagreement measure is *graded* rather than all-or-nothing, a run that writes "I feel calm and relaxed" where another wrote "I feel calm, relaxed" is penalised only a little, while a completely different sentence is penalised heavily.

---

## Scales

The scales extractor returns a tree: top-level scales, each of which may contain subscales, which may in turn contain sub-subscales.

### Count agreement

**Question.** Did every run find the same number of nodes in the tree (scales plus subscales at every level)?

**How it works.** Identical to item count agreement, applied to the total node count.

```
Count agreement = (number of run pairs with identical node counts)
                  / (total number of run pairs)
```

### Tree overlap

**Question.** Is the *structure* of the scale tree the same: the same scale names, nested the same way?

**How it works.** Two trees are compared node by node. The similarity of two nodes is the average of two things:

1. how similar their names are (Levenshtein similarity), and
2. how similar their children are, computed the same way, recursively.

Nodes from the two trees are then paired up one-to-one, most similar pairs first. The similarity of two trees is the sum of the paired similarities divided by the size of the *larger* tree, so a scale that appears in only one run counts as a zero for that run pair.

```
node similarity(a, b) = 0.5 × name similarity(a, b)
                      + 0.5 × tree similarity(children of a, children of b)

tree similarity(A, B) = (sum of node similarities over matched pairs)
                        / max(number of nodes in A, number of nodes in B)

Tree overlap = average tree similarity over all run pairs
```

Two empty trees are perfectly similar (1.0); an empty tree versus a non-empty one is entirely dissimilar (0.0).

**Example.** Run 1 finds *Anxiety* (with subscales *Worry* and *Physical*) and *Depression*. Run 2 finds *Anxiety* (with subscale *Worry* only), *Depression*, and *Stress*.

- *Anxiety* vs *Anxiety*: names identical (1.0). Children: *Worry* matches *Worry* (1.0) but *Physical* is unmatched, so children similarity = 1.0 / 2 = 0.5. Node similarity = 0.5 × 1.0 + 0.5 × 0.5 = 0.75.
- *Depression* vs *Depression*: names identical, both childless. Node similarity = 1.0.
- *Stress* has no partner.

Tree similarity = (0.75 + 1.0) / 3 = 0.58.

**Reading it.** 1.0 means identical trees. The score drops both for renamed scales (gradually, in proportion to the renaming) and for structural differences such as an extra subscale or a subscale placed under a different parent. The pass threshold is 0.80.

---

## Instrument

The instrument extractor assigns each item to one or more scales and marks whether the item is reverse-scored. In reliability testing the item list and scale tree are held fixed (taken from a curated fixture), so item and scale identifiers mean the same thing in every run. Only the *assignments* vary.

### Count agreement

**Question.** Did every run make the same number of item-to-scale assignments?

Same formula as above, applied to the number of assignments. This one is reported for diagnostic purposes but is not used in the pass/fail decision.

### Assignment alpha

**Question.** Was each item assigned to the same scale(s) in every run?

**How it works.** This is Krippendorff's alpha at the nominal level, which is exactly the setting psychologists use for categorical inter-rater reliability. The units are items, the raters are runs, and each run's "category" for an item is the set of scales it assigned that item to. An item that a run left unassigned gets the category "none". Two runs agree on an item only if their scale sets are identical; there is no partial credit.

```
alpha = 1 − (observed disagreement) / (expected disagreement)

disagreement between two categories = 0 if identical, 1 otherwise
```

**Example.** Four items, three runs. Cells show the scale(s) each run assigned.

| Item | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| 1 | Anxiety | Anxiety | Anxiety |
| 2 | Anxiety | Anxiety | Depression |
| 3 | Depression | Depression | Depression |
| 4 | Depression | none | Depression |

Items 1 and 3 show perfect agreement; items 2 and 4 each have one dissenting run. Alpha will be clearly below 1 but well above 0, reflecting mostly consistent assignment.

**Reading it.** As for any nominal alpha: 1 is perfect, 0 is chance, 0.80 is the conventional bar for reliable coding and is the pass threshold. If every cell is identical the score is 1 by definition.

### Reverse-coding agreement

**Question.** When runs agree that an item belongs to a scale, do they also agree on whether it is reverse-scored?

**How it works.** Collect every (item, scale) pairing that appears in at least one run. For each pairing, look at the reverse-coding flag recorded by each run that made that pairing. The pairing counts as consistent if all those flags are the same.

```
Reverse-coding agreement = (number of item–scale pairings with consistent reverse flags)
                           / (number of item–scale pairings seen in any run)
```

A pairing that only some runs made is *not* penalised here; that disagreement already shows up in assignment alpha. This metric isolates the reverse-scoring decision.

**Example.** Across three runs the pairings (item 5, Anxiety) and (item 6, Anxiety) appear. All runs mark item 5 as reversed. Two runs mark item 6 as not reversed and one marks it reversed. Reverse-coding agreement = 1 / 2 = 0.50.

**Reading it.** 1.0 means every reverse-scoring decision was unanimous. The pass threshold is 0.90.

---

## Meta-data

The meta-data extractor classifies each document on three categorical fields: its language (an ISO code such as `en` or `de`), whether it is an intake form, and whether it is an objective measure rather than a self-report.

### Language, intake-form and objective-measure agreement

**Question.** Did every run give the same classification?

**How it works.** For each field separately, the score is the proportion of run pairs with identical values.

```
Agreement = (number of run pairs with the identical value)
            / (total number of run pairs)
```

**Example.** Three runs report language `en`, `en`, `de`. One pair agrees out of three. Language agreement = 0.33.

**Reading it.** These are single categorical judgements that should be nearly deterministic, so the pass threshold is high at 0.95, which in practice means all runs must agree.

---

## Across the whole test set

The per-document metrics tell us whether a *particular* document is extracted consistently. After all documents have been processed, the framework also summarises reliability over the test set as a whole.

**ICC(2,1) on item counts.** Item counts from every run of every document are entered into a two-way random-effects intraclass correlation with *documents as targets* and *runs as raters*, absolute-agreement, single-measure form. This is the ICC that Shrout and Fleiss label ICC(2,1). It answers the question: "if you picked one run at random, how well would its item counts reproduce the true item counts across documents?" Values near 1 mean that variation in item counts comes from genuine differences between documents, not from run-to-run noise. A 95% confidence interval is reported alongside. It needs at least two documents.

```
ICC(2,1) = (MS_documents − MS_error)
           / (MS_documents + (k − 1) × MS_error + k × (MS_runs − MS_error) / n)

k = number of runs per document, n = number of documents
```

**Mean and spread.** For tree overlap, assignment alpha, and each of the three meta-data agreements, the framework reports the mean and standard deviation across documents together with the lowest and highest document score. This shows whether a mediocre average comes from uniformly moderate reliability or from a few problem documents.

## How to read a reliability report

Each document receives one line per extractor listing every metric, its value, and whether it met its threshold, followed by a PASS or FAIL verdict. Documents run fewer than two times are marked SKIPPED because no agreement can be assessed. The suite summary then counts passes and fails and prints the cross-document statistics above.

A FAIL is a prompt to look at the run-by-run output for that document, which the log also contains: a table of item texts, scale trees, and, for the instrument extractor, a grid showing for every item and scale which runs made the assignment and which marked it reversed.
