# How often Synthometricon shows related scales: summary

Summary of [`search_validity.Rmd`](search_validity.Rmd), which renders to `search_validity.html`
(`Rscript -e 'rmarkdown::render("analyses/search_validity.Rmd")'`). The inputs are the published
data of the SurveyBot3000 paper; the Rmd downloads them from the SurveyBot3000 repositories, pinned
to a commit. The pilot study (22 well-known personality and well-being questionnaires, 113 scales,
N = 493) is probably a best case. The validation study (57 scales sampled from APA PsycTests,
N = 387) is closer to the database as a whole. A related scale counts as shown at a displayed value
if it reaches that value with the correct sign.

The engine represents a scale the way a scale score is formed: the SurveyBot3000 vectors of
reverse-keyed items are multiplied by −1 and all item vectors are averaged. The displayed value is the
cosine between two such vectors. For the study scale pairs it equals the paper's synthetic scale
correlation (largest difference .0002), so the paper's accuracy applies to the displayed values:
.87 [.83, .90] in the pilot study and .83 [.75, .89] in the validation study.

Brackets give 95% confidence intervals from a cluster bootstrap over scales (1,000 replicates;
each replicate resamples a study's scales with replacement, keeps all pairs among them, and draws
one set of true correlations from the observed correlations and their standard errors; for short
queries and missing keys, the median over the random item or key draws within each replicate).

## Key numbers

Share of related scales shown at or above a displayed value:

| True correlation at least | Study | Pairs | Shown at .30 | Shown at .40 | Shown at .50 |
|---|---|---|---|---|---|
| .70 | Pilot | 46 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | .93 [.77, 1.00] |
| .70 | Validation | 44 | .88 [.65, 1.00] | .70 [.41, .95] | .48 [.21, .77] |
| .50 | Pilot | 431 | .89 [.81, .95] | .74 [.62, .84] | .48 [.37, .60] |
| .50 | Validation | 182 | .73 [.58, .87] | .47 [.32, .64] | .25 [.13, .41] |

![Share of related scales shown at each displayed value](output/search_validity/sensitivity-plot-1.png)

Displayed value a user needs to read down to in order to see 80%, 90% or 95% of the related scales:

| True correlation at least | Share | Pilot | Validation |
|---|---|---|---|
| .70 | 80% | .57 [.49, .65] | .32 [.23, .47] |
| .70 | 90% | .54 [.41, .61] | .29 [.14, .42] |
| .70 | 95% | .49 [.41, .59] | .23 [.14, .40] |
| .50 | 80% | .36 [.30, .42] | .25 [.17, .32] |
| .50 | 90% | .29 [.23, .36] | .17 [.08, .27] |
| .50 | 95% | .24 [.14, .30] | .12 [.06, .23] |

![How far down the result list to read](output/search_validity/read-down-plot-1.png)

The validation study needs a much lower cutoff although its accuracy is similar (.83 vs .87),
mainly because its displayed values are compressed toward zero: a pair with a true correlation of
.70 is displayed at about .43 on average, against .57 in the pilot study.

Short queries, validation study (full scale in parentheses):

| Query items | Scales | Accuracy | Strongly related shown at .30 |
|---|---|---|---|
| 1 | 57 | .72 [.64, .78] (.83 [.75, .89]) | .73 [.51, .96] (.88 [.68, 1.00]) |
| 2 | 57 | .80 [.72, .85] (.83 [.75, .89]) | .77 [.55, .95] (.88 [.67, 1.00]) |
| 3 | 32 | .81 [.73, .87] (.85 [.78, .90]) | .78 [.57, .96] (.85 [.59, 1.00]) |
| 5 | 20 | .84 [.77, .90] (.86 [.78, .91]) | .83 [.62, 1.00] (.91 [.72, 1.00]) |

In the pilot study, short queries cost little: single items reached an accuracy of .77 [.72, .81]
(full scales .87 [.83, .89]) and still showed .95 [.78, 1.00] of the strongly related scales at .30.

Reverse-keyed items treated as positively keyed, median over random draws. Unmarked query: the user's
"(R)" marks are missing, and the database scales keep their documented keying (each pair searched in
both directions):

| Share unmarked | Pilot: accuracy | Validation: accuracy | Validation: strongly related shown at .30 |
|---|---|---|---|
| 0% | .87 [.83, .90] | .83 [.75, .89] | .88 [.67, 1.00] |
| 25% | .82 [.78, .86] | .80 [.71, .86] | .83 [.62, 1.00] |
| 50% | .72 [.66, .78] | .74 [.64, .82] | .79 [.57, .98] |
| 75% | .55 [.44, .65] | .66 [.52, .75] | .76 [.53, .97] |
| 100% | .36 [.22, .50] | .55 [.38, .68] | .71 [.47, .93] |

Unmarked query searched against an unkeyed source (e.g. SemanticNet/ALIGNS): the keys are missing on
both sides of every pair:

| Share unmarked | Pilot: accuracy | Validation: accuracy | Validation: strongly related shown at .30 |
|---|---|---|---|
| 0% | .87 [.83, .90] | .83 [.75, .89] | .88 [.67, 1.00] |
| 25% | .78 [.73, .83] | .77 [.68, .84] | .80 [.58, 1.00] |
| 50% | .62 [.53, .69] | .67 [.54, .77] | .75 [.49, .97] |
| 75% | .40 [.28, .52] | .53 [.35, .67] | .71 [.46, .95] |
| 100% | .23 [.13, .35] | .37 [.17, .56] | .68 [.39, .93] |

## Proposed default marker

A line at a displayed value of .30, labelled along the lines of "strongly related scales rarely
appear below this line". In the validation study 9 of 10 scales with a true correlation of .70 or
more were shown at or above .30, and in the pilot study all of them. The line is not a relevance
threshold: many unrelated scales will also score above it. We do not suggest rescaling the
displayed values: the two studies would call for different rescalings, and the .30 line is already
set by the more compressed validation study.

## Draft FAQ text: "How trustworthy and accurate are the results?"

Synthometricon estimates how strongly your items would correlate with each scale in the database if
both were given to the same people. We checked these estimates against real data in two studies. In
the first, with 22 well-known personality and well-being questionnaires (493 participants), the
estimates correlated .87 with the observed correlations between scales. In the second, with 57
scales drawn from the database (387 participants), they correlated .83. A typical estimate was off by
about .12 in the first study and .16 in the second. The first study is a best case; the second is
closer to what you can expect for the database as a whole.

**Will I find scales that measure something very similar to mine?** Of 10 scales that truly correlate
.70 or more with yours, about 9 appeared with a displayed value of .30 or higher in the second study,
and about 7 at .40 or higher. In the first study, all of them reached .40. Of scales that correlate
.50 or more, about 7 in 10 appeared at .30 or higher in the second study. If nothing in your results
lies above .30, a strongly overlapping measure is unlikely, but not ruled out.

**How far down the list should I read?** To see 9 in 10 of the scales that truly correlate .70 or more
with yours, read down to a displayed value of about .30. For scales that correlate .50 or more, the
same share requires reading down to about .17. For well-known personality measures, about .55 and
.30 were enough. The engine shows how many results lie above any value, and the filter box helps
narrow long lists.

**Does it matter how many items I enter?** Yes. With a single item, 7 in 10 strongly related scales
reached .30 in the second study, compared with 9 in 10 when the full scale was entered. Enter more
than one item if you can; the more items, the closer the results come to those for a full scale.

**Do I need to mark reverse-keyed items?** Yes. Mark every reverse-keyed item with "(R)". The engine
reverses marked items before combining them, as you would when scoring the scale. If half of the
reverse-keyed items in our second study were left unmarked, the estimates' correlation with the
observed correlations fell from .83 to .74; with none marked, to .55.

## Draft methods paragraph

Synthometricon represents each scale by the mean of its SurveyBot3000 item vectors, with the vectors
of reverse-keyed items multiplied by −1, and displays the cosine between the query's and each
database scale's vector. To describe how often related scales appear in the result list, we reused
the scale pairs of the pilot (113 scales, 6,245 pairs) and validation studies (57 scales, 1,568
pairs). A pair
counted as related if its true correlation was at least .50 (or .70) in absolute value, and as shown
at a displayed value *c* if its synthetic correlation had the correct sign and an absolute value of
at least *c*. To account for sampling error in the empirical correlations, we drew each pair's true
correlation 1,000 times from a normal distribution with the observed correlation as its mean and its
standard error as its standard deviation, and report medians across draws. We obtained 95%
confidence intervals from a cluster bootstrap over scales (1,000 replicates), in which each replicate
resampled the scales of a study with replacement, kept all pairs among the resampled scales and drew
one set of true correlations. To
examine short queries, we drew up to 10 random sets of 1, 2, 3 or 5 items from each scale, scored
each set against all other scales in the same study, and compared accuracy and the share of related
scales shown with those of the full scales the items were drawn from. To examine missing keying, we
treated a random quarter, half, three quarters or all of the reverse-keyed items of the query scale as
positively keyed (50 draws per share) and recomputed the displayed values against the database scales
with their documented keying, searching each pair in both directions. To approximate sources that
record no keying, we repeated this with the keys removed from both scales of every pair.

## Caveats

- Each study contains only about 45 strongly related pairs, so shares for true correlations of .70
  or more are estimated from few pairs; their confidence intervals are correspondingly wide (e.g.,
  .88 [.65, 1.00] shown at .30 in the validation study).
- The share of related scales shown does not depend on the size of the database, but the number of
  other results a user has to screen does. This analysis does not estimate that number; the engine
  reports it for each query.
