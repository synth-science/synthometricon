# Reverse-keyed item pooling

How the pool stage builds `item_pooled_*` (`assemble/pool.py::keyed_centroid`). It mirrors scale scoring: reverse the reverse-keyed items, then sum with equal weight.

1. Unit-normalize every item vector. The encoder is trained so that the *angle* between item vectors predicts their correlation; length carries no information.
2. Negate every item documented as reverse-keyed.
3. Average.

## Why negation

cos(−v, w) = −cos(v, w), which is exactly what reverse-scoring does to an item's correlations (−.36 → +.36). Only the sign changes, never the magnitude. It uses only the model's predictions for the item as written, and trusts the documented keying the way a scoring key is trusted.

## Interpretation

With unit items whose cosines approximate correlations, the cosine between two pooled vectors equals the correlation between two ±1-weighted composites of standardized items (the scale-level estimator of Hommel & Arslan, 2025). Read "cosine between pooled scale vectors" as "predicted correlation between scale scores".

## Cost

The target is empirical correlation, not semantic opposition: "I talk a lot" vs. "I am quiet" correlate around −.4, not −.8. A flipped reverse item is therefore a moderately loading item and dilutes the pooled vector like a moderate item dilutes a real scale. The alternatives are worse: dropping reverse items discards information (as the pre-2026-09-18 hyperplane reflection did), and leaving them unflipped points them at the wrong pole.

## Notes

- No special cases: scales without reverse items get the plain mean of unit vectors; all-reverse scales point at the pole the scale label names.
- Compare pooled vectors by cosine and don't re-normalize: the length reflects the average keyed inter-item correlation (predicted internal consistency).
- `keying_disagreements_*` counts items whose keyed vector points against the keyed average of the other items — the analogue of a negative corrected item-total correlation. It never changes pooling but ships in the release. It flags ~31 % of reverse-keyed and ~1 % of positive items, reflecting the encoder's weak negative predictions rather than documentation errors; treat it as a review pointer and use a cut stricter than zero for likely keying errors.
- Valid only for an encoder whose cosine approximates a *signed* correlation (`surveybot3000`). For a generic semantic model, negating a reverse item is wrong — don't add one to `encode.item_models` without revisiting pooling.
- Robustness to wrong keys (single-flip displacement, dose–response, search stability, corpus-wide Monte Carlo, permutation test against same-count alternative keys) was measured in `analyses/reverse_coding_perturbation.ipynb`.
