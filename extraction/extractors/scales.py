"""ScaleExtractor — extract the hierarchical scale/subscale structure of a questionnaire."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
from pydantic import BaseModel

from ..models import RawSurvey, Scale, Scales, resolve_scales
from ..text_utils import Tee, extract_json, lev_dist

from ._render import (
    count_scale_nodes,
    dump_result_to_log,
    flatten_scale_names,
)
from .base import Extractor


SYSTEM_PROMPT = (
    "You are a precise data-extraction assistant specialising in "
    "psychological assessments."
)

USER_PROMPT = """\
Analyse the provided PDF pages and extract the complete scale structure of the questionnaire, survey, or test.

## What to extract
A questionnaire groups its items into named scales and subscales. Extract this hierarchy — do NOT extract the individual items or their wording.

For each scale or subscale, provide:
- `name`: The label used for the scale in the document, in Title Case.
  - Use the label exactly as it appears in the document when an overt heading exists (e.g. "Emotional Attachment Measure", "Anxiety Subscale").
  - If no overt heading exists but a legend, scoring key, or footnote names the subscale (e.g. "r = reactions subscale"), derive the name from that wording in Title Case (e.g. "Reactions Subscale").
- `construct`: The underlying psychological construct the scale measures (e.g. "Depression", "Anxiety", "Extraversion").
  - Use the label stated explicitly in the document when one exists.
  - If the `name` is already the name of the psychological construct (e.g., "Ability Utilization"), the `construct` should mirror or closely match it. Do not look for a longer explanation to fill this field.
  - If it is not stated but can be reasonably inferred from context (item content, scale description, or established instrument names), infer it.
  - Set to null only when the construct is genuinely unknowable from the available information.
- `subscales`: A list of nested subscales. **Required at every node.**
  - If the scale is a PARENT (groups named or implied sub-components), list its children here, recursively.
  - If the scale is a LEAF (no sub-components), emit an empty list: `"subscales": []`.
  - Never collapse a parent and its first/only child into a single flat scale. If the document identifies a questionnaire-level grouping with named subscales, that grouping is a parent node with the subscales as children — not a single scale carrying the construct of one of the children.
  - Nesting may be arbitrarily deep — represent the full hierarchy.

Do not emit any other fields per scale. Numeric IDs are assigned automatically downstream — do not include them.

## Evidence checklist for subscales

Before deciding the tree is flat, scan the document carefully for any of these cues. ANY of them implies a parent-with-subscales structure:

(a) **Per-item code or letter prefixes paired with a legend.** Items tagged with short codes like `r`, `e`, `rR`, `A1`, `EXT` on each line, accompanied (often in a footnote, header, or scoring key) by a key that maps each code to a subscale name (e.g. "Note. r = reactions subscale, e = events subscale"). The legend is often in tiny print — read footnotes and scoring-key boxes deliberately, even when they look like fine print.
(b) **Scoring instructions that reference per-subscale totals or per-subscale reversal.** Phrases like "sum items 1, 4, 7 for the Anxiety subscale", "reverse-score items belonging to the Reactions subscale", or "subscale scores are computed by …".
(c) **Section headers, blank lines, or typographic groupings** that partition items into named blocks (e.g. an italicised "Emotional Distress" header above a run of items, then a "Functional Impairment" header above the next run).

When any cue fires, `scales` MUST contain a parent node with one named child per cue group. First fill the `subscale_evidence` field with one short sentence per observed cue (or "none observed"); then emit `scales` consistently with that evidence.

## Items not assignable to a scale
Some questionnaires include items that won't fit into any scale of the tree you produce. Two kinds:
- items with no psychometric content — demographic questions (age, gender, education), administrative items, doctor's / examiner's notes, plain instructions;
- psychometric single-item measures embedded alongside the scaled questionnaire ("orphan" items — they ask about a feeling, attitude, behaviour, belief, or symptom but do not belong to any of the scales you identify);
- items from an **initial, draft, or earlier version** of the questionnaire appended to the same document (candidate or pilot item pools). Do NOT invent a scale to hold them.

In either case: do NOT invent a scale to hold such items, and set `has_unscaled_items` to true. The downstream instrument extractor classifies each one into the appropriate bucket.

## Example 1 — overtly-named subscales

A document titled "Academic Self-Efficacy Battery" is organised into three named subscales — "Math Self-Efficacy", "Verbal Self-Efficacy", and "Science Self-Efficacy". The questionnaire-level title is a PARENT, not a flat scale:

{
  "subscale_evidence": "Three section headers split the items into Math, Verbal, and Science blocks.",
  "scales": [
    {
      "name": "Academic Self-Efficacy Battery",
      "construct": "Academic Self-Efficacy",
      "subscales": [
        { "name": "Math Self-Efficacy Scale", "construct": "Math Self-Efficacy", "subscales": [] },
        { "name": "Verbal Self-Efficacy Scale", "construct": "Verbal Self-Efficacy", "subscales": [] },
        { "name": "Science Self-Efficacy Scale", "construct": "Science Self-Efficacy", "subscales": [] }
      ]
    }
  ],
  "has_unscaled_items": false
}

## Example 2 — subscales encoded by per-item codes + legend footnote

A hypothetical document titled "Workplace Resilience Inventory" has no section headers, but every item line is prefixed with `t` or `b` (sometimes followed by `R` for reverse), and a footnote at the bottom reads "Note. t = thoughts subscale, b = behaviors subscale; R = reverse item for scoring." The codes + legend mean the questionnaire has two subscales, named from the legend wording:

{
  "subscale_evidence": "Per-item t/b/tR/bR prefixes plus a footnote 't = thoughts subscale, b = behaviors subscale' partition items into two subscales.",
  "scales": [
    {
      "name": "Workplace Resilience Inventory",
      "construct": "Workplace Resilience",
      "subscales": [
        { "name": "Thoughts Subscale", "construct": "Thoughts", "subscales": [] },
        { "name": "Behaviors Subscale", "construct": "Behaviors", "subscales": [] }
      ]
    }
  ],
  "has_unscaled_items": false
}

If any subscale itself had further sub-components, those would be listed in its own `subscales` array, recursively.

## What NOT to do
- Do not use scale abbrevations as scale names.
- Do not include versions in scale names (e.g., avoid "--Revised Version" suffixes).
- Do not fabricate scale names that are absent from the document and cannot be clearly inferred from headings, legends, or scoring keys.
- Do not include contextual modifiers, target populations, or situational constraints in the `construct_name` (e.g., avoid suffixes/prefixes like 'in elderly populations', 'in school settings').
- Do not extract individual item texts.
- If the questionnaire has no discernible scale structure (no headers, no legend, no scoring instructions referencing subscales), set `subscale_evidence` to "none observed", return `scales: []` (or a single flat scale for the whole instrument, if appropriate), and set `has_unscaled_items` as appropriate.
"""


def _tree_similarity(scales_a: list[Scale], scales_b: list[Scale]) -> float:
    """Tree structure overlap score in ``[0, 1]``.

    Names are compared with Levenshtein distance; children are matched
    greedily by descending similarity to penalise structural differences.
    """
    if not scales_a and not scales_b:
        return 1.0
    if not scales_a or not scales_b:
        return 0.0

    n, m = len(scales_a), len(scales_b)
    sim = np.zeros((n, m))
    for i, sa in enumerate(scales_a):
        for j, sb in enumerate(scales_b):
            name_sim = 1.0 - lev_dist(sa.scale_name or "", sb.scale_name or "")
            child_sim = _tree_similarity(sa.subscales or [], sb.subscales or [])
            sim[i, j] = 0.5 * name_sim + 0.5 * child_sim

    used_i: set[int] = set()
    used_j: set[int] = set()
    total = 0.0
    for score, i, j in sorted(
        ((sim[i, j], i, j) for i in range(n) for j in range(m)), reverse=True
    ):
        if i not in used_i and j not in used_j:
            total += score
            used_i.add(i)
            used_j.add(j)

    return total / max(n, m)


class ScaleExtractor(Extractor):
    name = "scales"
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT
    result_model = Scales

    def llm_schema(self) -> dict:
        return RawSurvey.inlined_schema()

    def parse_response(
        self,
        content_text: str,
        context: dict[str, BaseModel] | None = None,
    ) -> Scales:
        raw = RawSurvey.model_validate_json(extract_json(content_text))
        return resolve_scales(raw)

    def signal(self, result: Scales) -> list[Scale]:
        return result.scales

    def reliability(self, signals: list[list[Scale]]) -> tuple[str, dict[str, float]]:
        """Pairwise tree-structure similarity across runs."""
        out = Tee()
        n = len(signals)
        out.write("\n" + "=" * 60)
        out.write(f"RELIABILITY ANALYSIS  ({n} runs)")
        out.write("=" * 60)

        counts = [count_scale_nodes(t) for t in signals]
        out.write("\nTotal scale-node counts per run:")
        for i, c in enumerate(counts, 1):
            out.write(f"  Run {i}: {c} node(s)")
        if len(set(counts)) == 1:
            out.write("  → All runs agree on node count.")
        else:
            out.write(f"  → Count mismatch! range = [{min(counts)}, {max(counts)}]")

        scores: dict[str, float] = {}

        if n < 2:
            out.write("\n(Need at least 2 runs for similarity comparisons.)")
            out.write("=" * 60 + "\n")
            return out.dump(), scores

        pairs = list(combinations(range(n), 2))
        count_agreement = sum(counts[i] == counts[j] for i, j in pairs) / len(pairs)
        scores["count_agreement"] = count_agreement
        out.write(f"\nCount agreement (pair-exact): {count_agreement:.4f}")

        out.write("\nPairwise tree-structure similarities (Levenshtein-weighted names):")
        pair_sims = [_tree_similarity(signals[i], signals[j]) for i, j in pairs]
        for (i, j), sim_score in zip(pairs, pair_sims):
            out.write(f"  Run {i+1} vs Run {j+1}: similarity={sim_score:.4f}")

        if pair_sims:
            arr = np.array(pair_sims)
            scores["tree_overlap"] = float(arr.mean())
            out.write("\nOverall tree-overlap (mean pairwise similarity):")
            out.write(f"  {arr.mean():.4f} ± {arr.std():.4f}")
        else:
            scores["tree_overlap"] = float("nan")
            out.write("\n(No pairs — overall statistics unavailable.)")

        out.write("=" * 60 + "\n")
        return out.dump(), scores

    def suite_reliability(self, all_signals: list[list[list[Scale]]]) -> str:
        """Mean ± SD of per-document tree-overlap across the suite."""
        per_doc_overlaps: list[float] = []
        for doc_signals in all_signals:
            if len(doc_signals) < 2:
                continue
            pair_sims = [
                _tree_similarity(doc_signals[i], doc_signals[j])
                for i, j in combinations(range(len(doc_signals)), 2)
            ]
            if pair_sims:
                per_doc_overlaps.append(float(np.mean(pair_sims)))

        if not per_doc_overlaps:
            return "  Suite tree overlap: no multi-run documents available."
        arr = np.array(per_doc_overlaps)
        return (
            f"  Mean tree overlap across docs: {arr.mean():.4f} ± {arr.std():.4f}"
            f"  (range [{arr.min():.4f}, {arr.max():.4f}])"
        )

    def format_run(
        self,
        result: Scales,
        run_idx: int,
        num_runs: int,
        run_elapsed: float,
        log: Callable[[str], None],
    ) -> None:
        n_top = len(result.scales)
        n_total = len(flatten_scale_names(result.scales))
        log(
            f"  Run {run_idx + 1}: {run_elapsed:.2f}s  "
            f"({n_top} top-level scale(s), {n_total} total)"
        )
        dump_result_to_log(
            result, log,
            stdout_header=f"Extracted scale structure ({n_top} top-level, {n_total} total):",
            log_header=f"  Scale structure ({n_top} top-level, {n_total} total):",
        )
