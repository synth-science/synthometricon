"""ItemExtractor — extract questionnaire items verbatim from PDF pages."""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import krippendorff
import numpy as np
import pandas as pd
import pingouin as pg
from pydantic import BaseModel

from ..models import Items, RawItems, resolve_items
from ..text_utils import Tee, extract_json, lev_dist

from ._render import dump_result_to_log
from .base import Extractor


SYSTEM_PROMPT = (
    "You are a precise data-extraction assistant specialising in "
    "psychological assessments."
)

USER_PROMPT = """\
Extract EVERY questionnaire item visible across the provided PDF pages, in page order, producing the verbatim wording shown to respondents.

## General rules
- Preserve original capitalisation and punctuation.
- **`text` MUST always be in English.** If the original is not in English, translate it and put ONLY the English translation in `text` (never the source-language wording). Then ALSO set `lang` to the ISO 639-1 code of the original language (e.g. `de` for German) to record what it was translated from. The `lang` flag does NOT replace translation — it is metadata recorded alongside the already-translated English `text`. For items already in English, leave `text` as-is and omit `lang` entirely.
- Do NOT extract section headers, instructions, demographic questions, item numbers, or other peripheral text — unless they are themselves scored items.

## What counts as item text
The item text is the verbatim statement, question, or prompt the respondent reads BEFORE choosing a response. Extract only this as `text`.

Do NOT treat the following as item text, even when they are the only text visible for an item:
- Semantic-differential endpoints (e.g., "Happy 1 2 3 4 5 Sad" — the words are anchors).
- Likert/rating-scale anchors ("Strongly disagree … Strongly agree", "Never … Always", "0 = Not at all").
- Single- or multiple-choice option labels and checkbox labels.
- Rating-matrix column/row headers describing the response dimension (frequency, intensity, agreement, etc.).
- Units, time frames, or scale instructions ("in the past 7 days", "% of the time").

Do NOT transcribe or describe non-textual stimuli (images, diagrams, symbols, audio/video cues, example faces, etc.).

Never fabricate, paraphrase, or infer a stem from context, the construct name, or response labels. When an item has only a non-textual stimulus (e.g. an image with no text and no labeled options), do NOT skip it — see "Image-only items" below for how to encode it.

## Administration notes
Some items contain text directed at the test administrator, not the respondent — e.g. skip/branching instructions, interviewer cues, age-gate conditions, procedural annotations. Common forms: parenthetical "(only administer if …)", bracketed "[skip if …]", prefixed "Interviewer: …", or margin notes near the item.

When present, extract this text into `note` and strip it from `text` so `text` contains only what the respondent reads. If no administration note is present, omit `note` entirely.

## Image-only items
Some instruments present items whose entire stimulus is an image — a picture, drawing, photograph, face, scene, symbol, or diagram — with NO verbatim stem and NO labeled response options. Respondents rate the image using a shared scale defined elsewhere in the document. Each distinct image stimulus is ONE item.

For each such item, emit a single object with `img: true` and OMIT both `text` and `options`. The `img: true` flag (together with the absent `text`) IS the encoding — do NOT fabricate, paraphrase, or describe the image in `text`, and do NOT invent option labels.

When an item carries BOTH a verbatim stem AND a paired image (e.g. a stem question accompanied by a picture), emit `text` with the verbatim stem and also set `img: true`.

## Response options: when to capture them
Capture an item's response options in `options` (a list of strings, in scoring order — low score to high score) in exactly these cases:

1. **Choice items (stem present).** The respondent SELECTS one (or several) from an enumerated, item-specific answer set — e.g. a multiple-choice question with labeled options, or a checkbox/select-all list. Emit `text` with the stem AND `options` with the selectable answers, and set `type: "choice"`. Example: `If 4 oranges cost 20¢, what is the cost of 1 orange? (a) 4¢ (b) 5¢ (c) 6¢ (d) 10¢` → `{"text": "If 4 oranges cost 20¢, what is the cost of 1 orange?", "type": "choice", "options": ["4¢", "5¢", "6¢", "10¢"]}`. Strip leading option markers (a/b/c, 1/2/3) — keep only the answer label.

2. **Labeled multi-choice with no stem.** Each item is a number (e.g., "1.", "2.") followed by a vertical list of fully-spelled response-option labels (often re-numbered 0/1/2 …):
     `1.  0 I don't feel discouraged about my future.`
     `    1 I sometimes feel discouraged about my future.`
     `    2 I often feel discouraged about my future.`
   This is ONE choice item with three options, NOT three items. Emit `{"type": "choice", "options": ["I don't feel discouraged about my future.", "I sometimes feel discouraged about my future.", "I often feel discouraged about my future."]}` and OMIT `text`. Never split the option labels into separate items.

3. **Semantic differential (no stem).** Each line is a pair of bipolar adjective anchors with a shared numeric rating scale between them:
     `Intelligent 1 2 3 4 5 6 7 Unintelligent`
     `Untrained 1 2 3 4 5 6 7 Trained`
   Each line is ONE item. Emit `{"options": ["Intelligent", "Unintelligent"]}` and `{"options": ["Untrained", "Trained"]}`. Do NOT emit a `text` field, and do NOT set `type` (a semantic differential is a rating scale). The two adjectives are the entire per-item content; the 1-7 numbers are the shared scale.

Do NOT put scale-shared rating anchors (e.g. "Strongly disagree … Strongly agree", "Never … Always", "0 = Not at all") into `options`. Those anchors describe the whole scale, not the per-item content, and belong nowhere in this output — the item stays a plain rating-scale item.

**Choice options vs. fill-in-the-blank fillers (critical).** A list of candidate words/phrases near an item is one of two very different things:
- If the respondent picks ONE (or several) of them as the answer to a SINGLE question, they are **choice options** → set `type: "choice"` and put them in `options`.
- If each candidate would turn the stem into a DIFFERENT rated statement that the respondent rates SEPARATELY, they are **fill-in-the-blank fillers** → expand into separate items (see "Fill-in-the-blank template" below), one per filler, each a plain rating-scale item. NEVER collapse fillers into a single item's `options`. Example: stem "It's important to me to be ___" with {Popular, smart, cool, good-looking} → FOUR separate rating-scale items ("It's important to me to be popular", … ), NOT one item with four options.

## Shared text across multiple items
Some instruments present one piece of text once and many per-item pieces alongside it. Resolve as follows:

- **Sentence-completion prefix.** A single opening phrase (often ending in a trailing-join marker — an ellipsis "…", a run of dots "....", a colon ":", a dash "—", or a blank "___") is followed by a numbered or bulleted list of completions, each of which is grammatically a fragment that only makes sense when read after the prefix. Prepend the prefix to every completion so item_text reads as the complete sentence the respondent sees. Examples: prefix "I am someone who…" + "is often afraid" → "I am someone who is often afraid"; prefix "I am satisfied with...." + "how the team works to be the best." → "I am satisfied with how the team works to be the best."; prefix "It bothers me when:" + "people are late" → "It bothers me when people are late". The prefix may appear ONCE at the start of the list, OR it may be repeated at the top of EVERY page as a running header above the response-scale anchors — both are the same pattern. Treat any opening phrase that ends in a trailing-join marker and is followed by lowercase-initial completions as a sentence-completion prefix, even when it sits in the page-header position; do NOT mistake it for an instruction or a section title and drop it.

- **Fill-in-the-blank template.** A template sentence has a placeholder where a word/phrase should go. Treat any of these as a placeholder: a run of two or more underscores of ANY length (`__`, `_____`, `__________` are all the same signal), a bracketed wildcard (`[X]`, `[___]`), or an inline run of dots/spaces sitting where words ought to be (e.g., "How often does your child ...... because of X?"). The moment you spot a placeholder, scan the lines immediately above and below — you WILL find a small enumerated set of fillers that all fit grammatically into the gap. The fillers may appear as a vertical list (numbered, bulleted, or plain) OR as an inline enumeration separated by commas or slashes; differences in capitalisation, extra whitespace around separators, or stray punctuation do not change this — treat it as one filler list. Expansion is MULTIPLICATIVE: emit ONE separate item PER filler, each carrying the entire template with the placeholder replaced by EXACTLY ONE filler in isolation (the other fillers do NOT appear in that item). N fillers → N items. Never stuff multiple fillers into the placeholder slot of a single item (e.g., emitting "How A, B, C are kids…?" as one item instead of three separate items is wrong and forbidden). If the same filler list applies to multiple templates on the same page, expand each template independently — N templates × M fillers = N×M total items. Lowercase each substituted filler unless it is a proper noun. Intervening labels (italicised domain names, subscale headings, formatting changes) do not break the pattern. Emitting the template with the placeholder still in it is NEVER valid; if you cannot find fillers, the placeholder is not a placeholder and you should not have flagged it. Fillers are NOT choice options: they expand into separate rating-scale items, never into one item's `options` (see "Choice options vs. fill-in-the-blank fillers" above).
- **Rating matrix with a framing stem.** The shared text is a question or instruction framing the response ("How often have you felt…", "Rate how much each describes you") and each row is the rated content. Extract the row alone — do NOT prepend the framing stem.

When combining (the first two cases), the output must read as a single natural sentence. Keep both parts verbatim except: remove any punctuation that only marked the join (ellipses, trailing colons/dashes, blank markers like "___" or "[X]"), and re-case the per-item text to fit its new position (typically lowercase its first letter, unless it is a proper noun).

**Completeness check — apply to every item before emitting it.** Verify the item reads as a complete, grammatical sentence or question that a respondent could actually read and answer on its own. Two failure signals:
  1. The item still contains a gap — a placeholder, brackets, underscores, or any inline sequence of dots/spaces sitting where words ought to be.
  2. The item is a fragment that cannot stand alone — typically a verb phrase, dependent clause, or noun phrase with no subject (e.g., something that starts with a lowercase verb and obviously continues a sentence begun elsewhere).
If either signal fires, you have missed a shared-text pattern. Look nearby — above, below, or across an intervening label — for the matching template or filler, and emit the combined sentence instead. Do NOT emit the broken item as-is.

Rule of thumb: if the shared text is part of the rated sentence itself (completing it at the end or filling a blank in the middle), combine. If it is a question/instruction wrapped around standalone rated content, don't.

## Item type
Classify each item's response format with `type`. The DEFAULT is a rating scale; emit `type` ONLY when the item is NOT a rating scale, and **omit it entirely** otherwise.
- *(default — omit `type`)* **Rating scale.** Anything quantifiable on a continuum: Likert/agreement/frequency/intensity scales, semantic differentials, and binary yes/no or true/false items.
- `type: "choice"` — the respondent SELECTS one or many categorical options (single- or multiple-choice, checkbox/select-all). Pair with `options` (the selectable answers).
- `type: "open"` — free-form input: open text, essay, drawing, or any other open response with no fixed options.
- `type: "other"` — cannot be classified as rating scale, choice, or open.

## Output format
For every item emit an object with:
- `text`: the verbatim item string. Required for stemmed items; **omit entirely** for stem-less items (semantic differential, labeled-only multi-choice, image-only).
- `type`: response format — set to `"choice"`, `"open"`, or `"other"` ONLY when the item is not a rating scale; **omit entirely** for rating-scale items (the default).
- `img`: set to `true` ONLY for items presented with an image; otherwise **omit this field entirely** (do not write `"img": false`).
- `options`: list of per-item response-option / choice / SD-anchor labels in scoring order. Emit for choice items (the selectable answers) and for stem-less SD-anchor / labeled-multi-choice items; **omit entirely** (do not write `null` or `[]`) for ordinary rating-scale items whose response anchors are scale-shared, and for image-only items.
- `note`: administration note directed at the test administrator (skip instructions, interviewer cues, age-gate conditions, procedural annotations). Emit ONLY when present; **omit entirely** otherwise.
- `lang`: ISO 639-1 code of the item's ORIGINAL language, set ONLY when you translated the item from another language into English. `text` must STILL hold the English translation — `lang` records provenance, it does not license leaving `text` untranslated. Emit ONLY for translated items; **omit entirely** for items already in English.
Every item must carry at least one of `text`, `options`, or `img: true`:
- stemmed items → `text` (and `img: true` too if the stem is paired with an image);
- stem-less items with per-item option/anchor labels → `options`;
- image-only items → `img: true` alone.
Do not emit any other fields. IDs are assigned automatically downstream — do not include them.
"""


def _krippendorff_alpha_on_texts(signals: list[list[str]], min_items: int) -> float:
    """Krippendorff's α (squared Levenshtein) on the first ``min_items`` items of every run.

    Truncating avoids missing values; count mismatches are scored by ``count_agreement``.
    """
    data = np.array([[run[k] for k in range(min_items)] for run in signals])
    # krippendorff can't infer value_domain for string data with a callable metric.
    value_domain = np.array(sorted({text for run in signals for text in run}))

    def _lev_sq(a, b, **_):
        return np.vectorize(lambda x, y: lev_dist(str(x), str(y)) ** 2)(a, b)

    return float(krippendorff.alpha(
        reliability_data=data,
        level_of_measurement=_lev_sq,
        value_domain=value_domain,
    ))


class ItemExtractor(Extractor):
    name = "items"
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT
    result_model = Items

    def llm_schema(self) -> dict:
        return RawItems.inlined_schema()

    def parse_response(
        self,
        content_text: str,
        context: dict[str, BaseModel] | None = None,
    ) -> Items:
        raw = RawItems.model_validate_json(extract_json(content_text))
        return resolve_items(raw)

    def signal(self, result: Items) -> list[str]:
        return [
            item.item_text or " | ".join(item.options or []) or ""
            for item in result.items
        ]

    def reliability(self, signals: list[list[str]]) -> tuple[str, dict[str, float]]:
        out = Tee()
        n = len(signals)
        out.write("\n" + "=" * 60)
        out.write(f"RELIABILITY ANALYSIS  ({n} runs)")
        out.write("=" * 60)

        counts = [len(r) for r in signals]
        out.write("\nItem counts per run:")
        for i, c in enumerate(counts, 1):
            out.write(f"  Run {i}: {c} items")
        if len(set(counts)) == 1:
            out.write("  → All runs agree on count.")
        else:
            out.write(f"  → Count mismatch! range = [{min(counts)}, {max(counts)}]")

        scores: dict[str, float] = {}

        if n < 2:
            out.write("\n(Need at least 2 runs for reliability measures.)")
            out.write("=" * 60 + "\n")
            return out.dump(), scores

        pairs = list(combinations(range(n), 2))
        count_agreement = sum(counts[i] == counts[j] for i, j in pairs) / len(pairs)
        scores["count_agreement"] = count_agreement
        out.write(f"\nCount agreement (pair-exact): {count_agreement:.4f}")

        min_items = min(counts)
        if min_items > 0:
            try:
                alpha = _krippendorff_alpha_on_texts(signals, min_items)
                scores["levenshtein_alpha"] = alpha
                out.write(
                    f"Krippendorff's α (Levenshtein, {min_items} aligned items): "
                    f"{alpha:.4f}"
                )
            except Exception as exc:
                scores["levenshtein_alpha"] = float("nan")
                out.write(f"Krippendorff's α: unavailable ({exc})")
        else:
            # All runs agree on 0 items — perfect reliability by definition.
            scores["levenshtein_alpha"] = 1.0
            out.write("Krippendorff's α: 1.0000 (all runs extracted 0 items)")

        out.write("=" * 60 + "\n")
        return out.dump(), scores

    def suite_reliability(self, all_signals: list[list[list[str]]]) -> str:
        """ICC(2,1) on item counts across all documents."""
        num_docs = len(all_signals)
        if num_docs < 2:
            return "  Suite ICC: requires ≥ 2 documents — skipped."

        records = [
            {"target": doc_idx, "rater": run_idx, "rating": float(len(run_signal))}
            for doc_idx, doc_signals in enumerate(all_signals)
            for run_idx, run_signal in enumerate(doc_signals)
        ]
        df = pd.DataFrame(records)
        try:
            icc = pg.intraclass_corr(
                data=df, targets="target", raters="rater", ratings="rating"
            )
            # Labels differ across pingouin versions ("ICC(A,1)"/"ICC2"; "CI95%"/"CI95").
            row = icc[icc["Type"].isin(["ICC(A,1)", "ICC2"])].iloc[0]
            ci_lo, ci_hi = row["CI95%"] if "CI95%" in row.index else row["CI95"]
            return (
                f"  ICC(2,1) on item counts: {float(row['ICC']):.4f}"
                f"  (95% CI [{ci_lo:.4f}, {ci_hi:.4f}])"
            )
        except Exception as exc:
            return f"  ICC(2,1): unavailable ({exc})"

    def format_run(
        self,
        result: Items,
        run_idx: int,
        num_runs: int,
        run_elapsed: float,
        log: Callable[[str], None],
    ) -> None:
        n = len(result.items)
        log(f"  Run {run_idx + 1}: {run_elapsed:.2f}s  ({n} items extracted)")
        dump_result_to_log(
            result, log,
            stdout_header=f"Extracted {n} item(s):",
            log_header=f"  Extracted items ({n}):",
        )
