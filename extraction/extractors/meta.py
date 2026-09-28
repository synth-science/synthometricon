"""MetaExtractor — document metadata from page-1 regex + an LLM call (``RawMeta``).

See docs/extractors.md#meta.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Callable

import numpy as np
from ..models import (
    Items,
    Meta,
    RawMeta,
    Scales,
    resolve_meta,
)
from ..text_utils import Tee, extract_json

from ._render import dump_result_to_log, render_items_block, render_scales_block
from .base import Extractor


SYSTEM_PROMPT = (
    "You are a precise data-extraction assistant specialising in "
    "psychological assessments."
)

USER_PROMPT_BODY = """\
You are given the PDF pages of a questionnaire along with the items already extracted from it (with IDs) and the scale structure already extracted (with IDs).

Your tasks:
1. Identify the **language** of the original document body text and return it as an ISO 639-1 code (two lowercase letters; e.g. 'en', 'de', 'fr', 'si'). If the title names a translated version (e.g. "Italian Version"), still emit the ISO code of the document body — not the title-named language.
2. Determine whether the document is a **clinical intake / medical history form** rather than a psychometric instrument with scorable scales.
3. Determine whether the instrument is an **objective measure** — i.e. an ability test, cognitive test, knowledge test, or any measure where participant responses have objectively correct or incorrect answers.

## Output rules
- `language`: required. Always lowercase two-letter ISO 639-1.
- `intake_form`: set to `true` ONLY if the document is a medical history form, patient intake form, or case history questionnaire that collects clinical/biographical information without psychometrically scorable scales. If the instrument contains any scales that are psychometrically scored (sum scores, subscale scores, Likert ratings), omit this field entirely. When in doubt, omit.
- `objective_measure`: set to `true` ONLY if the instrument is an ability test, cognitive test, knowledge test, or any measure where responses can be objectively scored as correct or incorrect. Omit for self-report questionnaires, rating scales, or any subjective measure. When in doubt, omit.
"""


# Page-1 regexes, tuned to the APA PsycTests template; best-effort (None on no match).

_CITATION_HEADER_RE = re.compile(r"(?:APA\s+)?PsycTESTS\s+Citation:", re.IGNORECASE)
_INSTRUMENT_TYPE_HEADER_RE = re.compile(r"Instrument Type:\s*\n\s*([^\n]+)")

_DOI_RE = re.compile(
    r"doi:\s*https?://(?:dx\.)?doi\.org/(10\.1037/t\d{4,6}-\d{3})",
    re.IGNORECASE,
)
_YEAR_IN_PARENS_RE = re.compile(r"\(\s*((?:19|20)\d{2})\s*\)")
_AUTHORS_BEFORE_YEAR_RE = re.compile(r"^(.+?)\(\s*(?:19|20)\d{2}\s*\)", re.DOTALL)
_TEST_FORMAT_RE = re.compile(
    r"Test Format:\s*\n(.+?)(?=\n[A-Z][a-zA-Z ]+:\s*\n)", re.DOTALL
)
_SOURCE_RE = re.compile(
    r"Source:\s*\n(.+?)(?=\n[A-Z][a-zA-Z ]+:\s*\n)", re.DOTALL
)
_ORIGINAL_PUBLICATION_RE = re.compile(
    r"Original Publication:\s*\n(.+?)(?=\n[A-Z][a-zA-Z ]+:\s*\n)", re.DOTALL
)
_PERMISSIONS_RE = re.compile(
    r"Permissions:\s*\n(.+?)(?=\n(?:[A-Z][a-zA-Z ]+:\s*\n|PsycTESTS™|APA PsycTests®|\Z))",
    re.DOTALL,
)
# "<Adjective> Version"; only accepted if the adjective is in _LANGUAGE_NAMES.
_VERSION_QUALIFIER_RE = re.compile(r"([A-Z][a-zA-Z]+)\s+Version\b")
_LANGUAGE_NAMES = frozenset(
    s.lower() for s in (
        "Arabic Bengali Bulgarian Burmese Catalan Chinese Croatian Czech "
        "Danish Dutch English Estonian Farsi Filipino Finnish French Gaelic "
        "Galician German Greek Gujarati Hebrew Hindi Hungarian Icelandic "
        "Indonesian Irish Italian Japanese Kannada Khmer Korean Lao Latvian "
        "Lithuanian Malay Malayalam Marathi Mongolian Norwegian Persian "
        "Polish Portuguese Punjabi Romanian Russian Serbian Sinhala Slovak "
        "Slovenian Spanish Swahili Swedish Tagalog Tamil Telugu Thai Turkish "
        "Ukrainian Urdu Vietnamese Welsh"
    ).split()
)
# "<Journal>, [Vol ]<vol>(<issue>), <pages>" — deliberately broad; stored as raw text.
_VENUE_RE = re.compile(
    r"([A-Z][^.]*?,\s*(?:Vol\s+)?\d+(?:\(\d+\))?(?:[,\s]+\d+[-–—]\d+)?)",
    re.DOTALL,
)


_REGEX_FIELDS: tuple[str, ...] = (
    "title_raw",
    "doi_raw",
    "instrument_type_raw",
    "publication_year_raw",
    "authors_raw",
    "journal_venue_raw",
    "test_format_raw",
    "source_raw",
    "permissions_raw",
    "language_raw",
)


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _extract_regex_fields(text: str) -> dict:
    """Apply the page-1 regexes; returns every ``_REGEX_FIELDS`` key (None if unmatched)."""
    out: dict = dict.fromkeys(_REGEX_FIELDS)

    stripped = text.strip()
    if not stripped:
        return out

    # Title: first non-empty line, skipping a leading "Note:" line.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        if first.lower().startswith("note:") and len(lines) > 1:
            first = lines[1]
        out["title_raw"] = first

    # First match is the PsycTESTS record DOI.
    if (doi_m := _DOI_RE.search(text)):
        out["doi_raw"] = doi_m.group(1)

    if (it_m := _INSTRUMENT_TYPE_HEADER_RE.search(text)):
        out["instrument_type_raw"] = it_m.group(1).strip()

    # Year + authors come from the citation block (header .. "Instrument Type:").
    cit_start = _CITATION_HEADER_RE.search(text)
    cit_end = _INSTRUMENT_TYPE_HEADER_RE.search(text)
    if cit_start and cit_end and cit_end.start() > cit_start.end():
        cit_slice = text[cit_start.end():cit_end.start()]
        if (year_m := _YEAR_IN_PARENS_RE.search(cit_slice)):
            try:
                out["publication_year_raw"] = int(year_m.group(1))
            except ValueError:
                pass
        if (auth_m := _AUTHORS_BEFORE_YEAR_RE.search(cit_slice.lstrip())):
            authors = _collapse_ws(auth_m.group(1)).rstrip(",").strip()
            if authors:
                out["authors_raw"] = authors

    if (m := _TEST_FORMAT_RE.search(text)):
        out["test_format_raw"] = _collapse_ws(m.group(1))
    if (m := _SOURCE_RE.search(text)):
        out["source_raw"] = _collapse_ws(m.group(1))
    if (m := _PERMISSIONS_RE.search(text)):
        out["permissions_raw"] = _collapse_ws(m.group(1))

    # Venue is in Source:, or in Original Publication: when Source: is "Supplied by author."
    orig_pub_m = _ORIGINAL_PUBLICATION_RE.search(text)
    orig_pub_raw = _collapse_ws(orig_pub_m.group(1)) if orig_pub_m else ""
    for candidate in (out["source_raw"], orig_pub_raw):
        if not candidate:
            continue
        if (venue_m := _VENUE_RE.search(candidate)):
            out["journal_venue_raw"] = _collapse_ws(venue_m.group(1))
            break

    # Language hint from the title only.
    if out["title_raw"]:
        for m in _VERSION_QUALIFIER_RE.finditer(out["title_raw"]):
            candidate = m.group(1)
            if candidate.lower() in _LANGUAGE_NAMES:
                out["language_raw"] = candidate
                break

    return out


class MetaExtractor(Extractor):
    name = "meta"
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT_BODY
    result_model = Meta
    depends_on = ["items", "scales"]
    needs_first_page_text = True
    needs_doc_stats = True

    def llm_schema(self) -> dict:
        return RawMeta.inlined_schema()

    def parse_response(
        self,
        content_text: str,
        context: dict | None = None,
    ) -> Meta:
        if not context or "_first_page_text" not in context:
            raise RuntimeError(
                "MetaExtractor.parse_response requires '_first_page_text' "
                "injected by orchestrator in context."
            )
        raw = RawMeta.model_validate_json(extract_json(content_text))
        regex_data = _extract_regex_fields(context["_first_page_text"])
        doc_stats = context.get("_doc_stats")
        return resolve_meta(raw, regex_data, doc_stats)

    def build_user_prompt(self, context: dict | None = None) -> str:
        if not context or "items" not in context or "scales" not in context:
            return self.user_prompt
        items: Items = context["items"]
        scales: Scales = context["scales"]
        return (
            f"{self.user_prompt}\n"
            f"## Items (id. text)\n{render_items_block(items)}\n\n"
            f"## Scales (indented tree with [id=N])\n"
            f"{render_scales_block(scales)}\n"
        )

    def signal(self, result: Meta) -> dict:
        return {
            "language": result.language,
            "intake_form": result.intake_form,
            "objective_measure": result.objective_measure,
        }

    def reliability(self, signals: list[dict]) -> tuple[str, dict[str, float]]:
        out = Tee()
        n = len(signals)
        out.write("\n" + "=" * 60)
        out.write(f"RELIABILITY ANALYSIS  ({n} runs)")
        out.write("=" * 60)

        out.write("\nLanguage per run:")
        for i, sig in enumerate(signals, 1):
            out.write(f"  Run {i}: {sig['language']}")

        out.write("\nIntake form per run:")
        for i, sig in enumerate(signals, 1):
            out.write(f"  Run {i}: {sig['intake_form']}")

        out.write("\nObjective measure per run:")
        for i, sig in enumerate(signals, 1):
            out.write(f"  Run {i}: {sig['objective_measure']}")

        scores: dict[str, float] = {}

        if n < 2:
            out.write("\n(Need at least 2 runs for reliability measures.)")
            out.write("=" * 60 + "\n")
            return out.dump(), scores

        pairs = list(combinations(range(n), 2))
        lang_agreement = sum(
            signals[i]["language"] == signals[j]["language"] for i, j in pairs
        ) / len(pairs)
        scores["language_agreement"] = lang_agreement
        out.write(f"\nLanguage agreement (pair-exact): {lang_agreement:.4f}")

        intake_agreement = sum(
            signals[i]["intake_form"] == signals[j]["intake_form"] for i, j in pairs
        ) / len(pairs)
        scores["intake_form_agreement"] = intake_agreement
        out.write(f"Intake-form agreement (pair-exact): {intake_agreement:.4f}")

        obj_agreement = sum(
            signals[i]["objective_measure"] == signals[j]["objective_measure"]
            for i, j in pairs
        ) / len(pairs)
        scores["objective_measure_agreement"] = obj_agreement
        out.write(f"Objective-measure agreement (pair-exact): {obj_agreement:.4f}")

        out.write("=" * 60 + "\n")
        return out.dump(), scores

    def suite_reliability(self, all_signals: list[list[dict]]) -> str:
        per_doc_lang: list[float] = []
        per_doc_intake: list[float] = []
        per_doc_obj: list[float] = []
        for doc_signals in all_signals:
            if len(doc_signals) < 2:
                continue
            pairs = list(combinations(range(len(doc_signals)), 2))
            per_doc_lang.append(
                sum(
                    doc_signals[i]["language"] == doc_signals[j]["language"]
                    for i, j in pairs
                ) / len(pairs)
            )
            per_doc_intake.append(
                sum(
                    doc_signals[i]["intake_form"] == doc_signals[j]["intake_form"]
                    for i, j in pairs
                ) / len(pairs)
            )
            per_doc_obj.append(
                sum(
                    doc_signals[i]["objective_measure"] == doc_signals[j]["objective_measure"]
                    for i, j in pairs
                ) / len(pairs)
            )

        if not per_doc_lang:
            return "  Suite meta reliability: no multi-run documents available."

        lang_arr = np.array(per_doc_lang)
        intake_arr = np.array(per_doc_intake)
        obj_arr = np.array(per_doc_obj)
        return (
            f"  Mean language agreement across docs: {lang_arr.mean():.4f} ± "
            f"{lang_arr.std():.4f}  (range [{lang_arr.min():.4f}, {lang_arr.max():.4f}])\n"
            f"  Mean intake-form agreement across docs: {intake_arr.mean():.4f} ± "
            f"{intake_arr.std():.4f}  (range [{intake_arr.min():.4f}, {intake_arr.max():.4f}])\n"
            f"  Mean objective-measure agreement across docs: {obj_arr.mean():.4f} ± "
            f"{obj_arr.std():.4f}  (range [{obj_arr.min():.4f}, {obj_arr.max():.4f}])"
        )

    def format_run(
        self,
        result: Meta,
        run_idx: int,
        num_runs: int,
        run_elapsed: float,
        log: Callable[[str], None],
    ) -> None:
        flags = []
        if result.intake_form:
            flags.append("intake_form")
        if result.objective_measure:
            flags.append("objective_measure")
        flag_tag = f", {', '.join(flags)}" if flags else ""
        log(f"  Run {run_idx + 1}: {run_elapsed:.2f}s  (language={result.language}{flag_tag})")
        dump_result_to_log(
            result, log,
            stdout_header=f"Extracted meta (language={result.language}{flag_tag}):",
            log_header="  Meta:",
        )
