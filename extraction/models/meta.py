"""Meta: result and wire models for the document-metadata stage (LLM + page-1 regex)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ._schema import inline_schema


# Wire model (LLM output).

class RawMeta(BaseModel):
    """Wire model — only the LLM-emitted parts of the Meta result.

    Regex-extracted fields are merged in by ``resolve_meta``.
    """

    language: str = Field(
        description=(
            "ISO 639-1 code of the document's original body language "
            "(two lowercase letters, e.g. 'en', 'de', 'si')."
        )
    )
    intake_form: Optional[Literal[True]] = Field(
        default=None,
        description=(
            "Set to true ONLY if the document is a medical history form, "
            "patient intake form, or case history questionnaire with no "
            "psychometrically scorable scales. Omit (do not emit) otherwise."
        ),
    )
    objective_measure: Optional[Literal[True]] = Field(
        default=None,
        description=(
            "Set to true ONLY if the instrument is an ability test, cognitive "
            "test, knowledge test, or any measure where participant responses "
            "have objectively correct or incorrect answers. Omit (do not emit) "
            "for self-report questionnaires, rating scales, or subjective measures."
        ),
    )

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


# Result model (persisted).

class Meta(BaseModel):
    """Document-level metadata for one PDF.

    Combines page-1 regex extractions with the LLM-emitted language code.
    Conceptually attached to the Instrument tree top-level via
    ``apply_meta_to_instrument``.
    """

    # ---- LLM-populated ----
    language: str = Field(
        description="ISO 639-1 code of the document's original language."
    )
    intake_form: bool = Field(
        default=False,
        description=(
            "True when the document is a medical history form, patient intake "
            "form, or case history questionnaire without psychometrically "
            "scorable scales."
        ),
    )
    objective_measure: bool = Field(
        default=False,
        description=(
            "True when the instrument is an ability test, cognitive test, "
            "knowledge test, or any measure with objectively correct/incorrect "
            "answers (as opposed to self-report or subjective rating scales)."
        ),
    )

    # ---- Regex-populated from page 1; all optional (any field may be absent) ----
    title_raw: Optional[str] = None
    doi_raw: Optional[str] = Field(default=None, description="e.g. '10.1037/t08009-000'.")
    source_doi: Optional[str] = Field(
        default=None,
        description=(
            "DOI of the original publication from the page-1 'Source:' block "
            "(any registrant, e.g. '10.1016/j.paid.2009.01.001'); distinct from "
            "the PsycTESTS record DOI in doi_raw."
        ),
    )
    instrument_type_raw: Optional[str] = Field(
        default=None, description="Verbatim from 'Instrument Type:' line."
    )
    publication_year_raw: Optional[int] = None
    authors_raw: Optional[str] = Field(
        default=None, description="Raw authors string from the PsycTESTS citation."
    )
    journal_venue_raw: Optional[str] = Field(
        default=None,
        description=(
            "Raw journal/source citation "
            "(e.g. 'Journal of Research in Personality, Vol 43(3), 362-373')."
        ),
    )
    test_format_raw: Optional[str] = None
    source_raw: Optional[str] = None
    permissions_raw: Optional[str] = None
    language_raw: Optional[str] = Field(
        default=None,
        description=(
            "Informational only. Set when the title explicitly names a language "
            "version (e.g. 'Italian Version', 'Sinhala Version'). Never "
            "overrides `language`."
        ),
    )

    # ---- Deterministic document stats ----
    page_count: Optional[int] = Field(
        default=None, description="Total number of pages in the PDF."
    )
    image_count: Optional[int] = Field(
        default=None, description="Total number of image XObjects across all pages."
    )
    char_count_excl_first_page: Optional[int] = Field(
        default=None,
        description="Character count of the full document text excluding page 1.",
    )

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


def resolve_meta(
    raw: RawMeta,
    regex_data: dict,
    doc_stats: dict | None = None,
) -> Meta:
    """Combine page-1 regex output, LLM-emitted language code, and document stats."""
    return Meta(
        language=raw.language,
        intake_form=bool(raw.intake_form),
        objective_measure=bool(raw.objective_measure),
        **(regex_data | (doc_stats or {})),
    )
