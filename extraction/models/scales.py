"""Scales: result and wire models for the scale-structure stage."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._schema import inline_schema


# Result models (persisted).

class Scale(BaseModel):
    """A scale or subscale within a questionnaire."""

    id: Optional[int] = Field(
        default=None,
        description="Auto-assigned sequential ID (DFS order).",
    )
    scale_name: Optional[str] = Field(
        description="Name of the scale or subscale as it appears in the document."
    )
    construct_name: Optional[str] = Field(
        default=None,
        description=(
            "The psychological construct this scale measures. "
            "If not explicitly named in the document, infer it from context. "
            "Set to null only if the construct is genuinely unknowable."
        ),
    )
    subscales: Optional[List["Scale"]] = Field(
        default=None,
        description="Nested subscales, if any. Omit or set to null for leaf scales.",
    )


Scale.model_rebuild()


class Survey(BaseModel):
    """The scale structure of a questionnaire."""

    scales: List[Scale] = Field(description="Top-level scales of the questionnaire.")
    has_unscaled_items: bool = Field(
        default=False,
        description=(
            "True if the questionnaire contains items that won't fit into any "
            "scale of this tree — either items without psychometric content "
            "(demographic, administrative, notes) OR psychometric single-item "
            "measures embedded alongside the scaled questionnaire (orphans). "
            "Do NOT invent a scale to hold such items."
        ),
    )

    @model_validator(mode="after")
    def assign_sequential_scale_ids(self) -> "Survey":
        counter = 1

        def walk(scales: List[Scale]) -> None:
            nonlocal counter
            for scale in scales:
                scale.id = counter
                counter += 1
                if scale.subscales:
                    walk(scale.subscales)

        walk(self.scales)
        return self

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


# Alias used across extractors and tests.
Scales = Survey


# Wire models (LLM output).

class RawScale(BaseModel):
    # ``construct`` would shadow BaseModel.construct, so it is only the wire alias.
    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = Field(
        description="Name of the scale or subscale as it appears in the document."
    )
    construct_name: Optional[str] = Field(
        default=None,
        alias="construct",
        description=(
            "The psychological construct this scale measures. "
            "If not explicitly named in the document, infer it from context. "
            "Set to null only if the construct is genuinely unknowable."
        ),
    )
    subscales: List["RawScale"] = Field(
        default_factory=list,
        description=(
            "Nested subscales of this scale. "
            "If this scale is a PARENT (groups named or implied sub-components), "
            "emit them all here. "
            "If this scale is a LEAF (no sub-components), emit an empty list: []. "
            "Never collapse a parent and its children into a single node."
        ),
    )


RawScale.model_rebuild()


class RawSurvey(BaseModel):
    # subscale_evidence must stay first: see docs/wire-model-conventions.md (chain-of-thought slot).
    subscale_evidence: str = Field(
        description=(
            "One short sentence per observed structural cue: per-item code/letter "
            "prefixes paired with a legend, scoring instructions referencing per-"
            "subscale totals or per-subscale reversal, section headers or "
            "typographic groupings that partition items. Write 'none observed' if "
            "the document genuinely has no hierarchy. If any cue is present, "
            "`scales` MUST contain a parent with the named subscales as children."
        )
    )
    scales: List[RawScale] = Field(
        description="Top-level scales of the questionnaire."
    )
    has_unscaled_items: bool = Field(
        default=False,
        description=(
            "True if the questionnaire contains items that won't fit into any "
            "scale above — either non-psychometric items (demographic, "
            "administrative) OR psychometric single-item measures (orphans)."
        ),
    )

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


def resolve_scales(raw: RawSurvey) -> Survey:
    """Convert wire ``RawSurvey`` to ``Survey`` (drops ``subscale_evidence``)."""

    def to_scale(r: RawScale) -> Scale:
        return Scale(
            scale_name=r.name,
            construct_name=r.construct_name,
            subscales=[to_scale(s) for s in r.subscales] if r.subscales else None,
        )

    return Survey(
        scales=[to_scale(s) for s in raw.scales],
        has_unscaled_items=raw.has_unscaled_items,
    )
