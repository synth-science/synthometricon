"""Items: result and wire models for the item-extraction stage."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ._schema import inline_schema


# Result models (persisted).

class Item(BaseModel):
    """An item in a survey, questionnaire or test."""

    id: Optional[int] = Field(
        default=None, description="Auto-assigned sequential ID."
    )
    item_text: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim wording of the item, or null if the item has no stem — "
            "in which case the per-item response option / SD-anchor labels are "
            "carried by `options` instead."
        ),
    )
    has_image: Optional[bool] = Field(
        default=False,
        description="True if the item is presented with an image.",
    )
    item_type: Literal["rating_scale", "choice", "open", "other"] = Field(
        default="rating_scale",
        description=(
            "Response format of the item. 'rating_scale' (default): "
            "quantifiable on a continuum — Likert/agreement/frequency scales, "
            "semantic differentials, and binary (yes/no, true/false) items. "
            "'choice': respondent selects one or many categorical options "
            "(carried in `options`). 'open': free-form input (open text, essay, "
            "drawing). 'other': none of the above."
        ),
    )
    options: Optional[List[str]] = Field(
        default=None,
        description=(
            "Per-item response-option labels in scoring order (e.g. "
            "choice options, semantic-differential anchors, or labeled "
            "multi-choice options). None when the item uses scale-shared "
            "response anchors."
        ),
    )
    admin_note: Optional[str] = Field(
        default=None,
        description=(
            "Administration note — text directed at the test administrator, "
            "not the respondent. None when no admin note is present."
        ),
    )
    language: str = Field(
        default="en",
        description=(
            "ISO 639-1 code of the item's ORIGINAL language. 'en' when the "
            "item was already in English (the default); otherwise the source "
            "language the item was translated from (e.g. 'de' for a German "
            "original rendered into English)."
        ),
    )


class Items(BaseModel):
    """A collection of items."""

    items: List[Item] = Field(
        description=(
            "List of items in the survey, questionnaire or test. "
            "Verbatim wording of the item."
        )
    )

    @model_validator(mode="after")
    def assign_sequential_ids(self) -> "Items":
        """Assign 1-based sequential IDs."""
        for index, item in enumerate(self.items, start=1):
            item.id = index
        return self

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


# Wire models (LLM output).

class RawItem(BaseModel):
    text: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim wording of the item. Omit entirely when the item has no "
            "stem (only per-item response options or semantic-differential "
            "anchors) — in that case emit `options` instead."
        ),
    )
    img: Optional[Literal[True]] = Field(
        default=None,
        description=(
            "Set to true ONLY if the item is presented with an image. "
            "Omit otherwise."
        ),
    )
    type: Optional[Literal["choice", "open", "other"]] = Field(
        default=None,
        description=(
            "Response format, set ONLY when NOT a rating scale (omit otherwise "
            "— the default is a rating/Likert scale, which also covers semantic "
            "differentials and binary yes/no items). 'choice': respondent picks "
            "one or many categorical options (emit those in `options`). 'open': "
            "free-form input — open text, essay, drawing, or other open "
            "response. 'other': cannot be classified as rating/choice/open."
        ),
    )
    options: Optional[List[str]] = Field(
        default=None,
        description=(
            "Per-item response-option labels in scoring order. Emit for CHOICE "
            "items (the selectable answer options — works with or without a "
            "stem) and for stem-less semantic-differential anchor pairs. Omit "
            "entirely (do not emit null or []) for ordinary rating-scale items "
            "whose response anchors are scale-shared (e.g. 'Strongly disagree … "
            "Strongly agree')."
        ),
    )
    note: Optional[str] = Field(
        default=None,
        description=(
            "Administration note — text directed at the test administrator, "
            "not the respondent (e.g. skip instructions, interviewer cues, "
            "age-gate conditions). Omit entirely when no admin note is present."
        ),
    )
    lang: Optional[str] = Field(
        default=None,
        description=(
            "ISO 639-1 code of the item's ORIGINAL language, set ONLY when the "
            "item was translated from another language into English (e.g. 'de' "
            "for a German original). `text` must still hold the English "
            "translation — this field only records what it was translated from. "
            "Omit entirely for items already in English."
        ),
    )


class RawItems(BaseModel):
    items: List[RawItem] = Field(
        description=(
            "List of items in the survey, in page order, "
            "with verbatim wording."
        )
    )

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


def resolve_items(raw: RawItems) -> Items:
    """Convert wire ``RawItems`` to ``Items``, filling omitted-field defaults."""
    return Items(
        items=[
            Item(
                item_text=r.text,
                has_image=bool(r.img),
                item_type=r.type or "rating_scale",
                options=r.options,
                admin_note=r.note,
                language=r.lang or "en",
            )
            for r in raw.items
        ]
    )
