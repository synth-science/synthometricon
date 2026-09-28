"""Instrument: result and wire models for the item-to-scale mapping stage."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ._schema import inline_schema
from .items import Item, Items
from .meta import Meta
from .scales import Scale, Survey


# Wire models (LLM output).

class ItemScaleMapping(BaseModel):
    """A single item-to-scale assignment (wire form).

    Uses the full ``item_id`` / ``scale_id`` / ``reverse_coded`` keys (not
    short aliases): shorter keys measurably hurt the model's semantic
    anchoring on this task — bare ``item``/``scale`` lose the "this is an id
    reference" cue, and ``rev`` reads as "revision/review" rather than
    reverse-coded, biasing the model toward spurious truthy values.
    Tightening ``reverse_coded`` to ``Optional[Literal[True]]`` still prevents
    ``false`` from ever being emitted.
    """

    item_id: int = Field(description="ID of the item, as assigned by the items extractor.")
    scale_id: int = Field(description="ID of the scale the item belongs to.")
    reverse_coded: Optional[Literal[True]] = Field(
        default=None,
        description=(
            "Set to true ONLY if the item is reverse-coded against the scale "
            "(i.e. scored in the opposite direction). Omit the field entirely "
            "otherwise."
        ),
    )


class InstrumentMappings(BaseModel):
    """Wire model — flat list of (item_id, scale_id, reverse_coded?) triples
    that ``resolve_instrument`` joins against the upstream items/scales to
    build a nested ``Instrument``. Not persisted; not exposed to consumers."""

    mappings: List[ItemScaleMapping] = Field(
        description=(
            "One entry per (item, scale) assignment. An item may appear in "
            "multiple mappings if it belongs to more than one scale."
        )
    )
    orphan_item_ids: Optional[List[int]] = Field(
        default=None,
        description=(
            "IDs of items that have psychometric merit but do NOT belong to "
            "any defined scale (orphan items — e.g. a stand-alone single-item "
            "measure embedded in a larger questionnaire). Omit the field "
            "entirely when there are none — do not emit null or []. Items "
            "that are not in `mappings` and not in `orphan_item_ids` are "
            "treated as unscaled (demographic / administrative)."
        ),
    )

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


# Result models (persisted).

class ScaledItem(BaseModel):
    """An item as it sits inside one scale of a resolved Instrument."""

    item_id: int
    item_text: Optional[str] = None
    has_image: Optional[bool] = False
    item_type: Literal["rating_scale", "choice", "open", "other"] = Field(
        default="rating_scale",
        description=(
            "Response format of the item: 'rating_scale' (default, covers "
            "Likert/semantic-differential/binary), 'choice', 'open', or 'other'."
        ),
    )
    options: Optional[List[str]] = Field(
        default=None,
        description=(
            "Per-item response-option labels in scoring order. None when the "
            "item has a verbatim stem and uses scale-shared response anchors."
        ),
    )
    admin_note: Optional[str] = Field(
        default=None,
        description=(
            "Administration note — text directed at the test administrator. "
            "None when no admin note is present."
        ),
    )
    language: str = Field(
        default="en",
        description=(
            "ISO 639-1 code of the item's ORIGINAL language; 'en' unless the "
            "item was translated from another language into English."
        ),
    )
    reverse_coded: bool = False


class ScaleNode(BaseModel):
    """A scale (or subscale) carrying its assigned items and any nested children."""

    id: int
    scale_name: Optional[str]
    construct_name: Optional[str] = None
    items: List[ScaledItem] = Field(default_factory=list)
    subscales: List["ScaleNode"] = Field(default_factory=list)


ScaleNode.model_rebuild()


class Instrument(BaseModel):
    """Resolved item-to-scale structure of an instrument.

    Mirrors the scale tree extracted by ``scales``, with each ``ScaleNode``
    carrying the items assigned to it (duplicated under every scale they
    belong to when multi-scale). Items with no scale assignment are split
    between two buckets:

    - ``unscaled_items``: items with no psychometric content — demographic
      questions, administrative items, doctor's / examiner's notes, plain
      instructions, manifest-fact questions. They are not scored.
    - ``orphan_items``: items that ARE psychometric in nature (feelings,
      attitudes, behaviours, beliefs, symptoms, …) but are not assigned to
      any defined scale — typically a stand-alone single-item measure
      embedded alongside a multi-scale questionnaire.

    ``meta`` is populated only by ``apply_meta_to_instrument`` after the meta
    extractor has run. The instrument extractor never sets it —
    ``resolve_instrument`` leaves it at its ``None`` default.
    """

    scales: List[ScaleNode]
    unscaled_items: List[Item] = Field(default_factory=list)
    orphan_items: List[Item] = Field(default_factory=list)
    meta: Optional[Meta] = Field(
        default=None,
        description=(
            "Document-level metadata. Populated only by "
            "`apply_meta_to_instrument` after the meta extractor has run; "
            "the instrument extractor never sets this."
        ),
    )

    @classmethod
    def inlined_schema(cls) -> dict:
        return inline_schema(cls)


def resolve_instrument(
    mappings: InstrumentMappings, items: Items, scales: Survey
) -> Instrument:
    """Join wire mappings against upstream items + scales into an ``Instrument`` tree.

    Multi-scale items are duplicated per scale; unmapped items go to ``orphan_items``
    if listed in ``orphan_item_ids``, else ``unscaled_items``. Unknown ids are dropped.
    """
    item_by_id: dict[int, Item] = {
        item.id: item for item in items.items if item.id is not None
    }

    mappings_by_scale: dict[int, list[ItemScaleMapping]] = {}
    mapped_item_ids: set[int] = set()
    for m in mappings.mappings:
        mappings_by_scale.setdefault(m.scale_id, []).append(m)
        mapped_item_ids.add(m.item_id)

    def build(scale: Scale) -> ScaleNode:
        scaled_items: list[ScaledItem] = []
        for m in mappings_by_scale.get(scale.id, []):
            src = item_by_id.get(m.item_id)
            if src is None:
                continue
            scaled_items.append(
                ScaledItem(
                    item_id=m.item_id,
                    item_text=src.item_text,
                    has_image=src.has_image,
                    item_type=src.item_type,
                    options=src.options,
                    admin_note=src.admin_note,
                    language=src.language,
                    reverse_coded=bool(m.reverse_coded),
                )
            )
        return ScaleNode(
            id=scale.id,
            scale_name=scale.scale_name,
            construct_name=scale.construct_name,
            items=scaled_items,
            subscales=[build(s) for s in (scale.subscales or [])],
        )

    resolved_scales = [build(s) for s in scales.scales]
    orphan_ids = {i for i in (mappings.orphan_item_ids or []) if i not in mapped_item_ids}
    unmapped = [item for item in items.items if item.id not in mapped_item_ids]
    orphan_items = [it for it in unmapped if it.id in orphan_ids]
    unscaled_items = [it for it in unmapped if it.id not in orphan_ids]
    return Instrument(
        scales=resolved_scales,
        unscaled_items=unscaled_items,
        orphan_items=orphan_items,
    )
