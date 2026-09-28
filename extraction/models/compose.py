"""Compose per-extractor results into an Instrument, and derive upstream inputs back out.

Derivers let curated fixtures supply dependency context in reliability mode.
"""

from __future__ import annotations

from typing import Optional

from .instrument import Instrument
from .items import Item, Items
from .meta import Meta
from .scales import Scale, Survey


def apply_meta_to_instrument(instrument: Instrument, meta: Meta) -> Instrument:
    """Return a new Instrument with ``meta`` attached. Non-mutating."""
    return instrument.model_copy(update={"meta": meta})


def compose_extraction_output(
    instrument: Optional[Instrument],
    meta: Optional[Meta],
) -> Optional[Instrument]:
    """Unified output tree: ``None`` until the instrument exists; attaches ``meta`` if present."""
    if instrument is None:
        return None
    if meta is None:
        return instrument
    return apply_meta_to_instrument(instrument, meta)


def derive_items(instrument: Instrument) -> Items:
    """Rebuild ``Items`` (deduped, sorted by id) from an Instrument tree.

    ``model_construct`` skips the id-assigning validator, preserving fixture IDs.
    """
    by_id: dict[int, Item] = {}

    def visit(node: ScaleNode) -> None:
        for si in node.items:
            if si.item_id not in by_id:
                by_id[si.item_id] = Item(
                    id=si.item_id,
                    item_text=si.item_text,
                    has_image=si.has_image,
                    item_type=si.item_type,
                    options=si.options,
                    admin_note=si.admin_note,
                    language=si.language,
                )
        for sub in node.subscales:
            visit(sub)

    for top in instrument.scales:
        visit(top)

    for it in (*instrument.unscaled_items, *instrument.orphan_items):
        if it.id is not None and it.id not in by_id:
            by_id[it.id] = it

    items_in_order = [by_id[i] for i in sorted(by_id)]
    return Items.model_construct(items=items_in_order)


def derive_survey(instrument: Instrument) -> Survey:
    """Rebuild the ``Survey`` scale tree from an Instrument tree.

    ``has_unscaled_items`` is set by unscaled or orphan items; ``model_construct``
    skips the id-assigning validator, preserving fixture IDs.
    """

    def to_scale(node: ScaleNode) -> Scale:
        subs = [to_scale(s) for s in node.subscales] if node.subscales else None
        return Scale.model_construct(
            id=node.id,
            scale_name=node.scale_name,
            construct_name=node.construct_name,
            subscales=subs,
        )

    return Survey.model_construct(
        scales=[to_scale(n) for n in instrument.scales],
        has_unscaled_items=bool(instrument.unscaled_items) or bool(instrument.orphan_items),
    )


