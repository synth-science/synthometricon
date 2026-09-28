"""Result, wire, and composer models for the extraction pipeline, split by stage.

Wire-model design: docs/wire-model-conventions.md.
"""

from .compose import (
    apply_meta_to_instrument,
    compose_extraction_output,
    derive_items,
    derive_survey,
)
from .instrument import (
    Instrument,
    InstrumentMappings,
    ItemScaleMapping,
    ScaledItem,
    ScaleNode,
    resolve_instrument,
)
from .items import Item, Items, RawItem, RawItems, resolve_items
from .meta import Meta, RawMeta, resolve_meta
from .scales import RawScale, RawSurvey, Scale, Scales, Survey, resolve_scales


__all__ = [
    # Items
    "Item", "Items", "RawItem", "RawItems", "resolve_items",
    # Scales
    "Scale", "Survey", "Scales", "RawScale", "RawSurvey", "resolve_scales",
    # Meta
    "Meta", "RawMeta", "resolve_meta",
    # Instrument
    "Instrument", "ItemScaleMapping", "InstrumentMappings",
    "ScaledItem", "ScaleNode", "resolve_instrument",
    # Composition
    "apply_meta_to_instrument", "compose_extraction_output",
    "derive_items", "derive_survey",
]
