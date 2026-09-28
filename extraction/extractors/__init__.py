"""Extractor registry; insertion order is pipeline order (deps must come first)."""

from .base import Extractor
from .instrument import InstrumentExtractor
from .items import ItemExtractor
from .meta import MetaExtractor
from .scales import ScaleExtractor


REGISTRY: dict[str, Extractor] = {
    "items": ItemExtractor(),
    "scales": ScaleExtractor(),
    "instrument": InstrumentExtractor(),
    "meta": MetaExtractor(),
}

__all__ = ["Extractor", "REGISTRY"]
