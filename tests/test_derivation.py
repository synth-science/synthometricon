"""Tests for ``derive_items`` / ``derive_survey`` / ``resolve_instrument`` in ``extraction.models``."""

from extraction.models import (
    Instrument,
    InstrumentMappings,
    Items,
    Meta,
    Survey,
    derive_items,
    derive_survey,
    resolve_instrument,
)


# --- Fixtures (synthetic Instrument trees) ---

def _flat_instrument() -> Instrument:
    """One scale, two items, no subscales, no unscaled, no meta."""
    return Instrument.model_validate({
        "scales": [{
            "id": 1,
            "scale_name": "Extraversion",
            "construct_name": "Extraversion",
            "items": [
                {"item_id": 1, "item_text": "I am the life of the party.",
                 "has_image": False, "reverse_coded": False},
                {"item_id": 2, "item_text": "I don't talk a lot.",
                 "has_image": False, "reverse_coded": True},
            ],
            "subscales": [],
        }],
        "unscaled_items": [],
        "meta": None,
    })


def _multi_scale_instrument() -> Instrument:
    """Two top-level scales sharing item 2; subscale under scale 1; one unscaled item."""
    return Instrument.model_validate({
        "scales": [
            {
                "id": 1,
                "scale_name": "Big Five",
                "construct_name": None,
                "items": [
                    {"item_id": 1, "item_text": "Item one",
                     "has_image": False, "reverse_coded": False},
                ],
                "subscales": [
                    {
                        "id": 2,
                        "scale_name": "Extraversion",
                        "construct_name": "Extraversion",
                        "items": [
                            {"item_id": 2, "item_text": "Item two",
                             "has_image": False, "reverse_coded": False},
                        ],
                        "subscales": [],
                    },
                ],
            },
            {
                "id": 3,
                "scale_name": "Other",
                "construct_name": None,
                "items": [
                    {"item_id": 2, "item_text": "Item two",
                     "has_image": False, "reverse_coded": True},
                    {"item_id": 3, "item_text": "Item three",
                     "has_image": True, "reverse_coded": False},
                ],
                "subscales": [],
            },
        ],
        "unscaled_items": [
            {"id": 4, "item_text": "Demographic item", "has_image": False},
        ],
        "meta": {"language": "en"},
    })


# --- derive_items ---

def test_derive_items_flat():
    inst = _flat_instrument()
    items = derive_items(inst)
    assert isinstance(items, Items)
    assert [i.id for i in items.items] == [1, 2]
    assert [i.item_text for i in items.items] == [
        "I am the life of the party.", "I don't talk a lot.",
    ]


def test_derive_items_dedupes_across_scales():
    inst = _multi_scale_instrument()
    items = derive_items(inst)
    ids = [i.id for i in items.items]
    assert ids == [1, 2, 3, 4], f"expected unique sorted ids 1-4, got {ids}"
    assert sum(1 for i in items.items if i.id == 2) == 1


def test_derive_items_includes_unscaled():
    inst = _multi_scale_instrument()
    items = derive_items(inst)
    unscaled = next(i for i in items.items if i.id == 4)
    assert unscaled.item_text == "Demographic item"


def test_derive_items_includes_orphan():
    inst = Instrument.model_validate({
        "scales": [{
            "id": 1, "scale_name": "S", "construct_name": None,
            "items": [
                {"item_id": 1, "item_text": "scaled",
                 "has_image": False, "reverse_coded": False},
            ],
            "subscales": [],
        }],
        "unscaled_items": [{"id": 2, "item_text": "Age?", "has_image": False}],
        "orphan_items":   [{"id": 3, "item_text": "I feel anxious.", "has_image": False}],
        "meta": None,
    })
    items = derive_items(inst)
    ids = [i.id for i in items.items]
    assert ids == [1, 2, 3]
    orphan = next(i for i in items.items if i.id == 3)
    assert orphan.item_text == "I feel anxious."


def test_derive_survey_has_unscaled_items_flag_from_orphans_only():
    """Orphans alone set has_unscaled_items (drives the instrument prompt's unscaled hint)."""
    inst = Instrument.model_validate({
        "scales": [{
            "id": 1, "scale_name": "S", "construct_name": None,
            "items": [], "subscales": [],
        }],
        "unscaled_items": [],
        "orphan_items": [{"id": 1, "item_text": "Lone item.", "has_image": False}],
        "meta": None,
    })
    survey = derive_survey(inst)
    assert survey.has_unscaled_items is True


def test_derive_items_preserves_ids_without_reassignment():
    """Non-contiguous IDs survive (Items.assign_sequential_ids must be bypassed)."""
    inst = Instrument.model_validate({
        "scales": [{
            "id": 1, "scale_name": "S", "construct_name": None,
            "items": [
                {"item_id": 7, "item_text": "seven",
                 "has_image": False, "reverse_coded": False},
                {"item_id": 11, "item_text": "eleven",
                 "has_image": False, "reverse_coded": False},
            ],
            "subscales": [],
        }],
        "unscaled_items": [], "meta": None,
    })
    items = derive_items(inst)
    assert [i.id for i in items.items] == [7, 11]


# --- derive_survey ---

def test_derive_survey_flat():
    inst = _flat_instrument()
    survey = derive_survey(inst)
    assert len(survey.scales) == 1
    assert survey.scales[0].id == 1
    assert survey.scales[0].scale_name == "Extraversion"
    assert survey.scales[0].construct_name == "Extraversion"
    assert survey.scales[0].subscales is None  # leaf → None, not []
    assert survey.has_unscaled_items is False


def test_derive_survey_preserves_subscale_tree():
    inst = _multi_scale_instrument()
    survey = derive_survey(inst)
    assert [s.scale_name for s in survey.scales] == ["Big Five", "Other"]
    big_five = survey.scales[0]
    assert big_five.subscales is not None
    assert len(big_five.subscales) == 1
    assert big_five.subscales[0].scale_name == "Extraversion"
    assert big_five.subscales[0].subscales is None


def test_derive_survey_strips_items():
    inst = _multi_scale_instrument()
    survey = derive_survey(inst)
    # Pydantic v2: hasattr won't trip on undefined attrs; check via fields_set
    for scale in survey.scales:
        assert "items" not in scale.model_fields_set or not hasattr(scale, "items")


def test_derive_survey_has_unscaled_items_flag():
    inst = _multi_scale_instrument()
    survey = derive_survey(inst)
    assert survey.has_unscaled_items is True


def test_derive_survey_preserves_ids():
    """Scale IDs survive (Survey.assign_sequential_scale_ids must be bypassed)."""
    inst = _multi_scale_instrument()
    survey = derive_survey(inst)
    assert survey.scales[0].id == 1
    assert survey.scales[0].subscales[0].id == 2
    assert survey.scales[1].id == 3


# --- resolve_instrument — buckets unmapped items into unscaled vs orphan ---

def _resolver_inputs() -> tuple[Items, Survey]:
    # assign_sequential_ids gives these ids 1..4.
    items = Items.model_validate({"items": [
        {"item_text": "Mapped item one"},
        {"item_text": "Mapped item two"},
        {"item_text": "Age?"},
        {"item_text": "I feel anxious."},
    ]})
    survey = Survey.model_validate({
        "scales": [{"scale_name": "S", "construct_name": "C", "subscales": []}],
        "has_unscaled_items": True,
    })
    return items, survey


def test_resolve_instrument_default_bucket_is_unscaled():
    items, survey = _resolver_inputs()
    mappings = InstrumentMappings.model_validate({"mappings": [
        {"item_id": 1, "scale_id": 1},
        {"item_id": 2, "scale_id": 1},
    ]})
    inst = resolve_instrument(mappings, items, survey)
    assert [it.id for it in inst.unscaled_items] == [3, 4]
    assert inst.orphan_items == []


def test_resolve_instrument_orphan_ids_bucket_into_orphan_items():
    items, survey = _resolver_inputs()
    mappings = InstrumentMappings.model_validate({"mappings": [
        {"item_id": 1, "scale_id": 1},
        {"item_id": 2, "scale_id": 1},
    ], "orphan_item_ids": [4]})
    inst = resolve_instrument(mappings, items, survey)
    assert [it.id for it in inst.unscaled_items] == [3]
    assert [it.id for it in inst.orphan_items] == [4]
    assert inst.orphan_items[0].item_text == "I feel anxious."


def test_resolve_instrument_orphan_id_for_mapped_item_is_ignored():
    """Mapped items win over orphan_item_ids."""
    items, survey = _resolver_inputs()
    mappings = InstrumentMappings.model_validate({"mappings": [
        {"item_id": 1, "scale_id": 1},
        {"item_id": 2, "scale_id": 1},
    ], "orphan_item_ids": [2, 4]})
    inst = resolve_instrument(mappings, items, survey)
    scale_item_ids = [si.item_id for si in inst.scales[0].items]
    assert 2 in scale_item_ids
    assert [it.id for it in inst.orphan_items] == [4]
    assert [it.id for it in inst.unscaled_items] == [3]


def test_resolve_instrument_unknown_orphan_id_silently_dropped():
    items, survey = _resolver_inputs()
    mappings = InstrumentMappings.model_validate({"mappings": [
        {"item_id": 1, "scale_id": 1},
    ], "orphan_item_ids": [4, 99]})
    inst = resolve_instrument(mappings, items, survey)
    assert [it.id for it in inst.orphan_items] == [4]
    assert [it.id for it in inst.unscaled_items] == [2, 3]


def test_resolve_instrument_no_orphan_field_defaults_to_unscaled():
    """Omitted ``orphan_item_ids`` (older outputs) leaves all unmapped items unscaled."""
    items, survey = _resolver_inputs()
    mappings = InstrumentMappings.model_validate({"mappings": [
        {"item_id": 1, "scale_id": 1},
    ]})
    assert mappings.orphan_item_ids is None
    inst = resolve_instrument(mappings, items, survey)
    assert [it.id for it in inst.unscaled_items] == [2, 3, 4]
    assert inst.orphan_items == []
