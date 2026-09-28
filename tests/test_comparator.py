"""Offline tests for the ``validity`` comparator, fixture tags, and ``normalize_for_model_validation``."""

import pytest
import yaml

from extraction.models import Instrument
from extraction.validity import (
    AnywhereDict,
    AnywhereList,
    ExtrasList,
    MatchSpec,
    ValiditySafeLoader,
    normalize_for_model_validation,
    strip_match_specs,
    validate_extraction,
)


# --- Fixtures ---

def _make_instrument(d: dict) -> Instrument:
    return Instrument.model_validate(d)


_EXTRAVERSION_INSTRUMENT = {
    "scales": [
        {
            "id": 1,
            "scale_name": "Extraversion",
            "construct_name": "Extraversion",
            "items": [
                {
                    "item_id": 1,
                    "item_text": "I am the life of the party.",
                    "has_image": False,
                    "reverse_coded": False,
                },
                {
                    "item_id": 2,
                    "item_text": "I don't talk a lot.",
                    "has_image": False,
                    "reverse_coded": True,
                },
            ],
            "subscales": [],
        }
    ],
    "unscaled_items": [],
    "meta": None,
}


_NESTED_INSTRUMENT = {
    "scales": [
        {
            "id": 1,
            "scale_name": "Big Five",
            "construct_name": "Personality",
            "items": [],
            "subscales": [
                {
                    "id": 2,
                    "scale_name": "Extraversion",
                    "construct_name": "Extraversion",
                    "items": [
                        {"item_id": 1, "item_text": "I am the life of the party.",
                         "has_image": False, "reverse_coded": False},
                        {"item_id": 2, "item_text": "I don't talk a lot.",
                         "has_image": False, "reverse_coded": True},
                    ],
                    "subscales": [],
                },
                {
                    "id": 3,
                    "scale_name": "Neuroticism",
                    "construct_name": "Neuroticism",
                    "items": [
                        {"item_id": 3, "item_text": "I get stressed out easily.",
                         "has_image": False, "reverse_coded": False},
                    ],
                    "subscales": [],
                },
            ],
        },
    ],
    "unscaled_items": [],
    "meta": None,
}


# --- Core comparator behaviour ---

def test_identical_fixture_passes():
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = actual.model_dump(mode="json")
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()
    assert report.issues == []


def test_omitted_fields_are_dont_care():
    """A minimal fixture should pass against a fully populated Instrument."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": "I am the life of the party."},
                    {"item_text": "I don't talk a lot."},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()


def test_missing_expected_item_fails():
    """Expected lists an extra item that the actual doesn't have."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": "I am the life of the party."},
                    {"item_text": "I don't talk a lot."},
                    {"item_text": "I am cheerful all the time."},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL"
    assert any("missing in actual" in i.message for i in report.issues)


def test_extra_actual_item_fails():
    """Actual has an item that the fixture doesn't list (hallucination case)."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [{"item_text": "I am the life of the party."}],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL"
    assert any("extra in actual" in i.message for i in report.issues)


def test_reverse_coded_mismatch_fails():
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": "I am the life of the party.", "reverse_coded": True},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL"
    assert any("reverse_coded" in i.path for i in report.issues), report.render()


def test_subscale_nesting():
    """Recursion into subscales matches the same Hungarian/name pattern."""
    actual = _make_instrument({
        "scales": [
            {
                "id": 1, "scale_name": "Big Five", "construct_name": None,
                "items": [],
                "subscales": [
                    {
                        "id": 2, "scale_name": "Extraversion", "construct_name": None,
                        "items": [
                            {"item_id": 1, "item_text": "I am outgoing.",
                             "has_image": False, "reverse_coded": False},
                        ],
                        "subscales": [],
                    },
                ],
            }
        ],
        "unscaled_items": [], "meta": None,
    })
    expected = {
        "scales": [
            {
                "scale_name": "Big Five",
                "subscales": [
                    {
                        "scale_name": "Extraversion",
                        "items": [{"item_text": "I am outgoing."}],
                    }
                ],
            }
        ],
    }
    assert validate_extraction(expected, actual).status == "PASS"


def test_levenshtein_tolerates_minor_drift():
    """One-character drifts pass the default 0.80 threshold."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": "I am the life of the party"},
                    {"item_text": "I don't talk a lot"},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()


def test_path_format_is_jsonpath_style():
    """Issue paths should be human-readable JSONPath-style."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": "I am the life of the party.", "reverse_coded": True},
                    {"item_text": "I don't talk a lot."},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    bad = [i for i in report.issues if "reverse_coded" in i.path]
    assert bad, report.render()
    assert bad[0].path.startswith("scales[0].items["), bad[0].path


# --- !contains / !exact opt-out tags ---

def test_contains_tag_accepts_substring():
    """``!contains`` as a list key: Hungarian picks the item containing the fragment."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": MatchSpec(mode="contains", value="life of the party")},
                    {"item_text": MatchSpec(mode="contains", value="don't talk")},
                ],
            }
        ],
    }
    assert validate_extraction(expected, actual).status == "PASS"


def test_contains_tag_tolerates_minor_drift():
    """A near-miss ``!contains`` fragment passes under the default Levenshtein threshold."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    # one transposition
                    {"item_text": MatchSpec(mode="contains", value="life of teh party")},
                    {"item_text": MatchSpec(mode="contains", value="don't talk")},
                ],
            }
        ],
    }
    assert validate_extraction(expected, actual).status == "PASS"


def test_contains_tag_rejects_large_drift():
    """!contains with a completely different fragment still fails."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": MatchSpec(mode="contains", value="entirely different text")},
                    {"item_text": MatchSpec(mode="contains", value="don't talk")},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL"


def test_exact_tag_fails_on_mismatch():
    """!exact on a dict leaf — failure surfaces a '!exact mismatch' message."""
    actual = _make_instrument({
        "scales": [],
        "unscaled_items": [],
        "meta": {"language": "en"},
    })
    expected = {"meta": {"language": MatchSpec(mode="exact", value="de")}}
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL"
    assert any("!exact mismatch" in i.message for i in report.issues)


def test_exact_tag_passes_on_match():
    actual = _make_instrument({
        "scales": [],
        "unscaled_items": [],
        "meta": {"language": "en"},
    })
    expected = {"meta": {"language": MatchSpec(mode="exact", value="en")}}
    assert validate_extraction(expected, actual).status == "PASS"


# --- orphan_items + unscaled_items — symmetric comparator handling ---

def test_orphan_items_match_by_item_text():
    """``orphan_items`` match like ``unscaled_items`` (Hungarian on ``item_text``)."""
    actual = _make_instrument({
        "scales": [],
        "unscaled_items": [{"id": 1, "item_text": "Age?", "has_image": False}],
        "orphan_items":   [{"id": 2, "item_text": "I feel anxious.", "has_image": False}],
        "meta": None,
    })
    expected = {
        "unscaled_items": [{"item_text": "Age?"}],
        "orphan_items":   [{"item_text": "I feel anxious."}],
    }
    assert validate_extraction(expected, actual).status == "PASS"


def test_orphan_items_missing_expected_fails():
    actual = _make_instrument({
        "scales": [],
        "unscaled_items": [],
        "orphan_items": [],
        "meta": None,
    })
    expected = {
        "orphan_items": [{"item_text": "Single-item PSWQ."}],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL"
    assert any("orphan_items" in i.path for i in report.issues)


def test_normalize_handles_orphan_items_block():
    """Fixture ``orphan_items`` normalise into a valid Instrument with synthesised IDs."""
    partial = {
        "scales": [],
        "orphan_items": [{"item_text": "I worry a lot."}],
    }
    normalized = normalize_for_model_validation(partial)
    assert "orphan_items" in normalized
    assert normalized["orphan_items"][0]["item_text"] == "I worry a lot."
    assert isinstance(normalized["orphan_items"][0]["id"], int)
    inst = Instrument.model_validate(normalized)
    assert [it.item_text for it in inst.orphan_items] == ["I worry a lot."]


# --- normalize_for_model_validation — dep-context plumbing ---

def test_normalize_partial_fixture_fills_ids():
    """An ID-less curated fixture normalises into a valid Instrument (dep-context regression)."""
    partial = {
        "scales": [
            {
                "scale_name": "Anxiety",
                "items": [
                    {"item_text": "I worry a lot."},
                    {"item_text": "I feel tense."},
                ],
                "subscales": [
                    {
                        "scale_name": "Cognitive",
                        "items": [
                            {"item_text": "I worry a lot.", "reverse_coded": True},
                        ],
                    },
                ],
            },
        ],
    }
    normalized = normalize_for_model_validation(partial)
    instrument = Instrument.model_validate(normalized)

    # DFS scale IDs.
    assert instrument.scales[0].id == 1
    assert instrument.scales[0].subscales[0].id == 2

    # Same item_text under multiple scales shares an item_id.
    parent_first = instrument.scales[0].items[0]
    nested_only = instrument.scales[0].subscales[0].items[0]
    assert parent_first.item_text == nested_only.item_text == "I worry a lot."
    assert parent_first.item_id == nested_only.item_id

    # Distinct text → distinct id.
    assert instrument.scales[0].items[1].item_id != parent_first.item_id


def test_normalize_strips_meta():
    """``meta`` is dropped."""
    partial = {
        "scales": [],
        "meta": {"language": "en"},
    }
    normalized = normalize_for_model_validation(partial)
    assert normalized["meta"] is None


def test_normalize_preserves_existing_ids():
    """Explicit fixture IDs round-trip untouched; synthesis only fills gaps."""
    seeded = {
        "scales": [
            {
                "id": 7,
                "scale_name": "S",
                "items": [
                    {"item_id": 42, "item_text": "x", "reverse_coded": False},
                ],
                "subscales": [],
            },
        ],
        "unscaled_items": [],
    }
    normalized = normalize_for_model_validation(seeded)
    assert normalized["scales"][0]["id"] == 7
    assert normalized["scales"][0]["items"][0]["item_id"] == 42


def test_normalize_handles_match_spec_stripped_strings():
    """Output of ``strip_match_specs`` still normalises and validates."""
    tagged = {
        "scales": [
            {
                "scale_name": MatchSpec(mode="contains", value="Anx"),
                "items": [{"item_text": "I worry a lot."}],
                "subscales": [],
            },
        ],
    }
    clean = strip_match_specs(tagged)
    normalized = normalize_for_model_validation(clean)
    instrument = Instrument.model_validate(normalized)
    assert instrument.scales[0].scale_name == "Anx"


# --- !anywhere — relaxed positional fixtures ---

def test_anywhere_subscale_finds_at_any_depth():
    """A top-level ``!anywhere`` scale matches a deeply nested actual subscale."""
    actual = _make_instrument(_NESTED_INSTRUMENT)
    expected = {
        "scales": AnywhereList([
            {
                "scale_name": MatchSpec(mode="contains", value="Extraversion"),
                "items": [
                    {"item_text": MatchSpec(mode="contains", value="life of the party")},
                    {"item_text": MatchSpec(mode="contains", value="don't talk")},
                ],
            },
        ]),
    }
    assert validate_extraction(expected, actual).status == "PASS"


def test_anywhere_items_no_scale_structure():
    """``items: !anywhere`` matches items across scales, ignoring scale assignment."""
    actual = _make_instrument(_NESTED_INSTRUMENT)
    expected = {
        "scales": [
            {
                "items": AnywhereList([
                    {"item_text": MatchSpec(mode="contains", value="life of the party")},
                    {"item_text": MatchSpec(mode="contains", value="stressed out")},
                ]),
            },
        ],
    }
    assert validate_extraction(expected, actual).status == "PASS"


def test_anywhere_mixed_strict_and_loose():
    """Strict and ``!anywhere`` entries in one list: the loose one's hit is not flagged as extra."""
    actual = _make_instrument(_NESTED_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": MatchSpec(mode="contains", value="Big Five"),
                "subscales": [
                    {"scale_name": MatchSpec(mode="contains", value="Extraversion")},
                    AnywhereDict({
                        "scale_name": MatchSpec(mode="contains", value="Neuroticism"),
                    }),
                ],
            },
        ],
    }
    assert validate_extraction(expected, actual).status == "PASS"


def test_anywhere_loose_not_found_fails():
    """An unmatched ``!anywhere`` entry fails as 'missing in actual (anywhere)'."""
    actual = _make_instrument(_NESTED_INSTRUMENT)
    expected = {
        "scales": AnywhereList([
            {"scale_name": MatchSpec(mode="contains", value="Openness")},
        ]),
    }
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL"
    assert any("missing in actual (anywhere)" in i.message for i in report.issues), \
        report.render()


def test_anywhere_purely_loose_skips_extras_check():
    """A purely loose ``scales: !anywhere`` list never flags extras."""
    actual = _make_instrument(_NESTED_INSTRUMENT)
    expected = {
        "scales": AnywhereList([
            {"scale_name": MatchSpec(mode="contains", value="Extraversion")},
        ]),
    }
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()
    assert not any("extra in actual" in i.message for i in report.issues), report.render()


def test_anywhere_yaml_tag_loads_into_sentinels():
    """``!anywhere`` parses into ``AnywhereList`` / ``AnywhereDict``."""
    yaml_doc = """
    extraction_output:
      scales: !anywhere
        - scale_name: !contains "Extraversion"
        - !anywhere
          scale_name: !contains "Neuroticism"
    """
    loaded = yaml.load(yaml_doc, Loader=ValiditySafeLoader)
    scales = loaded["extraction_output"]["scales"]
    assert isinstance(scales, AnywhereList)
    assert isinstance(scales[1], AnywhereDict)
    assert type(scales[0]) is dict


_EXTRAS_INSTRUMENT = {
    "scales": [
        {
            "id": 1,
            "scale_name": "Sample",
            "construct_name": "Sample",
            "items": [
                {"item_id": i + 1, "item_text": f"item {i + 1}", "has_image": False,
                 "reverse_coded": False}
                for i in range(5)
            ],
            "subscales": [],
        }
    ],
    "unscaled_items": [],
    "meta": None,
}


def _extras_expected(named_items: list[str], budget: int) -> dict:
    """Expected fixture asserting ``named_items`` under ``Sample.items`` with ``!extras=budget``."""
    lst = ExtrasList([{"item_text": t} for t in named_items])
    lst.allow_extras = budget
    return {"scales": [{"scale_name": "Sample", "items": lst}]}


def test_extras_budget_absorbs_unmatched_actuals_within_limit():
    """``!extras=3`` tolerates up to 3 unmatched actual entries."""
    actual = _make_instrument(_EXTRAS_INSTRUMENT)
    expected = _extras_expected(["item 1", "item 2"], budget=3)
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()
    assert not any("extra in actual" in i.message for i in report.issues), report.render()


def test_extras_budget_flags_only_overflow():
    """Only extras beyond the budget are flagged."""
    actual = _make_instrument(_EXTRAS_INSTRUMENT)
    # 2 expected matched + 3 budget tolerated → 5th unmatched should flag.
    expected = _extras_expected(["item 1", "item 2"], budget=2)
    report = validate_extraction(expected, actual)
    extras = [i for i in report.issues if "extra in actual" in i.message]
    assert len(extras) == 1, report.render()
    assert report.status == "FAIL"


def test_extras_zero_matches_strict_default():
    """``!extras=0`` behaves like an untagged strict list."""
    actual = _make_instrument(_EXTRAS_INSTRUMENT)
    expected = _extras_expected(["item 1", "item 2"], budget=0)
    report = validate_extraction(expected, actual)
    extras = [i for i in report.issues if "extra in actual" in i.message]
    assert len(extras) == 3, report.render()  # items 3, 4, 5 all flagged
    assert report.status == "FAIL"


def test_extras_yaml_tag_loads_with_budget():
    """``!extras=N`` parses into an ``ExtrasList`` carrying N."""
    yaml_doc = """
    extraction_output:
      scales:
        - scale_name: Sample
          items: !extras=3
            - item_text: item 1
            - item_text: item 2
    """
    loaded = yaml.load(yaml_doc, Loader=ValiditySafeLoader)
    items = loaded["extraction_output"]["scales"][0]["items"]
    assert isinstance(items, ExtrasList)
    assert items.allow_extras == 3
    assert items == [{"item_text": "item 1"}, {"item_text": "item 2"}]


def test_strip_match_specs_unwraps_anywhere():
    """``strip_match_specs`` reduces ``AnywhereList`` / ``AnywhereDict`` to plain list / dict."""
    tagged = {
        "scales": AnywhereList([
            AnywhereDict({
                "scale_name": MatchSpec(mode="contains", value="Anx"),
                "items": [{"item_text": "I worry a lot."}],
            }),
        ]),
    }
    clean = strip_match_specs(tagged)
    assert type(clean["scales"]) is list
    assert type(clean["scales"][0]) is dict
    assert clean["scales"][0]["scale_name"] == "Anx"
    normalized = normalize_for_model_validation(clean)
    Instrument.model_validate(normalized)


# --- MatchRecord — per-pair Levenshtein scores retained for the --validity table ---

def test_match_records_capture_every_matched_pair():
    """One ``MatchRecord`` per matched scale/item, with similarity and edit distance."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = actual.model_dump(mode="json")
    report = validate_extraction(expected, actual)

    assert len(report.matches) == 3  # 1 scale + 2 items
    kinds = sorted(m.kind for m in report.matches)
    assert kinds == ["item", "item", "scale"]
    assert all(m.similarity == 1.0 and m.edits == 0 for m in report.matches)
    assert all(m.expected == m.actual for m in report.matches)

    scale = next(m for m in report.matches if m.kind == "scale")
    assert scale.path == "scales[0]"
    assert scale.expected == "Extraversion"


def test_match_record_scores_a_near_miss():
    """A passing near-miss records its exact similarity and edit count."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": "Extraversion",
                "items": [
                    {"item_text": "I am the life of the party!"},  # '.' -> '!'
                    {"item_text": "I don't talk a lot."},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()

    near = next(m for m in report.matches if m.expected.endswith("party!"))
    assert near.kind == "item"
    assert near.edits == 1
    assert 0.80 < near.similarity < 1.0
    assert near.actual == "I am the life of the party."

    # edits == 0 exactly where similarity == 1.0, across every record
    for m in report.matches:
        assert (m.edits == 0) == (m.similarity == 1.0)


def test_match_specs_and_dont_care_keys_are_not_recorded():
    """``!exact`` / ``!contains`` and omitted keys are not real distances, so are not recorded."""
    actual = _make_instrument(_EXTRAVERSION_INSTRUMENT)
    expected = {
        "scales": [
            {
                "scale_name": MatchSpec(mode="contains", value="Extra"),
                "items": [
                    {"item_text": MatchSpec(mode="exact",
                                            value="I am the life of the party.")},
                    {},  # no key at all -> don't care
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()
    assert report.matches == []


# --- Options-key vs real item_text — cross-type mismatch ---

def test_options_key_does_not_match_real_item_text():
    """Options-keyed actual vs. real expected item_text: similarity 0.0, issue names missing item_text."""
    actual = _make_instrument({
        "scales": [
            {
                "id": 1,
                "scale_name": "Depression",
                "construct_name": "Depression",
                "items": [
                    {
                        "item_id": 1,
                        "item_text": None,
                        "item_type": "choice",
                        "has_image": False,
                        "reverse_coded": False,
                        "options": ["Not depressed", "Somewhat", "Very depressed"],
                    },
                ],
                "subscales": [],
            }
        ],
        "unscaled_items": [],
        "meta": None,
    })
    expected = {
        "scales": [
            {
                "scale_name": "Depression",
                "items": [
                    {"item_text": "Viṣāditva"},
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "FAIL", report.render()
    failing = [i for i in report.issues if "item_text" in i.message]
    assert failing
    assert "no item_text" in failing[0].message
    assert "similarity" not in failing[0].message
    item_matches = [m for m in report.matches if m.kind == "item"]
    assert item_matches == []


def test_options_key_matches_other_options_key():
    """Two stemless items match via Levenshtein on their joined options."""
    actual = _make_instrument({
        "scales": [
            {
                "id": 1,
                "scale_name": "Depression",
                "construct_name": "Depression",
                "items": [
                    {
                        "item_id": 1,
                        "item_text": None,
                        "item_type": "choice",
                        "has_image": False,
                        "reverse_coded": False,
                        "options": ["Not depressed", "Somewhat", "Very depressed"],
                    },
                ],
                "subscales": [],
            }
        ],
        "unscaled_items": [],
        "meta": None,
    })
    expected = {
        "scales": [
            {
                "scale_name": "Depression",
                "items": [
                    {
                        "item_text": None,
                        "options": ["Not depressed", "Somewhat", "Very depressed"],
                    },
                ],
            }
        ],
    }
    report = validate_extraction(expected, actual)
    assert report.status == "PASS", report.render()
