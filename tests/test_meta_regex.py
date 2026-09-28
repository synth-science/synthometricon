"""Meta extractor page-1 regex fields, pinned to the real PsycTESTS cover page.

Regression: a blanket ``doi`` -> ``doi_raw`` rename (a6e4cf2) once corrupted the regex literal.
"""
from extraction.extractors.meta import _extract_regex_fields

# DOI follows "doi:" after a line break, on the dx.doi.org host.
PAGE_1 = """\
Mother-Daughter Synchrony Scale
PsycTESTS Citation:
Fisher, L., & Katz, J. (1998). Mother-Daughter Synchrony Scale [Database record].
Retrieved from PsycTESTS. doi:
https://dx.doi.org/10.1037/t08009-000
Instrument Type:
Rating Scale
Test Format:
Items are rated on a 5-point scale.
Source:
Fisher, Lawrence (1998). Journal of Family Psychology, Vol 12(2), 253-266.
Permissions:
Test content may be reproduced and used for non-commercial research and
educational purposes without seeking written permission.
PsycTESTS™ is a database of the American Psychological Association
"""


def test_doi_extracted_from_citation_block():
    fields = _extract_regex_fields(PAGE_1)
    assert fields["doi_raw"] == "10.1037/t08009-000"


def test_doi_extracted_without_dx_prefix():
    fields = _extract_regex_fields(PAGE_1.replace("dx.doi.org", "doi.org"))
    assert fields["doi_raw"] == "10.1037/t08009-000"


def test_doi_none_when_absent():
    text = PAGE_1.replace("doi:", "").replace("https://dx.doi.org/10.1037/t08009-000", "")
    assert _extract_regex_fields(text)["doi_raw"] is None


def test_doi_none_on_empty_text():
    assert _extract_regex_fields("")["doi_raw"] is None


def test_sibling_fields_still_extracted():
    fields = _extract_regex_fields(PAGE_1)
    assert fields["title_raw"] == "Mother-Daughter Synchrony Scale"
    assert fields["instrument_type_raw"] == "Rating Scale"
    assert fields["publication_year_raw"] == 1998
    assert fields["permissions_raw"].startswith("Test content may be reproduced")
