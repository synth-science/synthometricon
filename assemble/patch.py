"""Patch stage: value-level repair and backfill of the combined corpus.

Never drops rows; DOI backfill is fill-only-null. Steps run in ``STEPS`` order;
per-step semantics in docs/assemble-patch.md.

Usage:
    python -m assemble --step patch [--report-only] [--refresh]
    python -m assemble.patch --input X.parquet --output Y.parquet \
        --steps schema,language --report-only
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import httpx
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError

from extraction.storage import PATH_COL
from extraction.text_utils import fuzzy_contains, extract_json

from .combine import write_parquet
from .stats import write_stats

try:
    from langdetect import DetectorFactory, detect as _langdetect

    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:  # optional dependency (test group)
    _HAS_LANGDETECT = False


SOURCE_COL = "corpus_source"
APA_SOURCE = "apa-psyctests"
DOI_PSYCTESTS_COL = "doi_psyctests"
PATCHED_COL = "is_patched"
PDF_FULL_TEXT_COL = "pdf_full_text"


# ---------------------------------------------------------------------------
# Context and scoping
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    """Carries config and collects the report across steps."""

    cfg: dict                      # the ``patch:`` config block
    full_cfg: dict                 # the whole config (llama_cpp, model, ...)
    report_only: bool = False
    refresh: bool = False
    report: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def log(self, line: str) -> None:
        self.report.append(line)

    def stat(self, key: str, value) -> None:
        self.stats[key] = value


def apa_mask(df: pd.DataFrame) -> pd.Series:
    """NA-safe mask of apa-sourced rows; all-True when the frame is untagged."""
    if SOURCE_COL not in df.columns:
        return pd.Series(True, index=df.index)
    return df[SOURCE_COL].eq(APA_SOURCE).fillna(False).astype(bool)


def partial_mask(df: pd.DataFrame) -> pd.Series:
    """NA-safe mask of partial-sourced rows; all-True when untagged."""
    if SOURCE_COL not in df.columns:
        return pd.Series(True, index=df.index)
    src = df[SOURCE_COL]
    return (src.notna() & ~src.eq(APA_SOURCE)).astype(bool)


def _as_object(df: pd.DataFrame, *cols: str) -> None:
    """Cast to object so masked scalar writes don't fail on arrow-backed strings."""
    for col in cols:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(object)


def _doc_keys(df: pd.DataFrame) -> pd.Series:
    """Per-row document key: path, else source+title."""
    keys = (df[PATH_COL].astype(object) if PATH_COL in df.columns
            else pd.Series(None, index=df.index, dtype=object))
    fallback = (
        "doc::" + df[SOURCE_COL].astype("string").fillna("?")
        + "::" + df["meta_title_raw"].astype("string").fillna("?")
    )
    return keys.where(keys.notna(), fallback.astype(object))


def _first(series: pd.Series):
    """First non-null value of a series, else None."""
    nn = series.dropna()
    return nn.iloc[0] if len(nn) else None


# ---------------------------------------------------------------------------
# DOI helpers
# ---------------------------------------------------------------------------

# Kept identical to extraction/extractors/meta.py::_DOI_RE (asserted by test).
_PSYCTESTS_DOI_RE = re.compile(
    r"doi:\s*https?://(?:dx\.)?doi\.org/(10\.1037/t\d{4,6}-\d{3})",
    re.IGNORECASE,
)
_PSYCTESTS_RECORD_RE = re.compile(r"^10\.1037/t\d{4,6}-\d{3}$")
_ANY_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)")
_STEM_ID_RE = re.compile(r"^9999(\d{5})")


def _clean_doi(raw: str) -> str:
    return raw.rstrip(".,;:)]}\"'").lower()


def page1_dois(text: str) -> tuple[Optional[str], Optional[str]]:
    """(PsycTESTS record DOI, source-publication DOI) from page-1 text."""
    m = _PSYCTESTS_DOI_RE.search(text)
    record = m.group(1).lower() if m else None
    source = None
    for cand in _ANY_DOI_RE.findall(text):
        doi = _clean_doi(cand)
        if not _PSYCTESTS_RECORD_RE.match(doi):
            source = doi
            break
    return record, source


def doi_from_stem(path) -> Optional[str]:
    """``9999NNNNN_...`` filename stem -> ``10.1037/tNNNNN-000``."""
    if not isinstance(path, str) or not path:
        return None
    m = _STEM_ID_RE.match(Path(path).stem)
    return f"10.1037/t{m.group(1)}-000" if m else None


def doi_in_text(text) -> Optional[str]:
    """First non-PsycTESTS DOI inside free text (e.g. an APA citation)."""
    if not isinstance(text, str):
        return None
    for cand in _ANY_DOI_RE.findall(text):
        doi = _clean_doi(cand)
        if not _PSYCTESTS_RECORD_RE.match(doi):
            return doi
    return None


def doi_from_url(url) -> Optional[str]:
    """DOI embedded in a URL (``https://doi.org/10.xxxx/...`` and friends)."""
    if not isinstance(url, str):
        return None
    from urllib.parse import unquote

    return doi_in_text(unquote(url))


# ---------------------------------------------------------------------------
# Source-field classification
# ---------------------------------------------------------------------------

_YEAR_PARENS_RE = re.compile(r"\(\s*((?:18|19|20)\d{2})")
_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"^((?:www\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/\S*)?)(?:\s+\(.*\))?$"
)
_FILE_SUFFIXES = {"htm", "html", "pdf", "doc", "docx", "csv", "xls", "xlsx"}


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fix_csv_corruption(text: str) -> str:
    """Strip split-cell artifacts: leading ``,"``, doubled and orphan quotes."""
    t = text.strip()
    if t.startswith(","):
        t = t.lstrip(",").lstrip().lstrip('"')
    t = t.replace('""', '"')
    if t.count('"') == 1 and (t.startswith('"') or t.endswith('"')):
        t = t.strip('"')
    return _collapse_ws(t)


def classify_source(text: str) -> tuple[str, Optional[str]]:
    """Classify a source string -> (``url``/``domain``/``citation``/``junk``, normalized value)."""
    t = _collapse_ws(text)
    if not t:
        return "junk", None
    if _URL_RE.fullmatch(t):
        return "url", t
    m = _DOMAIN_RE.fullmatch(t)
    if m:
        host = m.group(1)
        if host.rsplit(".", 1)[-1].split("/")[0].lower() not in _FILE_SUFFIXES:
            return "domain", f"https://{host}"
    if _YEAR_PARENS_RE.search(t) and len(t) >= 30:
        return "citation", t
    return "junk", None


# ---------------------------------------------------------------------------
# Citations, authors, Crossref
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    family: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None


def parse_citation(text: str) -> Citation:
    """Best-effort parse of an APA-style citation into (family, year, title)."""
    m = _YEAR_PARENS_RE.search(text)
    if not m:
        return Citation()
    year = int(m.group(1))
    family = text[: m.start()].split(",")[0].strip() or None
    title = None
    tm = re.search(r"\)\s*\.\s*(.{5,300}?)[.?!](?:\s|$)", text[m.start():])
    if tm:
        title = _collapse_ws(tm.group(1)) or None
    return Citation(family=family, year=year, title=title)


def first_family(authors) -> Optional[str]:
    """First author's family name from an APA or pipe-delimited author string."""
    if not isinstance(authors, str) or not authors.strip():
        return None
    head = authors.split("|")[0].split(",")[0].split("&")[0].strip()
    return head or None


_INITIALS_RE = re.compile(r"^(?:[A-Z]\.?\s*)+$")


def _to_initials(given: str) -> str:
    parts = re.split(r"[\s.]+", given.strip())
    return " ".join(f"{p[0].upper()}." for p in parts if p)


def format_authors_apa(authors: list[dict]) -> Optional[str]:
    """Crossref author list -> ``Family, F. M., & Family2, G.``."""
    names = []
    for a in authors or []:
        family = (a.get("family") or "").strip()
        if not family:
            continue
        given = _to_initials(a.get("given") or "")
        names.append(f"{family}, {given}".rstrip(", "))
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def parse_pipe_authors(raw: str) -> Optional[str]:
    """Pipe-delimited author list -> APA string, or None when unparseable.

    Also handles scale-hunt's alternating family/initials/affiliation tokens.
    """
    drop = re.compile(
        r"@|univ|depart|school|institute|college|hospital|center|centre|"
        r"faculty|laborator|clinic", re.IGNORECASE,
    )
    tokens = [t.strip() for t in raw.split("|")]
    tokens = [t for t in tokens if t and not drop.search(t) and len(t) <= 40]
    if not tokens:
        return None
    authors: list[dict] = []
    if all("," in t for t in tokens):
        for t in tokens:
            family, _, given = t.partition(",")
            authors.append({"family": family.strip(), "given": given.strip()})
    else:
        pending: Optional[str] = None
        for t in tokens:
            if _INITIALS_RE.fullmatch(t):
                if pending:
                    authors.append({"family": pending, "given": t})
                    pending = None
            elif re.fullmatch(r"[A-Za-z][A-Za-z' -]*", t):
                if pending:
                    authors.append({"family": pending})
                pending = t
            else:
                pending = None
        if pending:
            authors.append({"family": pending})
    return format_authors_apa(authors)


class CrossrefClient:
    """Minimal throttled Crossref client; ``_get`` is the only HTTP call (monkeypatched in tests)."""

    BASE_URL = "https://api.crossref.org/works"
    SELECT = "DOI,title,author,issued,container-title"

    def __init__(self, mailto: str, throttle: float = 1.0):
        self.headers = {
            "User-Agent": f"synthometricon-source/0.1 (mailto:{mailto})"
        }
        self.throttle = throttle
        self._last = 0.0

    def _wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self.throttle:
            time.sleep(self.throttle - delta)
        self._last = time.monotonic()

    def _get(self, url: str, params: Optional[dict]) -> Optional[dict]:
        self._wait()
        resp = httpx.get(url, params=params, headers=self.headers, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def search(self, bibliographic: str, author: Optional[str] = None,
               rows: int = 5) -> list[dict]:
        params = {
            "query.bibliographic": bibliographic,
            "rows": rows,
            "select": self.SELECT,
        }
        if author:
            params["query.author"] = author
        data = self._get(self.BASE_URL, params)
        return (data or {}).get("message", {}).get("items", [])

    def lookup(self, doi: str) -> Optional[dict]:
        data = self._get(f"{self.BASE_URL}/{doi}", None)
        return data.get("message") if data else None


def _work_title(work: dict) -> str:
    return " ".join(work.get("title") or [])


def _work_year(work: dict) -> Optional[int]:
    try:
        return int(work["issued"]["date-parts"][0][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _work_families(work: dict) -> set[str]:
    return {
        a["family"].casefold()
        for a in work.get("author") or []
        if a.get("family")
    }


def _author_matches(family: Optional[str], work: dict) -> bool:
    return bool(family) and family.casefold() in _work_families(work)


def _year_matches(year: Optional[int], work: dict, tolerance: int) -> bool:
    wy = _work_year(work)
    return year is not None and wy is not None and abs(wy - year) <= tolerance


def verify_candidate(cit: Citation, work: dict, *, title_threshold: float,
                     year_tolerance: int) -> bool:
    """Citation-mode acceptance: title similarity + year + first author."""
    if not (cit.title and cit.year and cit.family):
        return False
    wt = _work_title(work)
    if not wt or fuzzy_contains(cit.title, wt) < title_threshold:
        return False
    return _year_matches(cit.year, work, year_tolerance) and _author_matches(
        cit.family, work
    )


def verify_structured(title: str, family: str, year: int, work: dict, *,
                      instrument_threshold: float, year_tolerance: int) -> bool:
    """Structured-mode acceptance: author + year + instrument name in title."""
    wt = _work_title(work)
    if not wt or fuzzy_contains(title, wt) < instrument_threshold:
        return False
    return _year_matches(year, work, year_tolerance) and _author_matches(
        family, work
    )


@dataclass
class Lookup:
    """Per-document Crossref lookup fields."""

    citation: Optional[str] = None
    title: Optional[str] = None
    family: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None

    @property
    def mode(self) -> str:
        if self.citation:
            return "citation"
        if self.title and self.family and self.year:
            return "structured"
        return "insufficient"

    def key(self) -> str:
        if self.citation:
            return "cite::" + _collapse_ws(self.citation)
        return "struct::" + "|".join(
            str(v or "").casefold()
            for v in (self.family, self.year, self.title, self.venue)
        )


def collect_lookup(doc: pd.DataFrame) -> Lookup:
    """Build the lookup fields for one document's rows."""
    raw = _first(doc.get("meta_source_raw", pd.Series(dtype=object)))
    citation = None
    if isinstance(raw, str) and classify_source(raw)[0] == "citation":
        citation = _collapse_ws(raw)
    year = _first(doc.get("meta_publication_year_raw", pd.Series(dtype=object)))
    return Lookup(
        citation=citation,
        title=_first(doc.get("meta_title_raw", pd.Series(dtype=object))),
        family=first_family(_first(doc.get("meta_authors_raw", pd.Series(dtype=object)))),
        year=int(year) if pd.notna(year) else None,
        venue=_first(doc.get("meta_journal_venue_raw", pd.Series(dtype=object))),
    )


def _accepted_verdict(work: dict, mode: str) -> dict:
    return {
        "doi": _clean_doi(work["DOI"]),
        "mode": mode,
        "authors": format_authors_apa(work.get("author")),
        "year": _work_year(work),
        "venue": (work.get("container-title") or [None])[0],
    }


def evaluate_lookup(lookup: Lookup, client: CrossrefClient, cfg: dict) -> dict:
    """Query Crossref for one lookup and return a verdict dict."""
    tt = float(cfg.get("title_threshold", 0.90))
    yt = int(cfg.get("year_tolerance", 1))
    it = float(cfg.get("instrument_threshold", 0.85))
    if lookup.mode == "citation":
        cit = parse_citation(lookup.citation)
        if not (cit.family and cit.year and cit.title):
            return {"doi": None, "mode": "citation", "reason": "unparseable"}
        for work in client.search(lookup.citation):
            if _PSYCTESTS_RECORD_RE.match(_clean_doi(work.get("DOI", ""))):
                continue
            if verify_candidate(cit, work, title_threshold=tt, year_tolerance=yt):
                return _accepted_verdict(work, "citation")
        return {"doi": None, "mode": "citation", "reason": "no match"}
    query = lookup.title if not lookup.venue else f"{lookup.title} {lookup.venue}"
    for work in client.search(query, author=lookup.family):
        if _PSYCTESTS_RECORD_RE.match(_clean_doi(work.get("DOI", ""))):
            continue
        if verify_structured(lookup.title, lookup.family, lookup.year, work,
                             instrument_threshold=it, year_tolerance=yt):
            return _accepted_verdict(work, "structured")
    return {"doi": None, "mode": "structured", "reason": "no match"}


def _load_cache(path) -> dict:
    if path and Path(path).exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def _save_cache(path, cache: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=1)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

_LANG_NAMES = {
    "arabic": "ar", "bengali": "bn", "bulgarian": "bg", "catalan": "ca",
    "chinese": "zh", "croatian": "hr", "czech": "cs", "danish": "da",
    "dutch": "nl", "english": "en", "estonian": "et", "farsi": "fa",
    "filipino": "tl", "finnish": "fi", "french": "fr", "german": "de",
    "greek": "el", "hebrew": "he", "hindi": "hi", "hungarian": "hu",
    "icelandic": "is", "indonesian": "id", "italian": "it", "japanese": "ja",
    "kannada": "kn", "korean": "ko", "kurdish": "ku", "lithuanian": "lt",
    "malay": "ms", "malayalam": "ml", "norwegian": "no", "persian": "fa",
    "polish": "pl", "portuguese": "pt", "romanian": "ro", "russian": "ru",
    "serbian": "sr", "sinhala": "si", "slovak": "sk", "slovenian": "sl",
    "spanish": "es", "swahili": "sw", "swedish": "sv", "tagalog": "tl",
    "thai": "th", "turkish": "tr", "ukrainian": "uk", "urdu": "ur",
    "vietnamese": "vi",
}


def normalize_language(value) -> Optional[str]:
    """Code, locale (``en-US``), JSON-leaked or named (``English (US)``) language -> ISO 639-1, else None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if re.fullmatch(r"[A-Za-z]{2}", v):
        return v.lower()
    m = re.match(r"^([A-Za-z]{2})[-_',’]", v)
    if m:
        return m.group(1).lower()
    m = re.match(r"^([A-Za-z]+)", v)
    if m and m.group(1).casefold() in _LANG_NAMES:
        return _LANG_NAMES[m.group(1).casefold()]
    return None


_TITLE_LANG_RE = re.compile(r"[-–—(]+\s*([A-Za-z]+) [Vv]ersion\b")


def title_language(title) -> Optional[tuple[str, str]]:
    """``... — Kurdish Version`` -> ("ku", "Kurdish Version"), else None."""
    if not isinstance(title, str):
        return None
    m = _TITLE_LANG_RE.search(title)
    if not m:
        return None
    name = m.group(1).casefold()
    code = _LANG_NAMES.get(name)
    if not code or code == "en":
        return None
    return code, f"{name.capitalize()} Version"


def _detect_language(text: str) -> Optional[str]:
    if not _HAS_LANGDETECT or len(text) < 40:
        return None
    try:
        code = _langdetect(text)
    except Exception:
        return None
    return code.split("-")[0] if code else None


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

_PERMISSIONS_VOCAB = {
    "Contact Corresponding Author.",
    "Contact Publisher.",
    "Contact Publisher and Corresponding Author.",
    "Not Specified.",
    "May use for Research/Teaching.",
}


def permissions_category(raw) -> Optional[str]:
    """Fixed mapping from a raw permissions string to its category."""
    if not isinstance(raw, str):
        return None
    low = _collapse_ws(raw).casefold()
    if low.startswith("test content may be reproduced"):
        return "public_domain" if "public domain" in low else "free_reproduction"
    if low.startswith("contact publisher and corresponding author"):
        return "contact_publisher_and_author"
    if low.startswith("contact publisher"):
        return "contact_publisher"
    if low.startswith("contact corresponding author"):
        return "contact_author"
    if low.startswith("not specified"):
        return "not_specified"
    if low.startswith("may use for research/teaching"):
        return "research_teaching"
    return None


# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------

_ITEM_NUM_RE = re.compile(r"^\s*(?:\(?\d{1,3}[.)]|[A-Za-z][.)])\s+")


def strip_item_number(text: str) -> str:
    """Strip a leading ``1. `` / ``2) `` / ``a. `` numbering prefix."""
    m = _ITEM_NUM_RE.match(text)
    if not m:
        return text
    rest = text[m.end():].strip()
    return rest if len(rest) >= 3 else text


def _is_empty_sequence(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0


def _unescape_html_fixpoint(text: str) -> str:
    for _ in range(4):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


# ---------------------------------------------------------------------------
# Name-column helpers: version-qualifier relocation and title casing
# ---------------------------------------------------------------------------

# A trailing qualifier is accepted as version information only when its last
# word is one of these (loose grammar, explicit separators only) ...
_VERSION_LAST_WORDS = {
    "version", "versions", "form", "forms", "edition", "editions",
    "translation", "translations", "adaptation", "adaptations",
    "revision", "revisions", "variant", "variants",
    "revised", "adapted", "modified", "abbreviated", "abridged",
}
# ... or when the whole tail matches one of these (tight grammar — the only
# grammar allowed after weak separators: comma, unspaced hyphen/en dash).
_TIGHT_TAIL_RES = [re.compile(p, re.IGNORECASE) for p in (
    r"(?:very\s+)?(?:short|long|brief|revised|adapted|modified|extended|"
    r"expanded|shortened)[\s-](?:version|form)",
    r"revised|adapted|modified",
    r"forms?\s[\w.()/-]{1,8}",
    r"versions?\s[\w.()/-]{1,12}",
    r"(?:\d+(?:st|nd|rd|th)|second|third|fourth|fifth)\sedition",
)]
# Explicit separators; a single en dash is excluded because it occurs in names (DSM–IV).
_NAME_DELIM_RE = re.compile(r"\s*(?:[-–—]{2,}|—|\s-\s?)\s*")
_PAREN_TAIL_RE = re.compile(r"^(.*\S)\s*\(([^()]{2,60})\)$")
# A loose tail containing an instrument noun is a subscale/sub-measure name,
# not a version qualifier (e.g. "Survey--X Behavior Scale Revised").
_INSTRUMENT_NOUNS = {
    "scale", "scales", "questionnaire", "inventory", "survey", "index",
    "checklist", "measure", "measures", "test", "task", "interview",
    "schedule", "screen", "battery", "instrument",
}


def _tight_tail_ok(tail: str) -> bool:
    t = _collapse_ws(tail)
    return any(r.fullmatch(t) for r in _TIGHT_TAIL_RES)


def _loose_tail_ok(tail: str) -> bool:
    if tail.count("(") != tail.count(")"):
        return False
    words = re.findall(r"[A-Za-z]+", tail.lower())
    if words and words[-1] in _VERSION_LAST_WORDS:
        return not _INSTRUMENT_NOUNS.intersection(words)
    return _tight_tail_ok(tail)


def _version_base_ok(base: str) -> bool:
    b = base.rstrip(" -–—,;:").strip()
    return len(b) >= 3 and any(c.isalpha() for c in b)


def _split_version_once(name: str) -> tuple[str, Optional[str]]:
    m = _PAREN_TAIL_RE.match(name)
    if m and _loose_tail_ok(m.group(2)) and _version_base_ok(m.group(1)):
        return m.group(1).rstrip(" -–—,;:"), _collapse_ws(m.group(2))
    matches = list(_NAME_DELIM_RE.finditer(name))
    if matches:
        m = matches[-1]
        base, tail = name[: m.start()], name[m.end():]
        if 0 < len(tail) <= 60 and _loose_tail_ok(tail) and _version_base_ok(base):
            return base.rstrip(" -–—,;:"), _collapse_ws(tail)
    if "," in name:
        base, _, tail = name.rpartition(",")
        tail = tail.strip()
        if 0 < len(tail) <= 60 and _tight_tail_ok(tail) and _version_base_ok(base):
            return base.rstrip(" -–—,;:"), tail
    i = max(name.rfind("-"), name.rfind("–"))
    if i > 0:
        base, tail = name[:i], name[i + 1:].strip()
        if 0 < len(tail) <= 60 and _tight_tail_ok(tail) and _version_base_ok(base):
            return base.rstrip(" -–—,;:"), tail
    return name, None


def split_version_qualifier(name) -> tuple:
    """``"X--adapted version"`` -> ``("X", "adapted version")``; multiple qualifiers joined with ``"; "``."""
    if not isinstance(name, str):
        return name, None
    base, quals = _collapse_ws(name), []
    while True:
        b, q = _split_version_once(base)
        if q is None:
            break
        base, quals = b, [q] + quals
    if not quals:
        return name, None
    return base, "; ".join(quals)


_MINOR_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "as", "at", "by", "in", "of", "off", "on", "per", "to", "up", "via", "vs",
    "de", "del", "der", "des", "di", "du", "la", "le", "van", "von",
    "und", "et", "y",
}


def _cap_part(part: str, first: bool) -> str:
    if not part or part != part.lower():
        return part  # any uppercase present: leave as-is (acronyms, McX, ...)
    if not first and part in _MINOR_WORDS:
        return part  # hyphenated minor part: "Finger-to-Nose"
    for i, ch in enumerate(part):
        if ch.isalpha():
            return part[:i] + ch.upper() + part[i + 1:]
    return part


def title_case_name(text):
    """APA-style title case that never downcases existing uppercase."""
    if not isinstance(text, str):
        return text
    tokens = text.split(" ")
    n = len(tokens)
    out = []
    force_cap = True  # first word, and words after a colon
    for i, tok in enumerate(tokens):
        core = tok.strip("()[]{}\"'.,;:!?")
        if (core.lower() in _MINOR_WORDS and not force_cap and i < n - 1):
            # keep lowercase; downcase a Capitalized (not all-caps) minor word
            if core == core.capitalize() and core != core.lower():
                tok = tok.replace(core, core.lower(), 1)
            out.append(tok)
        else:
            parts = re.split(r"([-/]+)", tok)
            out.append("".join(
                _cap_part(p, first=(j == 0)) if j % 2 == 0 else p
                for j, p in enumerate(parts)))
        if tok:
            force_cap = tok.endswith((":", ";"))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

_INT_COLS = [
    "item_item_id", "meta_publication_year_raw", "scale_id", "scale_depth",
    "meta_page_count", "meta_image_count", "meta_char_count_excl_first_page",
]


def step_schema(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Dtype and emptiness hygiene."""
    for col in ("meta_doi_raw", "meta_source_doi"):
        if col in df.columns and df[col].dtype.kind == "f" and df[col].isna().all():
            df[col] = df[col].astype(object)
    if "meta_source_doi" not in df.columns:
        df["meta_source_doi"] = pd.Series(None, index=df.index, dtype=object)
    for col in _INT_COLS:
        if col in df.columns and df[col].dtype.kind == "f":
            nn = df[col].dropna()
            if len(nn) == 0 or (nn % 1 == 0).all():
                df[col] = df[col].astype("Int64")
    blanked = 0
    for col in df.columns:
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        try:
            mask = df[col].str.fullmatch(r"\s*", na=False)
        except AttributeError:  # object column of non-strings (bools, lists)
            continue
        if mask.any():
            _as_object(df, col)
            df.loc[mask, col] = None
            blanked += int(mask.sum())
    ctx.log(f"schema: {blanked} blank/whitespace cells -> null; "
            f"int-cast {[c for c in _INT_COLS if c in df.columns]}")
    ctx.stat("schema.blank_cells_nulled", blanked)
    return df


_SENTINELS = {"None", "N/A", "UNKNOWN", "Unknown", "nan", "-"}
_SENTINEL_COLS = [
    "item_item_text", "item_admin_note", "item_language", "meta_title_raw",
    "scale_name", "scale_construct_name", "meta_permissions_raw",
]
_SOURCE_STUBS = {
    "supplied by author": "Supplied by author.",
    "supplied by authors": "Supplied by author.",
    "supplied by the author": "Supplied by author.",
    "supplied author": "Supplied by author.",
    "author supplied": "Supplied by author.",
    "author provided": "Supplied by author.",
    "provided by author": "Supplied by author.",
    "submitted by author": "Supplied by author.",
    "supplied by publisher": "Supplied by publisher.",
}
_PLACEHOLDER_ITEM_RE = re.compile(r"^Item \d+ statement group$")


def step_text_repair(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Sentinel strings, mojibake, escapes, placeholders, stub casing."""
    pmask = partial_mask(df)
    for col in _SENTINEL_COLS:
        if col not in df.columns:
            continue
        mask = df[col].map(lambda v: isinstance(v, str) and v.strip() in _SENTINELS)
        if mask.any():
            _as_object(df, col)
            df.loc[mask, col] = None
            ctx.log(f"text_repair: {col}: {int(mask.sum())} sentinel cells -> null")
            ctx.stat(f"text_repair.sentinel_cells_nulled.{col}", int(mask.sum()))

    if "version_attached" in df.columns:
        mask = df["version_attached"].eq("none")
        if mask.any():
            _as_object(df, "version_attached")
            df.loc[mask, "version_attached"] = None
            ctx.log(f"text_repair: version_attached: {int(mask.sum())} 'none' -> null")
            ctx.stat("text_repair.version_attached_none_nulled", int(mask.sum()))

    if "meta_source_raw" in df.columns:
        src = df[SOURCE_COL] if SOURCE_COL in df.columns else pd.Series(index=df.index)
        sn = src.eq("semanticnet").fillna(False)
        # fix source_url alongside meta_source_raw so byte-copy pairs stay in
        # sync for the source_fields step
        for col in ("meta_source_raw", "source_url"):
            if col not in df.columns:
                continue
            raw = df.loc[sn, col]
            fixed = raw.map(
                lambda v: re.sub(r"(?<=[A-Za-z])\?(?=[A-Za-z])", "-",
                                 v.replace("¶", " ")).strip()
                if isinstance(v, str) else v
            )
            changed = raw.notna() & raw.ne(fixed)
            if changed.any():
                _as_object(df, col)
                df.loc[changed[changed].index, col] = fixed[changed]
                ctx.log(f"text_repair: {col}: {int(changed.sum())} semanticnet "
                        "mojibake cells repaired ('?'->'-', pilcrow stripped)")
                ctx.stat(f"text_repair.semanticnet_mojibake_repaired.{col}",
                         int(changed.sum()))

        # apa stub canonicalization + placeholder nulling
        am = apa_mask(df)
        raw = df.loc[am, "meta_source_raw"]
        canon = raw.map(
            lambda v: _SOURCE_STUBS.get(_collapse_ws(v).rstrip(".").casefold())
            if isinstance(v, str) and len(v) < 40 else None
        )
        changed = canon.notna() & raw.ne(canon)
        if changed.any():
            _as_object(df, "meta_source_raw")
            df.loc[changed[changed].index, "meta_source_raw"] = canon[changed]
            ctx.log(f"text_repair: meta_source_raw: {int(changed.sum())} apa stub "
                    "casing variants canonicalized")
            ctx.stat("text_repair.apa_source_stubs_canonicalized",
                     int(changed.sum()))
        placeholder = df["meta_source_raw"].eq("(replace with real citation)")
        if placeholder.any():
            _as_object(df, "meta_source_raw")
            df.loc[placeholder, "meta_source_raw"] = None
            ctx.log(f"text_repair: meta_source_raw: {int(placeholder.sum())} "
                    "placeholder cells -> null")
            ctx.stat("text_repair.source_placeholders_nulled",
                     int(placeholder.sum()))

    if "retrieval_notes" in df.columns:
        notes = df.loc[pmask, "retrieval_notes"]
        fixed = notes.map(
            lambda v: _unescape_html_fixpoint(v) if isinstance(v, str) else v
        )
        changed = notes.notna() & notes.ne(fixed)
        if changed.any():
            _as_object(df, "retrieval_notes")
            df.loc[changed[changed].index, "retrieval_notes"] = fixed[changed]
            ctx.log(f"text_repair: retrieval_notes: {int(changed.sum())} cells "
                    "HTML-unescaped")
            ctx.stat("text_repair.retrieval_notes_unescaped", int(changed.sum()))

    if "item_admin_note" in df.columns:
        notes = df["item_admin_note"]
        mask = notes.map(lambda v: isinstance(v, str) and "\\n" in v)
        if mask.any():
            _as_object(df, "item_admin_note")
            df.loc[mask, "item_admin_note"] = notes[mask].str.replace("\\n", "\n")
            ctx.log(f"text_repair: item_admin_note: {int(mask.sum())} literal "
                    r"'\n' escapes -> newline")
            ctx.stat("text_repair.admin_note_escapes_fixed", int(mask.sum()))

    if "item_item_text" in df.columns:
        texts = df.loc[pmask, "item_item_text"]
        mask = texts.map(
            lambda v: isinstance(v, str) and bool(_PLACEHOLDER_ITEM_RE.match(v.strip()))
        )
        if mask.any():
            _as_object(df, "item_item_text")
            df.loc[mask[mask].index, "item_item_text"] = None
            ctx.log(f"text_repair: item_item_text: {int(mask.sum())} "
                    "'Item N statement group' placeholders -> null")
            ctx.stat("text_repair.item_placeholders_nulled", int(mask.sum()))
    return df


def step_meta_dois(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Re-read page 1 of every apa PDF; PDF DOIs are authoritative."""
    from extraction.pdf_io import pdf_first_page_text

    am = apa_mask(df)
    has_meta = df["meta_language"].notna() if "meta_language" in df.columns else True
    paths = df.loc[am & has_meta, PATH_COL].dropna().unique()
    found: dict[str, tuple[Optional[str], Optional[str]]] = {}
    missing = errors = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="meta_dois: page-1 reads", unit="pdf")
    except ImportError:
        iterator = paths
    for path in iterator:
        if not Path(path).exists():
            missing += 1
            continue
        try:
            found[path] = page1_dois(pdf_first_page_text(path))
        except Exception:
            errors += 1
    record = df.loc[am, PATH_COL].map({p: v[0] for p, v in found.items()})
    source = df.loc[am, PATH_COL].map({p: v[1] for p, v in found.items()})
    _as_object(df, "meta_doi_raw", "meta_source_doi")
    for col, pdf_vals in (("meta_doi_raw", record), ("meta_source_doi", source)):
        overwrites = int((pdf_vals.notna() & df.loc[am, col].ne(pdf_vals)).sum())
        df.loc[am, col] = pdf_vals.combine_first(df.loc[am, col])
        ctx.log(f"meta_dois: {col}: {overwrites} cells set from PDF page 1")
        ctx.stat(f"meta_dois.cells_set_from_pdf.{col}", overwrites)
    ctx.log(f"meta_dois: {len(found)} PDFs read, {missing} missing, {errors} errors")
    ctx.stat("meta_dois.pdfs", {"read": len(found), "missing": missing,
                                "errors": errors})
    return df


def step_pdf_full_text(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Read each apa PDF's body text (excluding page 1) into ``pdf_full_text``."""
    from extraction.pdf_io import pdf_text_excl_first_page

    am = apa_mask(df)
    paths = df.loc[am, PATH_COL].dropna().unique()
    found: dict[str, str] = {}
    missing = errors = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="pdf_full_text: full-text reads", unit="pdf")
    except ImportError:
        iterator = paths
    for path in iterator:
        if not Path(path).exists():
            missing += 1
            continue
        try:
            found[path] = pdf_text_excl_first_page(path)
        except Exception:
            errors += 1
    df[PDF_FULL_TEXT_COL] = pd.Series(pd.NA, index=df.index, dtype=object)
    df.loc[am, PDF_FULL_TEXT_COL] = df.loc[am, PATH_COL].map(found)
    lengths = pd.Series([len(t) for t in found.values()], dtype="float64")
    chars = int(lengths.sum())
    mean = float(lengths.mean()) if len(lengths) else 0.0
    sd = float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0
    q1, med, q3 = (float(v) for v in (lengths.quantile([0.25, 0.5, 0.75])
                                      if len(lengths) else [0.0] * 3))
    ctx.log(f"pdf_full_text: {len(found)} PDFs read ({chars:,} chars; per "
            f"PDF M {mean:,.0f} (SD {sd:,.0f}), Mdn {med:,.0f} "
            f"[IQR {q1:,.0f}-{q3:,.0f}]), {missing} missing, {errors} errors")
    ctx.stat("pdf_full_text.pdfs", {"read": len(found), "chars": chars,
                                    "chars_mean": round(mean, 1),
                                    "chars_sd": round(sd, 1),
                                    "chars_median": round(med, 1),
                                    "chars_iqr_q1": round(q1, 1),
                                    "chars_iqr_q3": round(q3, 1),
                                    "missing": missing, "errors": errors})
    return df


def _flatten_strings(value) -> list[str]:
    """Flatten the arbitrarily nested list-of-lists used by the meta records."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for v in value for s in _flatten_strings(v)]
    return []


def _load_meta_dois(path) -> dict[str, str]:
    """``data.meta`` -> {pdf basename: PsycTESTS DOI} from each record's ``TestFileList``."""
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    mapping: dict[str, str] = {}
    for record in records:
        doi = record.get("DOI")
        files = record.get("TestFileList")
        if not isinstance(doi, str) or not isinstance(files, dict):
            continue
        for key, value in files.items():
            if not key.startswith("TestFile"):
                continue
            for name in _flatten_strings(value):
                if name.lower().endswith(".pdf"):
                    mapping[name] = doi.lower()
    return mapping


def step_doi_psyctests(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Resolve the scalar PsycTESTS DOI column."""
    meta_dois = _load_meta_dois((ctx.full_cfg.get("data") or {}).get("meta"))
    am, pm = apa_mask(df), partial_mask(df)
    result = pd.Series(None, index=df.index, dtype=object)

    apa_paths = df.loc[am, PATH_COL]
    uniq = apa_paths.dropna().unique()
    stem = apa_paths.map({p: doi_from_stem(p) for p in uniq})
    meta = apa_paths.map({p: meta_dois.get(Path(p).name) for p in uniq})
    pdf = (df.loc[am, "meta_doi_raw"] if "meta_doi_raw" in df.columns
           else pd.Series(None, index=apa_paths.index, dtype=object))
    candidates = pd.concat(
        [stem.astype(object), meta.astype(object), pdf.astype(object)], axis=1)
    n_distinct = candidates.nunique(axis=1, dropna=True)
    agreed = candidates.bfill(axis=1).iloc[:, 0].where(n_distinct.eq(1))
    result.loc[am] = agreed
    disagreements = int(apa_paths[n_distinct > 1].nunique())

    stem_fill = df.loc[pm, PATH_COL].map(doi_from_stem)
    result.loc[pm] = stem_fill

    df[DOI_PSYCTESTS_COL] = result
    if "doi" in df.columns:
        df = df.drop(columns=["doi"])
        ctx.log("doi_psyctests: dropped legacy 'doi' column")
    ctx.log(f"doi_psyctests: {int(result[am].notna().sum())} apa rows resolved "
            f"({disagreements} docs with disagreeing candidates left null), "
            f"{int(result[pm].notna().sum())} partial rows filled from stem ids")
    ctx.stat("doi_psyctests.apa_rows_resolved", int(result[am].notna().sum()))
    ctx.stat("doi_psyctests.docs_with_disagreeing_candidates", disagreements)
    ctx.stat("doi_psyctests.partial_rows_filled_from_stem",
             int(result[pm].notna().sum()))
    return df


def step_source_fields(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Repair and reclassify partial meta_source_raw / source_url."""
    pm = partial_mask(df)
    if "meta_source_raw" not in df.columns:
        return df
    _as_object(df, "meta_source_raw", "source_url", "retrieval_notes")
    counts = {"citation": 0, "url": 0, "domain": 0, "junk": 0, "mismatch": 0}
    for i in df.index[pm]:
        raw = df.at[i, "meta_source_raw"]
        if not isinstance(raw, str):
            continue
        url = df.at[i, "source_url"] if "source_url" in df.columns else None
        url_is_copy = pd.isna(url) or url == raw
        if not url_is_copy:
            counts["mismatch"] += 1
            continue
        cleaned = fix_csv_corruption(raw)
        kind, value = classify_source(cleaned)
        counts[kind] += 1
        if kind == "citation":
            df.at[i, "meta_source_raw"] = value
            if pd.notna(url):
                df.at[i, "source_url"] = None
        elif kind in ("url", "domain"):
            df.at[i, "source_url"] = value
            df.at[i, "meta_source_raw"] = None
            if kind == "domain":
                note = f"source_url normalized from '{cleaned}' [patch]"
                existing = df.at[i, "retrieval_notes"] if "retrieval_notes" in df.columns else None
                if not isinstance(existing, str):
                    df.at[i, "retrieval_notes"] = note
                elif note not in existing:
                    df.at[i, "retrieval_notes"] = f"{existing}; {note}"
        else:  # junk
            df.at[i, "meta_source_raw"] = None
            if pd.notna(url):
                df.at[i, "source_url"] = None
    ctx.stat("source_fields.partial_rows_classified", dict(counts))
    ctx.log("source_fields: partial rows classified "
            f"citation={counts['citation']} url={counts['url']} "
            f"domain={counts['domain']} junk={counts['junk']}; "
            f"{counts['mismatch']} rows skipped (source_url differs from raw)")
    return df


def step_source_doi(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Fill meta_source_doi from DOIs already present in text/URLs."""
    _as_object(df, "meta_source_doi")
    filled_text = filled_url = 0
    if "meta_source_raw" in df.columns:
        target = df["meta_source_doi"].isna() & df["meta_source_raw"].notna()
        uniq = {v: doi_in_text(v) for v in df.loc[target, "meta_source_raw"].unique()}
        extracted = df.loc[target, "meta_source_raw"].map(uniq)
        filled_text = int(extracted.notna().sum())
        df.loc[target, "meta_source_doi"] = extracted
    if "source_url" in df.columns:
        target = df["meta_source_doi"].isna() & df["source_url"].notna()
        extracted = df.loc[target, "source_url"].map(doi_from_url)
        filled_url = int(extracted.notna().sum())
        df.loc[target, "meta_source_doi"] = extracted
    ctx.log(f"source_doi: filled {filled_text} rows from citation text, "
            f"{filled_url} from source_url")
    ctx.stat("source_doi.rows_filled", {"from_citation_text": filled_text,
                                        "from_source_url": filled_url})
    return df


def _propagate_doc_doi(df: pd.DataFrame, ctx: Ctx) -> None:
    """Fill null meta_source_doi from a unanimous value in the same doc."""
    keys = _doc_keys(df)
    s = df["meta_source_doi"]
    grp = s.groupby(keys)
    unanimous = grp.transform("nunique").eq(1) & grp.transform("first").notna()
    mask = s.isna() & unanimous
    df.loc[mask, "meta_source_doi"] = grp.transform("first")[mask]
    ctx.log(f"crossref: {int(mask.sum())} rows filled by intra-document propagation")
    ctx.stat("crossref.rows_filled_by_propagation", int(mask.sum()))


def step_crossref(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Crossref lookups for documents still missing a source DOI."""
    _as_object(df, "meta_source_doi")
    _propagate_doc_doi(df, ctx)

    cfg = ctx.cfg.get("crossref", {})
    if not cfg.get("enabled", True):
        ctx.log("crossref: disabled")
        return df
    cache = {} if ctx.refresh else _load_cache(cfg.get("cache"))
    live = bool(cfg.get("mailto")) and not ctx.report_only
    client = CrossrefClient(cfg["mailto"]) if live else None

    keys = _doc_keys(df)
    # Only docs with no DOI at all; partially-filled ones hold disagreeing values.
    doc_has_doi = df["meta_source_doi"].notna().groupby(keys).transform("any")
    missing = df["meta_source_doi"].isna() & ~doc_has_doi
    lookups: dict[str, tuple[Lookup, list]] = {}
    insufficient = 0
    for key, idx in df.loc[missing].groupby(keys[missing]).indices.items():
        rows = df.loc[missing].index[list(idx)]
        lookup = collect_lookup(df.loc[rows])
        if lookup.mode == "insufficient":
            insufficient += 1
            continue
        entry = lookups.setdefault(lookup.key(), (lookup, []))
        entry[1].extend(rows)

    stats = {"cached": 0, "queried": 0, "accepted": 0, "offline": 0, "errors": 0}
    dirty = False
    items = list(lookups.items())
    try:
        from tqdm import tqdm
        items = tqdm(items, desc="crossref: lookups", unit="doc")
    except ImportError:
        pass
    pm = partial_mask(df)
    _as_object(df, "meta_authors_raw", "meta_journal_venue_raw")
    for key, (lookup, rows) in items:
        if key in cache:
            verdict = cache[key]
            stats["cached"] += 1
        elif client is None:
            stats["offline"] += 1
            continue
        else:
            try:
                verdict = evaluate_lookup(lookup, client, cfg)
            except (httpx.HTTPError, KeyError, ValueError):
                stats["errors"] += 1
                continue
            cache[key] = verdict
            dirty = True
            stats["queried"] += 1
            if cfg.get("cache") and stats["queried"] % 50 == 0:
                _save_cache(cfg["cache"], cache)  # crash resilience
        if not verdict.get("doi"):
            continue
        stats["accepted"] += 1
        rows = pd.Index(rows)
        fill = rows[df.loc[rows, "meta_source_doi"].isna()]
        df.loc[fill, "meta_source_doi"] = verdict["doi"]
        # backfill bibliographic fields on partial rows only
        prows = rows[pm[rows]]
        for col, val in (("meta_authors_raw", verdict.get("authors")),
                         ("meta_publication_year_raw", verdict.get("year")),
                         ("meta_journal_venue_raw", verdict.get("venue"))):
            if val is None or col not in df.columns:
                continue
            null_rows = prows[df.loc[prows, col].isna()]
            df.loc[null_rows, col] = val
    if dirty and live and cfg.get("cache"):
        _save_cache(cfg["cache"], cache)
    ctx.stat("crossref.lookups", {"total": len(lookups), **stats,
                                  "insufficient_anchors": insufficient})
    ctx.log(f"crossref: {len(lookups)} lookups ({stats['cached']} cached, "
            f"{stats['queried']} live, {stats['offline']} skipped offline, "
            f"{stats['errors']} errors); {stats['accepted']} accepted; "
            f"{insufficient} docs had insufficient anchors")
    return df


class SourceGuess(BaseModel):
    """Wire model for the doi_probe LLM proposal."""

    author: str
    year: int
    title: str
    journal: Optional[str] = None
    doi: Optional[str] = None


_PROBE_SYSTEM = (
    "You are a precise bibliographic research assistant specialising in "
    "psychological assessment instruments. You identify the original "
    "publication in which a given instrument was first introduced."
)
_PROBE_RULES = """
Return a JSON object with fields:
- author: family name of the first author of the original publication
- year: publication year (integer)
- title: full title of the original article or book
- journal: journal or venue name, or null if unknown
- doi: the publication's DOI ONLY if you are certain you know it; otherwise
  null. NEVER construct, derive, or guess a DOI.
If you cannot identify the original publication, still give your best
single guess for author/year/title from the information provided.
"""


def _probe_user_prompt(doc: dict) -> str:
    lines = [f"Instrument name: {doc['title']}"]
    if doc.get("authors"):
        lines.append(f"Known authors: {doc['authors']}")
    if doc.get("year"):
        lines.append(f"Known publication year: {doc['year']}")
    if doc.get("venue"):
        lines.append(f"Known journal/venue: {doc['venue']}")
    if doc.get("source"):
        lines.append(f"Source note: {doc['source']}")
    lines.append(_PROBE_RULES)
    return "\n".join(lines)


def _ask_llm(doc: dict, full_cfg: dict) -> SourceGuess:
    from extraction.config import build_options, resolve_options, resolve_think
    from extraction.llm import stream_and_collect
    from extraction.models._schema import inline_schema

    options = build_options(resolve_options(full_cfg, "doi_probe"))
    think = resolve_think(full_cfg, "doi_probe")
    last_err: Exception = ValueError("no attempts")
    for _ in range(int(full_cfg.get("max_retries", 2)) + 1):
        _, content, _, _ = stream_and_collect(
            full_cfg["llama_cpp"]["base_url"], full_cfg.get("model", ""),
            _PROBE_SYSTEM, _probe_user_prompt(doc),
            images=[], image_format=full_cfg.get("images", {}).get("format", "png"),
            format_schema=inline_schema(SourceGuess), options=options,
            think=think, verbose=False, timeout=full_cfg.get("timeout"),
        )
        try:
            return SourceGuess.model_validate_json(extract_json(content))
        except (ValidationError, ValueError) as err:
            last_err = err
    raise last_err


def verify_guess(guess: SourceGuess, doc: dict, client: CrossrefClient,
                 cfg: dict, probe_cfg: dict) -> dict:
    """Crossref-verify an LLM proposal, anchored on row metadata."""
    yt = int(cfg.get("year_tolerance", 1))
    tt = float(cfg.get("title_threshold", 0.90))
    it = float(probe_cfg.get("instrument_threshold", 0.60))
    anchor_family = first_family(doc.get("authors")) or guess.author
    anchor_year = doc.get("year") or guess.year

    def _accept(work: dict) -> bool:
        doi = _clean_doi(work.get("DOI", ""))
        if not doi or _PSYCTESTS_RECORD_RE.match(doi):
            return False
        if not _author_matches(anchor_family, work):
            return False
        if not _year_matches(int(anchor_year), work, yt):
            return False
        wt = _work_title(work)
        return bool(wt) and (
            fuzzy_contains(doc["title"], wt) >= it
            or fuzzy_contains(guess.title, wt) >= tt
        )

    if guess.doi:
        work = client.lookup(_clean_doi(guess.doi))
        if work and _accept(work):
            return _accepted_verdict(work, "probe-doi")
    query = f"{guess.author} ({guess.year}). {guess.title}"
    if guess.journal:
        query += f" {guess.journal}"
    for work in client.search(query, author=guess.author):
        if _accept(work):
            return _accepted_verdict(work, "probe-search")
    return {"doi": None, "mode": "probe", "reason": "not verified"}


def _probe_offline_reason(ctx: Ctx, probe_cfg: dict) -> Optional[str]:
    if ctx.report_only:
        return "report-only"
    if not probe_cfg.get("enabled", True):
        return "doi_probe.enabled: false"
    if not ctx.cfg.get("crossref", {}).get("mailto"):
        return "no crossref.mailto for verification"
    base_url = ctx.full_cfg.get("llama_cpp", {}).get("base_url")
    if not base_url:
        return "no llama_cpp.base_url"
    try:
        httpx.get(f"{base_url.rstrip('/')}/models", timeout=3)
    except httpx.HTTPError:
        return f"llama-server unreachable at {base_url}"
    return None


def step_doi_probe(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """LLM-proposed, Crossref-verified source DOIs for partial docs."""
    probe_cfg = ctx.cfg.get("doi_probe", {})
    _as_object(df, "meta_source_doi")
    pm = partial_mask(df)
    keys = _doc_keys(df)

    docs: list[tuple[str, dict, pd.Index]] = []
    for key, idx in df.loc[pm].groupby(keys[pm]).indices.items():
        rows = df.loc[pm].index[list(idx)]
        sub = df.loc[rows]
        if sub["meta_source_doi"].notna().any():
            continue
        title = _first(sub["meta_title_raw"])
        if not title:
            continue
        path = _first(sub[PATH_COL])
        year = _first(sub.get("meta_publication_year_raw", pd.Series(dtype=object)))
        docs.append((
            Path(path).stem if isinstance(path, str) else str(key),
            {
                "title": title,
                "authors": _first(sub.get("meta_authors_raw", pd.Series(dtype=object))),
                "year": int(year) if pd.notna(year) else None,
                "venue": _first(sub.get("meta_journal_venue_raw", pd.Series(dtype=object))),
                "source": _first(sub.get("meta_source_raw", pd.Series(dtype=object))),
            },
            rows,
        ))

    cache = {} if ctx.refresh else _load_cache(probe_cfg.get("cache"))
    offline = _probe_offline_reason(ctx, probe_cfg)
    client = (CrossrefClient(ctx.cfg["crossref"]["mailto"])
              if offline is None else None)
    stats = {"cached": 0, "probed": 0, "accepted": 0, "offline": 0, "errors": 0}
    dirty = False
    iterator = docs
    try:
        from tqdm import tqdm
        iterator = tqdm(docs, desc="doi_probe", unit="doc")
    except ImportError:
        pass
    for stem, doc, rows in iterator:
        if stem in cache:
            verdict = cache[stem]
            stats["cached"] += 1
        elif client is None:
            stats["offline"] += 1
            continue
        else:
            try:
                guess = _ask_llm(doc, ctx.full_cfg)
                verdict = verify_guess(guess, doc, client,
                                       ctx.cfg.get("crossref", {}), probe_cfg)
                verdict["guess"] = guess.model_dump()
            except Exception as err:
                stats["errors"] += 1
                ctx.log(f"doi_probe: {stem}: error: {err}")
                continue
            cache[stem] = verdict
            dirty = True
            stats["probed"] += 1
            if probe_cfg.get("cache") and stats["probed"] % 10 == 0:
                _save_cache(probe_cfg["cache"], cache)  # crash resilience
        if verdict.get("doi"):
            stats["accepted"] += 1
            fill = rows[df.loc[rows, "meta_source_doi"].isna()]
            df.loc[fill, "meta_source_doi"] = verdict["doi"]
    if dirty and probe_cfg.get("cache"):
        _save_cache(probe_cfg["cache"], cache)
    note = f" (cache-only: {offline})" if offline else ""
    ctx.stat("doi_probe.docs", {"targets": len(docs), **stats,
                                "cache_only_reason": offline})
    ctx.log(f"doi_probe: {len(docs)} target docs{note}; {stats['cached']} cached, "
            f"{stats['probed']} probed live, {stats['offline']} skipped offline, "
            f"{stats['errors']} errors; {stats['accepted']} DOIs filled")
    return df


def step_language(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Sanitize, normalize, and backfill language columns."""
    pm = partial_mask(df)
    _as_object(df, "meta_language", "meta_language_raw", "item_language")

    # Partial meta_language_raw byte-copying meta_language is a scraper artifact.
    if {"meta_language", "meta_language_raw"} <= set(df.columns):
        copy_fill = (pm & df["meta_language_raw"].notna()
                     & df["meta_language_raw"].eq(df["meta_language"]))
        df.loc[copy_fill, "meta_language_raw"] = None
        ctx.log(f"language: {int(copy_fill.sum())} partial copy-filled "
                "meta_language_raw cells -> null")
        ctx.stat("language.copy_filled_raw_nulled", int(copy_fill.sum()))

    for col in ("meta_language", "item_language"):
        if col not in df.columns:
            continue
        vals = df[col]
        needs = vals.notna() & ~vals.map(
            lambda v: isinstance(v, str) and bool(re.fullmatch(r"[a-z]{2}", v))
        )
        normalized = vals[needs].map(normalize_language)
        unmapped = int(normalized.isna().sum())
        df.loc[needs[needs].index, col] = normalized
        ctx.log(f"language: {col}: {int(needs.sum())} non-ISO values normalized "
                f"({unmapped} unmapped -> null)")
        ctx.stat(f"language.non_iso_normalized.{col}",
                 {"normalized": int(needs.sum()), "unmapped": unmapped})

    # title scan (partials only; apa meta_language is PDF-observed)
    if "meta_title_raw" in df.columns:
        hits = df.loc[pm, "meta_title_raw"].map(title_language).dropna()
        for i, (code, note) in hits.items():
            df.at[i, "meta_language"] = code
            df.at[i, "meta_language_raw"] = note
        ctx.log(f"language: {len(hits)} partial rows set from '<Language> Version' titles")
        ctx.stat("language.rows_set_from_version_titles", len(hits))

    # langdetect fill-only-null, grouped per document
    if _HAS_LANGDETECT and "item_item_text" in df.columns:
        keys = _doc_keys(df)
        need = df["item_language"].isna() | df["meta_language"].isna()
        filled = disagreed = 0
        for key in keys[need].unique():
            rows = df.index[keys.eq(key)]
            texts = df.loc[rows, "item_item_text"].dropna().astype(str)
            detected = _detect_language(" ".join(texts.iloc[:30])[:2000])
            if not detected:
                continue
            existing = set(df.loc[rows, "meta_language"].dropna()) | set(
                df.loc[rows, "item_language"].dropna())
            if existing and detected not in existing:
                disagreed += 1
                continue
            for col in ("item_language", "meta_language"):
                nulls = rows[df.loc[rows, col].isna()]
                df.loc[nulls, col] = detected
                filled += len(nulls)
        ctx.log(f"language: langdetect filled {filled} null cells "
                f"({disagreed} docs disagreed, left as-is)")
        ctx.stat("language.langdetect", {"cells_filled": filled,
                                         "docs_disagreed": disagreed})
    elif not _HAS_LANGDETECT:
        ctx.log("language: langdetect unavailable, detection skipped")

    default = pm & df["meta_language"].isna()
    df.loc[default, "meta_language"] = "en"
    ctx.log(f"language: {int(default.sum())} partial null meta_language -> 'en'")
    ctx.stat("language.partial_defaulted_en", int(default.sum()))
    return df


def step_permissions(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Normalize partial permission strings; recompute the category."""
    if "meta_permissions_raw" not in df.columns:
        return df
    pm = partial_mask(df)
    _as_object(df, "meta_permissions_raw", "permissions_category")
    raw = df.loc[pm, "meta_permissions_raw"]
    appendable = raw.map(
        lambda v: isinstance(v, str) and f"{v.strip()}." in _PERMISSIONS_VOCAB
    )
    df.loc[appendable[appendable].index, "meta_permissions_raw"] = (
        raw[appendable].str.strip() + "."
    )
    recomputed = df["meta_permissions_raw"].map(permissions_category)
    if "permissions_category" in df.columns:
        prior = df["permissions_category"]
        changed = int((~(recomputed.eq(prior)
                         | (recomputed.isna() & prior.isna()))).fillna(True).sum())
    else:
        changed = len(df)
    df["permissions_category"] = recomputed
    unmapped = int((df["meta_permissions_raw"].notna() & recomputed.isna()).sum())
    ctx.stat("permissions", {"raw_normalized": int(appendable.sum()),
                             "category_changed": changed,
                             "raw_unmapped": unmapped})
    ctx.log(f"permissions: {int(appendable.sum())} partial raw values normalized; "
            f"category recomputed for all rows ({changed} changed, "
            f"{unmapped} non-null raw values unmapped)")
    return df


_FABRICATED_CONSTANTS: list[tuple[str, str, object]] = [
    ("semanticnet", "source_grade", "semanticnet"),
    ("semanticnet", "version", "semanticnet (version/form not recorded)"),
    ("aligns", "extraction_confidence", "medium"),
    ("semanticnet", "extraction_confidence", "medium"),
    ("aligns", "verbatim_claimed", True),
    ("semanticnet", "verbatim_claimed", True),
]


def step_fabricated(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Guarded nulling of scraper constant-fills carrying no information."""
    if SOURCE_COL not in df.columns:
        return df
    for source, col, const in _FABRICATED_CONSTANTS:
        if col not in df.columns:
            continue
        mask = df[SOURCE_COL].eq(source).fillna(False)
        values = df.loc[mask, col].dropna().unique()
        if len(values) == 0:
            continue
        if len(values) > 1 or values[0] != const:
            ctx.log(f"fabricated: {source}.{col}: values differ from constant "
                    f"{const!r} — left untouched (may be real data now)")
            continue
        _as_object(df, col)
        df.loc[mask, col] = None
        ctx.log(f"fabricated: {source}.{col}: {int(mask.sum())} constant "
                f"{const!r} fills -> null")
        ctx.stat(f"fabricated.nulled.{source}.{col}", int(mask.sum()))
    return df


# Rating-scale databases whose scrapers never record an item type.
_DEFAULT_ITEM_TYPE_SOURCES = ("aligns", "semanticnet")


def step_item_type(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Fill-only-null ``item_item_type = 'rating_scale'`` for ``_DEFAULT_ITEM_TYPE_SOURCES``."""
    col = "item_item_type"
    if SOURCE_COL not in df.columns or col not in df.columns:
        return df
    mask = (df[SOURCE_COL].isin(_DEFAULT_ITEM_TYPE_SOURCES).fillna(False)
            & df[col].isna())
    if mask.any():
        _as_object(df, col)
        df.loc[mask, col] = "rating_scale"
    ctx.stat("item_type.rows_defaulted_rating_scale", int(mask.sum()))
    ctx.log(f"item_type: {int(mask.sum())} null types filled with "
            f"'rating_scale' ({', '.join(_DEFAULT_ITEM_TYPE_SOURCES)})")
    return df


def step_items(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Item-level normalizations."""
    pm = partial_mask(df)
    if "item_options" in df.columns:
        empty = df["item_options"].map(_is_empty_sequence)
        if empty.any():
            _as_object(df, "item_options")
            df.loc[empty, "item_options"] = None
        ctx.log(f"items: {int(empty.sum())} empty option lists -> null")
        ctx.stat("items.empty_option_lists_nulled", int(empty.sum()))

    if "item_item_id" in df.columns:
        keys = _doc_keys(df)
        sub = df.loc[pm & df["item_item_id"].notna()]
        mins = sub["item_item_id"].groupby(keys[sub.index]).transform("min")
        shift = sub.index[mins.eq(0)]
        df.loc[shift, "item_item_id"] = df.loc[shift, "item_item_id"] + 1
        docs = keys[shift].nunique() if len(shift) else 0
        ctx.log(f"items: {docs} partial documents shifted 0-based -> 1-based ids")
        ctx.stat("items.docs_shifted_to_1_based", int(docs))

    if "item_item_text" in df.columns:
        texts = df.loc[pm, "item_item_text"]
        stripped = texts.map(
            lambda v: strip_item_number(v) if isinstance(v, str) else v
        )
        changed = texts.notna() & texts.ne(stripped)
        if changed.any():
            _as_object(df, "item_item_text")
            df.loc[changed[changed].index, "item_item_text"] = stripped[changed]
        ctx.log(f"items: {int(changed.sum())} partial item texts had numbering "
                "prefixes stripped")
        ctx.stat("items.numbering_prefixes_stripped", int(changed.sum()))
    return df


def step_reverse_coded(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Fill-only-null ``item_reverse_coded = False`` on every row with an ``item_item_id``.

    Keyed on the id, not the text, so image-only items count; itemless rows stay null.
    """
    cols = ("item_reverse_coded", "item_item_id")
    if any(c not in df.columns for c in cols):
        ctx.log("reverse_coded: required columns missing — skipped")
        return df
    mask = df["item_item_id"].notna() & df["item_reverse_coded"].isna()
    if mask.any():
        df.loc[mask, "item_reverse_coded"] = False
    by_bucket = ""
    if "bucket" in df.columns and mask.any():
        counts = df.loc[mask, "bucket"].value_counts(dropna=False)
        by_bucket = " (" + ", ".join(
            f"{k}: {v}" for k, v in counts.items()) + ")"
    ctx.stat("reverse_coded.rows_defaulted_false", int(mask.sum()))
    ctx.log(f"reverse_coded: {int(mask.sum())} null item rows filled with "
            f"False{by_bucket}; itemless rows left null")
    return df


def step_authors(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Pipe-delimited partial author lists -> APA format."""
    if "meta_authors_raw" not in df.columns:
        return df
    pm = partial_mask(df)
    raw = df.loc[pm, "meta_authors_raw"]
    piped = raw.map(lambda v: isinstance(v, str) and "|" in v)
    converted = raw[piped].map(parse_pipe_authors)
    ok = converted.notna()
    _as_object(df, "meta_authors_raw")
    df.loc[ok[ok].index, "meta_authors_raw"] = converted[ok]
    ctx.log(f"authors: {int(ok.sum())} pipe-delimited author lists converted to "
            f"APA format ({int((~ok).sum())} unparseable, left as-is)")
    ctx.stat("authors.pipe_lists", {"converted": int(ok.sum()),
                                    "unparseable": int((~ok).sum())})
    return df


_NAME_VERSION_COLS = ("scale_name", "meta_title_raw")
_TITLE_CASE_COLS = ("scale_name", "scale_construct_name", "meta_title_raw")
VERSION_COL = "version"
NAME_PATH_COL = "scale_name_path"


def _map_name_path(df: pd.DataFrame, fn) -> pd.Series:
    """Apply ``fn`` to each ``scale_name_path`` element; return the changed-row mask."""
    def _apply(arr):
        if not isinstance(arr, (list, tuple, np.ndarray)):
            return arr
        return [fn(x) if isinstance(x, str) else x for x in arr]

    before = df[NAME_PATH_COL]
    after = before.map(_apply)
    changed = pd.Series(
        [not _cell_eq(a, b) for a, b in zip(before.tolist(), after.tolist())],
        index=df.index,
    )
    df[NAME_PATH_COL] = after
    return changed


def step_version_split(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Move trailing version qualifiers from name columns into ``version`` (fill-only-null)."""
    present = [c for c in _NAME_VERSION_COLS if c in df.columns]
    if not present:
        return df
    if VERSION_COL not in df.columns:
        df[VERSION_COL] = pd.Series(None, index=df.index, dtype=object)
    _as_object(df, VERSION_COL, *present)

    splits: dict[str, tuple] = {}
    for col in present:
        for v in df[col].dropna().unique():
            if isinstance(v, str) and v not in splits:
                splits[v] = split_version_qualifier(v)

    quals = {col: df[col].map(
        lambda v: splits[v][1] if isinstance(v, str) else None)
        for col in present}
    any_qual = pd.Series(False, index=df.index)
    for q in quals.values():
        any_qual |= q.notna()

    moved = kept = 0
    for i in df.index[any_qual]:
        row_quals: list[str] = []
        for col in present:
            q = quals[col].at[i]
            if isinstance(q, str):
                tq = title_case_name(q)
                if all(tq.casefold() != x.casefold() for x in row_quals):
                    row_quals.append(tq)
        existing = df.at[i, VERSION_COL]
        if isinstance(existing, str) and existing.strip():
            if not all(q.casefold() in existing.casefold() for q in row_quals):
                kept += 1  # would lose information — leave the name intact
                continue
        else:
            df.at[i, VERSION_COL] = "; ".join(row_quals)
        for col in present:
            v = df.at[i, col]
            if isinstance(v, str):
                df.at[i, col] = splits[v][0]
        moved += 1

    path_changed = 0
    if NAME_PATH_COL in df.columns:
        path_changed = int(_map_name_path(
            df, lambda x: split_version_qualifier(x)[0]).sum())

    uniq_hits = sum(1 for b, q in splits.values() if q is not None)
    ctx.stat("version_split", {"unique_names_with_qualifier": uniq_hits,
                               "rows_relocated": moved, "rows_kept": kept,
                               "name_path_rows_cleaned": path_changed})
    ctx.log(f"version_split: {uniq_hits} unique names carried a version "
            f"qualifier; {moved} rows relocated into '{VERSION_COL}', "
            f"{kept} rows left as-is (version occupied by other text); "
            f"{path_changed} {NAME_PATH_COL} rows cleaned")
    return df


def step_title_case(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """APA-style title case for the name columns."""
    for col in _TITLE_CASE_COLS:
        if col not in df.columns:
            continue
        changed_map = {}
        for v in df[col].dropna().unique():
            if isinstance(v, str):
                t = title_case_name(v)
                if t != v:
                    changed_map[v] = t
        mask = df[col].isin(changed_map)
        if mask.any():
            _as_object(df, col)
            df.loc[mask, col] = df.loc[mask, col].map(changed_map)
        ctx.log(f"title_case: {col}: {len(changed_map)} unique names "
                f"({int(mask.sum())} rows) reformatted")
        ctx.stat(f"title_case.reformatted.{col}",
                 {"unique_names": len(changed_map), "rows": int(mask.sum())})
    if NAME_PATH_COL in df.columns:
        _as_object(df, NAME_PATH_COL)
        changed = int(_map_name_path(df, title_case_name).sum())
        ctx.log(f"title_case: {NAME_PATH_COL}: {changed} rows reformatted")
        ctx.stat("title_case.name_path_rows_reformatted", changed)
    return df


def step_anomalies(df: pd.DataFrame, ctx: Ctx) -> pd.DataFrame:
    """Report-only census of residual oddities. Mutates nothing."""
    def census(key: str, n: int, line: str) -> None:
        ctx.stat(f"anomalies.{key}", n)
        ctx.log(line)

    if "has_errors" in df.columns:
        n = int(df["has_errors"].fillna(False).astype(bool).sum())
        census("failed_extraction_shell_rows", n,
               f"anomalies: {n} failed-extraction shell rows (has_errors)")
    if PATH_COL in df.columns:
        stems = df[PATH_COL].dropna().map(
            lambda p: Path(p).stem.split("_")[0]).to_frame("id")
        stems["path"] = df[PATH_COL].dropna()
        dup_ids = stems.groupby("id")["path"].nunique()
        dupes = dup_ids[dup_ids > 1]
        census("duplicate_document_ids", len(dupes),
               f"anomalies: {len(dupes)} document ids with multiple ingested "
               "files" + (f" (e.g. {dupes.index[0]})" if len(dupes) else ""))
        n = int(df[PATH_COL].isna().sum())
        census("null_path_rows", n, f"anomalies: {n} rows with null path")
    dup_cols = [c for c in (SOURCE_COL, PATH_COL, "scale_name", "item_item_id",
                            "item_item_text") if c in df.columns]
    full_dups = int(df.duplicated(subset=dup_cols).sum())
    census("fully_duplicated_item_rows", full_dups,
           f"anomalies: {full_dups} fully duplicated item rows on {dup_cols}")
    if {"item_item_text", "item_options"} <= set(df.columns):
        n = int((df["item_item_text"].isna() & df["item_options"].notna()).sum())
        census("options_without_text_rows", n,
               f"anomalies: {n} rows with options but no item text")
    if {"item_item_type", "item_options"} <= set(df.columns):
        n = int((df["item_item_type"].eq("choice")
                 & df["item_options"].isna()).sum())
        census("choice_without_options_rows", n,
               f"anomalies: {n} choice-type rows without options")
    if "item_item_text" in df.columns:
        short = df["item_item_text"].map(
            lambda v: isinstance(v, str) and len(v.strip()) <= 3)
        census("short_item_texts", int(short.sum()),
               f"anomalies: {int(short.sum())} item texts of <= 3 characters")
    if "saved_files" in df.columns:
        leak = df["saved_files"].map(
            lambda v: isinstance(v, str) and v.startswith(("/Users/", "/home/")))
        census("leaked_absolute_paths", int(leak.sum()),
               f"anomalies: {int(leak.sum())} saved_files cells with absolute "
               "foreign paths")
    return df


# ---------------------------------------------------------------------------
# Registry and runners
# ---------------------------------------------------------------------------

Step = Callable[[pd.DataFrame, Ctx], pd.DataFrame]
STEPS: dict[str, Step] = {  # insertion order = execution order
    "schema": step_schema,
    "text_repair": step_text_repair,
    "meta_dois": step_meta_dois,
    "pdf_full_text": step_pdf_full_text,
    "doi_psyctests": step_doi_psyctests,
    "source_fields": step_source_fields,
    "source_doi": step_source_doi,
    "crossref": step_crossref,
    "doi_probe": step_doi_probe,
    "language": step_language,
    "permissions": step_permissions,
    "fabricated": step_fabricated,
    "item_type": step_item_type,
    "items": step_items,
    "reverse_coded": step_reverse_coded,
    "authors": step_authors,
    "version_split": step_version_split,
    "title_case": step_title_case,
    "anomalies": step_anomalies,
}
# Report labeling only — enforcement lives inside each step via the masks.
STEP_SCOPE: dict[str, str] = {
    "schema": "all", "text_repair": "all", "meta_dois": "apa",
    "pdf_full_text": "apa", "doi_psyctests": "all", "source_fields": "partials", "source_doi": "all",
    "crossref": "all", "doi_probe": "partials", "language": "all",
    "permissions": "partials+all", "fabricated": "partials",
    "item_type": "partials", "items": "all/partials", "reverse_coded": "all",
    "authors": "partials",
    "version_split": "all", "title_case": "all", "anomalies": "report-only",
}


def _cell_eq(x, y) -> bool:
    x_seq = isinstance(x, (list, tuple, np.ndarray))
    y_seq = isinstance(y, (list, tuple, np.ndarray))
    if x_seq or y_seq:
        if not (x_seq and y_seq):
            return False
        return len(x) == len(y) and all(_cell_eq(a, b) for a, b in zip(x, y))
    xna, yna = bool(pd.isna(x)), bool(pd.isna(y))
    if xna or yna:
        return xna and yna
    return bool(x == y)


def _changed_mask(before: pd.DataFrame, after: pd.DataFrame) -> pd.Series:
    """Value-level diff per row; dtype-only casts don't count, new columns count where non-null."""
    changed = pd.Series(False, index=after.index)
    for col in after.columns:
        # pdf_full_text is source text, not a repair.
        if col in (PATCHED_COL, PDF_FULL_TEXT_COL):
            continue
        before_col = col
        if col not in before.columns:
            # Diff against legacy 'doi' so the rename alone doesn't flag every row.
            if col == DOI_PSYCTESTS_COL and "doi" in before.columns:
                before_col = "doi"
            else:
                changed |= after[col].notna()
                continue
        b, a = before[before_col], after[col]
        try:
            neq = ~(b.eq(a) | (b.isna() & a.isna()))
        except (TypeError, ValueError):
            neq = pd.Series(
                [not _cell_eq(x, y) for x, y in zip(b.tolist(), a.tolist())],
                index=after.index,
            )
        changed |= neq.fillna(True).astype(bool)
    return changed


def _run_steps(df: pd.DataFrame, selected: list[str], ctx: Ctx) -> pd.DataFrame:
    before = df.copy(deep=True)
    for name in STEPS:  # registry order, regardless of CLI order
        if name not in selected:
            continue
        ctx.log(f"--- {name} [{STEP_SCOPE[name]}] ---")
        df = STEPS[name](df, ctx)
    changed = _changed_mask(before, df)
    prior = (before[PATCHED_COL].fillna(False).astype(bool)
             if PATCHED_COL in before.columns
             else pd.Series(False, index=df.index))
    df[PATCHED_COL] = (changed | prior).astype(bool)
    ctx.log("")
    ctx.log(f"is_patched: {int(changed.sum())} rows changed this run, "
            f"{int(df[PATCHED_COL].sum())} flagged in total")
    ctx.stat("is_patched.rows_changed_this_run", int(changed.sum()))
    ctx.stat("is_patched.rows_flagged_total", int(df[PATCHED_COL].sum()))
    return df


def _select_steps(steps_arg: Optional[str]) -> list[str]:
    if not steps_arg:
        return list(STEPS)
    selected = [s.strip() for s in steps_arg.split(",") if s.strip()]
    unknown = [s for s in selected if s not in STEPS]
    if unknown:
        sys.exit(f"Unknown steps: {', '.join(unknown)}. "
                 f"Available: {', '.join(STEPS)}")
    return selected


def run(cfg: dict, *, report_only: bool = False, refresh: bool = False,
        input_path=None, output_path=None, steps: Optional[str] = None) -> list[str]:
    """Stage entry point: ``data.assemble.combined`` -> ``data.assemble.patched``."""
    asm_cfg = cfg.get("data", {}).get("assemble", {})
    inp = Path(input_path or asm_cfg.get("combined", ""))
    out = Path(output_path or asm_cfg.get("patched", ""))
    if not str(inp) or not str(out) or str(inp) == "." or str(out) == ".":
        sys.exit("data.assemble.combined and data.assemble.patched must be set")
    if not inp.exists():
        sys.exit(f"input not found: {inp}")

    ctx = Ctx(cfg=cfg.get("patch", {}) or {}, full_cfg=cfg,
              report_only=report_only, refresh=refresh)
    ctx.log("corpus patch report")
    ctx.log(f"input: {inp}")
    df = pd.read_parquet(inp)
    ctx.log(f"{len(df):,} rows, {len(df.columns)} columns")

    selected = _select_steps(steps)
    if not report_only and len(selected) < len(STEPS):
        skipped = [s for s in STEPS if s not in selected]
        ctx.log(f"WARNING: running {len(selected)} of {len(STEPS)} steps — the "
                f"output is a PARTIAL patch (skipped: {', '.join(skipped)}) "
                f"and overwrites {out}. Downstream stages expect a fully "
                "patched corpus; use --output to write elsewhere when trying "
                "a step subset.")

    df = _run_steps(df, selected, ctx)

    if not report_only:
        write_parquet(df, out)
        ctx.log(f"wrote {out}")
        write_stats(out, "patch", ctx.stats, ctx.report)
    return ctx.report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone patch stage: repair an assembled corpus parquet.",
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument("--input", default=None,
                    help="Input parquet (default: data.assemble.combined).")
    ap.add_argument("--output", default=None,
                    help="Output parquet (default: data.assemble.patched).")
    ap.add_argument("--steps", default=None,
                    help=f"Comma-separated subset of: {', '.join(STEPS)}. "
                         "Always executed in registry order.")
    ap.add_argument("--report-only", action="store_true",
                    help="Print the report without writing anything; DOI "
                         "lookups run cache-only.")
    ap.add_argument("--refresh", action="store_true",
                    help="Ignore the Crossref/doi_probe caches.")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    lines = run(cfg, report_only=args.report_only, refresh=args.refresh,
                input_path=args.input, output_path=args.output,
                steps=args.steps)
    print("\n".join(lines))
    if args.report_only:
        print("(report-only: nothing written)")


if __name__ == "__main__":
    main()
