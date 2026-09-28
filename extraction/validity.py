"""End-to-end validity testing: compare composed ``extraction_output`` against curated fixtures.

Semantics and fixture tags: docs/testing.md; internals: docs/architecture-extraction.md#validity-comparator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from .models import Instrument
from .text_utils import fuzzy_contains, lev_dist, lev_edits


@dataclass(frozen=True)
class MatchSpec:
    """Explicit matcher attached to a leaf value in a validity fixture."""
    mode: Literal["contains", "exact"]
    value: str


class AnywhereList(list):
    """``!anywhere`` list: entries match against the global candidate pool; no extras check."""


class AnywhereDict(dict):
    """``!anywhere`` entry: matched globally rather than among its siblings."""


class ExtrasList(list):
    """``!extras=N`` list: strict matching, but tolerates ``allow_extras`` unmatched actual entries."""
    allow_extras: int = 0


class ValiditySafeLoader(yaml.SafeLoader):
    """SafeLoader with ``!contains``/``!exact``/``!anywhere``/``!extras=N``; subclassed to keep tags scoped."""


def _construct_match_spec(mode: str):
    def _constructor(loader: yaml.SafeLoader, node: yaml.Node) -> MatchSpec:
        return MatchSpec(mode=mode, value=loader.construct_scalar(node))
    return _constructor


def _construct_anywhere(loader: yaml.SafeLoader, node: yaml.Node):
    if isinstance(node, yaml.SequenceNode):
        return AnywhereList(loader.construct_sequence(node, deep=True))
    if isinstance(node, yaml.MappingNode):
        return AnywhereDict(loader.construct_mapping(node, deep=True))
    raise yaml.constructor.ConstructorError(
        None, None, "!anywhere expects a list or mapping", node.start_mark,
    )


def _construct_extras(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node):
    """Construct an ``ExtrasList`` from ``!extras=N`` (N is encoded in the tag itself)."""
    if not isinstance(node, yaml.SequenceNode):
        raise yaml.constructor.ConstructorError(
            None, None, "!extras expects a sequence", node.start_mark,
        )
    n = int(tag_suffix.lstrip("=_-"))
    lst = ExtrasList(loader.construct_sequence(node, deep=True))
    lst.allow_extras = n
    return lst


ValiditySafeLoader.add_constructor("!contains", _construct_match_spec("contains"))
ValiditySafeLoader.add_constructor("!exact", _construct_match_spec("exact"))
ValiditySafeLoader.add_constructor("!anywhere", _construct_anywhere)
ValiditySafeLoader.add_multi_constructor("!extras", _construct_extras)


def load_validity_fixture(fixture_path: Path) -> Optional[dict]:
    """Return a fixture file's tagged ``extraction_output:`` block, or ``None`` if absent."""
    if not fixture_path.exists():
        return None
    with open(fixture_path) as f:
        loaded = yaml.load(f, Loader=ValiditySafeLoader) or {}
    block = loaded.get("extraction_output")
    return block if isinstance(block, dict) else None


def load_full_fixture_with_tags(fixture_path: Path) -> dict:
    """Load a whole fixture file via :class:`ValiditySafeLoader` (``{}`` if absent)."""
    if not fixture_path.exists():
        return {}
    with open(fixture_path) as f:
        return yaml.load(f, Loader=ValiditySafeLoader) or {}


def strip_match_specs(obj: Any) -> Any:
    """Replace ``MatchSpec`` leaves with their values and unwrap tag sentinels to plain list/dict."""
    if isinstance(obj, MatchSpec):
        return obj.value
    if isinstance(obj, dict):
        return {k: strip_match_specs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_match_specs(v) for v in obj]
    return obj


def normalize_for_model_validation(raw: dict) -> dict:
    """Fill missing IDs/defaults on a partial fixture so ``Instrument.model_validate`` succeeds.

    Call after :func:`strip_match_specs`. Rules: docs/architecture-extraction.md#validity-comparator.
    """
    def _item_dedup_key(it: dict) -> str:
        """Dedup key: item_text, else joined options, else ``""``."""
        text = it.get("item_text")
        if isinstance(text, str) and text:
            return text
        options = it.get("options")
        if isinstance(options, list) and options:
            return " | ".join(str(o) for o in options)
        return ""

    scale_counter = [0]
    item_text_to_id: dict[str, int] = {}
    item_counter = [0]

    used_scale_ids: set[int] = set()
    used_item_ids: set[int] = set()

    def _pre_collect_scale_ids(scales: list) -> None:
        for s in scales or []:
            if isinstance(s, dict) and s.get("id") is not None:
                used_scale_ids.add(int(s["id"]))
            _pre_collect_scale_ids((s or {}).get("subscales") or [])

    def _pre_collect_item_ids(scales: list) -> None:
        for s in scales or []:
            for it in (s or {}).get("items") or []:
                if isinstance(it, dict) and it.get("item_id") is not None:
                    used_item_ids.add(int(it["item_id"]))
                    key = _item_dedup_key(it)
                    if key and key not in item_text_to_id:
                        item_text_to_id[key] = int(it["item_id"])
            _pre_collect_item_ids((s or {}).get("subscales") or [])

    _pre_collect_scale_ids(raw.get("scales") or [])
    _pre_collect_item_ids(raw.get("scales") or [])
    for it in raw.get("unscaled_items") or []:
        if isinstance(it, dict) and it.get("id") is not None:
            used_item_ids.add(int(it["id"]))

    def _next_scale_id() -> int:
        scale_counter[0] += 1
        while scale_counter[0] in used_scale_ids:
            scale_counter[0] += 1
        used_scale_ids.add(scale_counter[0])
        return scale_counter[0]

    def _id_for_item_text(text: str) -> int:
        if text in item_text_to_id:
            return item_text_to_id[text]
        item_counter[0] += 1
        while item_counter[0] in used_item_ids:
            item_counter[0] += 1
        used_item_ids.add(item_counter[0])
        item_text_to_id[text] = item_counter[0]
        return item_counter[0]

    def _normalize_scale(s: dict) -> dict:
        sid = int(s["id"]) if s.get("id") is not None else _next_scale_id()
        items_out: list[dict] = []
        for it in s.get("items") or []:
            text = it.get("item_text") if isinstance(it.get("item_text"), str) else None
            options = it.get("options") if isinstance(it.get("options"), list) else None
            key_for_id = _item_dedup_key(it)
            iid = int(it["item_id"]) if it.get("item_id") is not None else _id_for_item_text(key_for_id)
            items_out.append({
                "item_id": iid,
                "item_text": text,
                "has_image": bool(it.get("has_image") or False),
                "item_type": it.get("item_type") or "rating_scale",
                "options": options,
                "admin_note": it.get("admin_note"),
                "language": it.get("language") or "en",
                "reverse_coded": bool(it.get("reverse_coded") or False),
            })
        subscales_out = [_normalize_scale(sub) for sub in (s.get("subscales") or [])]
        return {
            "id": sid,
            "scale_name": s.get("scale_name"),
            "construct_name": s.get("construct_name"),
            "items": items_out,
            "subscales": subscales_out,
        }

    scales_out = [_normalize_scale(s) for s in (raw.get("scales") or [])]

    def _normalize_loose_items(key: str) -> list[dict]:
        out: list[dict] = []
        for it in raw.get(key) or []:
            text = it.get("item_text") if isinstance(it.get("item_text"), str) else None
            options = it.get("options") if isinstance(it.get("options"), list) else None
            key_for_id = _item_dedup_key(it)
            iid = int(it["id"]) if it.get("id") is not None else _id_for_item_text(key_for_id)
            out.append({
                "id": iid,
                "item_text": text,
                "has_image": bool(it.get("has_image") or False),
                "item_type": it.get("item_type") or "rating_scale",
                "options": options,
                "admin_note": it.get("admin_note"),
                "language": it.get("language") or "en",
            })
        return out

    return {
        "scales": scales_out,
        "unscaled_items": _normalize_loose_items("unscaled_items"),
        "orphan_items": _normalize_loose_items("orphan_items"),
        "meta": None,
    }


@dataclass
class Issue:
    path: str
    severity: str
    message: str

    def format(self) -> str:
        return f"  [{self.severity}] {self.path or '<root>'}: {self.message}"


@dataclass
class MatchRecord:
    """One expected↔actual pair from the Hungarian matcher, with its Levenshtein scores."""

    kind: str            # "item" | "scale"
    path: str            # e.g. "scales[0].items[3]"
    expected: str
    actual: str
    similarity: float    # 1.0 - lev_dist(...)
    edits: int           # lev_edits(...)


@dataclass
class ValidityReport:
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "PASS"
    matches: list[MatchRecord] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Status: {self.status}"]
        if self.metrics:
            lines.append("Metrics: " + "  ".join(f"{k}={v:.4f}" for k, v in self.metrics.items()))
        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            lines.extend(issue.format() for issue in self.issues)
        return "\n".join(lines)


DEFAULT_STRING_SIMILARITY: float = 0.80


# Lists keyed by a named field — children matched via Hungarian on the key.
_KEY_FIELD_BY_LIST: dict[str, str] = {
    "scales": "scale_name",
    "subscales": "scale_name",
    "items": "item_text",
    "unscaled_items": "item_text",
    "orphan_items": "item_text",
}

# MatchRecord.kind per keyed list.
_KIND_BY_LIST: dict[str, str] = {
    "scales": "scale",
    "subscales": "scale",
    "items": "item",
    "unscaled_items": "item",
    "orphan_items": "item",
}


def _collect_all_scales(root_actual: dict) -> list[dict]:
    """DFS-collect every scale and subscale node under the actual root."""
    out: list[dict] = []

    def walk(scales: Any) -> None:
        if not isinstance(scales, list):
            return
        for s in scales:
            if not isinstance(s, dict):
                continue
            out.append(s)
            walk(s.get("subscales"))

    walk(root_actual.get("scales") if isinstance(root_actual, dict) else None)
    return out


def _collect_all_items(root_actual: dict) -> list[dict]:
    """DFS-collect every item under all scales plus ``unscaled_items`` / ``orphan_items``."""
    out: list[dict] = []

    def walk_scales(scales: Any) -> None:
        if not isinstance(scales, list):
            return
        for s in scales:
            if not isinstance(s, dict):
                continue
            for it in s.get("items") or []:
                if isinstance(it, dict):
                    out.append(it)
            walk_scales(s.get("subscales"))

    if isinstance(root_actual, dict):
        walk_scales(root_actual.get("scales"))
        for key in ("unscaled_items", "orphan_items"):
            for it in root_actual.get(key) or []:
                if isinstance(it, dict):
                    out.append(it)
    return out


_CANDIDATE_COLLECTOR_BY_LIST = {
    "scales": _collect_all_scales,
    "subscales": _collect_all_scales,
    "items": _collect_all_items,
    "unscaled_items": _collect_all_items,
    "orphan_items": _collect_all_items,
}


def _last_segment(path: str) -> str:
    if not path:
        return ""
    last = path.split(".")[-1]
    return last.split("[")[0]


def _validate(
    expected: Any,
    actual: Any,
    path: str,
    thresholds: dict,
    issues: list[Issue],
    matches: list[MatchRecord],
    root_actual: Optional[dict] = None,
) -> None:
    if isinstance(expected, MatchSpec):
        threshold = float(thresholds.get("string_similarity", DEFAULT_STRING_SIMILARITY))
        _compare_with_spec(expected, actual, path, issues, threshold=threshold)
        return

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            issues.append(Issue(path, "FAIL", f"expected object, got {type(actual).__name__}"))
            return
        for key, child_expected in expected.items():
            child_path = f"{path}.{key}" if path else key
            if key not in actual:
                issues.append(Issue(child_path, "FAIL", "missing in actual"))
                continue
            _validate(child_expected, actual[key], child_path, thresholds, issues, matches, root_actual)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            issues.append(Issue(path, "FAIL", f"expected list, got {type(actual).__name__}"))
            return
        field_name = _last_segment(path)
        key_field = _KEY_FIELD_BY_LIST.get(field_name)
        if key_field:
            _match_named_list(
                expected, actual, path, key_field, thresholds, issues, matches, root_actual
            )
        else:
            _match_positional_list(expected, actual, path, thresholds, issues, matches, root_actual)
        return

    if isinstance(expected, str):
        if not isinstance(actual, str):
            issues.append(Issue(
                path, "FAIL", f"expected string {expected!r}, got {type(actual).__name__}",
            ))
            return
        threshold = float(thresholds.get("string_similarity", DEFAULT_STRING_SIMILARITY))
        sim = 1.0 - lev_dist(expected, actual)
        if sim < threshold:
            issues.append(Issue(
                path,
                "FAIL",
                f"string mismatch (sim={sim:.2f} < {threshold:.2f}): "
                f"expected {expected!r}, got {actual!r}",
            ))
        return

    # Scalar leaf: bool, int, float, None.
    if expected != actual:
        issues.append(Issue(path, "FAIL", f"expected {expected!r}, got {actual!r}"))


def _compare_with_spec(
    spec: MatchSpec, actual: Any, path: str, issues: list[Issue],
    threshold: float = DEFAULT_STRING_SIMILARITY,
) -> None:
    if not isinstance(actual, str):
        issues.append(Issue(
            path, "FAIL", f"!{spec.mode} expects a string, got {type(actual).__name__}",
        ))
        return
    if spec.mode == "exact":
        if spec.value != actual:
            issues.append(Issue(
                path, "FAIL", f"!exact mismatch: expected {spec.value!r}, got {actual!r}",
            ))
    elif spec.mode == "contains":
        sim = fuzzy_contains(spec.value, actual)
        if sim < threshold:
            issues.append(Issue(
                path, "FAIL",
                f"!contains mismatch (sim={sim:.2f} < {threshold:.2f}): "
                f"{spec.value!r} not in {actual!r}",
            ))


def _spec_accepts(spec: MatchSpec, actual: Any,
                   threshold: float = DEFAULT_STRING_SIMILARITY) -> bool:
    if not isinstance(actual, str):
        return False
    if spec.mode == "exact":
        return spec.value == actual
    if spec.mode == "contains":
        return fuzzy_contains(spec.value, actual) >= threshold
    return False


class _OptionsKey(str):
    """Key synthesized from response options when item_text is null (never compared to real text)."""


def _extract_key(entry: Any, key_field: str) -> Any:
    if not isinstance(entry, dict):
        return None
    key = entry.get(key_field)
    if key is None and key_field == "item_text":
        options = entry.get("options")
        if isinstance(options, list) and options:
            return _OptionsKey(" | ".join(str(o) for o in options))
    return key


def _similarity(expected_key: Any, actual_key: Any) -> float:
    """Key similarity for Hungarian matching: omitted key → 1.0, options-key vs text → 0.0."""
    if isinstance(expected_key, MatchSpec):
        if expected_key.mode == "contains" and isinstance(actual_key, str):
            return fuzzy_contains(expected_key.value, actual_key)
        return 1.0 if _spec_accepts(expected_key, actual_key) else 0.0
    if expected_key is None:
        return 1.0
    if isinstance(expected_key, str) and isinstance(actual_key, str):
        if isinstance(expected_key, _OptionsKey) != isinstance(actual_key, _OptionsKey):
            return 0.0
        return 1.0 - lev_dist(expected_key, actual_key)
    if expected_key == actual_key:
        return 1.0
    return 0.0


def _match_named_list(
    expected_list: list[Any],
    actual_list: list[Any],
    path: str,
    key_field: str,
    thresholds: dict,
    issues: list[Issue],
    matches: list[MatchRecord],
    root_actual: Optional[dict] = None,
) -> None:
    """Hungarian-match a named-entity list on its key field, then check extras.

    Strict entries match locally (below-threshold pairs reported as "matched but failing");
    ``!anywhere`` entries match against the global pool of ``root_actual``.
    """
    threshold = float(thresholds.get("string_similarity", DEFAULT_STRING_SIMILARITY))

    list_is_anywhere = isinstance(expected_list, AnywhereList)
    field_name = _last_segment(path)
    kind = _KIND_BY_LIST.get(field_name, field_name)

    def record(child_path: str, exp_key: Any, act_key: Any, sim: float) -> None:
        """Record a pair only when ``sim`` is a genuine Levenshtein score (no specs/omitted/cross-type)."""
        if isinstance(exp_key, MatchSpec) or not isinstance(exp_key, str):
            return
        if not isinstance(act_key, str):
            return
        if isinstance(exp_key, _OptionsKey) != isinstance(act_key, _OptionsKey):
            return
        matches.append(MatchRecord(
            kind=kind,
            path=child_path,
            expected=exp_key,
            actual=act_key,
            similarity=sim,
            edits=lev_edits(exp_key, act_key),
        ))

    strict: list[tuple[int, Any]] = []
    loose: list[tuple[int, Any]] = []
    for i, entry in enumerate(expected_list):
        if list_is_anywhere or isinstance(entry, AnywhereDict):
            loose.append((i, entry))
        else:
            strict.append((i, entry))

    n_strict = len(strict)
    m = len(actual_list)
    consumed_actual: set[int] = set()
    actual_id_to_idx = {id(actual_list[j]): j for j in range(m)}

    # --- Strict phase: positional Hungarian over actual_list ---------------
    if n_strict > 0 and m > 0:
        size = max(n_strict, m)
        cost = np.ones((size, size), dtype=float)
        for si, (_, entry) in enumerate(strict):
            exp_key = _extract_key(entry, key_field)
            for j in range(m):
                act_key = _extract_key(actual_list[j], key_field)
                cost[si, j] = 1.0 - _similarity(exp_key, act_key)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_strict: set[int] = set()
        for ri, cj in zip(row_ind, col_ind):
            si, j = int(ri), int(cj)
            if si >= n_strict or j >= m:
                continue
            matched_strict.add(si)
            consumed_actual.add(j)
            i = strict[si][0]
            entry = strict[si][1]
            sim = 1.0 - cost[si, j]
            exp_key = _extract_key(entry, key_field)
            act_key = _extract_key(actual_list[j], key_field)
            child_path = f"{path}[{i}]"
            record(child_path, exp_key, act_key, sim)
            if sim < threshold:
                if not isinstance(exp_key, _OptionsKey) and isinstance(act_key, _OptionsKey):
                    detail = (
                        f"matched actual[{j}] but actual has no {key_field} "
                        f"(expected {exp_key!r}, actual has only response options)"
                    )
                elif isinstance(exp_key, _OptionsKey) and not isinstance(act_key, _OptionsKey):
                    detail = (
                        f"matched actual[{j}] but expected has no {key_field} "
                        f"(expected has only response options, actual {act_key!r})"
                    )
                else:
                    detail = (
                        f"matched actual[{j}] but {key_field} similarity {sim:.2f} < {threshold:.2f} "
                        f"(expected {exp_key!r}, actual {act_key!r})"
                    )
                issues.append(Issue(child_path, "FAIL", detail))
            if isinstance(entry, dict) and isinstance(actual_list[j], dict):
                child_expected = {k: v for k, v in entry.items() if k != key_field}
                _validate(
                    child_expected, actual_list[j], child_path, thresholds, issues, matches, root_actual
                )

        for si in range(n_strict):
            if si not in matched_strict:
                i, entry = strict[si]
                exp_key = _extract_key(entry, key_field)
                issues.append(Issue(
                    f"{path}[{i}]", "FAIL", f"missing in actual: {key_field}={exp_key!r}",
                ))
    elif n_strict > 0:  # m == 0
        for i, entry in strict:
            exp_key = _extract_key(entry, key_field)
            issues.append(Issue(
                f"{path}[{i}]", "FAIL", f"missing in actual: {key_field}={exp_key!r}",
            ))

    # --- Loose phase: Hungarian over the global candidate pool -------------
    if loose:
        collector = _CANDIDATE_COLLECTOR_BY_LIST.get(field_name)
        candidates: list[dict] = collector(root_actual) if (collector and root_actual is not None) else []
        n_loose = len(loose)
        c = len(candidates)
        if c == 0:
            for i, entry in loose:
                exp_key = _extract_key(entry, key_field)
                issues.append(Issue(
                    f"{path}[~{i}]", "FAIL",
                    f"missing in actual (anywhere): {key_field}={exp_key!r}",
                ))
        else:
            size = max(n_loose, c)
            cost = np.ones((size, size), dtype=float)
            for li, (_, entry) in enumerate(loose):
                exp_key = _extract_key(entry, key_field)
                for k in range(c):
                    act_key = _extract_key(candidates[k], key_field)
                    cost[li, k] = 1.0 - _similarity(exp_key, act_key)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_loose: set[int] = set()
            for ri, cj in zip(row_ind, col_ind):
                li, k = int(ri), int(cj)
                if li >= n_loose or k >= c:
                    continue
                i, entry = loose[li]
                sim = 1.0 - cost[li, k]
                exp_key = _extract_key(entry, key_field)
                cand = candidates[k]
                child_path = f"{path}[~{i}]"
                # Loose below-threshold = not found (not "matched but failing").
                if sim < threshold:
                    issues.append(Issue(
                        child_path, "FAIL",
                        f"missing in actual (anywhere): {key_field}={exp_key!r}",
                    ))
                    continue
                matched_loose.add(li)
                record(child_path, exp_key, _extract_key(cand, key_field), sim)
                # Loose match on a local entry: consume it so it isn't flagged as extra.
                if id(cand) in actual_id_to_idx:
                    consumed_actual.add(actual_id_to_idx[id(cand)])
                if isinstance(entry, dict):
                    child_expected = {kk: v for kk, v in entry.items() if kk != key_field}
                    _validate(child_expected, cand, child_path, thresholds, issues, matches, root_actual)
            for li in range(n_loose):
                if li not in matched_loose:
                    # Only flag entries Hungarian never paired; below-threshold already reported.
                    if any(int(ri) == li for ri in row_ind):
                        continue
                    i, entry = loose[li]
                    exp_key = _extract_key(entry, key_field)
                    issues.append(Issue(
                        f"{path}[~{i}]", "FAIL",
                        f"missing in actual (anywhere): {key_field}={exp_key!r}",
                    ))

    # --- Extras: skipped only for purely-loose lists; !extras=N grants a budget ---
    if not list_is_anywhere:
        budget = getattr(expected_list, "allow_extras", 0)
        seen = 0
        for j in range(m):
            if j in consumed_actual:
                continue
            seen += 1
            if seen <= budget:
                continue
            act_key = _extract_key(actual_list[j], key_field)
            issues.append(Issue(
                f"{path}[+{j}]", "FAIL", f"extra in actual: {key_field}={act_key!r}",
            ))


def _match_positional_list(
    expected_list: list[Any],
    actual_list: list[Any],
    path: str,
    thresholds: dict,
    issues: list[Issue],
    matches: list[MatchRecord],
    root_actual: Optional[dict] = None,
) -> None:
    if len(expected_list) != len(actual_list):
        issues.append(Issue(
            path,
            "FAIL",
            f"list length mismatch: expected {len(expected_list)}, got {len(actual_list)}",
        ))
    for i, (e, a) in enumerate(zip(expected_list, actual_list)):
        _validate(e, a, f"{path}[{i}]", thresholds, issues, matches, root_actual)


def validate_extraction(
    expected: dict,
    actual: Instrument,
    thresholds: Optional[dict] = None,
) -> ValidityReport:
    """Compare a raw (tagged) fixture dict against the composed ``Instrument``.

    ``expected`` must not be Pydantic-validated: defaults would erase absent-vs-present.
    """
    thresholds = thresholds or {}
    actual_dict = actual.model_dump(mode="json")
    issues: list[Issue] = []
    matches: list[MatchRecord] = []
    _validate(expected, actual_dict, "", thresholds, issues, matches, actual_dict)

    n_missing = sum(1 for i in issues if "missing in actual" in i.message)
    n_extra = sum(1 for i in issues if "extra in actual" in i.message)
    metrics = {
        "n_issues": float(len(issues)),
        "n_missing": float(n_missing),
        "n_extra": float(n_extra),
    }
    status = "FAIL" if issues else "PASS"
    return ValidityReport(issues=issues, metrics=metrics, status=status, matches=matches)
