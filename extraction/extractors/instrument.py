"""InstrumentExtractor — map extracted items onto extracted scales and flag reverse-coding.

See docs/extractors.md#instrument.
"""

from __future__ import annotations

import dataclasses
from itertools import combinations
from typing import Callable

import krippendorff
import numpy as np
from pydantic import BaseModel

from ..models import (
    Instrument,
    InstrumentMappings,
    Items,
    ScaleNode,
    Scales,
    resolve_instrument,
)
from ..text_utils import Tee, extract_json

from ._render import (
    dump_result_to_log,
    render_items_block,
    render_scales_block,
)
from .base import Extractor


SYSTEM_PROMPT = (
    "You are a precise data-extraction assistant specialising in "
    "psychological assessments."
)

USER_PROMPT_BODY = """\
You are given the PDF pages of a questionnaire, the list of items already extracted from it, and the scale structure already extracted from it. Both items and scales carry numeric IDs.

Your task: produce the item-to-scale structure of the instrument — for every item, identify the scale(s) it belongs to, and flag reverse-coded items.

## Output rules
- Emit one `mappings` entry per (item, scale) assignment.
- Most items belong to exactly one scale. In rare cases an item is part of multiple scales — emit one entry per scale.
- Reference items by `item_id` and scales by `scale_id` (numeric IDs as shown below). Do not invent IDs.
- Set `reverse_coded: true` ONLY for items scored in the opposite direction of the scale. **Omit the field entirely otherwise** — do not write `"reverse_coded": false`.
- If the document does not state or imply that any items are reverse-coded, do not emit `reverse_coded` for any mapping.

## Items that do not belong to any scale

Some items will not be assigned to any scale above. Classify each such item into ONE of the two categories below:

- **Unscaled items** — items with NO psychometric merit. They are not scored on any latent psychological construct: demographic questions (age, gender, education), administrative items (date, ID number, examiner), doctor's or examiner's notes, plain instructions, manifest-fact questions. These items must NOT appear in `mappings` and must NOT appear in `orphan_item_ids` — just leave their IDs out of both.

- **Orphan items** — items that ARE psychometric in nature (they ask about a feeling, attitude, behaviour, belief, symptom, preference, etc.) but are not part of any defined scale above. Two common patterns:
  - Stand-alone single-item measures embedded inside a larger questionnaire.
  - Items from an **initial, draft, or earlier version** of the questionnaire that appear in the same document (e.g., appended at the end as a pool of candidate or pilot items). Even when these items are psychometrically similar to the scaled items, they are not part of the defined scale structure and must be classified as orphans, not mapped to any scale.
  Emit their IDs in `orphan_item_ids`. They must NOT appear in `mappings`.

When in doubt between unscaled and orphan, prefer orphan. Default to "unscaled" only when an item is clearly non-psychometric.

Omit `orphan_item_ids` entirely when there are none — do not emit `null` or `[]`.
"""

_UNSCALED_HINT = (
    "(This questionnaire also contains items that are not part of any scale "
    "above. Classify each such item as either an orphan item with "
    "psychometric merit (emit its id in `orphan_item_ids`) or an unscaled "
    "demographic / administrative item (leave its id out of both `mappings` "
    "and `orphan_item_ids`). Neither category appears in `mappings`.)"
)


@dataclasses.dataclass
class InstrumentSignal:
    mappings: dict[int, frozenset[tuple[int, bool]]]
    item_texts: dict[int, str]
    scale_names: dict[int, str]
    scale_order: list[int]
    scale_depths: dict[int, int]


def _build_signal(nodes: list[ScaleNode]) -> InstrumentSignal:
    """Collect reliability/tree-comparison data from a resolved Instrument tree."""
    mappings: dict[int, set[tuple[int, bool]]] = {}
    item_texts: dict[int, str] = {}
    scale_names: dict[int, str] = {}
    scale_order: list[int] = []
    scale_depths: dict[int, int] = {}

    def visit(node: ScaleNode, depth: int) -> None:
        scale_names[node.id] = node.scale_name
        scale_order.append(node.id)
        scale_depths[node.id] = depth
        for scaled in node.items:
            mappings.setdefault(scaled.item_id, set()).add(
                (node.id, bool(scaled.reverse_coded))
            )
            item_texts[scaled.item_id] = scaled.item_text or f"(image item {scaled.item_id})"
        for sub in node.subscales:
            visit(sub, depth + 1)

    for n in nodes:
        visit(n, 0)
    return InstrumentSignal(
        mappings={k: frozenset(v) for k, v in mappings.items()},
        item_texts=item_texts,
        scale_names=scale_names,
        scale_order=scale_order,
        scale_depths=scale_depths,
    )


# Tree-comparison grid layout (logged when num_runs >= 2).
_LABEL_W = 58  # label column width
_COL_W = 4     # per-run column width


def _format_tree_comparison(signals: list[InstrumentSignal]) -> list[str]:
    n = len(signals)

    # Reversed so first-run values win.
    scale_names: dict[int, str] = {}
    scale_depths: dict[int, int] = {}
    item_texts: dict[int, str] = {}
    for sig in reversed(signals):
        scale_names.update(sig.scale_names)
        scale_depths.update(sig.scale_depths)
        item_texts.update(sig.item_texts)

    seen_scales: set[int] = set()
    all_scale_ids: list[int] = []
    for sig in signals:
        for sid in sig.scale_order:
            if sid not in seen_scales:
                all_scale_ids.append(sid)
                seen_scales.add(sid)

    scale_to_items: dict[int, list[int]] = {sid: [] for sid in all_scale_ids}
    seen_for_scale: dict[int, set[int]] = {sid: set() for sid in all_scale_ids}
    for sig in signals:
        for item_id, entries in sig.mappings.items():
            for scale_id, _rev in entries:
                if scale_id in seen_for_scale and item_id not in seen_for_scale[scale_id]:
                    scale_to_items[scale_id].append(item_id)
                    seen_for_scale[scale_id].add(item_id)
    for sid in all_scale_ids:
        scale_to_items[sid].sort()

    # run_lookup[i][item_id][scale_id] = reverse_coded
    run_lookup: list[dict[int, dict[int, bool]]] = []
    for sig in signals:
        d: dict[int, dict[int, bool]] = {}
        for item_id, entries in sig.mappings.items():
            d[item_id] = {sid: rev for sid, rev in entries}
        run_lookup.append(d)

    sep = "─" * (_LABEL_W + n * _COL_W)
    header_cols = "".join(f"R{i + 1:<{_COL_W - 1}}" for i in range(n))

    lines = [
        "",
        f"TREE COMPARISON  ({n} run{'s' if n != 1 else ''})",
        sep,
        f"{'':>{_LABEL_W}}{header_cols}",
        sep,
    ]

    any_deviation = False
    for sid in all_scale_ids:
        depth = scale_depths.get(sid, 0)
        sname = scale_names.get(sid, f"Scale {sid}")
        scale_indent = "  " * depth
        lines.append(f"{scale_indent}[id={sid}] {sname}")

        for item_id in scale_to_items[sid]:
            item_indent = "  " * (depth + 1)
            prefix = f"{item_indent}item {item_id:>3}  "
            max_text = _LABEL_W - len(prefix)
            raw = item_texts.get(item_id, f"(item {item_id})")
            text = raw[: max_text - 1] + "…" if len(raw) > max_text else raw
            label = f"{prefix}{text}"

            cells: list[str] = []
            for run in run_lookup:
                scale_map = run.get(item_id, {})
                if sid in scale_map:
                    cells.append("R" if scale_map[sid] else "✓")
                else:
                    cells.append("-")

            deviation = len(set(cells)) > 1
            any_deviation = any_deviation or deviation
            flag = "  ◄" if deviation else ""
            col_str = "".join(f"{c:<{_COL_W}}" for c in cells)
            lines.append(f"{label:<{_LABEL_W}}{col_str}{flag}")

    lines.append(sep)
    lines.append(
        "  ◄ = assignment or reverse-coding disagrees between runs"
        if any_deviation
        else "  All assignments and reverse flags consistent across runs."
    )
    return lines


def _format_resolved_tree(nodes: list[ScaleNode], indent: int = 0) -> list[str]:
    """Pretty-print the resolved Instrument tree."""
    lines: list[str] = []
    prefix = "  " * indent
    for node in nodes:
        construct = f" [{node.construct_name}]" if node.construct_name else ""
        lines.append(f"{prefix}- [id={node.id}] {node.scale_name}{construct}")
        for scaled in node.items:
            rev = "  [REV]" if scaled.reverse_coded else ""
            text = scaled.item_text or f"(image item {scaled.item_id})"
            lines.append(
                f"{prefix}    · item {scaled.item_id:>3}: {text}{rev}"
            )
        if node.subscales:
            lines.extend(_format_resolved_tree(node.subscales, indent + 1))
    return lines


def _assignment_label(mapping: dict, item_id: int) -> str:
    """Sorted scale-id set of an item in one run, as a nominal label."""
    entry = mapping.get(item_id)
    if not entry:
        return "∅"
    return ",".join(str(sid) for sid, _rev in sorted(entry))


def _assignment_alpha(mappings: list[dict]) -> tuple[float | None, int, bool]:
    """Krippendorff's α (nominal) on per-item scale-id sets across runs.

    Returns ``(alpha, n_items, all_identical)``; ``alpha`` is None if the library raises.
    """
    all_items = sorted({item_id for m in mappings for item_id in m})
    n_items = len(all_items)
    if n_items == 0:
        return 1.0, 0, True

    data = np.array([[_assignment_label(m, i) for i in all_items] for m in mappings])
    if len(set(data.flatten())) <= 1:
        return 1.0, n_items, True
    try:
        alpha = float(krippendorff.alpha(
            reliability_data=data, level_of_measurement="nominal",
        ))
        return alpha, n_items, False
    except Exception:
        return None, n_items, False


def _reverse_agreement(mappings: list[dict]) -> tuple[float, int, int]:
    """Share of (item, scale) pairs whose ``reverse_coded`` agrees across runs.

    Returns ``(fraction, agreed, n_pairs)``.
    """
    pair_universe: set[tuple[int, int]] = {
        (item_id, scale_id)
        for m in mappings
        for item_id, entries in m.items()
        for scale_id, _rev in entries
    }
    if not pair_universe:
        return 1.0, 0, 0

    agreed = 0
    for item_id, scale_id in pair_universe:
        rev_values = [
            rev
            for m in mappings
            for sid, rev in m.get(item_id, set())
            if sid == scale_id
        ]
        if len(set(rev_values)) <= 1:
            agreed += 1
    return agreed / len(pair_universe), agreed, len(pair_universe)


class InstrumentExtractor(Extractor):
    name = "instrument"
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT_BODY
    result_model = Instrument
    depends_on = ["items", "scales"]

    def llm_schema(self) -> dict:
        return InstrumentMappings.inlined_schema()

    def parse_response(
        self,
        content_text: str,
        context: dict[str, BaseModel] | None = None,
    ) -> Instrument:
        if not context or "items" not in context or "scales" not in context:
            raise RuntimeError(
                "InstrumentExtractor.parse_response requires 'items' and 'scales' "
                "in context — the orchestrator should have gated this via depends_on."
            )
        mappings = InstrumentMappings.model_validate_json(extract_json(content_text))
        return resolve_instrument(mappings, context["items"], context["scales"])

    def build_user_prompt(self, context: dict | None = None) -> str:
        if not context or "items" not in context or "scales" not in context:
            return self.user_prompt
        items: Items = context["items"]
        scales: Scales = context["scales"]
        return (
            f"{self.user_prompt}\n"
            f"## Items (id. text)\n{render_items_block(items)}\n\n"
            f"## Scales (indented tree with [id=N])\n"
            f"{render_scales_block(scales, _UNSCALED_HINT)}\n"
        )

    def signal(self, result: Instrument) -> InstrumentSignal:
        return _build_signal(result.scales)

    def reliability(
        self, signals: list[InstrumentSignal]
    ) -> tuple[str, dict[str, float]]:
        out = Tee()
        n = len(signals)
        mappings = [sig.mappings for sig in signals]

        out.write("\n" + "=" * 60)
        out.write(f"RELIABILITY ANALYSIS  ({n} runs)")
        out.write("=" * 60)

        mapping_counts = [sum(len(v) for v in m.values()) for m in mappings]
        out.write("\nMapping counts per run:")
        for i, c in enumerate(mapping_counts, 1):
            out.write(f"  Run {i}: {c} mapping(s)")

        scores: dict[str, float] = {}

        if n < 2:
            out.write("\n(Need at least 2 runs for reliability measures.)")
            out.write("=" * 60 + "\n")
            return out.dump(), scores

        pairs = list(combinations(range(n), 2))
        count_agreement = sum(
            mapping_counts[i] == mapping_counts[j] for i, j in pairs
        ) / len(pairs)
        scores["count_agreement"] = count_agreement
        out.write(f"\nCount agreement (pair-exact): {count_agreement:.4f}")

        alpha, n_items, all_identical = _assignment_alpha(mappings)
        if alpha is None:
            scores["assignment_alpha"] = float("nan")
            out.write("Krippendorff's α: unavailable")
        else:
            scores["assignment_alpha"] = alpha
            if n_items == 0:
                out.write("Krippendorff's α: 1.0000 (all runs agree: no items mapped)")
            elif all_identical:
                out.write(
                    f"Krippendorff's α (nominal, {n_items} items): 1.0000"
                    " (all items assigned identically across all runs)"
                )
            else:
                out.write(f"Krippendorff's α (nominal, {n_items} items): {alpha:.4f}")

        rev_score, agreed, universe = _reverse_agreement(mappings)
        scores["reverse_agreement"] = rev_score
        if universe == 0:
            out.write("Reverse-code agreement: 1.0000 (all runs agree: no mappings)")
        else:
            out.write(
                f"Reverse-code agreement: {rev_score:.4f}  "
                f"({agreed}/{universe} pairs consistent)"
            )

        out.writelines(_format_tree_comparison(signals))
        out.write("=" * 60 + "\n")
        return out.dump(), scores

    def suite_reliability(
        self, all_signals: list[list[InstrumentSignal]]
    ) -> str:
        """Mean ± SD of assignment_alpha across documents."""
        per_doc_alpha: list[float] = []
        for doc_signals in all_signals:
            if len(doc_signals) < 2:
                continue
            doc_mappings = [sig.mappings for sig in doc_signals]
            alpha, _n_items, _all_identical = _assignment_alpha(doc_mappings)
            if alpha is not None and not np.isnan(alpha):
                per_doc_alpha.append(alpha)

        if not per_doc_alpha:
            return "  Suite assignment α: no multi-run documents available."
        arr = np.array(per_doc_alpha)
        return (
            f"  Mean assignment α across docs: {arr.mean():.4f} ± {arr.std():.4f}  "
            f"(range [{arr.min():.4f}, {arr.max():.4f}])"
        )

    def format_run(
        self,
        result: Instrument,
        run_idx: int,
        num_runs: int,
        run_elapsed: float,
        log: Callable[[str], None],
    ) -> None:
        sig = _build_signal(result.scales)
        n_mappings = sum(len(v) for v in sig.mappings.values())
        n_items = len(sig.mappings)
        n_multi = sum(1 for v in sig.mappings.values() if len(v) > 1)
        n_reverse = sum(1 for v in sig.mappings.values() for _sid, rev in v if rev)
        n_unscaled = len(result.unscaled_items)
        n_orphan = len(result.orphan_items)

        log(
            f"  Run {run_idx + 1}: {run_elapsed:.2f}s  "
            f"({n_mappings} mapping(s) covering {n_items} item(s); "
            f"{n_multi} multi-scale, {n_reverse} reverse-coded, "
            f"{n_unscaled} unscaled, {n_orphan} orphan)"
        )
        dump_result_to_log(
            result, log,
            stdout_header=(
                f"Extracted {n_mappings} mapping(s) "
                f"({n_items} unique items, {n_multi} multi-scale, "
                f"{n_reverse} reverse-coded, {n_unscaled} unscaled, "
                f"{n_orphan} orphan):"
            ),
            log_header=(
                f"  Resolved instrument "
                f"({n_mappings} mapping(s), {n_items} unique items, "
                f"{n_multi} multi-scale, {n_reverse} reverse-coded, "
                f"{n_unscaled} unscaled, {n_orphan} orphan):"
            ),
        )
