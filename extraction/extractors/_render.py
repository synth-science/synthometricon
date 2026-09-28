"""Shared rendering helpers for extractor prompts and per-run log output."""

from __future__ import annotations

from typing import Callable

import yaml
from pydantic import BaseModel

from ..models import Items, Scale, Scales


def _format_scale_lines(
    scales: list[Scale],
    indent: int = 0,
    with_ids: bool = False,
) -> list[str]:
    """Render a scale hierarchy as indented ``- ...`` lines, optionally ``[id=N]``-prefixed."""
    out: list[str] = []
    prefix = "  " * indent
    for scale in scales:
        construct = f" [{scale.construct_name}]" if scale.construct_name else ""
        id_label = f"[id={scale.id}] " if with_ids else ""
        out.append(f"{prefix}- {id_label}{scale.scale_name}{construct}")
        if scale.subscales:
            out.extend(_format_scale_lines(scale.subscales, indent + 1, with_ids))
    return out


def format_scale_tree(scales: list[Scale]) -> str:
    """Scale tree for log output (no IDs)."""
    return "\n".join(_format_scale_lines(scales, with_ids=False))


def format_scale_tree_with_ids(scales: list[Scale]) -> str:
    """Scale tree with ``[id=N]`` prefixes for prompt context."""
    return "\n".join(_format_scale_lines(scales, with_ids=True))


def flatten_scale_names(scales: list[Scale]) -> list[str]:
    """All scale names in DFS order."""
    out: list[str] = []
    for scale in scales:
        out.append(scale.scale_name)
        if scale.subscales:
            out.extend(flatten_scale_names(scale.subscales))
    return out


def count_scale_nodes(scales: list[Scale]) -> int:
    """Count all nodes in a scale tree."""
    return sum(1 + count_scale_nodes(s.subscales or []) for s in scales)


def render_items_block(items: Items) -> str:
    """Render items as ``N. text`` lines.

    Stem-less items show their options or an image placeholder so downstream
    extractors can still judge scale assignment and reverse-coding.
    """
    lines: list[str] = []
    for item in items.items:
        if item.item_text:
            stem = item.item_text
        elif item.options:
            stem = f"(options: {' | '.join(item.options)})"
        elif item.has_image:
            stem = "(image stimulus)"
        else:
            stem = "(no stem)"
        lines.append(f"{item.id}. {stem}")
    return "\n".join(lines)


def render_scales_block(scales: Scales, unscaled_hint: str | None = None) -> str:
    """ID-labelled scale tree, plus ``unscaled_hint`` if ``has_unscaled_items``."""
    tree = format_scale_tree_with_ids(scales.scales)
    if scales.has_unscaled_items and unscaled_hint:
        tree += f"\n\n{unscaled_hint}"
    return tree


def dump_result_to_log(
    result: BaseModel,
    log: Callable[[str], None],
    *,
    stdout_header: str,
    log_header: str,
) -> None:
    """YAML-dump a result to stdout and (indented) to the log."""
    dump = yaml.dump(result.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
    print(f"\n{stdout_header}\n")
    print(dump)

    log(log_header)
    for line in dump.splitlines():
        log(f"    {line}")
    log("")
