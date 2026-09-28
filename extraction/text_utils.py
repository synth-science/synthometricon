"""Dependency-light text and I/O helpers shared across the project."""

from __future__ import annotations

import os
from typing import Iterable

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def lev_dist(a: str, b: str) -> float:
    """Normalised Levenshtein distance in ``[0, 1]`` after lowercasing."""
    return Levenshtein.normalized_distance(a.lower(), b.lower())


def fuzzy_contains(needle: str, haystack: str) -> float:
    """Best lowercased similarity of *needle* against any same-length window of *haystack*."""
    n = needle.lower()
    h = haystack.lower()
    if n in h:
        return 1.0
    nl = len(n)
    if nl == 0:
        return 1.0
    if nl >= len(h):
        return 1.0 - Levenshtein.normalized_distance(n, h)
    best = 0.0
    for i in range(len(h) - nl + 1):
        sim = 1.0 - Levenshtein.normalized_distance(n, h[i : i + nl])
        if sim > best:
            best = sim
            if best == 1.0:
                return 1.0
    return best


def best_partial_distance(needle: str, haystack: str) -> tuple[int, float] | None:
    """Return ``(edits, normalized)`` distance of *needle* to its best window of *haystack*.

    Case-sensitive; uses ``partial_ratio_alignment`` so it scales to full documents.
    ``None`` when either side is empty.
    """
    if not needle or not haystack:
        return None
    aln = fuzz.partial_ratio_alignment(needle, haystack)
    if aln is None:
        return None
    window = haystack[aln.dest_start:aln.dest_end]
    return (Levenshtein.distance(needle, window),
            Levenshtein.normalized_distance(needle, window))


def lev_edits(a: str, b: str) -> int:
    """Raw Levenshtein edit count after lowercasing (companion to :func:`lev_dist`)."""
    return Levenshtein.distance(a.lower(), b.lower())


def extract_json(text: str) -> str:
    """Strip prose and markdown code fences around a JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class Tee:
    """Accumulate log lines; ``write`` also prints to stdout, ``log`` does not."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def write(self, s: str = "") -> None:
        print(s, flush=True)
        self._lines.append(s)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def log(self, s: str = "") -> None:
        self._lines.append(s)

    def loglines(self, lines: Iterable[str]) -> None:
        self._lines.extend(lines)

    def dump(self) -> str:
        return "\n".join(self._lines)


def write_log(log_path: str, content: str) -> None:
    """Write ``content`` to ``log_path``, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(content)


# Run-log placement and scratch retention — see docs/architecture-extraction.md#run-logs

LOGS_DIR = "logs"
SCRATCH_SUBDIR = "scratch"

#: Defaults for ``logs.scratch_below_files`` / ``logs.scratch_keep``.
SCRATCH_BELOW_FILES = 10
SCRATCH_KEEP = 50


def _logs_cfg(config: dict | None, key: str, default: int) -> int:
    """Read ``logs.<key>`` from ``config``, falling back to ``default``."""
    try:
        return int((config or {}).get("logs", {}).get(key, default))
    except (TypeError, ValueError):
        return default


def is_scratch_run(n_files: int, config: dict | None = None) -> bool:
    """True when a run over ``n_files`` PDFs is a smoke test (threshold ``0`` disables)."""
    below = _logs_cfg(config, "scratch_below_files", SCRATCH_BELOW_FILES)
    return below > 0 and n_files < below


def prune_scratch_logs(keep: int = SCRATCH_KEEP, logs_dir: str = LOGS_DIR) -> int:
    """Delete all but the ``keep`` newest logs in ``logs/scratch/``; return the count removed."""
    scratch = os.path.join(logs_dir, SCRATCH_SUBDIR)
    if keep < 0 or not os.path.isdir(scratch):
        return 0
    entries = [
        os.path.join(scratch, name)
        for name in os.listdir(scratch)
        if name.endswith(".log")
    ]
    entries = [p for p in entries if os.path.isfile(p)]
    entries.sort(key=os.path.getmtime, reverse=True)
    removed = 0
    for path in entries[keep:]:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def write_run_log(
    content: str,
    basename: str,
    n_files: int,
    config: dict | None = None,
    logs_dir: str = LOGS_DIR,
    allow_scratch: bool = True,
) -> str:
    """Write a run log to ``logs/`` or (smoke runs) ``logs/scratch/``; return its path.

    ``allow_scratch=False`` forces a permanent top-level log (used by live mode).
    """
    if allow_scratch and is_scratch_run(n_files, config):
        path = os.path.join(logs_dir, SCRATCH_SUBDIR, basename)
        write_log(path, content)
        prune_scratch_logs(
            _logs_cfg(config, "scratch_keep", SCRATCH_KEEP), logs_dir=logs_dir
        )
        return path
    path = os.path.join(logs_dir, basename)
    write_log(path, content)
    return path
