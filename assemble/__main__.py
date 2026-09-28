"""CLI entrypoint for the assemble pipeline (explode → combine → patch → postprocess → encode → pool).

Usage: ``python -m assemble [--step NAME] [--report-only] [--refresh] [--no-logs]``.
See docs/architecture-assemble.md; publishing is ``python -m publish``.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Callable

import yaml

from extraction.text_utils import write_log

from . import combine, encode, explode, patch, pool, postprocess

Stage = Callable[[dict, bool], list[str]]
STAGES: dict[str, Stage] = {
    "explode": explode.run,
    "combine": combine.run,
    "patch": patch.run,
    "postprocess": postprocess.run,
    "encode": encode.run,
    "pool": pool.run,
}

PIPELINE: tuple[str, ...] = ("explode", "combine", "patch", "postprocess",
                             "encode", "pool")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dataset build pipeline: explode, combine, patch, "
                    "postprocess, encode, pool. Publishing the result is "
                    "`python -m publish`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument(
        "--explode", action="store_true",
        help=(
            "Alias for --step explode: explode the raw extraction store "
            "(data.extractions) to a per-item parquet "
            "(data.assemble.exploded)."
        ),
    )
    ap.add_argument(
        "--step", default=None,
        help=f"Run a single stage: {', '.join(STAGES)}.",
    )
    ap.add_argument(
        "--report-only", action="store_true",
        help="Print reports without writing anything.",
    )
    ap.add_argument(
        "--refresh", action="store_true",
        help="Patch stage only: ignore the Crossref/doi_probe caches.",
    )
    ap.add_argument(
        "--no-logs", action="store_true",
        help="Skip writing the stage reports to logs/.",
    )
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}

    if args.explode:
        if args.step and args.step != "explode":
            sys.exit(f"--explode conflicts with --step {args.step}")
        args.step = "explode"

    if args.step:
        if args.step not in STAGES:
            sys.exit(f"Unknown stage: {args.step}. Available: {', '.join(STAGES)}")
        selected = [args.step]
    else:
        selected = list(PIPELINE)

    if args.refresh and "patch" not in selected:
        sys.exit("--refresh only applies to the patch stage")

    report: list[str] = []
    for name in selected:
        extra = {"refresh": args.refresh} if name == "patch" else {}
        lines = STAGES[name](cfg, report_only=args.report_only, **extra)
        print("\n".join(lines))
        print()
        report += [f"=== {name} ==="] + lines + [""]

    if args.report_only:
        print("(report-only: nothing written)")
    elif not args.no_logs:
        log_path = ("logs/assemble-"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        write_log(log_path, "\n".join(report))
        print(f"[log written to {log_path}]")


if __name__ == "__main__":
    main()
