#!/usr/bin/env python3
"""CLI entry point for the extraction pipeline (``python -m extraction``).

Modes and flags: docs/commands.md.
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

from .extractors import REGISTRY
from .orchestrator import (
    build_dependency_context,
    run_extractor,
    run_live,
    run_seed_validity_fixtures,
    run_status,
    run_validity,
)
from .config import DEFAULT_FIXTURES_DIR, load_config, load_test_cases


TEST_FILES_YAML = Path("tests/test_files.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one or more extractors against the PDFs listed in the config."
    )
    for name in REGISTRY:
        parser.add_argument(
            f"--{name}",
            action="store_true",
            help=f"Run the {name} extractor",
        )
    parser.add_argument(
        "--think",
        action="store_true",
        help=(
            "Override config: enable model thinking mode for every extractor "
            "in this run. Without --think, per-extractor thinking is taken "
            "from config.yaml's `think:` field (bool or {name: bool, default: bool})."
        ),
    )
    parser.add_argument("--no-logs", action="store_true", help="Skip writing log files")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--reliability",
        action="store_true",
        help="Run reliability tests over tests/test_files.yaml",
    )
    mode_group.add_argument(
        "--validity",
        action="store_true",
        help=(
            "Run validity tests: full pipeline per PDF, compared to the "
            "extraction_output: block in tests/fixtures/<stem>.yaml."
        ),
    )
    mode_group.add_argument(
        "--seed-validity-fixtures",
        action="store_true",
        help=(
            "Run the full pipeline per PDF and seed the extraction_output: "
            "block of tests/fixtures/<stem>.yaml (skip when present unless "
            "--force is also passed). For expert review before --validity."
        ),
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help=(
            "Print a done/failed/pending tally of the live parquet "
            "(data.extractions) plus a histogram of error strings, then exit."
        ),
    )
    parser.add_argument(
        "--no-fixtures",
        action="store_true",
        help=(
            "With --reliability: bypass YAML fixtures and run upstream "
            "dependencies live every time (end-to-end mode)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Live mode: re-run selected extractors even if their cell is "
            "already populated. --seed-validity-fixtures: overwrite an "
            "existing extraction_output: block."
        ),
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help=(
            "Live mode: use the documents already in the parquet "
            "(data.extractions) as the worklist instead of config input_files. "
            "Combine with --retry-failed / --retry-error to re-attempt failures "
            "without re-listing paths."
        ),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Live mode: re-attempt cells that previously failed (content null, "
            "error recorded). Without this flag failed cells are left alone."
        ),
    )
    parser.add_argument(
        "--retry-error",
        metavar="PATTERN",
        default=None,
        help=(
            "Live mode: re-attempt failed cells whose recorded error string "
            "matches the regex PATTERN (e.g. 'TimeoutError' or 'context'). "
            "Implies retrying only the matching failures."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Live mode: cap how many times a failed cell is retried. A cell "
            "that has reached this many attempts is skipped even under "
            "--retry-failed / --retry-error. Default: unlimited."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Live mode: split the worklist into N disjoint shards and process "
            "only this task's shard (see --shard-index). Default 1 (no "
            "sharding). Used for cluster array jobs: each array task takes one "
            "shard. The slice is paths[shard_index::num_shards] over the "
            "deterministic (sorted, deduped) input list, so shards are stable "
            "and disjoint across tasks."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help=(
            "Live mode: 0-based index of the shard this invocation processes "
            "(0 <= shard-index < num-shards). Maps to $SLURM_ARRAY_TASK_ID."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Live mode: override data.extractions from the config. Lets each "
            "shard write its own parquet (e.g. extractions.shard-0007.parquet) "
            "so concurrent workers never share a file."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Override llama_cpp.base_url from the config (OpenAI-compatible "
            "endpoint, e.g. http://127.0.0.1:20007/v1). Lets each array task "
            "point at its own per-task llama-server port."
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")

    selected = [name for name in REGISTRY if getattr(args, name)]
    if not selected:
        selected = list(REGISTRY)
    selected_extractors = [REGISTRY[name] for name in selected]
    all_extractors = list(REGISTRY.values())

    config = load_config(args.config)
    if args.think:
        config["think"] = True
    if args.base_url:
        config.setdefault("llama_cpp", {})["base_url"] = args.base_url
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fixtures_dir = config.get("testing", {}).get("fixtures_dir", DEFAULT_FIXTURES_DIR)

    output_path = Path(
        args.output_path
        if args.output_path
        else config.get("data", {}).get("extractions", "./data/extractions.parquet")
    )

    if args.status:
        run_status(all_extractors=all_extractors, output_path=output_path)
        sys.exit(0)

    if args.reliability:
        test_cases = load_test_cases(str(TEST_FILES_YAML), fixtures_dir=fixtures_dir)
        pdf_paths = [c["path"] for c in test_cases]
        num_runs = config.get("testing", {}).get("num_runs", 2)
        all_thresholds: dict = (
            config.get("testing", {}).get("reliability", {}).get("thresholds", {})
        )
        all_passed = True
        for name in selected:
            extractor = REGISTRY[name]
            contexts: dict | None = None
            if extractor.depends_on:
                contexts = {}
                for case in test_cases:
                    try:
                        ctx = build_dependency_context(
                            pdf_path=case["path"],
                            deps=extractor.depends_on,
                            registry=REGISTRY,
                            config=config,
                            case=case,
                            fixtures_dir=fixtures_dir,
                            force_live=args.no_fixtures,
                        )
                        contexts[case["path"]] = ctx
                    except Exception as exc:
                        print(
                            f"  [warn] could not build dependency context for "
                            f"{case['path']}: {exc} — file will be skipped"
                        )
            result = run_extractor(
                extractor,
                config,
                run_timestamp=run_timestamp,
                pdf_paths=pdf_paths,
                num_runs=num_runs,
                no_logs=args.no_logs,
                reliability_thresholds=all_thresholds.get(name),
                contexts=contexts,
            )
            if not result["all_passed"]:
                all_passed = False
        sys.exit(0 if all_passed else 1)

    if args.validity:
        test_cases = load_test_cases(str(TEST_FILES_YAML), fixtures_dir=fixtures_dir)
        pdf_paths = [c["path"] for c in test_cases]
        result = run_validity(
            pdf_paths=pdf_paths,
            all_extractors=all_extractors,
            config=config,
            test_cases=test_cases,
            no_logs=args.no_logs,
            run_timestamp=run_timestamp,
        )
        sys.exit(0 if result["all_passed"] else 1)

    if args.seed_validity_fixtures:
        test_cases = load_test_cases(str(TEST_FILES_YAML), fixtures_dir=fixtures_dir)
        pdf_paths = [c["path"] for c in test_cases]
        run_seed_validity_fixtures(
            pdf_paths=pdf_paths,
            all_extractors=all_extractors,
            config=config,
            fixtures_dir=fixtures_dir,
            force=args.force,
            no_logs=args.no_logs,
        )
        sys.exit(0)

    # Live mode
    if args.retry:
        from .storage import read_paths
        pdf_paths = read_paths(output_path)
        if not pdf_paths:
            print(f"--retry: no documents in {output_path} yet — nothing to retry.")
            sys.exit(1)
    else:
        pdf_paths = config.get("input_files", [])
        if not pdf_paths:
            print("No input_files specified in config. Add files under input_files: to run in live mode.")
            sys.exit(1)

    # Stable, disjoint stride over the deterministic worklist for cluster array jobs.
    if args.num_shards > 1:
        full_count = len(pdf_paths)
        pdf_paths = pdf_paths[args.shard_index :: args.num_shards]
        print(
            f"shard     : {args.shard_index + 1}/{args.num_shards} "
            f"({len(pdf_paths)} of {full_count} documents)"
        )
        if not pdf_paths:
            print("This shard has no documents — nothing to do.")
            sys.exit(0)

    result = run_live(
        selected_extractors=selected_extractors,
        all_extractors=all_extractors,
        pdf_paths=pdf_paths,
        output_path=output_path,
        config=config,
        no_logs=args.no_logs,
        run_timestamp=run_timestamp,
        force=args.force,
        retry_failed=args.retry_failed,
        retry_error=args.retry_error,
        max_attempts=args.max_attempts,
    )
    sys.exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
