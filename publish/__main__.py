"""CLI entrypoint for the publication pipeline: publish -> upload.

Deliberately separate from ``python -m assemble`` (Hub commits are permanent).
See docs/architecture-publish.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import yaml

from extraction.text_utils import write_log

from . import hub, publish

STEPS = {"publish": publish.run, "upload": hub.run}

#: Upload reads what publish wrote.
PIPELINE: tuple[str, ...] = ("publish", "upload")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Publication pipeline: build the encrypted release "
                    "artifact from the pooled corpus, then publish it to the "
                    "private HuggingFace dataset repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument(
        "--step", default=None,
        help=f"Run a single step: {', '.join(STEPS)}.",
    )
    ap.add_argument(
        "--report-only", action="store_true",
        help="Print reports without writing or uploading anything. Makes no "
             "network calls and never generates a key.",
    )
    ap.add_argument(
        "--check-remote", action="store_true",
        help="Upload step only: with --report-only, also read the live repo "
             "state (visibility, whether it already holds this corpus).",
    )
    ap.add_argument(
        "--yes", action="store_true",
        help="Upload step only: skip the interactive confirmation.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Upload step only: publish even when the remote already holds "
             "this corpus, or the pipeline tree was dirty when it was built.",
    )
    ap.add_argument(
        "--skip-verify", action="store_true",
        help="Upload step only: skip re-hashing the artifact against its "
             "manifest before committing.",
    )
    ap.add_argument(
        "--key-path", default=None,
        help="Fernet key file (default: data.publish with a .key "
             "suffix). Overridden by "
             "$SYNTHOMETRICON_RELEASE_KEY when that is set.",
    )
    ap.add_argument(
        "--no-logs", action="store_true",
        help="Skip writing the step reports to logs/.",
    )
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}

    if args.step:
        if args.step not in STEPS:
            sys.exit(f"Unknown step: {args.step}. Available: "
                     f"{', '.join(STEPS)}")
        selected = [args.step]
    else:
        selected = list(PIPELINE)

    upload_only = {"yes", "check_remote", "force", "skip_verify"}
    if "upload" not in selected:
        for flag in sorted(upload_only):
            if getattr(args, flag):
                sys.exit(f"--{flag.replace('_', '-')} only applies to the "
                         f"upload step")

    report: list[str] = []
    for name in selected:
        extra = {"key_path": args.key_path}
        if name == "upload":
            extra |= {"yes": args.yes, "check_remote": args.check_remote,
                      "force": args.force, "skip_verify": args.skip_verify}
        lines = STEPS[name](cfg, report_only=args.report_only, **extra)
        print("\n".join(lines))
        print()
        report += [f"=== {name} ==="] + lines + [""]

    if args.report_only:
        print("(report-only: nothing written or uploaded)")
    elif not args.no_logs:
        log_path = ("logs/publish-"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        write_log(log_path, "\n".join(report))
        print(f"[log written to {log_path}]")


if __name__ == "__main__":
    main()
