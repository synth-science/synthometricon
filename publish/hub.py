"""Upload step: commit the encrypted artifact, dataset card (``README.md``) and
manifest to the private HuggingFace dataset repo in one atomic commit.

Mostly refusals — Hub commits are permanent. Rationale and gotchas:
docs/architecture-publish.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from . import keys, manifest
from .paths import key_path as key_path_for, release_path

CARD_FILE = Path(__file__).resolve().parent / "dataset_card.md"

# The Hub renders the repo's README.md as the dataset card.
CARD_PATH_IN_REPO = "README.md"
MANIFEST_PATH_IN_REPO = "manifest.json"

REPO_TYPE = "dataset"


def _cfg(cfg: dict) -> dict:
    return cfg.get("publish", {}) or {}


def resolve_token(cfg: dict) -> str:
    """HF token from ``$publish.token_env``, else the cached CLI login. Never reported."""
    from huggingface_hub import get_token

    env_name = _cfg(cfg).get("token_env") or "HF_TOKEN"
    token = (os.environ.get(env_name) or "").strip() or get_token()
    if not token:
        sys.exit(f"no HuggingFace token: set ${env_name} or run "
                 f"`huggingface-cli login` (a write token is required)")
    return token


def scrub(text: str, token: str) -> str:
    """Remove a token from text bound for stdout or a log file."""
    return text.replace(token, "<token>") if token else text


def verify_repo(api, repo_id: str, *, private: bool, create: bool,
                report: list[str]):
    """Create the repo if asked, verify it is private; return the parent commit oid.

    ``create_repo(private=True, exist_ok=True)`` does not flip an existing public repo."""
    from huggingface_hub.errors import RepositoryNotFoundError

    namespace = repo_id.split("/")[0] if "/" in repo_id else None
    if namespace:
        who = api.whoami()
        owned = {who.get("name")} | {o.get("name")
                                     for o in who.get("orgs", []) or []}
        if namespace not in owned:
            sys.exit(f"the token's account cannot write to '{namespace}' "
                     f"(it can write to: {', '.join(sorted(filter(None, owned)))}). "
                     f"A user-scoped token passes authentication and only "
                     f"fails at the commit, after the upload.")
        report.append(f"authenticated as {who.get('name')} "
                      f"(namespace {namespace} writable)")

    if create:
        api.create_repo(repo_id, repo_type=REPO_TYPE, private=private,
                        exist_ok=True)
    try:
        info = api.repo_info(repo_id, repo_type=REPO_TYPE)
    except RepositoryNotFoundError:
        sys.exit(f"{repo_id} does not exist and publish.create_repo is false")

    if private and info.private is not True:
        sys.exit(f"{repo_id} is PUBLIC on the Hub. create_repo(private=True, "
                 f"exist_ok=True) does not change an existing repo's "
                 f"visibility — fix it in the Hub settings before publishing.")
    report.append(f"repo {repo_id}: private={info.private}, "
                  f"parent commit {(info.sha or '')[:12] or 'none'}")
    return info.sha


def assert_no_key_leak(ops, key: bytes, key_path) -> None:
    """Refuse to commit a key file, or text payloads containing the key."""
    key_names = {Path(key_path).name} if key_path else set()
    for op in ops:
        name = Path(op.path_in_repo).name
        if name.endswith(".key") or name in key_names:
            raise ValueError(f"refusing to commit a key file: "
                             f"{op.path_in_repo}")
        payload = getattr(op, "path_or_fileobj", None)
        if isinstance(payload, (bytes, bytearray)) and key in payload:
            raise ValueError(f"the release key appears in {op.path_in_repo} — "
                             f"refusing to upload")


def build_operations(artifact, card_bytes: bytes, manifest_bytes: bytes,
                     path_in_repo: str):
    """The three commit operations; the blob is a ``Path`` so it is not held in memory."""
    from huggingface_hub import CommitOperationAdd

    return [
        CommitOperationAdd(path_in_repo=path_in_repo,
                           path_or_fileobj=Path(artifact)),
        CommitOperationAdd(path_in_repo=CARD_PATH_IN_REPO,
                           path_or_fileobj=card_bytes),
        CommitOperationAdd(path_in_repo=MANIFEST_PATH_IN_REPO,
                           path_or_fileobj=manifest_bytes),
    ]


def remote_plaintext_sha(api, repo_id: str, report: list[str]):
    """``plaintext_sha256`` of the release currently on the Hub, or None."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    try:
        path = hf_hub_download(repo_id, MANIFEST_PATH_IN_REPO,
                               repo_type=REPO_TYPE, token=api.token)
    except (EntryNotFoundError, HfHubHTTPError):
        report.append("remote has no manifest yet (first release)")
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return (payload.get("artifact") or {}).get("plaintext_sha256")


def confirm(lines: list[str], *, yes: bool) -> None:
    """Interactive gate. ``--yes`` skips; a non-tty without it aborts."""
    if yes:
        return
    if not sys.stdin.isatty():
        sys.exit("refusing to upload non-interactively without --yes")
    print("\n".join(lines))
    print()
    answer = input("Type 'upload' to publish, anything else to abort: ")
    if answer.strip() != "upload":
        sys.exit("aborted")


def run(cfg: dict, *, report_only: bool = False, yes: bool = False,
        check_remote: bool = False, force: bool = False,
        skip_verify: bool = False, artifact_path=None, key_path=None,
        api=None) -> list[str]:
    """Step entry point: publish the release artifact to the Hub."""
    pub = _cfg(cfg)
    artifact = Path(artifact_path or release_path(cfg) or "")
    repo_id = pub.get("repo_id")
    path_in_repo = pub.get("path_in_repo")
    if str(artifact) == "." or not repo_id or not path_in_repo:
        sys.exit("data.publish, publish.repo_id and "
                 "publish.path_in_repo must be set")

    report = ["corpus upload report", f"repo: {repo_id} ({REPO_TYPE})",
              f"artifact: {artifact}"]

    payload = manifest.read_manifest(artifact)
    if payload is None:
        # In a full --report-only run the publish step has not written it either.
        message = (f"no manifest beside {artifact} — run "
                   f"`python -m publish --step publish` first")
        if report_only:
            return report + [f"WARNING: {message}",
                             "(report-only: nothing to plan yet)"]
        sys.exit(message)
    art_meta = payload.get("artifact") or {}
    frame = payload.get("frame") or {}
    plaintext_sha = art_meta.get("plaintext_sha256")
    report.append(f"manifest: {frame.get('rows', 0):,} rows x "
                  f"{frame.get('columns', 0)} columns, plaintext sha256 "
                  f"{str(plaintext_sha)[:12]}, key "
                  f"{art_meta.get('key_fingerprint')}")

    if not artifact.exists():
        if report_only:
            report.append(f"artifact missing locally; would upload "
                          f"{art_meta.get('bytes', 0) / 1e6:,.0f} MB per manifest")
        else:
            sys.exit(f"artifact not found: {artifact}")
    else:
        size = artifact.stat().st_size
        report.append(f"local artifact: {size / 1e6:,.0f} MB -> {path_in_repo}")
        if manifest.manifest_stale(artifact):
            sys.exit("the artifact is newer than its manifest — re-run "
                     "`python -m publish --step publish`")
        pooled = (cfg.get("data", {}).get("assemble", {}) or {}).get("pooled")
        if pooled and Path(pooled).exists() and \
                Path(pooled).stat().st_mtime > artifact.stat().st_mtime:
            sys.exit(f"{Path(pooled).name} is newer than the release artifact "
                     f"— re-run `python -m publish --step publish`")
        if not skip_verify and not report_only:
            actual = manifest.sha256_file(artifact)
            if actual != art_meta.get("sha256"):
                sys.exit(f"artifact sha256 {actual[:12]} does not match the "
                         f"manifest ({str(art_meta.get('sha256'))[:12]}) — the "
                         f"file changed since it was built")
            report.append("artifact sha256 matches the manifest")

    git = payload.get("pipeline") or {}
    if git.get("dirty"):
        note = (f"the pipeline checkout was dirty at commit "
                f"{git.get('commit')} when this artifact was built — its "
                f"provenance does not reproduce")
        if report_only:
            report.append(f"WARNING: {note}; a real run needs --force")
        elif not force:
            sys.exit(f"{note}. Commit your changes and re-run the publish "
                     f"step, or pass --force.")
        else:
            report.append(f"WARNING: {note} (--force)")

    card_bytes = CARD_FILE.read_bytes()
    manifest_bytes = (json.dumps(payload, indent=2, default=str) + "\n").encode()

    key_path = key_path or key_path_for(artifact)
    key, _ = keys.resolve_key(cfg, report, allow_create=False,
                              key_path=key_path)
    if art_meta.get("key_fingerprint") != f"sha256:{keys.fingerprint(key)}":
        sys.exit(f"the resolved key ({keys.fingerprint(key)}) is not the one "
                 f"this artifact was encrypted with "
                 f"({art_meta.get('key_fingerprint')}) — publishing it would "
                 f"ship data nobody can decrypt")

    ops_preview = [path_in_repo, CARD_PATH_IN_REPO, MANIFEST_PATH_IN_REPO]
    report.append(f"would commit: {', '.join(ops_preview)}")

    if report_only and not check_remote:
        report.append("(report-only: no network calls made)")
        return report

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    token = resolve_token(cfg)
    api = api or HfApi(token=token, library_name="synth-net-pipline",
                       library_version=str(git.get("commit", "unknown")))
    try:
        parent = verify_repo(api, repo_id,
                             private=bool(pub.get("private", True)),
                             create=bool(pub.get("create_repo", True))
                             and not report_only,
                             report=report)
        remote_sha = remote_plaintext_sha(api, repo_id, report)
        if remote_sha and remote_sha == plaintext_sha and not force:
            report.append("remote already holds this exact corpus "
                          f"(plaintext sha256 {remote_sha[:12]}) — nothing to "
                          "upload. Pass --force to push anyway.")
            return report

        if report_only:
            report.append("(report-only: remote checked, nothing uploaded)")
            return report

        ops = build_operations(artifact, card_bytes, manifest_bytes,
                               path_in_repo)
        assert_no_key_leak(ops, key, key_path)

        confirm(report + [
            "",
            f"About to publish to {repo_id} (private, {REPO_TYPE}).",
            f"  {path_in_repo}  {artifact.stat().st_size / 1e6:,.0f} MB",
            f"  {CARD_PATH_IN_REPO}  {len(card_bytes) / 1e3:,.0f} kB",
            f"  {MANIFEST_PATH_IN_REPO}  {len(manifest_bytes) / 1e3:,.0f} kB",
            "Hub commits are permanent: an uploaded blob stays in history.",
        ], yes=yes)

        api.preupload_lfs_files(repo_id, [ops[0]], repo_type=REPO_TYPE,
                                revision=pub.get("revision"))
        mode = getattr(ops[0], "_upload_mode", None)
        if mode == "regular":
            sys.exit(f"the Hub classified {path_in_repo} as a regular git "
                     f"object, not LFS — committing a gigabyte that way would "
                     f"wreck the repo. Use a path ending in .bin.")
        report.append(f"blob uploaded (mode {mode or 'unknown'})")

        commit = api.create_commit(
            repo_id, ops, repo_type=REPO_TYPE,
            revision=pub.get("revision"),
            parent_commit=parent,
            commit_message=(f"Release {payload.get('generated', '')[:10]} — "
                            f"{frame.get('rows', 0):,} rows x "
                            f"{frame.get('columns', 0)} columns"),
            commit_description=(
                f"pipeline commit: {git.get('commit')}\n"
                f"plaintext sha256: {plaintext_sha}\n"
                f"key fingerprint: {art_meta.get('key_fingerprint')}\n"),
        )
        report.append(f"committed {str(commit.oid)[:12]}")

        tag = pub.get("tag")
        if tag:
            try:
                api.create_tag(repo_id, tag=str(tag), revision=commit.oid,
                               repo_type=REPO_TYPE)
                report.append(f"tagged {tag}")
            except HfHubHTTPError as err:
                report.append(f"WARNING: could not tag {tag}: "
                              f"{scrub(str(err), token)}")
    except HfHubHTTPError as err:
        sys.exit(scrub(str(err), token))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Standalone upload step: publish the encrypted release "
                    "artifact, its dataset card and its manifest to the "
                    "private HuggingFace dataset repo.",
    )
    ap.add_argument("--config", default="config.yaml", help="Config YAML path.")
    ap.add_argument("--artifact", default=None,
                    help="Encrypted artifact (default: data.publish).")
    ap.add_argument("--key-path", default=None,
                    help="Fernet key file (default: <artifact>.key).")
    ap.add_argument("--report-only", action="store_true",
                    help="Print the plan; make no network calls and upload "
                         "nothing.")
    ap.add_argument("--check-remote", action="store_true",
                    help="With --report-only: also read the live repo state.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the interactive confirmation.")
    ap.add_argument("--force", action="store_true",
                    help="Upload even when the remote already holds this "
                         "corpus, or the build tree was dirty.")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip re-hashing the artifact against the manifest.")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    lines = run(cfg, report_only=args.report_only, yes=args.yes,
                check_remote=args.check_remote, force=args.force,
                skip_verify=args.skip_verify, artifact_path=args.artifact,
                key_path=args.key_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
