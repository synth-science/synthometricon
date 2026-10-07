"""Tests for ``publish/``: key handling, column contract, manifest, dataset card, and a stubbed ``HfApi``.

Offline; keys live only in ``tmp_path``. See docs/testing.md for the invariants guarded here.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from cryptography.fernet import Fernet

from publish import hub, keys, manifest, paths
from publish import publish as step

KEY = Fernet.generate_key()
OTHER_KEY = Fernet.generate_key()


# --- Fixtures ---

def tiny_pooled(n: int = 6) -> pd.DataFrame:
    """Full release-contract frame with sentinel values so manifest leaks are visible."""
    rows = []
    for i in range(n):
        instrument = i % 3 == 0
        rows.append({
            "version": "SECRET_VERSION_XYZ",
            "meta_language": "en" if i % 2 else "de",
            "public_year": 2000 + i,
            "flag_item_count_deviation": bool(i % 2),
            "flag_scale_count_deviation": False,
            "flag_item_text_deviation": None if i == 1 else False,
            "flag_item_translated": False,
            "path": f"/abs/secret/path_{i}.pdf",
            "corpus_source": "apa-psyctests" if i % 2 else "scale-hunt",
            "public_doi": f"10.1037/SECRETDOI{i}",
            "doi_psyctests": f"10.1037/tSECRET{i}",
            "meta_title_raw": "SECRET_TITLE_XYZ",
            "is_instrument": instrument,
            "scale_id": None if instrument else float(i),
            "scale_name": None if instrument else "SECRET_SCALE_XYZ",
            "scale_depth": None if instrument else 1.0,
            "n_items": 10 + i,
            "n_scales": 2,
            "item_pooled_m": [0.1 * i, 0.2, 0.3, 0.4],
            "keying_disagreements_m": i,
            "scale_pooled_m": [0.5, 0.6, 0.7, 0.8],
            "scale_embedding_m": None if instrument else [0.1, 0.2, 0.3, 0.4],
            "instrument_embedding_m": [0.9, 0.8, 0.7, 0.6] if instrument else None,
        })
    return pd.DataFrame(rows)


def cfg_for(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "data": {"assemble": {"pooled": str(tmp_path / "pooled.parquet")},
                 "publish": str(tmp_path / "rel.parquet")},
        "publish": {"key_env": "TEST_RELEASE_KEY",
                    "repo_id": "org/corpus",
                    "path_in_repo": "data/rel.parquet",
                    "private": True, "create_repo": True, "revision": "main"},
        "encode": {"item_models": [{"name": "m", "path": "/home/models/secret"}],
                   "scale_models": [{"name": "m", "path": "/home/models/secret"}]},
    }
    cfg["publish"].update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("TEST_RELEASE_KEY", raising=False)


# --- publish.keys — resolution order and the no-echo guarantee ---

def test_env_key_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RELEASE_KEY", KEY.decode())
    report = []
    key, source = keys.resolve_key(cfg_for(tmp_path), report,
                                   allow_create=False)
    assert key == KEY and source == "env:TEST_RELEASE_KEY"
    assert keys.fingerprint(KEY) in "\n".join(report)


def test_key_file_used_when_env_unset(tmp_path):
    (tmp_path / "rel.key").write_bytes(KEY + b"\n")  # trailing newline stripped
    report = []
    key, source = keys.resolve_key(cfg_for(tmp_path), report,
                                   allow_create=False)
    assert key == KEY and source.startswith("file:")


def test_env_and_file_disagreement_aborts(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RELEASE_KEY", KEY.decode())
    (tmp_path / "rel.key").write_bytes(OTHER_KEY)
    with pytest.raises(SystemExit) as excinfo:
        keys.resolve_key(cfg_for(tmp_path), [], allow_create=False)
    message = str(excinfo.value)
    # Both fingerprints, neither key.
    assert keys.fingerprint(KEY) in message
    assert keys.fingerprint(OTHER_KEY) in message
    assert KEY.decode() not in message and OTHER_KEY.decode() not in message


def test_invalid_env_key_does_not_echo_the_value(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RELEASE_KEY", "definitely-not-a-fernet-key")
    with pytest.raises(SystemExit) as excinfo:
        keys.resolve_key(cfg_for(tmp_path), [], allow_create=False)
    assert "definitely-not-a-fernet-key" not in str(excinfo.value)


def test_generate_writes_0600_and_never_overwrites(tmp_path):
    report = []
    key, source = keys.resolve_key(cfg_for(tmp_path), report, allow_create=True)
    path = tmp_path / "rel.key"
    assert source.startswith("new:")
    assert path.stat().st_mode & 0o777 == 0o600
    # A second resolution must find the same key, not mint another one.
    again, source2 = keys.resolve_key(cfg_for(tmp_path), [], allow_create=True)
    assert again == key and source2.startswith("file:")


def test_refuses_to_create_when_not_allowed(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        keys.resolve_key(cfg_for(tmp_path), [], allow_create=False)
    assert "TEST_RELEASE_KEY" in str(excinfo.value)


def test_no_branch_ever_reports_the_key(tmp_path, monkeypatch):
    """The umbrella guarantee: run every successful branch, scan every line."""
    lines = []
    monkeypatch.setenv("TEST_RELEASE_KEY", KEY.decode())
    keys.resolve_key(cfg_for(tmp_path), lines, allow_create=False)
    monkeypatch.delenv("TEST_RELEASE_KEY")
    keys.resolve_key(cfg_for(tmp_path), lines, allow_create=True)   # generates
    keys.resolve_key(cfg_for(tmp_path), lines, allow_create=False)  # reads back
    lines.append(keys.describe_key(cfg_for(tmp_path), lines))
    blob = "\n".join(lines)
    generated = (tmp_path / "rel.key").read_bytes().strip()
    assert KEY.decode() not in blob and generated.decode() not in blob


def test_world_readable_key_file_warns(tmp_path):
    path = tmp_path / "rel.key"
    path.write_bytes(KEY)
    path.chmod(0o644)
    report = []
    keys.resolve_key(cfg_for(tmp_path), report, allow_create=False)
    assert any("group/world-accessible" in line for line in report)


# --- publish.publish — the column contract and the round trip ---

def test_missing_release_column_aborts():
    df = tiny_pooled().drop(columns=["n_scales"])
    with pytest.raises(SystemExit) as excinfo:
        step.reduce_columns(df, [])
    assert "n_scales" in str(excinfo.value)


def test_extra_columns_dropped_and_reported():
    df = tiny_pooled()
    df["pdf_full_text"] = "copyrighted body text"
    report = []
    out = step.reduce_columns(df, report)
    assert "pdf_full_text" not in out.columns
    assert any("pdf_full_text" in line for line in report)


def test_frame_without_embeddings_is_not_a_pooled_parquet():
    df = tiny_pooled().drop(columns=["item_pooled_m", "scale_pooled_m",
                                     "scale_embedding_m",
                                     "instrument_embedding_m"])
    with pytest.raises(SystemExit) as excinfo:
        step.reduce_columns(df, [])
    assert "embedding" in str(excinfo.value)


def test_release_columns_come_out_in_contract_order():
    out = step.reduce_columns(tiny_pooled(), [])
    assert list(out.columns)[:len(step.RELEASE_COLS)] == \
        list(step.RELEASE_COLS)
    assert list(out.columns)[len(step.RELEASE_COLS):] == [
        "item_pooled_m", "keying_disagreements_m", "scale_pooled_m",
        "scale_embedding_m", "instrument_embedding_m"]


def test_encrypt_frame_round_trips():
    df = step.reduce_columns(tiny_pooled(), [])
    raw = Fernet(KEY).decrypt(step.encrypt_frame(df, KEY))
    back = pd.read_parquet(io.BytesIO(raw))
    assert back.shape == df.shape
    assert list(back.columns) == list(df.columns)


def test_ciphertext_differs_between_runs():
    """Fernet re-IVs every call — this is why idempotency compares plaintext."""
    df = step.reduce_columns(tiny_pooled(), [])
    assert step.encrypt_frame(df, KEY) != step.encrypt_frame(df, KEY)


def test_run_writes_artifact_and_manifest(tmp_path):
    cfg = cfg_for(tmp_path)
    tiny_pooled().to_parquet(cfg["data"]["assemble"]["pooled"], index=False)
    (tmp_path / "rel.key").write_bytes(KEY)
    report = step.run(cfg)
    artifact = Path(cfg["data"]["publish"])
    assert artifact.exists()
    payload = manifest.read_manifest(artifact)
    assert payload["frame"]["rows"] == 6
    assert payload["frame"]["columns"] == 23
    assert payload["artifact"]["key_fingerprint"] == \
        f"sha256:{keys.fingerprint(KEY)}"
    assert not manifest.manifest_stale(artifact)
    assert any("round-trip verified" in line for line in report)


def test_run_writes_the_release_directory(tmp_path):
    """Artifact, manifest, key, descriptives and README side by side."""
    cfg = cfg_for(tmp_path)
    tiny_pooled().to_parquet(cfg["data"]["assemble"]["pooled"], index=False)
    step.run(cfg)  # no key yet: generates rel.key beside the artifact
    artifact = Path(cfg["data"]["publish"])
    key = paths.key_path(artifact)
    assert key == tmp_path / "rel.key" and key.exists()
    assert paths.report_path(artifact).read_text().startswith(
        "# Assembly descriptives")
    readme = paths.readme_path(artifact).read_text()
    generated = manifest.read_manifest(artifact)["generated"]
    assert generated in readme
    for name in ("rel.parquet", "rel.parquet.manifest.json", "rel.key",
                 "rel.md"):
        assert f"`{name}`" in readme
    assert key.read_bytes().strip().decode() not in readme


def test_run_mirrors_only_the_descriptives_to_reports_dir(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg["data"]["reports_dir"] = str(tmp_path / "reports")
    tiny_pooled().to_parquet(cfg["data"]["assemble"]["pooled"], index=False)
    step.run(cfg, report_only=True)
    assert not (tmp_path / "reports").exists()
    step.run(cfg)
    md = paths.report_path(cfg["data"]["publish"])
    assert [p.name for p in (tmp_path / "reports").iterdir()] == [md.name]
    assert (tmp_path / "reports" / md.name).read_bytes() == md.read_bytes()


def test_read_release_projects_columns(tmp_path):
    cfg = cfg_for(tmp_path)
    tiny_pooled().to_parquet(cfg["data"]["assemble"]["pooled"], index=False)
    (tmp_path / "rel.key").write_bytes(KEY)
    step.run(cfg)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    out = step.read_release(config=str(config_path), columns=["path"])
    assert list(out.columns) == ["path"] and len(out) == 6


def test_report_only_writes_nothing_and_generates_no_key(tmp_path):
    cfg = cfg_for(tmp_path)
    tiny_pooled().to_parquet(cfg["data"]["assemble"]["pooled"], index=False)
    report = step.run(cfg, report_only=True)
    assert not Path(cfg["data"]["publish"]).exists()
    assert not (tmp_path / "rel.key").exists()
    assert any("would generate one" in line for line in report)


# --- publish.manifest — what it must and must not contain ---

def test_manifest_path_does_not_eat_a_double_suffix():
    assert manifest.manifest_path("a.enc.bin") == Path("a.enc.bin.manifest.json")
    assert manifest.manifest_path("a.enc.parquet") == \
        Path("a.enc.parquet.manifest.json")


def test_manifest_carries_no_row_values_or_local_paths(tmp_path):
    df = step.reduce_columns(tiny_pooled(), [])
    payload = manifest.build_manifest(
        df, cfg_for(tmp_path), "/home/research/data/rel.enc.bin", KEY,
        token_bytes=10, token_sha256="a" * 64,
        plaintext_bytes=5, plaintext_sha256="b" * 64,
        source_artifact="/home/research/data/pooled.parquet")
    blob = json.dumps(payload)
    for sentinel in ("SECRET_TITLE_XYZ", "SECRET_SCALE_XYZ", "SECRET_VERSION_XYZ",
                     "/abs/secret", "SECRETDOI", "tSECRET"):
        assert sentinel not in blob, sentinel
    assert "/home/research" not in blob and "/home/models" not in blob
    assert KEY.decode() not in blob
    assert payload["artifact"]["name"] == "rel.enc.bin"


def test_manifest_describes_the_frame(tmp_path):
    df = step.reduce_columns(tiny_pooled(), [])
    payload = manifest.build_manifest(
        df, cfg_for(tmp_path), tmp_path / "rel.enc.bin", KEY,
        token_bytes=10, token_sha256="a" * 64,
        plaintext_bytes=5, plaintext_sha256="b" * 64)
    frame = payload["frame"]
    assert frame["rows"] == 6
    assert frame["instrument_rows"] + frame["scale_rows"] == 6
    assert frame["embedding_dims"]["item_pooled_m"] == 4
    assert frame["corpus_source_counts"]["apa-psyctests"] == 3
    assert frame["public_year"]["min"] == 2000
    assert frame["flag_true_counts"]["flag_item_count_deviation"] == 3
    assert [m["name"] for m in payload["models"]] == ["m", "m"]


def test_sha256_file_streams_across_chunks(tmp_path):
    import hashlib
    path = tmp_path / "blob"
    path.write_bytes(b"x" * 1000)
    assert manifest.sha256_file(path, chunk=64) == \
        hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_stale_when_artifact_is_newer(tmp_path):
    artifact = tmp_path / "rel.enc.bin"
    artifact.write_bytes(b"x")
    assert manifest.manifest_stale(artifact)          # no sidecar yet
    manifest.write_manifest(artifact, {"a": 1})
    assert not manifest.manifest_stale(artifact)
    import os
    os.utime(artifact, (10 ** 10, 10 ** 10))          # artifact touched after
    assert manifest.manifest_stale(artifact)


# --- The hard-coded dataset card ---

def card_parts() -> tuple[dict, str]:
    text = hub.CARD_FILE.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    return yaml.safe_load(front), body


def test_card_frontmatter_disables_the_viewer():
    front, _ = card_parts()
    # The payload is a Fernet token; the viewer and `configs`/`dataset_info` would try to load it.
    assert front["viewer"] is False
    assert "configs" not in front and "dataset_info" not in front


def test_card_language_tags_are_iso639_1():
    front, _ = card_parts()
    for tag in front["language"]:
        assert re.fullmatch(r"[a-z]{2}", str(tag)), tag


def test_card_documents_every_released_column():
    """Anti-drift: the hand-written card must name every column in the release contract."""
    _, body = card_parts()
    missing = [c for c in step.RELEASE_COLS if c not in body]
    missing += [p for p in step.MODEL_COL_PREFIXES if p not in body]
    assert not missing, f"undocumented in the dataset card: {missing}"


def test_card_leaks_nothing_local():
    text = hub.CARD_FILE.read_text(encoding="utf-8")
    assert "/home/research" not in text and "/home/models" not in text
    assert ".key\"" not in text or "synthometricon-release.key" in text


# --- publish.hub — refusals, against a stub ---

class StubInfo:
    def __init__(self, private=True, sha="parent000000"):
        self.private, self.sha = private, sha


class StubCommit:
    oid = "commit000000"


class StubApi:
    """Records calls for assertions on repo_type and ordering."""

    token = "stub-token"

    def __init__(self, private=True):
        self.calls: list[tuple] = []
        self._private = private
        self.ops = None

    def whoami(self):
        self.calls.append(("whoami", {}))
        return {"name": "someone", "orgs": [{"name": "org"}]}

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", kw))

    def repo_info(self, repo_id, **kw):
        self.calls.append(("repo_info", kw))
        return StubInfo(private=self._private)

    def preupload_lfs_files(self, repo_id, ops, **kw):
        self.calls.append(("preupload_lfs_files", kw))
        ops[0]._upload_mode = "lfs"

    def create_commit(self, repo_id, ops, **kw):
        self.calls.append(("create_commit", kw))
        self.ops = ops
        return StubCommit()


def build_release(tmp_path, monkeypatch, **overrides) -> dict:
    """Build a tiny artifact + manifest; git provenance is stubbed clean (upload refuses dirty trees)."""
    monkeypatch.setattr(manifest, "git_provenance",
                        lambda repo_root=None: {"commit": "abc1234",
                                                "dirty": False})
    cfg = cfg_for(tmp_path, **overrides)
    tiny_pooled().to_parquet(cfg["data"]["assemble"]["pooled"], index=False)
    (tmp_path / "rel.key").write_bytes(KEY)
    step.run(cfg)
    monkeypatch.setattr(hub, "resolve_token", lambda cfg: "stub-token")
    return cfg


def prepared(tmp_path, monkeypatch, *, remote_sha=None, **overrides):
    """A built release plus a stubbed remote that holds ``remote_sha``."""
    cfg = build_release(tmp_path, monkeypatch, **overrides)
    monkeypatch.setattr(hub, "remote_plaintext_sha",
                        lambda api, repo_id, report: remote_sha)
    return cfg


def test_report_only_makes_no_calls(tmp_path, monkeypatch):
    cfg = prepared(tmp_path, monkeypatch)
    api = StubApi()
    report = hub.run(cfg, report_only=True, api=api)
    assert api.calls == []
    assert any("no network calls" in line for line in report)
    assert any("data/rel.parquet" in line for line in report)


def test_public_repo_aborts_before_any_commit(tmp_path, monkeypatch):
    cfg = prepared(tmp_path, monkeypatch)
    api = StubApi(private=False)
    with pytest.raises(SystemExit) as excinfo:
        hub.run(cfg, yes=True, api=api)
    assert "PUBLIC" in str(excinfo.value)
    assert not any(call == "create_commit" for call, _ in api.calls)


def test_every_call_targets_a_dataset_repo(tmp_path, monkeypatch):
    cfg = prepared(tmp_path, monkeypatch)
    api = StubApi()
    hub.run(cfg, yes=True, api=api)
    typed = [kw for call, kw in api.calls if call != "whoami"]
    assert typed and all(kw.get("repo_type") == "dataset" for kw in typed)


def test_single_commit_with_exactly_three_files(tmp_path, monkeypatch):
    cfg = prepared(tmp_path, monkeypatch)
    api = StubApi()
    hub.run(cfg, yes=True, api=api)
    assert sum(call == "create_commit" for call, _ in api.calls) == 1
    assert {op.path_in_repo for op in api.ops} == {
        "data/rel.parquet", "README.md", "manifest.json"}


def test_blob_operation_holds_a_path_not_bytes(tmp_path, monkeypatch):
    """Bytes would stay resident for the whole upload and break retries."""
    cfg = prepared(tmp_path, monkeypatch)
    api = StubApi()
    hub.run(cfg, yes=True, api=api)
    blob = next(op for op in api.ops if op.path_in_repo == "data/rel.parquet")
    assert isinstance(blob.path_or_fileobj, (str, Path))


def test_matching_remote_is_skipped_unless_forced(tmp_path, monkeypatch):
    cfg = build_release(tmp_path, monkeypatch)
    sha = manifest.read_manifest(
        cfg["data"]["publish"])["artifact"]["plaintext_sha256"]
    monkeypatch.setattr(hub, "remote_plaintext_sha",
                        lambda api, repo_id, report: sha)

    api = StubApi()
    report = hub.run(cfg, yes=True, api=api)
    assert not any(call == "create_commit" for call, _ in api.calls)
    assert any("nothing to" in line for line in report)

    forced = StubApi()
    hub.run(cfg, yes=True, force=True, api=forced)
    assert any(call == "create_commit" for call, _ in forced.calls)


def test_wrong_key_for_this_artifact_aborts(tmp_path, monkeypatch):
    cfg = prepared(tmp_path, monkeypatch)
    (tmp_path / "rel.key").unlink()
    monkeypatch.setenv("TEST_RELEASE_KEY", OTHER_KEY.decode())
    api = StubApi()
    with pytest.raises(SystemExit) as excinfo:
        hub.run(cfg, yes=True, api=api)
    assert "nobody can decrypt" in str(excinfo.value)


def test_stale_artifact_aborts(tmp_path, monkeypatch):
    import os
    cfg = prepared(tmp_path, monkeypatch)
    os.utime(cfg["data"]["assemble"]["pooled"], (10 ** 10, 10 ** 10))
    with pytest.raises(SystemExit) as excinfo:
        hub.run(cfg, yes=True, api=StubApi())
    assert "newer than the release artifact" in str(excinfo.value)


def test_key_leak_guard_rejects_a_key_file_and_key_bearing_text():
    from huggingface_hub import CommitOperationAdd
    key_op = CommitOperationAdd(path_in_repo="secrets/rel.key",
                                path_or_fileobj=b"x")
    with pytest.raises(ValueError, match="key file"):
        hub.assert_no_key_leak([key_op], KEY, "rel.key")
    card_op = CommitOperationAdd(path_in_repo="README.md",
                                 path_or_fileobj=b"the key is " + KEY)
    with pytest.raises(ValueError, match="appears in"):
        hub.assert_no_key_leak([card_op], KEY, None)


def test_non_tty_upload_refuses_without_yes(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        hub.confirm(["plan"], yes=False)
    assert "--yes" in str(excinfo.value)


def test_token_is_scrubbed_from_error_text():
    assert hub.scrub("failed: hf_secrettoken bad", "hf_secrettoken") == \
        "failed: <token> bad"


# --- publish.__main__ ---

def test_unknown_step_lists_the_valid_ones(monkeypatch, capsys):
    from publish.__main__ import main
    monkeypatch.setattr(sys, "argv", ["publish", "--step", "nope"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert "publish" in str(excinfo.value) and "upload" in str(excinfo.value)


def test_upload_only_flags_rejected_on_a_publish_run(monkeypatch):
    from publish.__main__ import main
    monkeypatch.setattr(sys, "argv",
                        ["publish", "--step", "publish", "--yes"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert "upload step" in str(excinfo.value)


def test_dirty_build_tree_blocks_upload(tmp_path, monkeypatch):
    """A release whose provenance commit does not reproduce needs --force."""
    cfg = prepared(tmp_path, monkeypatch)
    artifact = cfg["data"]["publish"]
    payload = manifest.read_manifest(artifact)
    payload["pipeline"]["dirty"] = True
    manifest.write_manifest(artifact, payload)
    with pytest.raises(SystemExit, match="does not reproduce"):
        hub.run(cfg, yes=True, api=StubApi())
    forced = StubApi()
    hub.run(cfg, yes=True, force=True, api=forced)
    assert any(call == "create_commit" for call, _ in forced.calls)
