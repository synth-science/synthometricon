# publish/ — publication pipeline

Turns the pooled corpus into the encrypted release and pushes it to the private HuggingFace dataset repo (`publish.repo_id`), in two steps: **publish → upload**. Entry point `python -m publish`; each step also runs standalone (`python -m publish.publish`, `python -m publish.hub`). Deliberately separate from `python -m assemble` (which ends at `pooled`): building is local and repeatable, publishing is outward-facing and permanent — a Hub commit's LFS blobs survive in history even after the file is removed. Reports go to `logs/publish-<ts>.log` (real runs only).

`STEPS` / `PIPELINE` in `publish/__main__.py` mirror assemble's `STAGES` / `PIPELINE`; each step is `run(cfg, *, report_only, ...) -> list[str]`.

```bash
poetry run python -m publish                      # publish, then upload
poetry run python -m publish --step publish       # build the release directory only
poetry run python -m publish --step upload --yes  # push without the prompt
poetry run python -m publish --report-only        # dry run: no writes, no network
poetry run python -m publish --report-only --check-remote   # + live repo state
```

## Release directory (`publish/paths.py`)

`data.publish` names the artifact; everything else is derived from it and sits beside it. No other path is configured.

| file | contents |
|---|---|
| `<stem>.parquet` | release frame: parquet bytes, Fernet-encrypted (not readable as parquet) |
| `<stem>.parquet.manifest.json` | manifest sidecar: shape, dtypes, counts, hashes, pipeline commit, key fingerprint |
| `<stem>.key` | the Fernet key (only generated if none exists) — private |
| `<stem>.md` | descriptives report (`assemble.report`), built in the same run so it cannot drift |
| `README.md` | build time, pipeline commit, shape, one line per file (fingerprint only) |

## Publish step (`publish/publish.py`)

Reads `data.assemble.pooled` (already at release granularity: scale rows at every hierarchy level plus one instrument row per document; `n_items` / `n_scales` come from pool).

- **reduce** — keep the scalar `RELEASE_COLS` (all required) plus every column matching `MODEL_COL_PREFIXES`, in this order: `item_pooled_*`, `keying_disagreements_*`, `scale_pooled_*`, `scale_embedding_*`, `instrument_embedding_*`. At least one embedding column must exist, else the input is not a pooled parquet. Anything else is dropped and reported, so internal columns never leak.
- **mutate** — home for release-only derived variables; currently empty.
- **encrypt** — serialize to parquet in memory, Fernet-encrypt, write atomically (tmp + `os.replace`, after a `shutil.disk_usage` pre-check).
- **round-trip verify** — decrypt the written file and check row count, column count and plaintext SHA-256. The one guard against shipping an undecryptable blob.

Output is a Fernet token (base64, ~33% larger than the parquet). Peak memory 8–10 GB (frame, ~750 MB parquet buffer, ~1 GB token, then the token read back and decrypted).

**Reading a release.** In-repo: `publish.publish.read_release(path=None, key_path=None, config="config.yaml", columns=None)` — defaults from config, never generates a key; `columns` projects at the parquet read (embeddings are nearly all the bytes). Without this package:

```python
from cryptography.fernet import Fernet
import io, pandas as pd
raw = Fernet(key).decrypt(Path(enc_path).read_bytes())
df = pd.read_parquet(io.BytesIO(raw))
```

## Key (`publish/keys.py`)

The only module that touches the key. One stable **random** key (`Fernet.generate_key`) encrypts every release; it cannot be re-derived, so losing it locks out every consumer and rotating it silently breaks them.

Resolution (`resolve_key`):
1. `$publish.key_env` (default `SYNTHOMETRICON_RELEASE_KEY`);
2. the key file: `--key-path`, else `<stem>.key` (outside the repo; `*.key` gitignored as backstop). If both env and file exist and differ, abort showing both fingerprints rather than guess;
3. generate (`O_EXCL`, mode 0600) **only when `allow_create`** — true only for a real `--step publish`. Dry runs (`describe_key`), `hub.run` and `read_release` pass False, so they can never mint a second key.

`validate_key` constructs `Fernet(key)` so a truncated key fails at resolution, not at decrypt. A group/world-readable key file is warned about.

**Never print, log, commit or upload the key.** Only `fingerprint(key)` = `sha256(key).hexdigest()[:12]` may reach a report, log, manifest or card; `test_no_branch_ever_reports_the_key` enforces this.

## Manifest (`publish/manifest.py`)

`<artifact>.manifest.json` — built by publish, read and shipped by upload, so upload never loads pandas or decrypts. Holds schema version, build time, git commit + dirty flag, artifact basename/bytes/sha256, `plaintext_bytes` / `plaintext_sha256`, key fingerprint, a frame description (rows/columns, instrument/scale split, dtypes, embedding dims, null counts, `corpus_source` and language counts, year range, `flag_*` true counts) and encode model names.

- **No row values, local paths (basename only; model `path` omitted) or key.** `test_manifest_carries_no_row_values_or_local_paths` enforces it.
- `artifact.sha256` is upload's integrity check; `artifact.plaintext_sha256` is the idempotency key.
- The sidecar path appends `.manifest.json` rather than using `Path.with_suffix`, which would eat the `.bin` of a double-suffixed name.
- `sha256_file` streams 8 MiB chunks — never `read_bytes()` a gigabyte to hash it.

## Dataset card (`publish/dataset_card.md`)

Hard-coded prose committed as the Hub repo's `README.md` (which *is* the card). Not templated: per-release numbers live in the manifest, which the card points at. Front matter sets `viewer: false` and omits `configs` / `dataset_info` (either would make the Hub try to load the Fernet token). **Every released column must be documented in it** — `test_card_documents_every_released_column` checks every `RELEASE_COLS` name and `MODEL_COL_PREFIXES` entry; there is no generation step.

## Upload step (`publish/hub.py`)

One atomic `create_commit` of three files: the blob (at `publish.path_in_repo`), the card as `README.md`, and `manifest.json`; `parent_commit` acts as an optimistic lock; optional `publish.tag`. Mostly refusals, in order:

1. manifest present (a dry run warns instead);
2. artifact present, not older than its manifest, not older than the pooled input, and hashing to the manifest's `sha256` (`--skip-verify` skips the hash);
3. build tree not dirty (`--force` overrides);
4. resolved key's fingerprint matches the artifact's (else nobody could decrypt it);
5. token's account can write the namespace (a user-scoped token otherwise fails only at commit, after the upload);
6. repo actually private;
7. remote manifest's `plaintext_sha256` differs (`--force` overrides);
8. no op points at a `*.key` / the key file, and the key bytes are not in the card or manifest (the blob is not scanned — a ciphertext cannot contain its own key);
9. interactive confirmation, or `--yes`; a non-tty without `--yes` exits rather than blocking on `input()`.

The HF token comes from `$publish.token_env` (default `HF_TOKEN`) or the cached CLI login and is scrubbed from error text.

## Things that will bite

- **`Fernet.encrypt` is non-deterministic** (fresh IV per call): an unchanged corpus re-encrypts to different bytes. Idempotency compares `plaintext_sha256`, never ciphertext or size; without it a re-run costs a permanent gigabyte of Hub history.
- **`create_repo(private=True, exist_ok=True)` does not make an existing public repo private.** `verify_repo` reads `repo_info(...).private` back and aborts. `update_repo_settings` is deliberately not called — no automatic visibility changes.
- **`repo_type="dataset"` on every call** — the default is `"model"`. `test_every_call_targets_a_dataset_repo` enforces it.
- **The blob path must end in `.bin`.** The Hub's LFS sniff on a 512-byte sample never fires on ASCII base64; `*.bin` is in HF's default `.gitattributes`. After preupload the step aborts if the mode is `"regular"`.
- **Pass the blob as a `Path`, not `bytes`** — bytes stay resident for the whole upload and a retry finds the buffer freed.
- **Don't use `datasets`.** `push_to_hub` would shard to parquet and enable the viewer. (Declared in `pyproject.toml`, unused.)
- **`--report-only` makes zero network calls**; `--check-remote` opts into reading live repo state.
