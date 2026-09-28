# Synthometricon

The Synthometricon ("dictionary of measured synthetics"; formally known as **SynthNet**) is a search engine designed to help researchers, practitioners, and students discover scales from a database of more than 28,000 questionnaires, surveys, and tests.

This repository contains the data extraction pipeline (which extracts structured items, scales, and metadata from PDF test manuals) and the processing pipeline (which combines them with other sources into an analysis-ready corpus), as well as various descriptive analyses. It also contains code to publish the resulting encrypted dataset on Hugging Face.

The web app is currently hosted on Hugging Face Spaces at: [https://huggingface.co/spaces/magnolia-psychometrics/synthometricon](https://huggingface.co/spaces/magnolia-psychometrics/synthometricon)

## What is (not) included

This is a read-only snapshot of the pipeline code. Source PDFs, extraction outputs, test fixtures and evaluation logs are not included because they contain copyright-protected material. The published dataset is encrypted; see [`publish/dataset_card.md`](publish/dataset_card.md) for how to request access.

## Setup

```bash
poetry install --with test
./start_server.sh            # llama-server, needed for extraction
```

`config.yaml` holds the settings used for the published corpus. Local paths point to `data/`; set `input_files` to your PDFs and `patch.crossref.mailto` to your email address before running.

## Pipelines

| Pipeline | Stages | Entry point |
| --- | --- | --- |
| [`extraction/`](extraction/README.md) | items → scales → instrument → meta | `python -m extraction` |
| `assemble/` | explode → combine → patch → postprocess → encode → pool | `python -m assemble` |
| `publish/` | publish → upload (encrypt, push to Hugging Face) | `python -m publish` |

```bash
poetry run python -m extraction                 # extract config input_files
poetry run python -m assemble --report-only     # dry run
poetry run pytest tests/ -v -k "not reliability and not validity"   # offline suite
```

The reliability and validity suites need a running llama-server and your own PDFs listed in `tests/test_files.yaml`, with fixtures under `tests/validation-fixtures/`. See [docs/testing.md](docs/testing.md).

## Documentation

- [commands.md](docs/commands.md) — CLI reference
- [configuration.md](docs/configuration.md) — `config.yaml` keys
- [architecture-extraction.md](docs/architecture-extraction.md), [architecture-assemble.md](docs/architecture-assemble.md), [architecture-publish.md](docs/architecture-publish.md) — how each pipeline works
- [extractors.md](docs/extractors.md), [wire-model-conventions.md](docs/wire-model-conventions.md) — extractor prompts and models
- [reverse-keyed-pooling.md](docs/reverse-keyed-pooling.md) — pooled item vectors and reverse keying
- [extraction-output.md](docs/extraction-output.md) — exploded parquet columns
- [testing.md](docs/testing.md), [reliability-metrics.md](docs/reliability-metrics.md) — test suites and metrics

## License

[MIT](LICENSE)
