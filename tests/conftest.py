from pathlib import Path

import pytest

from extraction.config import load_config, load_test_cases, resolve_fixtures_dir

_ROOT = Path(__file__).parent.parent
_TEST_FILES_YAML = Path(__file__).parent / "test_files.yaml"


@pytest.fixture(scope="session")
def config():
    return load_config(str(_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def fixtures_dir(config):
    """``testing.fixtures_dir`` from config.yaml — the same human-coded fixtures the CLI uses."""
    return resolve_fixtures_dir(config, _ROOT)


@pytest.fixture(scope="session")
def test_cases(fixtures_dir):
    """Session-shared; mutated in place when fixtures are auto-built."""
    return load_test_cases(str(_TEST_FILES_YAML), fixtures_dir=fixtures_dir)
