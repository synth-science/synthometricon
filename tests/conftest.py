from pathlib import Path

import pytest

from extraction.config import load_config, load_test_cases

_ROOT = Path(__file__).parent.parent
_TEST_FILES_YAML = Path(__file__).parent / "test_files.yaml"


@pytest.fixture(scope="session")
def config():
    return load_config(str(_ROOT / "config.yaml"))


@pytest.fixture(scope="session")
def test_cases():
    """Session-shared; mutated in place when fixtures are auto-built."""
    return load_test_cases(str(_TEST_FILES_YAML))
