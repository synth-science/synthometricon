"""End-to-end validity: full pipeline per PDF vs. its curated ``extraction_output`` fixture.

Needs a running llama-server; the comparator itself is covered offline by ``test_comparator.py``.
"""

from datetime import datetime
from pathlib import Path

import pytest

from extraction.config import load_config, load_test_cases, resolve_fixtures_dir
from extraction.extractors import REGISTRY
from extraction.orchestrator import run_validity


ROOT = Path(__file__).parent.parent
TEST_FILES_YAML = Path(__file__).parent / "test_files.yaml"
# Same fixtures as the CLI (config.yaml testing.fixtures_dir: the human coding).
FIXTURES_DIR = resolve_fixtures_dir(load_config(str(ROOT / "config.yaml")), ROOT)
_TEST_CASES = load_test_cases(str(TEST_FILES_YAML), fixtures_dir=FIXTURES_DIR)

_VALIDITY_PARAMS = [
    pytest.param(Path(case["path"]).stem, id=Path(case["path"]).stem)
    for case in _TEST_CASES
    if (case.get("fixtures") or {}).get("extraction_output") is not None
]


@pytest.mark.parametrize("stem", _VALIDITY_PARAMS)
def test_validity_e2e(stem, config, test_cases):
    case = next(c for c in test_cases if Path(c["path"]).stem == stem)
    pdf_path = case["path"]
    result = run_validity(
        pdf_paths=[pdf_path],
        all_extractors=list(REGISTRY.values()),
        config=config,
        test_cases=[case],
        no_logs=False,
        run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    statuses = [r["status"] for r in result["doc_results"]]
    if statuses == ["SKIPPED"]:
        pytest.skip(f"no extraction_output fixture for {stem}")
    failed = [r for r in result["doc_results"] if r["status"] == "FAIL"]
    if failed:
        details = "\n".join(
            f"  {Path(r['path']).name}: {len(r['issues'])} issue(s):\n    "
            + "\n    ".join(r["issues"][:5])
            for r in failed
        )
        pytest.fail(f"Validity check failed:\n{details}")
