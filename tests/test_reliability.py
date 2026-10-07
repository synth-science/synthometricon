"""Run each extractor ``num_runs`` times per test PDF and assert reliability thresholds.

Needs a running llama-server; see docs/testing.md for fixture and skip semantics.
"""

from datetime import datetime
from pathlib import Path

import pytest

from extraction.extractors import REGISTRY
from extraction.orchestrator import build_dependency_context, run_extractor
from extraction.config import load_config, load_test_cases, resolve_fixtures_dir

ROOT = Path(__file__).parent.parent
TEST_FILES_YAML = Path(__file__).parent / "test_files.yaml"
# Same fixtures as the CLI (config.yaml testing.fixtures_dir: the human coding), so the
# dependency context of instrument/meta runs is the human-validated item and scale set.
FIXTURES_DIR = resolve_fixtures_dir(load_config(str(ROOT / "config.yaml")), ROOT)
_TEST_CASES = load_test_cases(str(TEST_FILES_YAML), fixtures_dir=FIXTURES_DIR)

_params = [
    pytest.param(name, Path(case["path"]).stem, id=f"{name}-{Path(case['path']).stem}")
    for name in REGISTRY
    for case in _TEST_CASES
]


@pytest.mark.parametrize("extractor_name,case_stem", _params)
def test_reliability(extractor_name, case_stem, config, test_cases, fixtures_dir):
    case = next(c for c in test_cases if Path(c["path"]).stem == case_stem)
    extractor = REGISTRY[extractor_name]
    testing_cfg = config.get("testing", {})
    num_runs = testing_cfg.get("num_runs", 2)
    thresholds: dict = (
        testing_cfg.get("reliability", {}).get("thresholds", {}).get(extractor_name, {})
    )

    contexts = None
    if extractor.depends_on:
        try:
            ctx = build_dependency_context(
                pdf_path=case["path"],
                deps=extractor.depends_on,
                registry=REGISTRY,
                config=config,
                case=case,
                fixtures_dir=fixtures_dir,
            )
        except Exception as exc:
            pytest.fail(f"dependency context build failed: {exc}")
        contexts = {case["path"]: ctx}

    result = run_extractor(
        extractor,
        config,
        run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
        pdf_paths=[case["path"]],
        num_runs=num_runs,
        no_logs=False,
        reliability_thresholds=thresholds or None,
        contexts=contexts,
    )

    statuses = [r["status"] for r in result["doc_results"]]

    if all(s == "SKIPPED" for s in statuses):
        pytest.skip(f"num_runs={num_runs} < 2 — reliability not assessed")

    failed = [r for r in result["doc_results"] if r["status"] == "FAIL"]
    if failed:
        details = "\n".join(
            f"{Path(r['path']).name}\n  " + "\n  ".join(r["status_lines"])
            for r in failed
        )
        pytest.fail(f"Reliability thresholds not met:\n{details}")
