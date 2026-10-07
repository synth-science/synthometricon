"""Offline tests for evaluation-harness helpers: fixtures directory, input discovery, items suite ICC."""

from pathlib import Path

from extraction.config import (
    DEFAULT_FIXTURES_DIR,
    expand_input_files,
    load_config,
    resolve_fixtures_dir,
)
from extraction.extractors import REGISTRY

_ROOT = Path(__file__).parent.parent


def test_fixtures_dir_reads_config_and_defaults_to_human_fixtures(tmp_path):
    assert resolve_fixtures_dir({}) == DEFAULT_FIXTURES_DIR == "tests/validation-fixtures"
    assert resolve_fixtures_dir({"testing": {"fixtures_dir": "x/y"}}) == "x/y"
    assert resolve_fixtures_dir({"testing": {"fixtures_dir": "x/y"}}, tmp_path) == str(tmp_path / "x/y")
    assert resolve_fixtures_dir({"testing": {"fixtures_dir": "/abs"}}, tmp_path) == "/abs"


def test_pytest_route_uses_the_cli_fixtures():
    """The e2e tests must read the human coding that config.yaml points the CLI at."""
    cfg = load_config(str(_ROOT / "config.yaml"))
    fixtures = Path(resolve_fixtures_dir(cfg, _ROOT))
    assert fixtures.name == "validation-fixtures"
    assert fixtures.is_dir()


def test_expand_input_files_skips_os_copies(tmp_path, capsys):
    for name in ["a.pdf", "a (2).pdf", "b (2).pdf", "C.PDF", "C (1).pdf", "d (copy).pdf"]:
        (tmp_path / name).write_bytes(b"%PDF-1.4")
    names = sorted(Path(p).name for p in expand_input_files([str(tmp_path / "*.[pP][dD][fF]")]))
    # copies of a listed original go; a "(N)" file without an original stays
    assert names == ["C.PDF", "a.pdf", "b (2).pdf", "d (copy).pdf"]
    assert "skipped 2 OS duplicate copies" in capsys.readouterr().out


def test_items_suite_icc_is_reported():
    """ICC(2,1) on item counts is computed (pingouin renamed CI95% -> CI95 in 0.7)."""
    report = REGISTRY["items"].suite_reliability([
        [["a"] * 3, ["a"] * 3, ["a"] * 4],
        [["a"] * 10, ["a"] * 9, ["a"] * 10],
        [["a"] * 5, ["a"] * 5, ["a"] * 5],
    ])
    assert "ICC(2,1) on item counts:" in report and "95% CI" in report, report
