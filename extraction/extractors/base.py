"""Extractor ABC: prompts, result model, reliability signal/metrics, run formatting.

Extractors never persist; the orchestrator (storage.py) does.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel

from ..text_utils import extract_json


class Extractor(ABC):
    name: str
    system_prompt: str
    user_prompt: str
    result_model: type[BaseModel]
    depends_on: list[str] = []
    needs_first_page_text: bool = False  # when True, orchestrator injects context["_first_page_text"]

    def llm_schema(self) -> dict:
        """JSON schema for grammar-constrained decoding; override to emit a wire model."""
        return self.result_model.inlined_schema()

    def parse_response(
        self,
        content_text: str,
        context: dict[str, BaseModel] | None = None,
    ) -> BaseModel:
        """Parse the raw LLM response into `result_model`; override to resolve a wire model."""
        return self.result_model.model_validate_json(extract_json(content_text))

    @abstractmethod
    def signal(self, result: BaseModel) -> Any:
        """Reduce a result to the unit compared across runs for reliability."""

    @abstractmethod
    def reliability(self, signals: list) -> tuple[str, dict[str, float]]:
        """Per-document cross-run reliability; returns (report, {metric: score})."""

    def suite_reliability(self, all_signals: list[list]) -> str:
        """Suite-level reliability report; ``all_signals[doc_idx][run_idx]``."""
        return ""

    @abstractmethod
    def format_run(
        self,
        result: BaseModel,
        run_idx: int,
        num_runs: int,
        run_elapsed: float,
        log: Callable[[str], None],
    ) -> None:
        """Print a per-run summary and append it to the log."""

    def build_user_prompt(self, context: dict[str, BaseModel] | None = None) -> str:
        """Render the user prompt; dependent extractors inject ``context`` results."""
        return self.user_prompt
