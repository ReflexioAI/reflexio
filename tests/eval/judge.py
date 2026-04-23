"""LLM-as-judge scorer for golden-set evaluation.

Takes a rubric (prompt template + judge model) and an (expected, actual)
pair, renders the prompt, and parses the judge response into a
``JudgeScore``. Used by the comparison harness in Task 5.7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient


class JudgeScore(BaseModel):
    """Judge's per-case numerical verdict.

    Args:
        signal_f1 (float): Extraction signal recall vs expected signals, in [0, 1].
            Always 0 for search-rubric scores.
        answer_correctness (float): Search top-rank correctness, in [0, 1].
            Always 0 for extraction-rubric scores.
        grounded_rate (float): Fraction of emitted items that are grounded in
            the source (no hallucinated IDs or source_spans), in [0, 1].
        rationale (str): One-paragraph explanation of the scores.
    """

    signal_f1: float
    answer_correctness: float
    grounded_rate: float
    rationale: str


class LLMJudge:
    """Wraps a ``LiteLLMClient`` + rubric and produces ``JudgeScore`` results.

    The rubric dict has two required keys: ``prompt`` (a template with
    ``{expected}`` / ``{actual}`` substitution placeholders) and
    ``judge_model`` (model name override).

    Args:
        client: Any client exposing ``generate_chat_response(messages,
            response_format, ...)`` — in practice a ``LiteLLMClient`` or a
            ``MagicMock`` in unit tests.
        rubric (dict): Parsed rubric YAML.
    """

    def __init__(self, *, client: LiteLLMClient | Any, rubric: dict[str, Any]) -> None:
        self.client = client
        self.rubric = rubric

    def score(self, *, expected: Any, actual: Any) -> JudgeScore:
        """Render the rubric prompt and return the parsed judge score.

        Raises:
            TypeError: When the client returns a plain string instead of a
                structured ``JudgeScore`` (misconfigured response_format).
        """
        prompt = (
            self.rubric["prompt"]
            .replace("{expected}", str(expected))
            .replace("{actual}", str(actual))
        )
        result = self.client.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            response_format=JudgeScore,
            model_name_override=self.rubric.get("judge_model"),
        )
        if isinstance(result, JudgeScore):
            return result
        if isinstance(result, BaseModel):
            return JudgeScore.model_validate(result.model_dump())
        raise TypeError(f"LLMJudge expected JudgeScore, got {type(result).__name__}")
