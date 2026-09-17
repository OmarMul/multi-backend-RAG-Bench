"""Tests for RAGAS evaluation engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.eval.ragas_eval import RagasEvaluator


def test_eval_single_mock():
    """Test single score calculation with mocked RAGAS evaluate."""
    mock_ragas_scores = {
        "faithfulness": 0.95,
        "answer_relevancy": 0.91,
        "context_precision": 0.88,
        "context_recall": 0.92,
    }

    with patch("src.eval.ragas_eval.evaluate", return_value=mock_ragas_scores):
        evaluator = RagasEvaluator(model_name="openai/gpt-4o-mini")
        scores = evaluator.score_single(
            question="What is Agentic AI?",
            answer="Agentic AI utilizes planning, memory, and tools.",
            contexts=["Agentic AI systems use planning and memory to achieve goals."],
            ground_truth="Agentic AI systems combine planning, memory, and tools.",
        )

        assert isinstance(scores, dict)
        assert 0.0 <= scores["faithfulness"] <= 1.0
        assert 0.0 <= scores["answer_relevancy"] <= 1.0
        assert 0.0 <= scores["context_precision"] <= 1.0
        assert 0.0 <= scores["context_recall"] <= 1.0
        assert scores["faithfulness"] == 0.95


def test_eval_questions_file_exists():
    """Verify ground truth evaluation dataset exists and has valid format."""
    eval_file = Path("data/raw_docs/eval_questions.json")
    assert eval_file.exists()

    import json
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) > 0
    assert "question" in data[0]
    assert "ground_truth" in data[0]
