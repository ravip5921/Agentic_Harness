"""Tests for the QA stage."""

from harness.models import QAFeedback, QAResult, QAVerdict
from harness.stages.qa import qa


def test_qa_returns_valid_result():
    result = qa({})

    assert "qa_result" in result
    assert isinstance(result["qa_result"], QAResult)


def test_qa_is_blocked_until_implemented():
    result = qa({})["qa_result"]

    assert result.verdict == QAVerdict.BLOCKED
    assert result.report_path is None
    assert result.feedback == QAFeedback(
        summary="Test execution is not yet available.",
        development_attempt=0,
    )
