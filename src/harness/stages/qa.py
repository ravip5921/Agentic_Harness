
"""QA stage of the generic agentic harness.

Responsibilities:
    - Validate generated implementation artifacts.
    - Execute applicable tests in an isolated environment.
    - Record executed commands and test results.
    - Produce a QAResult with a verifiable verdict.

"""

from harness.models import QAResult, QAVerdict
from harness.state import HarnessState


def qa(state: HarnessState) -> dict:
    """Execute the QA stage."""

    result = QAResult(
        verdict=QAVerdict.BLOCKED,
        summary="QA stage is not yet implemented.",
        report_path=None,
        feedback=["Test execution is not yet available."],
    )

    return {"qa_result": result}
