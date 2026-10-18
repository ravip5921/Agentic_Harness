from harness.graph import build_graph
from harness.models import (
    TaskRequest, ResearchResult, DevelopmentResult, QAResult, ReleaseResult,
    StageOutcome, QAVerdict, ReleaseStatus, RunStatus,
)


def test_successful_run_stages_only_when_approved():
    calls = []
    def research(_):
        calls.append("research")
        return {"research_result": ResearchResult(outcome=StageOutcome.COMPLETED)}
    def developer(_):
        calls.append("developer")
        return {"development_result": DevelopmentResult(outcome=StageOutcome.COMPLETED)}
    def qa(_):
        calls.append("qa")
        return {"qa_result": QAResult(verdict=QAVerdict.PASS)}
    def release(_):
        calls.append("release")
        return {"release_result": ReleaseResult(status=ReleaseStatus.STAGED)}
    graph = build_graph(research=research, developer=developer, qa=qa, release=release)
    result = graph.invoke({"task": TaskRequest(task="Generic demo"), "release_approved": True})
    assert result["status"] == RunStatus.STAGED
    assert calls == ["research", "developer", "qa", "release"]
    calls.clear()
    result = graph.invoke({"task": TaskRequest(task="Generic demo")})
    assert result["status"] == RunStatus.STAGED
    assert calls == ["research", "developer", "qa", "release"]


def test_retry_limit_never_releases():
    calls = []
    def research(_):
        return {"research_result": ResearchResult(outcome=StageOutcome.COMPLETED)}
    def developer(_):
        calls.append("developer")
        return {"development_result": DevelopmentResult(outcome=StageOutcome.COMPLETED)}
    def qa(_):
        return {"qa_result": QAResult(verdict=QAVerdict.FAIL)}
    def release(_):
        raise AssertionError("release must not execute")
    graph = build_graph(research=research, developer=developer, qa=qa, release=release)
    result = graph.invoke({"task": TaskRequest(task="Generic demo", max_dev_revisions=2)})
    assert result["status"] == RunStatus.FAILED
    assert result["revision_count"] == 2
    assert len(calls) == 3  # Initial development + 2 revisions


def test_blocked_research_stops():
    def research(_):
        return {"research_result": ResearchResult(outcome=StageOutcome.BLOCKED)}
    def unexpected(_):
        raise AssertionError("must not execute")
    graph = build_graph(research=research, developer=unexpected, qa=unexpected, release=unexpected)
    result = graph.invoke({"task": TaskRequest(task="Generic demo")})
    assert result["status"] == RunStatus.BLOCKED
