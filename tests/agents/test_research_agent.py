"""Research request format and local validation without network access."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from harness.agents.research_agent import GroqResearchAgent, ResearchGenerationError
from harness.models import TaskRequest


@pytest.mark.parametrize("payload,valid", [
    ('{"status":"blocked","reason":"Missing task details","specification":null}', True),
    ('{"status":"ready","reason":"","specification":null}', False),
])
def test_research_uses_strict_schema_and_keeps_semantic_validation(
    monkeypatch, payload, valid,
):
    create = Mock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=payload)),
    ]))
    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create),
    ))
    monkeypatch.setattr("harness.agents.research_agent.Groq", lambda **kwargs: client)
    agent = GroqResearchAgent(model="test-model")
    task = TaskRequest(task="Example task")

    if valid:
        assert agent.generate(task, None, []).status == "blocked"
    else:
        with pytest.raises(ResearchGenerationError, match="ValidationError"):
            agent.generate(task, None, [])

    response_format = create.call_args.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "research_assessment"
    assert response_format["json_schema"]["strict"] is True
    assert "specification" in response_format["json_schema"]["schema"]["required"]
