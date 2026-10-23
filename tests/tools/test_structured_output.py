"""Offline checks of the schemas sent to Groq's strict output decoder."""

from copy import deepcopy

import pytest

from harness.models import (
    DevelopmentPlan,
    QAFeedback,
    QARequirementAssessment,
    ReleaseAssessment,
    ResearchAssessment,
)
from harness.tools.structured_output import strict_json_schema, strict_response_format


@pytest.mark.parametrize("model", [
    ResearchAssessment,
    DevelopmentPlan,
    QAFeedback,
    QARequirementAssessment,
    ReleaseAssessment,
])
def test_agent_schemas_are_strict_without_mutating_original(model):
    original = model.model_json_schema()
    snapshot = deepcopy(original)
    response_format = strict_response_format("assessment", original)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "assessment"
    assert response_format["json_schema"]["strict"] is True

    def check(node):
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(response_format["json_schema"]["schema"])
    assert original == snapshot


def test_schema_conversion_preserves_nullable_refs_and_constraints():
    schema = strict_json_schema(ResearchAssessment.model_json_schema())

    # A blocked Research response can still use specification: null.
    assert schema["properties"]["specification"]["anyOf"] == [
        {"$ref": "#/$defs/EngineeringSpec"},
        {"type": "null"},
    ]
    assert schema["properties"]["status"]["enum"] == ["ready", "blocked"]
    spec = schema["$defs"]["EngineeringSpec"]
    assert spec["properties"]["title"]["minLength"] == 1
    assert spec["properties"]["requirements"]["minItems"] == 1
    assert "source_ids" in spec["required"]
