"""Shared strict JSON response formats for schema-constrained agent output.

The configured Groq model must support strict structured outputs. Agents
still validate responses locally to enforce application-specific rules.
"""

from copy import deepcopy


def strict_json_schema(schema: dict) -> dict:
    """Adapt a schema without mutating it, including nested definitions."""
    result = deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            node.pop("default", None)

            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))

            for value in node.values():
                visit(value)

        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def strict_response_format(name: str, schema: dict) -> dict:
    """Build the API response format for a named strict JSON schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": strict_json_schema(schema),
        },
    }
