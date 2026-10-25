"""Baseline contract tests that must survive connector integration."""

import pytest

from integrations import Connector, User


def test_connector_requires_an_implementation():
    with pytest.raises(TypeError):
        Connector()


def test_connector_returns_shared_user_type():
    class MemoryConnector(Connector):
        def list_users(self) -> list[User]:
            return [User("1", "octocat", "https://github.com/octocat")]

    assert MemoryConnector().list_users() == [
        User(id="1", username="octocat", profile_url="https://github.com/octocat")
    ]
