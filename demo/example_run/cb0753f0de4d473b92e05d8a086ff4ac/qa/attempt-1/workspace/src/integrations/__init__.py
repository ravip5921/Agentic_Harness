"""Shared types for platform integrations."""

from integrations.base import Connector, User
from integrations.github import GitHubConnector

__all__ = ["Connector", "User", "GitHubConnector"]