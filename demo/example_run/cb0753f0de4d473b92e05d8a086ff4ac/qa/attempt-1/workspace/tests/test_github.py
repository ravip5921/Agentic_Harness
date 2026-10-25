import pytest
import json
import urllib.error
from unittest.mock import patch, MagicMock
from integrations.github import GitHubConnector
from integrations.base import User

def test_github_connector_initialization():
    connector = GitHubConnector(organization="test-org", token="secret-token")
    assert connector.organization == "test-org"
    assert connector.token == "secret-token"

def test_github_connector_invalid_args():
    with pytest.raises(ValueError):
        GitHubConnector(organization="", token="token")
    with pytest.raises(ValueError):
        GitHubConnector(organization="org", token="")

@patch("urllib.request.urlopen")
def test_list_users_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps([
        {"id": 123, "login": "user1", "html_url": "https://github.com/user1"},
        {"id": 456, "login": "user2", "html_url": "https://github.com/user2"}
    ]).encode("utf-8")
    mock_response.headers = {}
    
    # Mock context manager
    mock_urlopen.return_value.__enter__.return_value = mock_response

    connector = GitHubConnector(organization="test-org", token="token")
    users = connector.list_users()

    assert len(users) == 2
    assert users[0] == User(id="123", username="user1", profile_url="https://github.com/user1")
    assert users[1] == User(id="456", username="user2", profile_url="https://github.com/user2")

@patch("urllib.request.urlopen")
def test_list_users_pagination(mock_urlopen):
    # First page
    resp1 = MagicMock()
    resp1.read.return_value = json.dumps([{"id": 1, "login": "u1", "html_url": "url1"}]).encode("utf-8")
    resp1.headers = {"Link": '<https://api.github.com/page2>; rel="next"'}
    
    # Second page
    resp2 = MagicMock()
    resp2.read.return_value = json.dumps([{"id": 2, "login": "u2", "html_url": "url2"}]).encode("utf-8")
    resp2.headers = {}

    # Mock context manager for each call
    mock_urlopen.return_value.__enter__.side_effect = [resp1, resp2]

    connector = GitHubConnector(organization="test-org", token="token")
    users = connector.list_users()

    assert len(users) == 2
    assert users[0].username == "u1"
    assert users[1].username == "u2"
    assert mock_urlopen.call_count == 2

@patch("urllib.request.urlopen")
def test_list_users_http_error(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.HTTPError(
        url="http://test", code=404, msg="Not Found", hdrs={}, fp=None
    )
    connector = GitHubConnector(organization="test-org", token="token")
    with pytest.raises(RuntimeError, match="404"):
        connector.list_users()

@patch("urllib.request.urlopen")
def test_list_users_timeout(mock_urlopen):
    # Simulate TimeoutError which is now caught by the connector
    mock_urlopen.side_effect = TimeoutError("Connection timed out")
    connector = GitHubConnector(organization="test-org", token="token")
    with pytest.raises(RuntimeError, match="Failed to communicate"):
        connector.list_users()

@patch("urllib.request.urlopen")
def test_list_users_malformed_json(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b"invalid json"
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    connector = GitHubConnector(organization="test-org", token="token")
    with pytest.raises(RuntimeError):
        connector.list_users()