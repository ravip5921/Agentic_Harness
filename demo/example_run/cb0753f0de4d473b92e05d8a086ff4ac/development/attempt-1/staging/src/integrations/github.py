import json
import re
import urllib.request
import urllib.error
from typing import List, Optional
from integrations.base import Connector, User

class GitHubConnector(Connector):
    """
    Connector to fetch organization members from GitHub REST API.
    
    Usage:
        connector = GitHubConnector(organization="my-org", token="ghp_...")
        users = connector.list_users()
    """

    def __init__(self, organization: str, token: str, timeout: float = 10.0):
        if not isinstance(organization, str) or not organization.strip():
            raise ValueError("Organization must be a non-empty string.")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Token must be a non-empty string.")
        
        self.organization = organization
        self.token = token
        self.timeout = timeout
        self.base_url = f"https://api.github.com/orgs/{self.organization}/members"

    def list_users(self) -> List[User]:
        users: List[User] = []
        next_url: Optional[str] = self.base_url

        while next_url:
            request = urllib.request.Request(
                next_url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                }
            )

            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    
                    for item in data:
                        users.append(User(
                            id=str(item["id"]),
                            username=item["login"],
                            profile_url=item["html_url"]
                        ))

                    # Handle pagination via Link header
                    link_header = response.headers.get("Link")
                    next_url = self._parse_next_link(link_header) if link_header else None

            except urllib.error.HTTPError as e:
                raise RuntimeError(f"GitHub API request failed with status {e.code}: {e.reason}") from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Failed to communicate with GitHub API: {str(e)}") from e

        return users

    def _parse_next_link(self, link_header: str) -> Optional[str]:
        """Parses the Link header to find the URL for the next page."""
        parts = link_header.split(",")
        for part in parts:
            if 'rel="next"' in part:
                match = re.search(r'<(.*?)>', part)
                if match:
                    return match.group(1)
        return None