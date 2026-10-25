# Implement GitHubConnector for integration-demo

        ## Objective

        Extend the integration-demo library with a GitHubConnector that fetches organization members using the GitHub REST API.

        ## Requirements

        - Implement GitHubConnector in src/integrations/github.py inheriting from integrations.base.Connector.
- Constructor must accept organization: str, token: str, and timeout: float = 10.0.
- Validate that organization and token are non-empty strings.
- Use standard library urllib.request for HTTP calls.
- Implement pagination using the Link header or by following the API specification for GET /orgs/{org}/members.
- Set headers: Authorization: Bearer <token> and Accept: application/vnd.github+json.
- Apply the configured timeout to all requests.
- Map GitHub response fields: id (as string) to User.id, login to User.username, and html_url to User.profile_url.
- Raise exceptions for all HTTP errors (4xx, 5xx) and transport errors.
- Ensure no partial results are returned on failure.
- Do not log sensitive tokens.
- Add comprehensive pytest tests using mocking for network calls.
- Include documentation with placeholder credentials.

        ## Repository Context

        D:\Agentic_Harness\examples\integration-demo

        ## Technical Approach

        Create src/integrations/github.py. Use urllib.request.Request and urllib.request.urlopen for network communication. Implement a loop to handle pagination by checking the 'Link' header or simply requesting pages until an empty list is returned. Use unittest.mock to patch urlopen in tests to simulate various API responses (success, pagination, 401, 403, 404, 429, malformed JSON).

        ## Implementation Steps

        - Create src/integrations/github.py and define GitHubConnector class.
- Implement constructor with validation for organization and token.
- Implement list_users method with pagination logic and header management.
- Implement error handling for HTTP status codes and network exceptions.
- Create tests/test_github.py with mocks for successful, paginated, and error scenarios.
- Update documentation to include usage examples.

        ## Acceptance Criteria

        - GitHubConnector correctly maps GitHub API response to User objects.
- GitHubConnector handles pagination correctly when multiple pages exist.
- GitHubConnector raises appropriate exceptions on 4xx/5xx errors.
- GitHubConnector respects the configured timeout.
- All baseline tests in tests/test_base.py and tests/test_utils.py pass.
- New tests in tests/test_github.py pass with 100% coverage of the new connector logic.
- No runtime dependencies are added to pyproject.toml.

        ## Assumptions

        - None

        ## Constraints

        - None

        ## Evidence References

        - None
