# Existing-repository integration demo

`integration-demo/` is the target repository directory. It intentionally has
only a connector contract and baseline tests; the harness generates the GitHub
implementation. Git initialization is optional for the current staging flow.

Run these commands from the harness root using its installed environment:

```bash
.venv/bin/python -m pytest -q examples/integration-demo
docker image inspect harness-qa-python:3.11
# Build the trusted QA image if it is missing:
docker build -f docker/qa-python.Dockerfile -t harness-qa-python:3.11 .

.venv/bin/python -m harness.main "$(cat examples/github-members-task.txt)" \
  --repository examples/integration-demo \
  --documentation-url 'https://docs.github.com/en/rest/orgs/members' \
  --runs-dir runs/integration-demo \
  --max-dev-revisions 2
```

Requires a configured `GROQ_API_KEY`, external access to Groq and documentation,
and a usable Docker daemon with the QA image. The documentation fetcher rejects
responses above 100,000 bytes and retains only the first 6,000 characters. If
the official page exceeds that limit, Research will block; inspect the actual
Research result before treating documentation retrieval as successful.

A successful run must show Research and Developer outcomes `completed`, QA
verdict `pass`, and Release status `staged`. Inspect the engineering spec under `research/`, the generated
manifest and files under `development/attempt-N/`, the actual executed checks
and requirement findings in `qa/attempt-N/test_report.json`, and the candidate
and release report under `release/attempt-N/`. The final CLI output gives their
exact paths. Check that the candidate preserves the baseline files and adds
the connector and tests, and that the original target remains unchanged.

STAGED means an isolated candidate has been prepared. It does not mean `git
add`, commit, merge, deployment, or modification of the target directory.
Mocked connector tests establish implementation behavior without exercising
live GitHub authentication or organization permissions.

## Initial execution result

On October 23, 2026, the two baseline tests passed. CLI run
`4baac1a9b9b042a581907e8b4ca4c554` inspected all four target files, then returned
`blocked` (exit code 2): `Could not resolve documentation host.` Developer,
QA, and Release did not execute. Docker image inspection also failed with
permission denied connecting to `/var/run/docker.sock`, including outside the
sandbox; image availability could not be established.

Automatic approval review initially rejected the network-enabled retry because
it would send repository context and task text to Groq without specific approval
for that external transfer. The user subsequently approved sending only the new
demo repository files and task text. Network-enabled run
`f4a655669d3f480e8062c165efbf7d4d` then resolved the documentation host, but
Research returned `blocked` (exit code 2): `Documentation exceeds retrieval
limit.` The official page exceeds the fetcher's 100,000-byte limit. No model
generation, successful end-to-end result, or staged candidate has been
established. Release implementation was not changed.
