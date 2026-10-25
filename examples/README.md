# GitHub organization connector demo

This example uses the [GitHub members task](github-members-task.txt) to extend an existing Python library with a `GitHubConnector`. The connector lists members of a configured GitHub organization, maps them to the library's `User` model, follows pagination, and handles HTTP and transport errors.

The supplied [integration-demo/](integration-demo/) contains the connector contract and baseline tests. The harness generates the GitHub implementation and its tests while preserving the existing contract.

## Recorded demo outcome

Run `cb0753f0de4d473b92e05d8a086ff4ac` completed the workflow with final release status **`pr_created`**. The saved QA and release reports record:

| Step | Recorded result |
| --- | --- |
| Research | Produced the engineering specification for `GitHubConnector` |
| Developer, attempt 0 | Generated the connector, tests, and package export changes |
| QA, attempt 0 | Syntax passed; pytest reported 12 passed and 2 failed, involving pagination test mocking and timeout handling |
| Developer, attempt 1 | Revised the output in response to QA feedback |
| QA, attempt 1 | Syntax passed; all 14 tests passed; requirements assessment returned `pass` |
| Release | Prepared the candidate, recorded human approval to create a PR, and reported draft PR creation |

The release result records [integration-demo pull request #5](https://github.com/ravip5921/integration-demo/pull/5). The changes create `src/integrations/github.py` and `tests/test_github.py`, and modify `src/integrations/__init__.py` to export the connector.

This run demonstrates the QA repair loop as well as generation and human-approved release. It completed after one Developer revision.

See [demo materials](../demo/README.md) for the slides, saved run artifacts, and recording availability.

## Run the example

Complete the root README's [setup instructions](../README.md#setup), including model provider configuration and starting Docker. Run the following from the harness root with the environment active.

Check the supplied baseline:

```bash
python -m pytest -q examples/integration-demo
```

On Windows PowerShell:

```powershell
$taskText = Get-Content examples/github-members-task.txt -Raw
harness "$taskText" --repository examples/integration-demo --runs-dir runs --max-dev-revisions 2
```

On macOS or Linux:

```bash
harness "$(cat examples/github-members-task.txt)" --repository examples/integration-demo --runs-dir runs --max-dev-revisions 2
```

The harness builds its QA Docker image automatically if needed. After QA passes, approve the release and select `staged` to keep the candidate locally. Git initialization is optional for this staging flow. To publish a draft PR as in the recorded demo, use a clean, synchronized Git checkout with a GitHub origin, push access, and authenticated `gh`; see [release prerequisites](../README.md#release).

Model outputs can vary, so a rerun may produce different files or require a different number of revisions.

## Inspect the output

Artifacts are saved under `runs/<run-id>/`. For the recorded run, the directory is `runs/cb0753f0de4d473b92e05d8a086ff4ac/`, and the final attempt is `attempt-1`.

| Artifact path within the run | What it contains |
| --- | --- |
| `research/engineering_spec.md` and `engineering_spec.json` | Engineering design and requirements |
| `development/attempt-1/manifest.json` and `staging/` | Generated file manifest and implementation |
| `qa/attempt-0/test_report.json` | Initial failures that triggered revision |
| `qa/attempt-1/test_report.json` | Final execution results and requirements assessment |
| `release/attempt-1/candidate/` | Prepared project candidate |
| `release/attempt-1/release_report.md` | Changes, testing summary, and review notes |
| `release/attempt-1/release_decision.json` and `release_result.json` | Human decision, final status, and PR reference |

The `runs/` directory is excluded from version control. A copy of this demo run is provided in [demo/example_run/](../demo/example_run/cb0753f0de4d473b92e05d8a086ff4ac/) for review. The CLI prints the artifact locations for each new run.

## Validation boundaries

- The suite contains 14 passing tests, including the existing baseline tests; connector tests use mocked HTTP calls. This result does not establish live GitHub authentication or organization permission behavior, and it is not a coverage measurement.
- The connector's intended scope is organization members visible to the supplied token, rather than every GitHub account or guaranteed visibility of private members.
- The recorded Research evidence file is empty. This run demonstrates specification generation, but does not establish successful retrieval of external documentation.
- `staged` keeps a candidate locally; `pr_created` records draft PR creation. Neither status means the changes were merged or deployed.
