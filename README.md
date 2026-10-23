# Agentic Harness

Run the harness with the existing Groq configuration (`GROQ_API_KEY` in your
environment or `.env`):

```bash
python -m pip install -e . -r requirements.txt
harness "Implement an addition utility with tests" --runs-dir runs
harness "Update the client" --repository /path/to/repository --documentation-url https://example.com/docs --max-dev-revisions 2
```

The CLI displays operation progress as it happens, including validated file
plans, file generation, actual QA check results, retries, and release staging.
Interactive terminals show a current-operation indicator; redirected output
uses plain lines without animation. The original final JSON summary still
includes all stage results, artifact paths, run ID, and revision count.
Exit codes remain 0 for STAGED, 2 for BLOCKED, and 1 for failure or invalid input.

Stages and the Developer Agent emit small best-effort events through
`harness.events.emit_event`, using LangGraph's `get_stream_writer()`. Calls
outside a graph safely do nothing. The CLI consumes `custom` for operations,
`updates` for completed nodes and retries, and `values` for the complete final
graph state. Events are never stored in workflow state or artifacts. This
uses the LangGraph 1.x streaming API, verified with 1.2.16, and Rich 13–14.

Run offline tests (mocked generation and sandbox execution):

```bash
RUN_LIVE_GROQ=0 python -m pytest -q
RUN_LIVE_GROQ=0 python -m pytest -q tests/test_events.py tests/test_progress.py tests/test_cli.py tests/test_generation_events.py tests/test_qa_release_events.py tests/integration/test_streaming.py
```

Existing live Groq tests remain opt-in with `RUN_LIVE_GROQ=1`.

### Project Requirements

```
    Build an agentic harness which has four stages.

    1. Research
    2. Developer
    3. QA
    4. Release

    Each stage can have one or more agents as you design.

    Now, we will give a task say, given a name of system such as GitHub or Gitlab or Salesforce or AWS or Google, and the information to extract such as “Users on the platform”

    1. Researcher will research and create an engg design spec file
    2. Developer will build based on spec
    3. QA will test
    4. Releaser will add that to the existing code
```
