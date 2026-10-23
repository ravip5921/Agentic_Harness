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

All Groq requests share a global gate within the Python process. Successful
requests proceed immediately. A **429 response** triggers a wait based on
Groq's `Retry-After` / `retry-after-ms` headers or the error's "try again in"
duration, plus **2 seconds of padding**. If multiple hints are present, the
longest is respected. Without a valid hint, backoff starts at 5 seconds and
doubles to a maximum of 60 seconds, plus padding.

The harness allows up to **5 retries** for temporary rate limits. If the error
says the individual requested token count exceeds the TPM limit, it allows
only **2 retries** for that condition. Exhaustion returns the error to the
existing stage failure or documentation fallback handling. SDK automatic
retries are disabled so these caps reflect actual API attempts. Other errors
propagate immediately. Model settings, prompts, and response formats are unchanged.

This covers Research, Developer planning and each file, QA assessments and
feedback, and Release documentation. Rich shows a yellow countdown and retry
messages; redirected output prints each wait once. There is no fixed cooldown;
the former `GROQ_COOLDOWN_SECONDS` setting is no longer used. Separate harness
processes do not share the gate. See [Groq's rate-limit guidance](https://console.groq.com/docs/rate-limits).

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
