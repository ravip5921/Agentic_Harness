# Agentic Harness

Install the project and configure a model provider in your environment or the
project's root `.env` file (see `.env.example`). Existing Groq configuration
continues to work without changes:

```bash
python -m pip install -e .
harness "Implement an addition utility with tests" --runs-dir runs
harness "Update the client" --repository /path/to/repository --documentation-url https://example.com/docs --max-dev-revisions 2
```

`src/harness/config.py` centrally loads `.env` with `python-dotenv` when the
harness starts through `main.py` or the CLI. Values already exported in the
process environment take precedence. Loaded values are available throughout
the process through environment access.
You can also run the entry point with `python -m harness.main`.

### Model providers

All four agents use the provider-independent `TextGenerator.complete()`
interface in `src/harness/llm/`. Adapters translate messages, structured schemas,
output-token limits, authentication, and responses. Select a provider once;
Research, Developer, QA, and Release use that configuration. There is no
automatic fallback to another provider.

For Gemini, set:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.8-flash
LLM_MAX_OUTPUT_TOKENS=4096
```

For Groq, use `LLM_PROVIDER=groq` (the default), `GROQ_API_KEY`, and optionally
`GROQ_MODEL` (default `openai/gpt-oss-20b`). A Groq key is not required for Gemini.
Gemini's default model is `gemini-3.8-flash`; choose a model available to your
account. The native adapter uses Google's
[generateContent API](https://ai.google.dev/api/generate-content), including
`responseJsonSchema` for structured output.

For a Chat Completions-compatible endpoint, set:

```dotenv
LLM_PROVIDER=openai-compatible
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-id
LLM_API_KEY=your-key
```

The adapter appends `/chat/completions` to this base URL. The endpoint must
support `max_completion_tokens` and strict `json_schema` responses for structured
calls. Protocol compatibility varies; unsupported capabilities fail rather than
silently weakening validation. APIs with a different protocol need an adapter.
The built-in Gemini adapter uses the existing `requests` dependency; no extra
SDK installation is required.

`LLM_MODEL` and `LLM_API_KEY`, if set, override provider-specific model/key
settings. Remove stale generic overrides when switching providers.
`LLM_MAX_OUTPUT_TOKENS` defaults to 4096 and `LLM_TIMEOUT_SECONDS` to 90.
Output budgets must fit the selected model and account limits; increasing a
budget can cause oversized requests. Requests ending with `length`,
`MAX_TOKENS`, or another non-normal completion status are rejected before
their content is accepted. This detects provider-reported truncation; normal
completion does not establish code correctness or test completeness.

Every built-in provider prints request and response diagnostics through the
CLI's progress stream. Requests identify the provider, model, file or operation,
configured output-token limit, timeout, and input character count (not a token
estimate). Responses show the finish reason, provider-reported prompt,
completion, reasoning and total tokens, output characters/bytes/lines, and
elapsed time. Missing metadata is labeled `unreported`. Python file responses
also receive a syntax diagnostic; this does not replace QA.

Developer planning and file-generation metadata is saved as JSON Lines under
`runs/<run-id>/diagnostics/development-attempt-N.jsonl` (or the selected runs
directory). These logs survive generation failures and are outside staging and
release candidates. They contain no prompts, source text, API keys, or raw
exception messages. Responses are logged before validation, including those
rejected for truncation. `output_limit_reached` identifies a provider-reported
token cutoff; `normal_stop_not_completeness_proof` and
`finish_reason_unreported` distinguish other cases. Token counters retain the
provider's definitions; reasoning tokens are not necessarily additive to
completion tokens.

Agents accept `generator=` for custom adapters and tests; `client=` injects a
native transport for the selected provider (Groq SDK client, or an HTTP client
with `post()` for Gemini/compatible endpoints). The old `Groq*Agent` class names
remain aliases. To add a new protocol, implement `TextGenerator`, add its settings
and factory selection, and test request/response translation. Agents and workflow
stages need no changes.

All runtime dependencies are declared in `pyproject.toml`. It is sufficient
for pip to install the project and its dependencies; a separate
`requirements.txt` would duplicate that list. Requirements files are useful
for installation constraints or pinned deployment environments, but this
project does not currently need one. For development and tests, include the
`dev` extra:

```bash
python -m pip install -e '.[dev]'
```

The CLI displays operation progress as it happens, including validated file
plans, file generation, actual QA check results, retries, and release staging.
Interactive terminals show a current-operation indicator; redirected output
uses plain lines without animation. The original final JSON summary still
includes all stage results, artifact paths, run ID, and revision count.
Exit codes remain 0 for STAGED, 2 for BLOCKED, and 1 for failure or invalid input.

All model requests share a global gate within the Python process. Successful
requests proceed immediately. A **429 response** triggers a wait based on
`Retry-After` / `retry-after-ms` headers, Gemini `RetryInfo.retryDelay`, or the error's "try again in"
duration, plus **2 seconds of padding**. If multiple hints are present, the
longest is respected. Without a valid hint, backoff starts at 5 seconds and
doubles to a maximum of 60 seconds, plus padding.

The harness allows up to **5 retries** for temporary rate limits. If the error
says the individual requested token count exceeds the TPM limit, it allows
only **2 retries** for that condition, whether the provider returns HTTP 429 or HTTP
413. A 413 is retried only when its TPM counts confirm `Requested > Limit`;
other 413 errors propagate immediately. Exhaustion returns the error to the
existing stage failure or documentation fallback handling. SDK automatic
retries are disabled so these caps reflect actual API attempts. Other errors
propagate immediately. The retry policy never shrinks or rewrites requests.

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

Structured outputs cover Research, Developer planning, QA requirements and
feedback, and Release documentation. Groq and compatible endpoints use the
strict JSON Schema helper in `src/harness/tools/structured_output.py`; the
configured model must support strict structured outputs. Gemini uses native
JSON-schema output with references inlined and unsupported decoder constraints
removed. Local Pydantic validation retains the complete original constraints
and application rules for every provider. Developer file generation continues
to return plain source text.

### QA dependencies

Build the Python QA image before running tasks that need containerized QA:

```bash
docker build -f docker/qa-python.Dockerfile -t harness-qa-python:3.11 .
```

The Developer plan's `dependencies` array is the canonical dependency
manifest for Python QA. Declare runtime and test dependencies as PEP 508
distribution requirements, for example `Pillow>=10` (the distribution that
provides `PIL`) and `pytest>=8`. Use an empty array for standard-library-only
implementations. Entries cannot be shell commands, URLs, local paths, or prose.
QA does not automatically install from generated `requirements.txt` files;
such a file is unnecessary solely to provision QA and may conflict with a
task's file-count constraint.

Before tests, QA installs declared dependencies in a separate network-enabled
setup container using wheels only. That container mounts no project sources
or secrets. Dependency artifacts live under `qa/attempt-N/dependencies`,
outside the generated workspace and release. Setup status and logs are
recorded in `qa/attempt-N/test_report.json`.
Tests run offline with the dependency directory mounted read-only. Neither
setup nor testing installs packages into the host Python environment.

A dependency setup failure returns QA `BLOCKED` and does not trigger a
Developer retry. Missing imports whose roots cannot be found in the workspace
also conservatively block QA; the harness does not guess distribution names
or install from test output. Missing submodules of identifiable local modules
remain implementation failures. Missing system shared libraries block QA and
must be supplied by the trusted QA Docker image, rather than Python packages.
Packages available only as source distributions cannot be installed through
this wheel-only setup. Review the dependency setup logs when resolution,
network access, or wheel availability prevents verification.

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
