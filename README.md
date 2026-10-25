# Agentic Harness

Agentic Harness is a Python command-line tool that turns a natural-language software task into a four-stage LangGraph workflow: Research creates a design specification, Developer implements it, QA tests and reviews the result in Docker, and Release prepares the output for human approval. It supports new standalone projects or changes to an existing repository, saves run artifacts for review, and works with Groq, Gemini, or a compatible model endpoint.

## Setup

You need Python 3.11 or newer, a model provider API key, and **Docker installed and running**. Start Docker Desktop (or the Docker daemon) before running the harness; QA requires access to it. The first run automatically builds the QA image and needs network access.

From the project root, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Alternatively, if you have Conda installed, create and activate a Conda environment with Python 3.12:

```bash
conda create -n harness python=3.12
conda activate harness
```

With either environment active, install the project and its dependencies:

```bash
python -m pip install -e .
```

Copy `.env.example` to `.env` in the project root (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` on macOS/Linux). Set `LLM_PROVIDER` and the corresponding API key. The default is Groq, so replace the `GROQ_API_KEY` placeholder with your key. The same [.env.example](.env.example) includes settings for Gemini and compatible endpoints.

Confirm Docker is reachable:

```bash
docker info
```

## Run

Run a task from the project root with the virtual environment active:

```bash
harness "Implement an addition utility with tests" --runs-dir runs
```

To work on an existing repository:

```bash
harness "Update the client" --repository /path/to/repository --documentation-url https://example.com/docs --max-dev-revisions 2
```

Artifacts are saved under `runs/<run-id>/`. After QA passes, the terminal prompts for release approval; see [Release](#release) below for staging and pull request options.

You can also run the CLI with `python -m harness.main`.

## Release

After QA passes, Release prepares a candidate and asks `Approve this release candidate? [y/N]`. If approved, choose one of these options:

- `staged` (default): keeps the approved output in the run's release directory.
- `pr` (requires `--repository`): pushes a new branch and opens a draft GitHub pull request for review and merging. The original working tree remains unchanged.

**Creating a pull request requires Git and `gh` (GitHub CLI) installed and available on your PATH.** Authenticate before running the harness:

```bash
gh auth login
```

Configure your Git author (`user.name` and `user.email`) and ensure you have push access to the target repository's GitHub `origin`. The repository must be clean, with its current branch synchronized with origin. Standalone runs support staging only.

Release requires interactive approval. Rejecting the candidate or running without terminal input blocks release and preserves the candidate for review. Release decisions and results are saved in the run's release directory.

## Development

Install the development dependencies and run the tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Live Groq tests are opt-in with `RUN_LIVE_GROQ=1`; leave it unset for offline tests.


## Project Requirements
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