"""Centralized environment initialization for the harness."""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment(dotenv_path: str | Path | None = None) -> bool:
    """Load .env values into os.environ, preserving existing environment variables.

    The default file is the project's .env, independent of the working directory.
    An explicit path can be supplied by scripts or tests. A missing file is fine
    when configuration is provided through the process environment.
    """
    env_file = PROJECT_ROOT / ".env" if dotenv_path is None else dotenv_path
    return load_dotenv(dotenv_path=env_file, override=False)
