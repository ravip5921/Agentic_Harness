
"""Application entry point for the Agentic Harness."""

import sys

from harness.cli import harness_cli


def main() -> int:
    """Launch the Agentic Harness command-line interface."""
    return harness_cli()


if __name__ == "__main__":
    sys.exit(main())
