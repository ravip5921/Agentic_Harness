import pytest
from harness.cli import harness_cli


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        harness_cli(["--help"])

    assert exc.value.code == 0
