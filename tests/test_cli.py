from typer.testing import CliRunner

from proofpath import __version__
from proofpath.cli import app

runner = CliRunner()


def test_version_flag_reports_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_bare_invocation_is_an_entry_point_not_a_help_screen() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert f"proofpath {__version__}" in result.stdout
    assert "the interactive TUI arrives in v0.2" in result.stdout
    assert "proofpath check paper.pdf" in result.stdout
