"""Command-line entry point.

The TUI and the one-shot commands are two front-ends over a single ``verify()``
call. Neither holds logic of its own; see the design spec, section 13.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from proofpath import __version__

app = typer.Typer(
    name="proofpath",
    help="Check whether the sources behind a claim actually say what the claim says.",
    no_args_is_help=False,
    add_completion=True,
)

# Exit codes are part of the interface: CI can gate on them without parsing text.
EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2

_NOT_BUILT_YET = (
    "proofpath is at the design stage — the verification pipeline is not implemented yet.\n"
    "\n"
    "  design spec  https://github.com/Yigtwxx/proofpath/blob/main/docs/superpowers/specs/2026-09-10-proofpath-design.md\n"
    "  plan         https://github.com/Yigtwxx/proofpath/blob/main/docs/superpowers/plans/2026-09-10-proofpath-implementation-plan.md\n"
    "  open items   https://github.com/Yigtwxx/proofpath/blob/main/docs/superpowers/OPEN-ITEMS.md\n"
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"proofpath {__version__}")
        raise typer.Exit(EXIT_CLEAN)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """Launch the interactive TUI when called with no subcommand."""
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(_NOT_BUILT_YET)
    raise typer.Exit(EXIT_CLEAN)


@app.command()
def check(
    target: Annotated[Path, typer.Argument(help="Document to check.")],
) -> None:
    """Verify every citation in a document and write a report."""
    typer.echo(_NOT_BUILT_YET, err=True)
    raise typer.Exit(EXIT_ERROR)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
