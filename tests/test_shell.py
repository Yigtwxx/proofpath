"""``shell.display_command``: a logged command has to paste back into this platform's
shell. The expectations are literal strings, never the function's own output."""

from __future__ import annotations

import pytest

from proofpath import shell

PIP = ["python", "-m", "pip", "install", "--upgrade", "scrapling[fetchers]>=0.4.15"]


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "on_windows", lambda: True)


@pytest.fixture
def posix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "on_windows", lambda: False)


@pytest.mark.usefixtures("windows")
def test_a_redirect_character_is_quoted_on_windows() -> None:
    # Bare, the `>` would send pip's output to a file named `=0.4.15`.
    assert shell.display_command(PIP) == (
        'python -m pip install --upgrade "scrapling[fetchers]>=0.4.15"'
    )


@pytest.mark.usefixtures("windows")
def test_an_extras_spec_stays_bare_on_windows() -> None:
    # Single quotes would become part of the argument in cmd.exe.
    assert shell.display_command(["uv", "tool", "install", "--upgrade", "proofpath[browser]"]) == (
        "uv tool install --upgrade proofpath[browser]"
    )


@pytest.mark.usefixtures("windows")
def test_two_extras_are_quoted_for_powershell() -> None:
    # Bare, PowerShell would read `browser,gpu` as an array and pass two arguments.
    assert shell.display_command(
        ["uv", "tool", "install", "--upgrade", "proofpath[browser,gpu]"]
    ) == ('uv tool install --upgrade "proofpath[browser,gpu]"')


@pytest.mark.usefixtures("windows")
def test_a_path_with_spaces_is_quoted_on_windows() -> None:
    assert shell.display_command([r"C:\Program Files\py\python.exe", "-V"]) == (
        r'"C:\Program Files\py\python.exe" -V'
    )


@pytest.mark.usefixtures("windows")
def test_a_trailing_backslash_cannot_swallow_the_closing_quote() -> None:
    assert shell.display_command(["a&b\\"]) == '"a&b\\\\"'


@pytest.mark.usefixtures("windows")
def test_an_inner_quote_is_escaped_on_windows() -> None:
    assert shell.display_command(['say "hi" & go']) == '"say \\"hi\\" & go"'


@pytest.mark.usefixtures("posix")
def test_posix_quoting_is_shlex() -> None:
    assert shell.display_command(PIP) == (
        "python -m pip install --upgrade 'scrapling[fetchers]>=0.4.15'"
    )
