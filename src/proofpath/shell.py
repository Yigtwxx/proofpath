"""Logged subprocess commands, shown the way this platform's shell takes them back.

A leaf: it imports nothing from the package, so ``browser`` (the browser install) and
``update`` (the self-update) can both run their commands through it without a cycle
between them. Both log every command as ``$ cmd`` with its exit code, and both may
have to tell the user to run one by hand, so the line has to paste: ``cmd.exe``
quoting on Windows, POSIX quoting elsewhere.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence


def on_windows() -> bool:
    """The platform question, asked through one hook a test can patch on any runner."""
    return sys.platform == "win32"


# Characters ``cmd.exe`` or PowerShell act on outside double quotes. ``list2cmdline``
# quotes only for whitespace, so ``scrapling[fetchers]>=0.4.15`` came out bare and,
# pasted back, redirected pip's output to a file named ``=0.4.15``. PowerShell also
# reads ``proofpath[browser,gpu]`` as an array (``,``) and ends a statement at ``;``.
_WINDOWS_SPECIAL = frozenset("<>|&^(),;")


def _windows_quoted(arg: str) -> str:
    """``arg`` in double quotes by the MS C runtime's rule, as ``list2cmdline`` quotes
    one: a backslash run is doubled before a quote (escaped or closing) and kept as is
    anywhere else, so a trailing backslash cannot swallow the closing quote."""
    out = ['"']
    backslashes = 0
    for char in arg:
        if char == "\\":
            backslashes += 1
            continue
        if char == '"':
            out.append("\\" * (backslashes * 2 + 1) + '"')
        else:
            out.append("\\" * backslashes + char)
        backslashes = 0
    out.append("\\" * (backslashes * 2) + '"')
    return "".join(out)


def _windows_arg(arg: str) -> str:
    if _WINDOWS_SPECIAL.intersection(arg):
        return _windows_quoted(arg)
    return subprocess.list2cmdline([arg])


def display_command(cmd: Sequence[str]) -> str:
    """``cmd`` as one line this platform's shell takes back as is. ``shlex.join``
    would wrap ``proofpath[gpu]`` in single quotes, which ``cmd.exe`` keeps as part
    of the argument. On Windows each argument is quoted as ``list2cmdline`` quotes it
    (``subprocess``'s own rule there), and also whenever it holds a character the
    shell would act on."""
    if on_windows():
        return " ".join(_windows_arg(arg) for arg in cmd)
    return shlex.join(cmd)


def run_logged(
    cmd: list[str],
    log: list[str],
    run: Callable[..., subprocess.CompletedProcess[str]],
    *,
    show: Callable[[Sequence[str]], str] = display_command,
) -> subprocess.CompletedProcess[str]:
    """Run one command with its output captured, and log it as ``$ cmd``, its exit
    code and, on failure, the last line of its stderr.

    Captured, never inherited: the TUI owns the terminal while these run.
    """
    # Quoted, not merely spaced: the log is what the user is shown when an install
    # fails, and a line they can paste back into a shell says more than one they
    # cannot. The -c program and the pip spec are each one argument.
    log.append(f"$ {show(cmd)}")
    try:
        result = run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        # e.g. the interpreter/binary named in cmd does not exist on this machine.
        log.append(f"exit ? ({type(exc).__name__}: {exc})")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
    log.append(f"exit {result.returncode}")
    if result.returncode != 0:
        stderr_lines = (result.stderr or "").strip().splitlines()
        if stderr_lines:
            log.append(stderr_lines[-1])
    return result


__all__ = ["display_command", "on_windows", "run_logged"]
