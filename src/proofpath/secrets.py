"""Where proofpath looks for a credential, and what it says when there is none.

Nothing in proofpath's default path needs a key (spec section 7): the optional LLM
judge takes one, and Reddit takes a client id and secret for the free app a user
registers in their own account (section 6.2). Both are found the same way — the
environment first, then a ``.env`` — so the lookup lives here once rather than in
each caller, and ``judge.py`` re-exports it for the callers that already import it
from there.

Two rules hold for everything in this module. A value is never printed, logged, put
in an exception or shown in a ``repr``: :class:`ApiKey` carries its *source* for
that, so a run can say where a key came from without saying what it is. And a
credential that is absent is *reported* rather than skipped —
:data:`CREDENTIALS_MISSING` is a spec section 15 state of its own, never collapsed
into "unreachable" or "blocked", because "nobody asked" and "the platform refused"
are different facts about a source (product rule 2).

One place writes a key rather than reading it: the ``/config`` panel's key row, which
saves a pasted key with :func:`save_dotenv_value` into :func:`user_dotenv_path` --
the config dir's ``.env``, owner-only -- and never into the project ``.env`` or the
TOML config, both of which are the kind of file that ends up committed.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from proofpath.paths import config_dir

#: A source nobody asked for, because this run held no credential to ask with. It is
#: its own spec section 15 state: an unreachable post may not exist and a blocked one
#: is being withheld, where this one was never requested and one line of ``.env``
#: would read it. Collapsing it into either would tell the reader something untrue
#: about the source and hide the only fix (product rules 2 and 6).
CREDENTIALS_MISSING = "UNVERIFIED (credentials missing)"

#: The free Reddit "script" app a user registers under their own account. Two values,
#: because Reddit's client-credentials grant is HTTP basic auth over both.
REDDIT_CLIENT_ID_ENV = "REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET_ENV = "REDDIT_CLIENT_SECRET"


def missing_hint(*env_names: str) -> str:
    """What to set, named. Never a value, and never a guess at why it is absent."""
    if not env_names:
        return "set the credential in .env"
    listed = env_names[0] if len(env_names) == 1 else " and ".join(env_names)
    return f"set {listed} in .env"


#: The one sentence the coverage block owes a reader whose run hit
#: :data:`CREDENTIALS_MISSING`. It lives beside the variable names so the block and
#: the reader can never name different ones.
REDDIT_HINT = missing_hint(REDDIT_CLIENT_ID_ENV, REDDIT_CLIENT_SECRET_ENV)
CREDENTIALS_NOTE = (
    f"Reddit posts are read with a free app you register yourself: {REDDIT_HINT} and run again."
)


@dataclass(frozen=True, repr=False)
class ApiKey:
    """A secret plus where it came from. ``repr``/``str`` never show the value."""

    value: str = field(repr=False)
    source: str

    def __repr__(self) -> str:
        return f"ApiKey(source={self.source!r})"

    __str__ = __repr__


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``.env`` reader: ``KEY=value``, optional ``export``, quotes, comments."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in {"'", '"'} and value.count(value[0]) >= 2:
            quote = value[0]
            value = value[1 : value.index(quote, 1)]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key.strip()] = value
    return values


def user_dotenv_path() -> Path:
    """The ``.env`` in the user config dir: the one file proofpath itself writes a key
    to. Never the project ``.env`` beside the working directory, which sits in a
    repository and is one ``git add .`` away from being published."""
    return config_dir() / ".env"


def default_dotenv_paths() -> list[Path]:
    """Project ``.env`` first, then the user config dir."""
    return [Path.cwd() / ".env", user_dotenv_path()]


class SecretValueError(ValueError):
    """A key was refused before it was written. The message never contains the value:
    it says what was wrong with it, which is all a log line may say about a secret."""


#: What a shell and every ``.env`` reader accept as a variable name.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Characters :func:`read_dotenv` would read differently from how they were written --
#: a quote opens a quoted value, `` #`` starts a comment -- plus anything that is not
#: part of a pasted key but of what was around it. Refusing them is cheaper than
#: quoting them, and no provider's key contains one.
_REFUSED = frozenset("\"'#")


def _dotenv_name(line: str) -> str | None:
    """The variable a ``.env`` line sets, or ``None`` for a blank line or a comment."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :]
    return stripped.partition("=")[0].strip()


def _write_private(path: Path, lines: Sequence[str], newline: str = "\n") -> None:
    """Write ``lines`` to ``path`` so that no other user can ever read it.

    The content goes to a fresh sibling from :func:`tempfile.mkstemp` -- unique,
    created exclusively (so a planted file or symlink of that name is never written
    through) and ``0o600`` from its first byte on POSIX -- and then replaces the
    file, so the key never sits in a file with the default, often world-readable,
    mode, and a crash mid-write leaves the old file whole. A failure anywhere after
    the sibling exists removes it: no half-written copy of a key is left behind.

    A ``.env`` that is a symlink (a dotfiles repository, say) is written *through*:
    the link's target is replaced and the link survives. The final ``chmod`` covers a
    file that existed with a looser mode; Windows has no such bits (its ACLs follow
    the user profile the config dir lives in), so it is POSIX-only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.resolve()
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=".env.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write("".join(f"{line}{newline}" for line in lines))
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        target.chmod(0o600)


def _read_lines(path: Path) -> tuple[list[str], str]:
    """The file's lines and the line ending it uses, so a rewrite keeps a CRLF file
    CRLF (a Windows editor's) and every other file ``\\n``. Read with ``newline=""``:
    universal-newline mode would have turned every ``\\r\\n`` into ``\\n`` already."""
    if not path.exists():
        return [], "\n"
    with path.open(encoding="utf-8", newline="") as handle:
        raw = handle.read()
    return raw.splitlines(), "\r\n" if "\r\n" in raw else "\n"


def save_dotenv_value(path: Path, name: str, value: str) -> None:
    """Set ``name=value`` in the ``.env`` at ``path``, keeping every other line.

    An existing line for ``name`` (with or without ``export``) is replaced where it
    stands, so a hand-kept file keeps its order and comments; a later duplicate is
    dropped, because :func:`read_dotenv` keeps the *last* value and a stale one below
    would quietly win over the key just saved. With no such line, one is appended.

    The value is stripped of the whitespace a paste brings with it and refused if
    anything is left that would not read back as written (see :data:`_REFUSED`) --
    before the file is opened, so a refusal writes nothing.
    """
    if not _ENV_NAME.fullmatch(name):
        raise SecretValueError(f"{name!r} is not a variable name; nothing was saved")
    value = value.strip()
    if not value:
        raise SecretValueError("the key is empty; nothing was saved")
    if any(char.isspace() or char in _REFUSED for char in value):
        raise SecretValueError(
            "the key has a space, a quote or a # in it, which a .env cannot hold as "
            "written; nothing was saved"
        )
    existing, newline = _read_lines(path)
    lines: list[str] = []
    replaced = False
    for line in existing:
        if _dotenv_name(line) != name:
            lines.append(line)
        elif not replaced:
            export = "export " if line.strip().startswith("export ") else ""
            lines.append(f"{export}{name}={value}")
            replaced = True
    if not replaced:
        lines.append(f"{name}={value}")
    _write_private(path, lines, newline)


def remove_dotenv_value(path: Path, name: str) -> bool:
    """Drop every line that sets ``name``; ``True`` if there was one. A file with no
    such line -- or no file at all -- is left exactly as it was."""
    lines, newline = _read_lines(path)
    kept = [line for line in lines if _dotenv_name(line) != name]
    if len(kept) == len(lines):
        return False
    _write_private(path, kept, newline)
    return True


def resolve_api_key(
    env_name: str,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_paths: Sequence[Path] | None = None,
) -> ApiKey | None:
    """Environment variable first, then each ``.env`` file in order."""
    if not env_name:
        return None
    environ = os.environ if environ is None else environ
    value = environ.get(env_name)
    if value:
        return ApiKey(value, source=f"environment variable {env_name}")
    for path in default_dotenv_paths() if dotenv_paths is None else dotenv_paths:
        value = read_dotenv(path).get(env_name)
        if value:
            return ApiKey(value, source=str(path))
    return None
