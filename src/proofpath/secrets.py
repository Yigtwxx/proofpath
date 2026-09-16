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
"""

from __future__ import annotations

import os
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


def default_dotenv_paths() -> list[Path]:
    """Project ``.env`` first, then the user config dir."""
    return [Path.cwd() / ".env", config_dir() / ".env"]


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
