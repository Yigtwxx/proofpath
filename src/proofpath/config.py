"""User configuration and the permission model (spec section 7.1).

The schema is small and flat on purpose. TOML is read with the standard library
(``tomli`` on 3.10) and written by hand, so no extra dependency is needed for the
handful of keys involved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args

from proofpath.paths import config_path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib  # type: ignore[import-not-found]

Permission = Literal["ask", "allow", "deny"]
PERMISSION_VALUES: tuple[str, ...] = get_args(Permission)

Outcome = Literal["allow", "deny", "prompt"]


class ConfigError(ValueError):
    """The config file exists but cannot be used. The message names the key."""


@dataclass(frozen=True)
class Permissions:
    # Step 3 of the fetch ladder: ~280 MB browser engine. Never installed silently.
    install_browser: Permission = "ask"
    network: Permission = "allow"


@dataclass(frozen=True)
class FetchConfig:
    respect_robots: bool = True


@dataclass(frozen=True)
class Contact:
    # Optional address for the Crossref / OpenAlex polite pools. Never hardcoded.
    email: str = ""


@dataclass(frozen=True)
class JudgeConfig:
    """Optional LLM second opinion (spec section 11). Off unless ``--judge`` is passed.

    The API key itself is never stored here: ``api_key_env`` names the environment
    variable (or ``.env`` entry) that holds it.
    """

    provider: str = "groq"
    model: str = "openai/gpt-oss-120b"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"


@dataclass(frozen=True)
class Config:
    permissions: Permissions = field(default_factory=Permissions)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    contact: Contact = field(default_factory=Contact)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


_SECTIONS: dict[str, type] = {
    "permissions": Permissions,
    "fetch": FetchConfig,
    "contact": Contact,
    "judge": JudgeConfig,
}


@dataclass(frozen=True)
class Decision:
    """What to do about a permission right now, and why, in words a report can print."""

    outcome: Outcome
    reason: str


def is_interactive() -> bool:
    """True only when both ends are a terminal. CI and pipes must never see a prompt."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def resolve_permission(value: Permission, *, interactive: bool) -> Decision:
    """Turn a stored setting into an outcome for this run."""
    if value == "allow":
        return Decision("allow", "permission is set to allow")
    if value == "deny":
        return Decision("deny", "permission is set to deny")
    if interactive:
        return Decision("prompt", "permission is set to ask")
    return Decision("deny", "permission is set to ask but there is no interactive terminal")


# --- loading -----------------------------------------------------------------


def _coerce(section: str, key: str, value: Any, expected: Any, path: Path) -> Any:
    where = f"{section}.{key}"
    if expected is Permission:
        if isinstance(value, str) and value in PERMISSION_VALUES:
            return value
        choices = ", ".join(PERMISSION_VALUES)
        raise ConfigError(f"{path}: {where} must be one of {choices}, got {value!r}")
    if expected is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: {where} must be true or false, got {value!r}")
    if expected is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: {where} must be a string, got {value!r}")
    raise ConfigError(f"{path}: {where} has an unsupported type")  # pragma: no cover


def _build_section(section: str, raw: Any, path: Path) -> Any:
    cls = _SECTIONS[section]
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: [{section}] must be a table")
    known = {f.name: f.type for f in fields(cls)}
    values: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            raise ConfigError(f"{path}: unknown key {section}.{key}")
        values[key] = _coerce(section, key, value, _resolve_type(known[key]), path)
    return cls(**values)


def _resolve_type(annotation: Any) -> Any:
    # ``from __future__ import annotations`` leaves field types as strings.
    if isinstance(annotation, str):
        return {"Permission": Permission, "bool": bool, "str": str}[annotation]
    return annotation


def load_config(path: Path | None = None) -> Config:
    """Read the config file, or return defaults when it does not exist."""
    path = path or config_path()
    if not path.exists():
        return Config()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: not valid TOML: {exc}") from exc
    sections: dict[str, Any] = {}
    for section, body in raw.items():
        if section not in _SECTIONS:
            raise ConfigError(f"{path}: unknown section [{section}]")
        sections[section] = _build_section(section, body, path)
    return Config(**sections)


# --- saving ------------------------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"cannot render {value!r} as TOML")  # pragma: no cover


def render_config(config: Config) -> str:
    lines = ["# proofpath configuration", "# permissions: ask | allow | deny", ""]
    for section in _SECTIONS:
        lines.append(f"[{section}]")
        body = getattr(config, section)
        for f in fields(body):
            lines.append(f"{f.name} = {_toml_value(getattr(body, f.name))}")
        lines.append("")
    return "\n".join(lines)


def save_config(config: Config, path: Path | None = None) -> Path:
    """Write the config file, creating its directory. Returns the path written."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config(config), encoding="utf-8")
    return path


def set_value(dotted_key: str, raw_value: str, path: Path | None = None) -> Config:
    """Update one ``section.key`` from a string, as typed on the command line."""
    path = path or config_path()
    section, _, key = dotted_key.partition(".")
    if section not in _SECTIONS or key not in {f.name for f in fields(_SECTIONS[section])}:
        raise ConfigError(f"unknown key {dotted_key}")
    expected = _resolve_type({f.name: f.type for f in fields(_SECTIONS[section])}[key])
    value: Any = raw_value
    if expected is bool:
        lowered = raw_value.strip().lower()
        if lowered not in {"true", "false"}:
            raise ConfigError(f"{dotted_key} must be true or false, got {raw_value!r}")
        value = lowered == "true"
    coerced = _coerce(section, key, value, expected, path)
    current = load_config(path)
    body = getattr(current, section)
    new_body = type(body)(**{**_as_dict(body), key: coerced})
    updated = Config(**{**_as_dict(current), section: new_body})
    save_config(updated, path)
    return updated


def _as_dict(obj: Any) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}
