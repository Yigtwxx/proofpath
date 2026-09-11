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
from proofpath import config as cfg
from proofpath import judge as judge_mod
from proofpath.cache import Cache
from proofpath.paths import config_path

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


permissions_app = typer.Typer(
    name="permissions",
    help="Show or change what proofpath is allowed to do on this machine.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(permissions_app)


@permissions_app.callback()
def permissions(ctx: typer.Context) -> None:
    """Print the current permissions and where they are stored."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        current = cfg.load_config()
    except cfg.ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    path = config_path()
    state = "" if path.exists() else "  (not written yet, showing defaults)"
    typer.echo(f"config  {path}{state}")
    typer.echo("")
    typer.echo("[permissions]")
    typer.echo(f"install_browser = {current.permissions.install_browser}")
    typer.echo(f"network         = {current.permissions.network}")
    typer.echo("")
    typer.echo("[fetch]")
    typer.echo(f"respect_robots  = {str(current.fetch.respect_robots).lower()}")
    typer.echo("")
    typer.echo("[contact]")
    typer.echo(f"email           = {current.contact.email!r}")


@permissions_app.command("set")
def permissions_set(
    key: Annotated[str, typer.Argument(help="Permission name, e.g. install_browser.")],
    value: Annotated[str, typer.Argument(help="ask | allow | deny")],
) -> None:
    """Change one permission without opening the config file."""
    try:
        cfg.set_value(f"permissions.{key}", value)
    except cfg.ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    typer.echo(f"permissions.{key} = {value}  ({config_path()})")


judge_app = typer.Typer(
    name="judge",
    help="Optional LLM second opinion: show settings, change provider, test the key.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(judge_app)


def _judge_settings() -> tuple[cfg.JudgeConfig, judge_mod.ApiKey | None]:
    try:
        current = cfg.load_config().judge
    except cfg.ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    return current, judge_mod.resolve_api_key(current.api_key_env)


@judge_app.callback()
def judge(ctx: typer.Context) -> None:
    """Print the judge provider, model and where the API key is looked for."""
    if ctx.invoked_subcommand is not None:
        return
    current, key = _judge_settings()
    typer.echo(f"provider    {current.provider}")
    typer.echo(f"model       {current.model}")
    typer.echo(f"base_url    {current.base_url}")
    typer.echo(f"api_key     {current.api_key_env or '(none needed)'}")
    typer.echo(f"key found   {key.source if key else 'no'}")
    typer.echo("")
    typer.echo("The key is read from the environment variable, then from:")
    for path in judge_mod.default_dotenv_paths():
        typer.echo(f"  {path}")


@judge_app.command("check")
def judge_check() -> None:
    """Send one tiny request to prove the provider, model and key work."""
    current, key = _judge_settings()
    if current.api_key_env and key is None:
        typer.echo(f"no API key found for {current.provider}.", err=True)
        typer.echo(f"Put a line like  {current.api_key_env}=...  in one of:", err=True)
        for path in judge_mod.default_dotenv_paths():
            typer.echo(f"  {path}", err=True)
        typer.echo(f"or export {current.api_key_env} in your shell.", err=True)
        raise typer.Exit(EXIT_ERROR)
    result = judge_mod.check(current, key)
    typer.echo(f"provider    {current.provider}")
    typer.echo(f"model       {result.model}")
    typer.echo(f"key from    {key.source if key else '(none needed)'}")
    if result.ok:
        typer.echo(f"status      ok  {result.latency_ms} ms")
        return
    typer.echo(f"status      FAILED  {result.detail}", err=True)
    raise typer.Exit(EXIT_ERROR)


@judge_app.command("set")
def judge_set(
    key: Annotated[str, typer.Argument(help="provider | model | base_url | api_key_env")],
    value: Annotated[str, typer.Argument()],
) -> None:
    """Change one judge setting. Setting the provider also applies its defaults."""
    try:
        if key == "provider":
            defaults = judge_mod.provider_defaults(value)
            for name in ("provider", "model", "base_url", "api_key_env"):
                cfg.set_value(f"judge.{name}", getattr(defaults, name))
        else:
            cfg.set_value(f"judge.{key}", value)
    except (cfg.ConfigError, judge_mod.JudgeError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(EXIT_ERROR) from exc
    typer.echo(f"judge.{key} = {value}  ({config_path()})")


cache_app = typer.Typer(
    name="cache",
    help="Inspect or clear the local cache (one SQLite file; any SQLite GUI can open it).",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(cache_app)


@cache_app.callback()
def cache(ctx: typer.Context) -> None:
    """Show where the cache lives and how much it holds."""
    if ctx.invoked_subcommand is not None:
        return
    with Cache() as db:
        entries = db.summary()
        typer.echo(f"cache     {db.path}")
        typer.echo(
            f"holds     {len(entries)} sources, {sum(e.chunks for e in entries)} chunks, "
            f"{sum(e.verdicts for e in entries)} verdicts"
        )
        typer.echo("open it with DB Browser for SQLite, TablePlus or DBeaver — plain tables.")


@cache_app.command("path")
def cache_path() -> None:
    """Print the SQLite file path, nothing else (pipeable)."""
    with Cache() as db:
        typer.echo(str(db.path))


@cache_app.command("ls")
def cache_ls() -> None:
    """List cached sources with chunk and verdict counts and text expiry."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with Cache() as db:
        entries = db.summary()
    if not entries:
        typer.echo("cache is empty")
        return
    for e in entries:
        if e.expires_at is None:
            expiry = "no raw text"
        elif e.expires_at <= now:
            expiry = "raw text expired"
        else:
            expiry = f"raw text until {e.expires_at[:10]}"
        typer.echo(
            f"{e.source_id}  {e.title or '(untitled)'}  [{e.scheme}/{e.text_kind}]  "
            f"{e.chunks} chunk(s), {e.verdicts} verdict(s), {expiry}"
        )


@cache_app.command("show")
def cache_show(
    source_id: Annotated[str, typer.Argument(help="Source id, as listed by ls.")],
) -> None:
    """Print a source's chunks and verdicts."""
    with Cache() as db:
        detail = db.detail(source_id)
    if detail is None:
        typer.echo(f"no cached source {source_id!r}", err=True)
        raise typer.Exit(EXIT_ERROR)
    s = detail.summary
    typer.echo(
        f"{s.source_id}  {s.title or '(untitled)'}  [{s.scheme}/{s.text_kind}]  "
        f"fetched {s.fetched_at[:19]}"
    )
    if detail.embed_models:
        typer.echo(f"embeddings  {', '.join(detail.embed_models)}  dim={detail.dim}")
    typer.echo("")
    typer.echo(f"chunks ({len(detail.chunks)})")
    for ordinal, text in detail.chunks:
        typer.echo(f"  [{ordinal}] {text if text is not None else '(text expired)'}")
    typer.echo("")
    typer.echo(f"verdicts ({len(detail.verdicts)})")
    for v in detail.verdicts:
        typer.echo(
            f"  {v.label:<9} {v.tier:<6} {v.score:.2f}  claim {v.claim_hash[:12]}…  "
            f"model {v.model_id}"
        )
        if v.passage_text is not None:
            typer.echo(f'            "{v.passage_text}"  [{v.passage_index}]')
        if v.reason:
            typer.echo(f"            {v.reason}")


@cache_app.command("clear")
def cache_clear(
    expired: Annotated[
        bool, typer.Option("--expired", help="Only sources whose raw text expired.")
    ] = False,
) -> None:
    """Delete cached sources with their text, chunks and verdicts."""
    with Cache() as db:
        removed = db.clear(expired_only=expired)
    typer.echo(f"removed {removed} source(s){' (expired only)' if expired else ''}")


@app.command()
def check(
    target: Annotated[Path, typer.Argument(help="Document to check.")],
) -> None:
    """Verify every citation in a document and write a report."""
    typer.echo(_NOT_BUILT_YET, err=True)
    raise typer.Exit(EXIT_ERROR)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
