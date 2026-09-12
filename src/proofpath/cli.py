"""Command-line entry point.

The TUI and the one-shot commands are two front-ends over a single ``verify()``
call. Neither holds logic of its own; see the design spec, section 13.
"""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer

from proofpath import __version__, events, ingest, ui
from proofpath import config as cfg
from proofpath import fetch as fetch_mod
from proofpath import judge as judge_mod
from proofpath import oa as oa_mod
from proofpath import report as report_mod
from proofpath import resolve as resolve_mod
from proofpath import verify as verify_mod
from proofpath.browser import ConsentGate
from proofpath.cache import Cache
from proofpath.paths import config_path
from proofpath.polite import PoliteClient

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

# The bare-invocation line, until Phase 8 opens the TUI here instead.
_NO_TUI_YET = (
    f"proofpath {__version__} — the interactive TUI arrives in v0.2; try: proofpath check paper.pdf"
)

# Where ``check`` writes its markdown when ``--out`` is not given (text mode only).
DEFAULT_REPORT = Path("report.md")


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
    no_color: Annotated[
        bool, typer.Option("--no-color", help="Plain text, no colour (NO_COLOR also works).")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Drop stage and note lines; findings remain.")
    ] = False,
) -> None:
    """Launch the interactive TUI when called with no subcommand."""
    ctx.obj = ui.build(no_color=no_color, quiet=quiet)
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(_NO_TUI_YET)
    raise typer.Exit(EXIT_CLEAN)


def _ui(ctx: typer.Context) -> ui.Ui:
    found = ctx.find_object(ui.Ui)
    assert found is not None, "Ui not built by the app callback"
    return found


def _fail(out: ui.Ui, exc: Exception) -> typer.Exit:
    ui.error(out, str(exc))
    return typer.Exit(EXIT_ERROR)


def _reject_sarif(out: ui.Ui, fmt: Format) -> None:
    """``sarif`` parses wherever ``Format`` does, but only ``check`` will ever emit it,
    and not before Phase 8. Every command says so rather than quietly printing text."""
    if fmt is Format.SARIF:
        ui.error(out, "--format sarif arrives in v0.2")
        raise typer.Exit(EXIT_ERROR)


def _load_config(out: ui.Ui) -> cfg.Config:
    try:
        return cfg.load_config()
    except cfg.ConfigError as exc:
        raise _fail(out, exc) from exc


config_app = typer.Typer(
    name="config",
    help="Show or change settings: permissions, contact address, judge provider.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(config_app)


@config_app.callback()
def config(ctx: typer.Context) -> None:
    """Print the config path and every section as TOML (same as ``config show``)."""
    if ctx.invoked_subcommand is None:
        config_show(ctx)


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the config path and every section as TOML."""
    out = _ui(ctx)
    current = _load_config(out)
    path = config_path()
    state = "" if path.exists() else "  (not written yet, showing defaults)"
    ui.kv(out, "config", f"{path}{state}")
    ui.blank(out)
    out.out.print(cfg.render_config(current))


@config_app.command("path")
def config_path_cmd(ctx: typer.Context) -> None:
    """Print the config file path, nothing else (pipeable)."""
    _ui(ctx).out.print(str(config_path()))


_JUDGE_PRESET_FIELDS = ("provider", "model", "base_url", "api_key_env")


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="SECTION.KEY, e.g. permissions.install_browser")],
    value: Annotated[str, typer.Argument(help="New value, as it would appear in the file.")],
) -> None:
    """Change one setting without opening the config file.

    ``judge.provider`` is special: it also applies that provider's model,
    base URL and API key env var, so switching provider does not leave a
    stale model/base_url pointed at the old one.
    """
    out = _ui(ctx)
    if key == "judge.provider":
        try:
            preset = judge_mod.provider_defaults(value)
        except judge_mod.JudgeError as exc:
            raise _fail(out, exc) from exc
        try:
            for name in _JUDGE_PRESET_FIELDS:
                cfg.set_value(f"judge.{name}", getattr(preset, name))
        except cfg.ConfigError as exc:
            raise _fail(out, exc) from exc
        for name in _JUDGE_PRESET_FIELDS:
            out.out.print(f"judge.{name} = {getattr(preset, name)}  ({config_path()})")
        return
    try:
        cfg.set_value(key, value)
    except cfg.ConfigError as exc:
        raise _fail(out, exc) from exc
    out.out.print(f"{key} = {value}  ({config_path()})")


@config_app.command("check")
def config_check(ctx: typer.Context) -> None:
    """Send one tiny request to prove the judge provider, model and key work."""
    out = _ui(ctx)
    current = _load_config(out).judge
    key = judge_mod.resolve_api_key(current.api_key_env)
    if current.api_key_env and key is None:
        ui.error(out, f"no API key found for {current.provider}.")
        out.err.print(f"Put a line like  {current.api_key_env}=...  in one of:")
        for path in judge_mod.default_dotenv_paths():
            out.err.print(f"  {path}")
        out.err.print(f"or export {current.api_key_env} in your shell.")
        raise typer.Exit(EXIT_ERROR)
    result = judge_mod.check(current, key)
    ui.kv(out, "provider", current.provider)
    ui.kv(out, "model", result.model)
    ui.kv(out, "key from", key.source if key else "(none needed)")
    if result.ok:
        ui.state_line(out, "status", "ok", f"{result.latency_ms} ms")
        return
    ui.state_line(out, "status", "FAILED", result.detail)
    raise typer.Exit(EXIT_ERROR)


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
    out = _ui(ctx)
    with Cache() as db:
        entries = db.summary()
        ui.kv(out, "cache", str(db.path))
        ui.kv(
            out,
            "holds",
            f"{len(entries)} sources, {sum(e.chunks for e in entries)} chunks, "
            f"{sum(e.verdicts for e in entries)} verdicts",
        )
        ui.hint(out, "open it with DB Browser for SQLite, TablePlus or DBeaver — plain tables.")


@cache_app.command("path")
def cache_path(ctx: typer.Context) -> None:
    """Print the SQLite file path, nothing else (pipeable)."""
    with Cache() as db:
        _ui(ctx).out.print(str(db.path))


@cache_app.command("ls")
def cache_ls(ctx: typer.Context) -> None:
    """List cached sources with chunk and verdict counts and text expiry."""
    from datetime import datetime, timezone

    out = _ui(ctx).out
    now = datetime.now(timezone.utc).isoformat()
    with Cache() as db:
        entries = db.summary()
    if not entries:
        out.print("cache is empty")
        return
    for e in entries:
        if e.expires_at is None:
            expiry = "no raw text"
        elif e.expires_at <= now:
            expiry = "raw text expired"
        else:
            expiry = f"raw text until {e.expires_at[:10]}"
        out.print(
            f"{e.source_id}  {e.title or '(untitled)'}  [{e.scheme}/{e.text_kind}]  "
            f"{e.chunks} chunk(s), {e.verdicts} verdict(s), {expiry}"
        )


@cache_app.command("show")
def cache_show(
    ctx: typer.Context,
    source_id: Annotated[str, typer.Argument(help="Source id, as listed by ls.")],
) -> None:
    """Print a source's chunks and verdicts."""
    with Cache() as db:
        detail = db.detail(source_id)
    if detail is None:
        ui.error(_ui(ctx), f"no cached source {source_id!r}")
        raise typer.Exit(EXIT_ERROR)
    out = _ui(ctx).out
    s = detail.summary
    out.print(
        f"{s.source_id}  {s.title or '(untitled)'}  [{s.scheme}/{s.text_kind}]  "
        f"fetched {s.fetched_at[:19]}"
    )
    if detail.embed_models:
        out.print(f"embeddings  {', '.join(detail.embed_models)}  dim={detail.dim}")
    out.print("")
    out.print(f"chunks ({len(detail.chunks)})")
    for ordinal, text in detail.chunks:
        out.print(f"  [{ordinal}] {text if text is not None else '(text expired)'}")
    out.print("")
    out.print(f"verdicts ({len(detail.verdicts)})")
    for v in detail.verdicts:
        out.print(
            f"  {v.label:<9} {v.tier:<6} {v.score:.2f}  claim {v.claim_hash[:12]}…  "
            f"model {v.model_id}"
        )
        if v.passage_text is not None:
            out.print(f'            "{v.passage_text}"  [{v.passage_index}]')
        if v.reason:
            out.print(f"            {v.reason}")


@cache_app.command("clear")
def cache_clear(
    ctx: typer.Context,
    expired: Annotated[
        bool, typer.Option("--expired", help="Only sources whose raw text expired.")
    ] = False,
) -> None:
    """Delete cached sources with their text, chunks and verdicts."""
    with Cache() as db:
        removed = db.clear(expired_only=expired)
    _ui(ctx).out.print(f"removed {removed} source(s){' (expired only)' if expired else ''}")


class Format(str, Enum):
    TEXT = "text"
    JSON = "json"
    SARIF = "sarif"  # ``check`` only, and Phase 8 fills it in (spec section 13.2)


@app.command()
def resolve(
    ctx: typer.Context,
    reference: Annotated[
        str, typer.Argument(help="One reference string, as it appears in a bibliography.")
    ],
    fmt: Annotated[Format, typer.Option("--format", help="Output format.")] = Format.TEXT,
) -> None:
    """Check whether a cited reference exists (Crossref, Semantic Scholar, arXiv, OpenAlex)."""
    out = _ui(ctx)
    _reject_sarif(out, fmt)
    contact = _load_config(out).contact.email
    resolver = resolve_mod.Resolver(contact_email=contact)
    result = resolver.resolve(reference)
    best = result.best
    retraction = resolver.retraction(best.doi) if best is not None and best.doi else None

    if fmt is Format.JSON:
        ui.emit_json(out, {"result": result, "retraction": retraction})
    else:
        _print_resolve(out, result, retraction)

    if result.state is resolve_mod.State.RESOLVED and retraction is None:
        return
    # Every other state is a finding, including "provider unavailable" (spec 13.3),
    # and so is a resolved but retracted source (spec 13.2 ``warning[retracted]``).
    raise typer.Exit(EXIT_FINDINGS)


def _print_resolve(
    out: ui.Ui, result: resolve_mod.ResolveResult, retraction: resolve_mod.Retraction | None
) -> None:
    ui.kv(out, "state", result.state.value, state=True)
    best = result.best
    if best is not None:
        ui.kv(out, "record", best.title)
        ui.kv(
            out,
            "",
            f"{best.first_author} {best.year or '?'} · {best.venue or '—'} · via {best.provider}",
        )
        ui.kv(out, "", ("https://doi.org/" + best.doi) if best.doi else best.url)
        if result.match is not None:
            m = result.match
            ui.kv(
                out,
                "agreement",
                f"title {m.title:.2f} · author {'yes' if m.author else 'no'} · "
                f"year {'yes' if m.year else 'no'}",
            )
        if best.doi:
            if retraction is None:
                ui.state_line(
                    out, "retraction", "not retracted", "(Crossref/Retraction Watch, OpenAlex)"
                )
            else:
                ui.state_line(
                    out,
                    "retraction",
                    "RETRACTED",
                    f"{retraction.date or ''} — {retraction.source}",
                )
    elif result.candidates:
        ui.kv(out, "checked", f"{len(result.candidates)} candidate(s), none agrees on the fields:")
        for c in result.candidates[:5]:
            ui.kv(out, "", f"- {c.title[:70]} ({c.first_author} {c.year or '?'}, {c.provider})")
    for note in result.notes:
        ui.note(out, note)


@app.command()
def fetch(
    ctx: typer.Context,
    target: Annotated[
        str, typer.Argument(help="An http(s) URL, a DOI, or an arXiv id (arXiv:2103.00020).")
    ],
    fmt: Annotated[Format, typer.Option("--format", help="Output format.")] = Format.TEXT,
    allow_browser: Annotated[
        bool, typer.Option("--allow-browser", help="Permit the browser step for this run.")
    ] = False,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Refuse the browser step for this run.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Neither read nor fill the local cache.")
    ] = False,
    show: Annotated[
        int, typer.Option("--show", help="Print the first N characters of the text.")
    ] = 0,
) -> None:
    """Fetch one source through the ladder (URL) or the open-access chain (DOI, arXiv)."""
    out = _ui(ctx)
    _reject_sarif(out, fmt)
    config = _load_config(out)
    if allow_browser and no_browser:
        ui.error(out, "--allow-browser and --no-browser cannot be combined.")
        raise typer.Exit(EXIT_ERROR)
    override = True if allow_browser else False if no_browser else None

    is_url = target.startswith(("http://", "https://"))
    doi = arxiv_id = None
    if not is_url:
        doi = resolve_mod.find_doi(target)
        arxiv_id = None if doi else resolve_mod.find_arxiv_id(target)
        if doi is None and arxiv_id is None:
            ui.error(out, f"not a URL, DOI or arXiv id: {target!r}")
            raise typer.Exit(EXIT_ERROR)

    interactive = cfg.is_interactive()
    gate = ConsentGate(
        config.permissions.install_browser, interactive=interactive, override=override
    )
    result: fetch_mod.Fetched | oa_mod.Evidence
    with ExitStack() as stack:
        cache = None if no_cache else stack.enter_context(Cache())
        fetcher = fetch_mod.Fetcher(config=config, gate=gate, cache=cache, interactive=interactive)
        stack.callback(fetcher.close)
        if is_url:
            result = fetcher.fetch(target)
        else:
            client = PoliteClient(contact_email=config.contact.email)
            stack.callback(client.client.close)
            chain = oa_mod.OpenAccess(
                fetcher, client, contact_email=config.contact.email, cache=cache
            )
            result = chain.fetch(doi, arxiv_id)
        stats = fetcher.summary()

    if fmt is Format.JSON:
        if show > 0:
            ui.hint(out, "--show is ignored under --format json", err=True)
        browser = {
            "decision": gate.decision,
            "skipped": gate.skipped,
            "skipped_urls": gate.skipped_urls,
            "install_log": gate.install_log,
        }
        ui.emit_json(out, {"target": target, "result": result, "stats": stats, "browser": browser})
    else:
        if isinstance(result, fetch_mod.Fetched):
            _print_fetched(out, result)
        else:
            _print_evidence(out, result)
        _print_gate(out, gate)
        if show > 0:
            ui.kv(out, "text", result.text[:show])

    if isinstance(result, fetch_mod.Fetched):
        clean = result.ok and result.words > 0  # reached with nothing to read is not clean
    else:
        clean = result.kind == "fulltext"
    if clean:
        return
    # Abstract-only and every UNVERIFIED state are findings (spec 13.3).
    raise typer.Exit(EXIT_FINDINGS)


def _print_fetched(out: ui.Ui, result: fetch_mod.Fetched) -> None:
    ui.kv(out, "outcome", result.outcome.value, state=True)
    ui.kv(out, "step", f"{result.step} ({fetch_mod.STEP_NAMES[result.step]})")
    ui.kv(out, "status", str(result.status) if result.status is not None else "—")
    ui.kv(out, "type", f"{result.content_type or '—'} · {result.kind}")
    ui.kv(out, "words", str(result.words))
    ui.kv(out, "url", result.final_url or "—")
    ui.kv(out, "cached", "yes" if result.from_cache else "no")
    for note in result.notes:
        ui.note(out, note)


def _print_evidence(out: ui.Ui, result: oa_mod.Evidence) -> None:
    if result.state:
        # abstract / none: the state word is the styled one, not the kind.
        ui.state_line(out, "evidence", result.state, prefix=f"{result.kind} — ")
    else:
        ui.state_line(out, "evidence", result.kind)  # fulltext: kind is the state word.
    ui.kv(out, "source", result.source or "—")
    ui.kv(out, "words", str(result.words))
    ui.kv(out, "url", result.url or "—")
    for attempt in result.attempts:
        ui.note(
            out,
            f"{attempt.location.label:<13} {attempt.outcome.value:<44} "
            f"step {attempt.step}   {attempt.words} words",
            key="attempt",
        )
    for note in result.notes:
        ui.note(out, note)


def _print_gate(out: ui.Ui, gate: ConsentGate) -> None:
    """The section 7.1 permission lines: only when the browser step mattered."""
    if gate.consulted:
        ui.kv(out, "browser", gate.decision.reason)
    if gate.skipped:
        ui.kv(out, "skipped", f"{gate.skipped} source(s) because the browser was not permitted")
    for line in gate.install_log:
        ui.kv(out, "install", line)


@app.command()
def check(
    ctx: typer.Context,
    target: Annotated[str, typer.Argument(help="Document to check, or - to read it from stdin.")],
    fmt: Annotated[Format, typer.Option("--format", help="Output format.")] = Format.TEXT,
    judge: Annotated[
        bool, typer.Option("--judge", help="Ask an LLM about the claims the models left open.")
    ] = False,
    summarize: Annotated[
        bool, typer.Option("--summarize", help="Add a model-written summary to the report.")
    ] = False,
    allow_browser: Annotated[
        bool, typer.Option("--allow-browser", help="Permit the browser step for this run.")
    ] = False,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Refuse the browser step for this run.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Neither read nor fill the local cache.")
    ] = False,
    out_path: Annotated[
        Path | None,
        typer.Option("--out", help="Markdown report path (default report.md; text mode)."),
    ] = None,
) -> None:
    """Verify every citation in a document and write a report."""
    out = _ui(ctx)
    if allow_browser and no_browser:
        ui.error(out, "--allow-browser and --no-browser cannot be combined.")
        raise typer.Exit(EXIT_ERROR)
    _reject_sarif(out, fmt)
    for flag, asked in (("--judge", judge), ("--summarize", summarize)):
        if asked:
            ui.error(out, f"{flag} arrives in v0.3")
            raise typer.Exit(EXIT_ERROR)

    source, name = _check_target(out, target)
    config = _load_config(out)
    override = True if allow_browser else False if no_browser else None
    json_out = fmt is Format.JSON
    # Under --format json stdout carries one document and nothing else, so the run's
    # human lines are printed against stderr instead (spec section 13.3).
    human = replace(out, out=out.err) if json_out else out

    def on_event(event: events.Event) -> None:
        # ``Emitted`` is not printed here: findings are shown at the end, in document
        # order. ``Progress`` has no bar in v0.1 — the TUI (Phase 8) draws one.
        if isinstance(event, events.StageEnd):
            ui.stage_row(human, event.name, event.by, event.summary, event.elapsed)
        elif isinstance(event, events.Note):
            ui.note(human, event.text)

    cancel = threading.Event()
    try:
        with (
            verify_mod.Engine.default(
                config,
                interactive=cfg.is_interactive(),  # rule 4: never sniffed further down
                browser=override,
                no_cache=no_cache,
            ) as engine,
            _interruptible(cancel),
        ):
            report = verify_mod.verify(source, engine, name=name, on_event=on_event, cancel=cancel)
    except (KeyboardInterrupt, events.Cancelled) as exc:
        # A stopped run is incomplete, not wrong (product rule 6): whatever it did
        # decide is still printed, and the footer says the run was cancelled. A Ctrl-C
        # the handler below could not turn into a cancel, or one taken in the I/O half,
        # leaves nothing to print, and then the line is all there is.
        cancel.set()
        partial = exc.report if isinstance(exc, events.Cancelled) else None
        if partial is None:
            ui.error(out, "cancelled")
            raise typer.Exit(EXIT_ERROR) from None
        report = partial
    except (ingest.IngestError, cfg.ConfigError, OSError) as exc:
        raise _fail(out, exc) from exc
    except typer.Exit:
        raise  # a deliberate exit from inside the run keeps its own code
    except Exception as exc:
        # Exit 1 is a statement about the document: it has findings. A provider, a
        # model or a parser failing in a way nobody foresaw must not be able to make
        # that statement, so anything unforeseen is reported as the tool failing (2).
        raise _fail(out, exc) from exc

    written = _write_report(out, report, out_path, default=not json_out)
    if json_out:
        ui.emit_json(out, report)
    else:
        ui.blank(out)
        for item in report_mod.render_diagnostics(report):
            ui.diagnostic(out, item)
        ui.blank(out)
        ui.footer(out, report_mod.render_footer(report, written=written))
    # Everything the run had to say is out, and the engine — with the ONNX sessions
    # it owned — was closed on the way out of the ``with`` block above. Flushing here
    # means the process has nothing left to do but exit, whatever the interpreter's
    # own teardown gets up to afterwards (spec section 13.3).
    _flush_streams()
    # A cancelled run is not a verdict on the document: 2 says the tool stopped early.
    raise typer.Exit(EXIT_ERROR if report.cancelled else report.exit_code())


def _flush_streams() -> None:
    """Push stdout and stderr out before the command exits.

    A run whose report is still sitting in a buffer has not finished reporting, and
    an exit code is worth nothing beside a half-written page. A stream that is
    already closed (``… | head``) is not an error at this point: the run is over.
    """
    for stream in (sys.stdout, sys.stderr):
        with suppress(ValueError, OSError):
            stream.flush()


def _check_target(out: ui.Ui, target: str) -> tuple[Path | str, str | None]:
    """The document and the name its locations carry.

    ``-`` is the document itself, arriving on stdin and named for where it came from;
    anything else is a file that has to exist, and a file names itself.
    """
    if target == "-":
        # Decoded here rather than by ``sys.stdin``, whose encoding is the locale's:
        # on Windows that is the ANSI code page, which turns a UTF-8 paper into
        # mojibake. Undecodable bytes become U+FFFD instead of ending the run.
        return sys.stdin.buffer.read().decode("utf-8", errors="replace"), "stdin"
    path = Path(target)
    if not path.exists():
        ui.error(out, f"no such file: {path}")
        raise typer.Exit(EXIT_ERROR)
    if not path.is_file():
        ui.error(out, f"not a file: {path}")
        raise typer.Exit(EXIT_ERROR)
    return path, None


@contextmanager
def _interruptible(cancel: threading.Event) -> Iterator[None]:
    """Turn Ctrl-C into the run's cancel event for the duration of the block.

    The pipeline checks ``cancel`` between units of work, so a run stopped this way
    finishes the claim it is on, keeps the verdicts it reached and hands them over in
    the report the ``Cancelled`` carries -- where a bare ``KeyboardInterrupt`` would
    have dropped all of it. ``signal.signal`` only works on the main thread; anywhere
    else (a TUI worker, an embedding host) the handler is skipped and the
    ``KeyboardInterrupt`` the caller sees is handled as before.

    A second Ctrl-C is the user saying the wait is over: the previous handler goes
    back and the ``KeyboardInterrupt`` is let through, so a stage stuck in a socket
    read cannot hold the terminal hostage. Whatever the run had decided is gone with
    it, which is the trade the second press asks for.
    """
    previous = signal.getsignal(signal.SIGINT)

    def stop(signum: int, frame: FrameType | None) -> None:
        if cancel.is_set():
            signal.signal(signal.SIGINT, previous)
            raise KeyboardInterrupt
        cancel.set()

    try:
        signal.signal(signal.SIGINT, stop)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _write_report(
    out: ui.Ui, report: report_mod.Report, path: Path | None, *, default: bool
) -> str | None:
    """Write the markdown report and return the path the footer should name.

    ``--format json`` already puts the whole report on stdout, so it writes a file
    only when asked for one; text mode always leaves one behind.
    """
    if path is None:
        if not default:
            return None
        path = DEFAULT_REPORT
    try:
        path.write_text(report_mod.render_markdown(report), encoding="utf-8")
    except OSError as exc:
        raise _fail(out, exc) from exc
    return str(path)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
