"""The mirrored verbs (spec section 13.3): run the library function, render what it answered.

Everything here runs on a worker thread and returns lines. The calls go to
:mod:`proofpath.commands`, which is what ``cli.py`` calls, so the two surfaces cannot
drift about what a verb does -- only about how it is drawn, which is all these are.
"""

from __future__ import annotations

from rich.text import Text

from proofpath import commands as library
from proofpath import ui
from proofpath.browser import ConsentGate
from proofpath.cache import SourceDetail, SourceSummary
from proofpath.config import Config, ConfigError
from proofpath.fetch import STEP_NAMES, Fetched
from proofpath.judge import JudgeError
from proofpath.paths import config_path
from proofpath.report import BROWSER_SKIPPED_REASON
from proofpath.tui import commands
from proofpath.tui.widgets._shared import _link


def _kv(out: ui.Ui, key: str, value: str | Text, *, state: bool = False) -> Text:
    """``ui.kv``'s line, built as a ``Text`` instead of printed to a console."""
    if isinstance(value, str):
        value = ui.style_state(out, value) if state else Text(value)
    return Text.assemble((key.ljust(ui.KEY_WIDTH) + " ", "bold" if out.color else ""), value)


def error_line(out: ui.Ui, text: str) -> Text:
    """``ui.error``'s line, as a widget's ``Text``. Red is a meaning, and ``ui`` owns it."""
    return Text.assemble(("error: ", ui.LEVEL_STYLES["error"] if out.color else ""), text)


def verb_lines(
    command: commands.Command,
    *,
    out: ui.Ui,
    config: Config,
    prompt: library.PromptFn | None = None,
) -> list[Text]:
    """Run one mirrored verb and render its result. Worker thread; never prints."""
    verb, arg = command.verb, command.arg
    if verb == "resolve":
        return _resolve_lines(out, library.resolve_reference(arg, config=config))
    if verb == "fetch":
        fetched = library.fetch_target(
            arg,
            config=config,
            # The TUI is a terminal and has a prompt of its own, so ``ask`` really can
            # be asked here -- of the block, never of stdin (rule 4).
            interactive=True,
            prompt=prompt,
        )
        return _fetch_lines(out, fetched)
    if verb == "config":
        return _config_lines(out, arg)
    return _cache_lines(out, arg)


def _resolve_lines(out: ui.Ui, resolved: library.Resolved) -> list[Text]:
    """What ``proofpath resolve`` prints, line for line (spec section 13.3)."""
    result, retraction = resolved.result, resolved.retraction
    lines = [_kv(out, "state", result.state.value, state=True)]
    best = result.best
    if best is not None:
        url = ("https://doi.org/" + best.doi) if best.doi else best.url
        lines.append(_kv(out, "record", best.title))
        byline = f"{best.first_author} {best.year or '?'} · {best.venue or '—'}"
        lines.append(_kv(out, "", f"{byline} · via {best.provider}"))
        lines.append(_kv(out, "", _link(url, url)))
        if result.match is not None:
            match = result.match
            lines.append(
                _kv(
                    out,
                    "agreement",
                    f"title {match.title:.2f} · author {'yes' if match.author else 'no'} · "
                    f"year {'yes' if match.year else 'no'}",
                )
            )
        if best.doi and resolved.retraction_error is not None:
            # Nobody answered: neither "not retracted" nor a notice (rule 2).
            lines.append(
                _kv(
                    out,
                    "retraction",
                    Text.assemble(
                        ui.style_state(out, "unavailable"), f" ({resolved.retraction_error})"
                    ),
                )
            )
        elif best.doi and retraction is None:
            lines.append(
                _kv(
                    out,
                    "retraction",
                    Text.assemble(
                        ui.style_state(out, "not retracted"),
                        " (Crossref/Retraction Watch, OpenAlex)",
                    ),
                )
            )
        elif best.doi:
            assert retraction is not None
            lines.append(
                _kv(
                    out,
                    "retraction",
                    Text.assemble(
                        ui.style_state(out, "RETRACTED"),
                        f" {retraction.date or ''} — {retraction.source}",
                    ),
                )
            )
    elif result.candidates:
        lines.append(
            _kv(
                out, "checked", f"{len(result.candidates)} candidate(s), none agrees on the fields:"
            )
        )
        lines.extend(
            _kv(out, "", f"- {c.title[:70]} ({c.first_author} {c.year or '?'}, {c.provider})")
            for c in result.candidates[:5]
        )
    lines.extend(_kv(out, "note", note) for note in result.notes)
    return lines


def _fetch_lines(out: ui.Ui, fetched: library.FetchOutcome) -> list[Text]:
    """What ``proofpath fetch`` prints: the result, then the section 7.1 lines."""
    result = fetched.result
    lines: list[Text] = []
    if isinstance(result, Fetched):
        lines.append(_kv(out, "outcome", result.outcome.value, state=True))
        lines.append(_kv(out, "step", f"{result.step} ({STEP_NAMES[result.step]})"))
        lines.append(_kv(out, "status", str(result.status) if result.status is not None else "—"))
        lines.append(_kv(out, "type", f"{result.content_type or '—'} · {result.kind}"))
        lines.append(_kv(out, "words", str(result.words)))
        lines.append(_kv(out, "url", _link(result.final_url or "—", result.final_url)))
        lines.append(_kv(out, "cached", "yes" if result.from_cache else "no"))
        lines.extend(_kv(out, "note", note) for note in result.notes)
    else:
        if result.state:
            # abstract / none: the state word is the styled one, not the kind.
            lines.append(
                _kv(
                    out,
                    "evidence",
                    Text.assemble(f"{result.kind} — ", ui.style_state(out, result.state)),
                )
            )
        else:
            lines.append(_kv(out, "evidence", result.kind, state=True))
        lines.append(_kv(out, "source", result.source or "—"))
        lines.append(_kv(out, "words", str(result.words)))
        lines.append(_kv(out, "url", _link(result.url or "—", result.url)))
        lines.extend(
            _kv(
                out,
                "attempt",
                f"{attempt.location.label:<13} {attempt.outcome.value:<44} "
                f"step {attempt.step}   {attempt.words} words",
            )
            for attempt in result.attempts
        )
        lines.extend(_kv(out, "note", note) for note in result.notes)
    lines.extend(_gate_lines(out, fetched.gate))
    return lines


def _gate_lines(out: ui.Ui, gate: ConsentGate) -> list[Text]:
    """The section 7.1 permission lines: only when the browser step mattered."""
    lines: list[Text] = []
    if gate.consulted:
        lines.append(_kv(out, "browser", gate.decision.reason))
    if gate.skipped:
        number = Text(f"{gate.skipped} source(s)")
        if out.color:
            number.stylize(ui.BROWSER_SKIPPED_STYLE)
        lines.append(
            _kv(
                out,
                "skipped",
                Text.assemble(
                    number, f" {BROWSER_SKIPPED_REASON} — /config set {ui.BROWSER_SETTING}"
                ),
            )
        )
    lines.extend(_kv(out, "install", line) for line in gate.install_log)
    return lines


def _config_lines(out: ui.Ui, arg: str) -> list[Text]:
    """``/config``, ``/config path|show|set K V|check`` — the CLI group, mirrored."""
    sub, *rest = arg.split() or ["show"]
    if sub == "path":
        return [_kv(out, "config", str(config_path()))]
    if sub == "set":
        if len(rest) != 2:
            return [error_line(out, "/config set wants a key and a value: SECTION.KEY VALUE")]
        try:
            written = library.config_set(rest[0], rest[1])
        except (ConfigError, JudgeError) as exc:
            return [error_line(out, str(exc))]
        return [_kv(out, "", f"{key} = {value}  ({config_path()})") for key, value in written]
    if sub not in ("check", "show"):
        return [error_line(out, f"/config {sub}: try show, path, set or check")]
    try:
        view = library.config_view()
    except ConfigError as exc:
        # An unreadable config file is reported the way the CLI reports it, not raised
        # as a traceback in the block.
        return [error_line(out, str(exc))]
    if sub == "check":
        return _judge_lines(out, library.config_check(view.config))
    state = "" if view.exists else "  (not written yet, showing defaults)"
    lines = [_kv(out, "config", f"{view.path}{state}"), Text("")]
    lines.extend(Text(row) for row in view.toml.splitlines())
    return lines


def _judge_lines(out: ui.Ui, checked: library.JudgeCheck) -> list[Text]:
    """``/config check``: the provider that would be used, and whether it answered."""
    if checked.result is None:
        lines = [error_line(out, f"no API key found for {checked.judge.provider}.")]
        lines.append(_kv(out, "", f"Put a line like  {checked.judge.api_key_env}=...  in one of:"))
        lines.extend(_kv(out, "", f"  {path}") for path in checked.dotenv_paths)
        return lines
    lines = [
        _kv(out, "provider", checked.judge.provider),
        _kv(out, "model", checked.result.model),
        _kv(out, "key from", checked.key.source if checked.key else "(none needed)"),
    ]
    if checked.ok:
        word, rest = "ok", f"{checked.result.latency_ms} ms"
    else:
        word, rest = "FAILED", checked.result.detail
    lines.append(_kv(out, "status", Text.assemble(ui.style_state(out, word), f" {rest}")))
    return lines


def _cache_lines(out: ui.Ui, arg: str) -> list[Text]:
    """``/cache``, ``/cache path|ls|show ID|clear [--expired]`` — the CLI group."""
    sub, *rest = arg.split() or [""]
    if sub == "path":
        return [_kv(out, "cache", str(library.cache_file()))]
    if sub == "ls":
        listing = library.cache_list()
        if listing.empty:
            return [_kv(out, "cache", "cache is empty")]
        lines = [
            Text(f"{entry.source_id}  {_entry_line(entry, listing.now)}")
            for entry in listing.entries
        ]
        lines.append(
            Text(f"{listing.resolutions} resolution(s), {listing.retractions} retraction check(s)")
        )
        return lines
    if sub == "show":
        if not rest:
            return [error_line(out, "/cache show wants a source id, as listed by /cache ls")]
        detail = library.cache_detail(rest[0])
        if detail is None:
            return [error_line(out, f"no cached source {rest[0]!r}")]
        return _detail_lines(out, detail)
    if sub == "clear":
        expired = "--expired" in rest
        removed = library.cache_clear(expired=expired)
        return [
            _kv(
                out,
                "removed",
                f"{removed.sources} source(s), {removed.resolutions} resolution(s), "
                f"{removed.retractions} retraction check(s)"
                f"{' (expired only)' if expired else ''}",
            )
        ]
    if sub:
        return [error_line(out, f"/cache {sub}: try path, ls, show or clear")]
    held = library.cache_overview()
    return [
        _kv(out, "cache", str(held.path)),
        _kv(
            out, "holds", f"{held.sources} sources, {held.chunks} chunks, {held.verdicts} verdicts"
        ),
        _kv(
            out, "lookups", f"{held.resolutions} resolutions, {held.retractions} retraction checks"
        ),
    ]


def _entry_line(entry: SourceSummary, now: str) -> str:
    """One ``/cache ls`` row, expiry measured against the listing's single instant."""
    if entry.expires_at is None:
        expiry = "no raw text"
    elif entry.expires_at <= now:
        expiry = "raw text expired"
    else:
        expiry = f"raw text until {entry.expires_at[:10]}"
    return (
        f"{entry.title or '(untitled)'}  [{entry.scheme}/{entry.text_kind}]  "
        f"{entry.chunks} chunk(s), {entry.verdicts} verdict(s), {expiry}"
    )


def _detail_lines(out: ui.Ui, detail: SourceDetail) -> list[Text]:
    summary = detail.summary
    lines = [
        _kv(
            out,
            "source",
            f"{summary.source_id}  {summary.title or '(untitled)'}  "
            f"[{summary.scheme}/{summary.text_kind}]  fetched {summary.fetched_at[:19]}",
        )
    ]
    chunks, verdicts = detail.chunks, detail.verdicts
    lines.append(_kv(out, "chunks", str(len(chunks))))
    lines.extend(
        Text(f"  [{ordinal}] {text if text is not None else '(text expired)'}")
        for ordinal, text in chunks
    )
    lines.append(_kv(out, "verdicts", str(len(verdicts))))
    for verdict in verdicts:
        lines.append(
            Text.assemble(
                "  ",
                ui.style_state(out, verdict.label),
                f"  {verdict.tier:<6} {verdict.score:.2f}  claim {verdict.claim_hash[:12]}…",
            )
        )
        if verdict.passage_text is not None:
            lines.append(Text(f'    "{verdict.passage_text}"  [{verdict.passage_index}]'))
    return lines
