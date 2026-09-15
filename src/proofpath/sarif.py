"""One report as a SARIF 2.1.0 log, for editors that show problems inline.

Spec sections 13 and 13.3: ``proofpath check draft.md --format sarif`` writes the run
in the format VS Code's Problems pane, GitHub code scanning and every other static
analysis viewer already read. This module turns a :class:`~proofpath.report.Report`
into that document and does nothing else: no file is opened, no path is resolved
against the working directory, and the same report always produces the same dict.

Three things the format does not enforce but the product does:

- A result that *asserts* something about a source — ``not-supported``,
  ``numeric-mismatch`` — puts the quoted passage in its message, because a viewer
  shows the message and nothing else (rule 1). The passage is already mandatory on
  those findings; here it has to survive the trip to the message line.
- An ``UNVERIFIED`` result carries the exact state it was given. "UNVERIFIED
  (blocked, robots.txt)" reaches the Problems pane in those words rather than as a
  generic warning, so a source that could not be read never reads like a source that
  was read and found wanting (rule 2).
- ``runs[0].properties`` carries the run's coverage, so a log opened without the
  terminal output still says how much of the bibliography was actually read (rule 6).

The SARIF ``level`` vocabulary — ``error``, ``warning``, ``note`` — is the one
:data:`~proofpath.report.LEVELS` already uses, so severity is taken from that table
rather than mapped here: there is no second opinion about how serious a kind is.
"""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import Any
from urllib.parse import quote

from proofpath import __version__

# ``_one_line`` is private to ``report`` but it is *the* definition of flattening a
# quoted passage onto one line, and the markdown renderer already uses it. Reusing it
# keeps SARIF and markdown quoting the same passage the same way.
from proofpath.report import LEVELS, Finding, Kind, Report, _one_line

SARIF_VERSION = "2.1.0"
SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"

TOOL_NAME = "proofpath"
INFORMATION_URI = "https://github.com/Yigtwxx/proofpath"
# Spec section 15 lists every honesty state a finding can carry; a reader who wants to
# know what a rule means is sent there rather than to a second, drifting copy of it.
HELP_URI = (
    "https://github.com/Yigtwxx/proofpath/blob/main/"
    "docs/superpowers/specs/2026-09-10-proofpath-design.md"
    "#15-error-handling-and-honesty-states"
)

# The base id a relative artifact path is resolved against. SARIF has no fixed
# vocabulary for these; ``%SRCROOT%`` is what the GitHub and VS Code tooling expects,
# and an absolute path gets none at all — it is already anchored.
SRCROOT = "%SRCROOT%"
# Absolute paths are emitted as file URIs; see ``_artifact_location``.
FILE_SCHEME = "file:///"
# What ``quote`` leaves alone in a path: the separators, and the colon of a Windows
# drive (``C:/``). Everything else a file name may hold -- a space, a ``#``, a ``?``
# -- would be read as URI syntax rather than as part of the name.
URI_SAFE = "/:"
SOURCE_ROOT_DESCRIPTION = "the directory proofpath was run from"

# What each kind means, in one line. These are rule descriptions and not finding
# titles: a rule describes the family, a finding describes the one case. The wording
# stays close to spec section 15 so a viewer's tooltip and the report agree.
RULE_DESCRIPTIONS: dict[Kind, str] = {
    Kind.GHOST: "the cited source does not exist",
    Kind.AMBIGUOUS: "the reference matches more than one record, so none was chosen",
    Kind.RETRACTED: "the cited source has been retracted",
    Kind.NUMERIC_MISMATCH: "a figure in the claim contradicts the figure in the source",
    Kind.NOT_SUPPORTED: "the cited source does not say what the claim says",
    Kind.NEI: "the source neither supports nor contradicts the claim",
    Kind.UNVERIFIED: "the source could not be read, so nothing was concluded from it",
    Kind.ABSTRACT_ONLY: "only the abstract was available, not the full text",
    Kind.PARAGRAPH_SCOPED: "the citation supports a paragraph, not one sentence",
    Kind.UNSUPPORTED_STYLE: "the citation style is not paired in this version",
    Kind.UNRESOLVED_MARKER: "the citation marker points at no bibliography entry",
    Kind.PARSE_ERROR: "part of the document could not be parsed",
    Kind.PROVIDER_UNAVAILABLE: "a bibliographic provider could not be consulted",
}


def to_sarif(report: Report, *, artifact: Path | str) -> dict[str, Any]:
    """One run as a SARIF 2.1.0 log document, ready for ``ui.emit_json``.

    ``artifact`` is the file the run was about, as the caller named it. It is spelled
    with forward slashes whichever platform produced it, so a log written on Windows
    opens in a viewer anywhere; a relative path is anchored to ``%SRCROOT%`` and an
    absolute one is left to stand on its own.

    Every value in the returned dict is a plain JSON type, so no custom encoder is
    needed to serialise it.
    """
    location = _artifact_location(artifact)
    kinds = list(report.counts())  # in ``Kind`` declaration order, present ones only
    index_of = {kind: index for index, kind in enumerate(kinds)}
    findings = [item for group in report.by_level().values() for item in group]
    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": TOOL_NAME,
                "version": __version__,
                "informationUri": INFORMATION_URI,
                "rules": [_rule(kind) for kind in kinds],
            }
        },
        "invocations": [{"executionSuccessful": not report.cancelled}],
        "results": [_result(item, index_of[item.kind], location) for item in findings],
        "properties": _run_properties(report),
    }
    # Declared only when something actually points at it: a base id no uri mentions is
    # a promise about a directory this function never looked at.
    if findings and location.get("uriBaseId") == SRCROOT:
        run["originalUriBaseIds"] = {SRCROOT: _source_root()}
    return {"$schema": SCHEMA_URI, "version": SARIF_VERSION, "runs": [run]}


def _source_root() -> dict[str, Any]:
    """What ``%SRCROOT%`` is, for a viewer opening the log somewhere else.

    The entry carries a description and no uri: naming one would mean reading the
    working directory, and this module reads nothing. ``originalUriBaseIds``
    entries may omit ``uri``, and a viewer then asks where the root is.
    """
    return {"description": {"text": SOURCE_ROOT_DESCRIPTION}}


def _rule(kind: Kind) -> dict[str, Any]:
    """One kind as a reporting descriptor: the id a result points back at."""
    return {
        "id": kind.value,
        "name": _camel_case(kind.value),
        "shortDescription": {"text": RULE_DESCRIPTIONS[kind]},
        "defaultConfiguration": {"level": LEVELS[kind]},
        "helpUri": HELP_URI,
    }


def _camel_case(value: str) -> str:
    """``ghost-reference`` -> ``GhostReference``, the shape SARIF rule names take."""
    return "".join(part.capitalize() for part in value.split("-"))


def _result(item: Finding, rule_index: int, location: dict[str, Any]) -> dict[str, Any]:
    return {
        "ruleId": item.kind.value,
        "ruleIndex": rule_index,
        "level": item.level,
        "message": {"text": _message(item)},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": location,
                    # The page is deliberately not folded in here: a viewer opens a
                    # text file at a line, and a PDF at neither, so the page travels
                    # in the properties below instead of distorting the region.
                    "region": {
                        "startLine": item.locator.line,
                        "startColumn": item.locator.column,
                    },
                }
            }
        ],
        "properties": {
            "page": item.locator.page,
            "state": item.state,
            "tier": item.tier,
            "sourceId": item.source_id,
            "fetchStep": item.fetch_step,
            "group": item.group,
            "reference": None if item.reference is None else item.reference.raw,
            "claim": None if item.claim is None else item.claim.text,
        },
    }


def _message(item: Finding) -> str:
    """The one line a viewer shows: the title, the evidence, and the honesty state.

    Rules 1 and 2 both land here, because a viewer that shows only this line is the
    whole report as far as its reader is concerned. The passage is quoted when the
    finding carries one — it is never invented, and the two asserting kinds cannot
    exist without it (:class:`~proofpath.report.Finding` refuses). The exact
    ``UNVERIFIED (...)`` string is appended for that family, whose flavour is the
    thing worth saying.
    """
    parts = [item.title]
    if item.verdict is not None and item.verdict.passage is not None:
        # Flattened: a viewer shows the first line of a message and drops the rest, and
        # a passage lifted from a PDF is full of hard line breaks. A quote cut in half
        # is evidence the reader cannot check (rule 1).
        parts.append(f'source says: "{_one_line(item.verdict.passage.text)}"')
    if item.kind is Kind.UNVERIFIED:
        parts.append(item.state)
    return " — ".join(parts)


def _artifact_location(artifact: Path | str) -> dict[str, Any]:
    """The file a result points at, spelled the one way every platform reads.

    A :class:`~pathlib.PurePath` is used as given rather than rebuilt: re-wrapping a
    ``PureWindowsPath`` on a POSIX host would turn its separators into part of a file
    name, which is exactly the corruption ``as_posix`` exists to prevent.
    """
    path = artifact if isinstance(artifact, PurePath) else Path(artifact)
    # A file name is not a URI: ``my draft #2.md`` percent-encodes, or a viewer opens
    # ``my draft`` and looks for the fragment ``2.md``.
    posix = quote(path.as_posix(), safe=URI_SAFE)
    if path.is_absolute():
        # An absolute path is spelled as a URI, because ``C:/srv/draft.md`` read as one
        # has a ``C:`` scheme. ``lstrip`` leaves the POSIX root as the URI's own slash.
        return {"uri": FILE_SCHEME + posix.lstrip("/")}
    return {"uri": posix, "uriBaseId": SRCROOT}


def _run_properties(report: Report) -> dict[str, Any]:
    """What the run has to say about itself, coverage first (rule 6)."""
    coverage = report.coverage
    fulltext, abstract, unverified = coverage.pct()
    return {
        "coverage": {
            "references": coverage.references,
            "fulltext": coverage.fulltext,
            "abstract": coverage.abstract,
            "unverified": coverage.unverified,
            "pct": {"fulltext": fulltext, "abstract": abstract, "unverified": unverified},
            # Rule 2 again, one level up: a log read without the terminal output still
            # says *why* each unread source went unread, and says that a permission
            # rather than the source itself is what stopped some of them.
            "reasons": dict(coverage.reasons),
            "browserSkipped": coverage.browser_skipped,
            "networkDenied": coverage.network_denied,
        },
        "models": dict(report.models),
        "apiCalls": report.api_calls,
        "elapsed": report.elapsed,
        "cancelled": report.cancelled,
    }
