"""The SARIF 2.1.0 writer: schema-valid output that keeps rules 1 and 2 intact.

The schema is vendored rather than fetched, so the test suite stays offline:

    https://json.schemastore.org/sarif-2.1.0.json
    sha256 7c9688f0a1c4a4e1649ecc78521087e664729c1dff56ee8212ff195c7b16132a

``tests/data/sarif-schema-2.1.0.json`` is that file byte for byte; the checksum above
is what a re-download has to match before the copy is replaced.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import jsonschema
import pytest

from proofpath import __version__
from proofpath.fetch import Outcome
from proofpath.models import Label, Passage, Verdict
from proofpath.report import (
    LEVELS,
    STATE_WORDS,
    Coverage,
    Finding,
    Kind,
    Report,
    judge_detail,
)
from proofpath.sarif import (
    RULE_DESCRIPTIONS,
    SARIF_VERSION,
    SCHEMA_URI,
    SRCROOT,
    to_sarif,
)
from tests.test_report import OPINION, PASSAGE, finding, report_with

SCHEMA_PATH = Path(__file__).parent / "data" / "sarif-schema-2.1.0.json"
SCHEMA_SHA256 = "7c9688f0a1c4a4e1649ecc78521087e664729c1dff56ee8212ff195c7b16132a"

# The exact UNVERIFIED flavour a finding of that family carries (spec section 15).
BLOCKED_BY_ROBOTS = Outcome.BLOCKED_ROBOTS.value

COVERAGE = Coverage(
    references=4,
    fulltext=2,
    abstract=1,
    unverified=1,
    reasons={BLOCKED_BY_ROBOTS: 1},
    browser_skipped=1,
    network_denied=False,
)


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def every_kind() -> tuple[Finding, ...]:
    """One finding of every kind, each on its own line, half of them on a page."""
    findings = []
    for index, kind in enumerate(Kind, start=1):
        item = finding(kind, line=index, page=4 if index % 2 == 0 else None, column=index)
        if kind is Kind.UNVERIFIED:
            # Rule 2: which UNVERIFIED it is has to survive the trip to SARIF.
            item = dataclasses.replace(item, state=BLOCKED_BY_ROBOTS)
        findings.append(item)
    return tuple(findings)


def full_report(*, cancelled: bool = False) -> Report:
    return dataclasses.replace(
        report_with(*every_kind(), cancelled=cancelled),
        coverage=COVERAGE,
        models={"nli": "onnx-nli", "device": "cpu"},
        api_calls=7,
    )


def results_by_kind(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result["ruleId"]: result for result in payload["runs"][0]["results"]}


# --- the vendored schema ------------------------------------------------------------


def test_the_vendored_schema_is_the_file_the_header_names() -> None:
    # Normalise line endings: a Windows checkout with autocrlf must hash the same bytes.
    raw = SCHEMA_PATH.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == SCHEMA_SHA256


def test_a_report_of_every_kind_validates_against_the_schema(schema: dict[str, Any]) -> None:
    payload = to_sarif(full_report(), artifact=Path("docs") / "draft.md")
    jsonschema.validate(payload, schema)


def test_an_empty_report_validates_too(schema: dict[str, Any]) -> None:
    payload = to_sarif(report_with(), artifact="draft.md")
    jsonschema.validate(payload, schema)
    assert payload["runs"][0]["results"] == []
    assert payload["runs"][0]["tool"]["driver"]["rules"] == []


def test_an_absolute_artifact_validates(schema: dict[str, Any]) -> None:
    payload = to_sarif(full_report(), artifact=PurePosixPath("/srv/papers/draft.md"))
    jsonschema.validate(payload, schema)


# --- the envelope -------------------------------------------------------------------


def test_the_envelope_names_the_schema_and_the_version() -> None:
    payload = to_sarif(report_with(), artifact="draft.md")
    assert payload["$schema"] == SCHEMA_URI
    assert payload["version"] == SARIF_VERSION == "2.1.0"
    assert len(payload["runs"]) == 1


def test_the_driver_names_this_build_of_proofpath() -> None:
    driver = to_sarif(report_with(), artifact="draft.md")["runs"][0]["tool"]["driver"]
    assert driver["name"] == "proofpath"
    assert driver["version"] == __version__
    assert driver["informationUri"].startswith("https://")


def test_the_payload_is_plain_json_with_no_custom_encoder() -> None:
    # ``ui.emit_json`` serialises this dict; nothing in it should need ``_json_default``.
    payload = to_sarif(full_report(), artifact=Path("docs") / "draft.md")
    assert json.loads(json.dumps(payload)) == payload


# --- rules --------------------------------------------------------------------------


def test_rule_descriptions_cover_every_kind() -> None:
    assert set(RULE_DESCRIPTIONS) == set(Kind)
    assert all(text for text in RULE_DESCRIPTIONS.values())


def test_only_the_kinds_present_in_the_report_become_rules() -> None:
    report = report_with(finding(Kind.GHOST, line=1), finding(Kind.NEI, line=2))
    rules = to_sarif(report, artifact="draft.md")["runs"][0]["tool"]["driver"]["rules"]
    assert [rule["id"] for rule in rules] == [Kind.GHOST.value, Kind.NEI.value]


def test_rule_ids_are_unique_even_when_a_kind_repeats() -> None:
    report = report_with(*(finding(Kind.GHOST, line=n) for n in (1, 2, 3)))
    rules = to_sarif(report, artifact="draft.md")["runs"][0]["tool"]["driver"]["rules"]
    ids = [rule["id"] for rule in rules]
    assert ids == [Kind.GHOST.value]
    assert len(ids) == len(set(ids))


def test_every_rule_carries_its_level_name_description_and_help() -> None:
    payload = to_sarif(full_report(), artifact="draft.md")
    for rule in payload["runs"][0]["tool"]["driver"]["rules"]:
        kind = Kind(rule["id"])
        assert rule["defaultConfiguration"]["level"] == LEVELS[kind]
        assert rule["shortDescription"]["text"] == RULE_DESCRIPTIONS[kind]
        assert rule["helpUri"].startswith("https://")
        # CamelCase of the kind: ghost-reference -> GhostReference.
        assert rule["name"] == "".join(part.capitalize() for part in kind.value.split("-"))


# --- results ------------------------------------------------------------------------


def test_every_finding_becomes_exactly_one_result() -> None:
    report = full_report()
    payload = to_sarif(report, artifact="draft.md")
    assert len(payload["runs"][0]["results"]) == len(report.findings)


def test_every_result_rule_index_points_at_its_own_rule() -> None:
    payload = to_sarif(full_report(), artifact="draft.md")
    rules = payload["runs"][0]["tool"]["driver"]["rules"]
    for result in payload["runs"][0]["results"]:
        assert rules[result["ruleIndex"]]["id"] == result["ruleId"]


def test_result_levels_match_the_report_table() -> None:
    payload = to_sarif(full_report(), artifact="draft.md")
    for result in payload["runs"][0]["results"]:
        assert result["level"] == LEVELS[Kind(result["ruleId"])]


def test_an_asserting_result_quotes_the_passage_it_rests_on() -> None:
    # Rule 1: no passage, no assertion — so the message carries the quote.
    results = results_by_kind(to_sarif(full_report(), artifact="draft.md"))
    for kind in (Kind.NOT_SUPPORTED, Kind.NUMERIC_MISMATCH):
        text = results[kind.value]["message"]["text"]
        assert f'"{PASSAGE.text}"' in text


def test_an_unverified_result_carries_the_exact_state_string() -> None:
    # Rule 2: "UNVERIFIED (blocked, robots.txt)" never degrades to "unverified".
    result = results_by_kind(to_sarif(full_report(), artifact="draft.md"))[Kind.UNVERIFIED.value]
    assert BLOCKED_BY_ROBOTS in result["message"]["text"]
    assert result["properties"]["state"] == BLOCKED_BY_ROBOTS


def test_a_result_message_starts_with_the_finding_title() -> None:
    item = finding(Kind.GHOST, line=3)
    payload = to_sarif(report_with(item), artifact="draft.md")
    assert payload["runs"][0]["results"][0]["message"]["text"].startswith(item.title)


def test_a_note_does_not_invent_a_passage_it_has_none_of() -> None:
    result = results_by_kind(to_sarif(full_report(), artifact="draft.md"))[Kind.NEI.value]
    assert '"' not in result["message"]["text"]


def test_a_passage_spanning_a_line_break_still_reaches_the_message_in_one_line() -> None:
    # A PDF passage is full of hard line breaks; a viewer shows the first line of a
    # message and drops the rest, which would cut the evidence rule 1 exists to show.
    broken = Passage(text="we observed\na 4-8%\n improvement", source_id="doi:10.1/x", index=6)
    item = dataclasses.replace(
        finding(Kind.NOT_SUPPORTED, line=1),
        verdict=Verdict(label=Label.REFUTED, score=0.88, tier="high", passage=broken),
    )
    text = to_sarif(report_with(item), artifact="draft.md")["runs"][0]["results"][0]["message"][
        "text"
    ]
    assert "\n" not in text
    assert '"we observed a 4-8% improvement"' in text


# --- locations ----------------------------------------------------------------------


def test_the_region_carries_the_locator_line_and_column() -> None:
    item = finding(Kind.GHOST, line=112, column=7)
    payload = to_sarif(report_with(item), artifact="draft.md")
    region = payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
    assert region == {"startLine": 112, "startColumn": 7}


def test_a_relative_artifact_is_anchored_to_the_source_root() -> None:
    payload = to_sarif(report_with(finding(Kind.GHOST, line=1)), artifact="docs/draft.md")
    location = payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
    assert location["artifactLocation"] == {"uri": "docs/draft.md", "uriBaseId": "%SRCROOT%"}


def test_an_absolute_artifact_has_no_base_id_to_be_relative_to() -> None:
    payload = to_sarif(
        report_with(finding(Kind.GHOST, line=1)), artifact=PurePosixPath("/srv/papers/draft.md")
    )
    location = payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
    assert location["artifactLocation"] == {"uri": "file:///srv/papers/draft.md"}


def test_the_source_root_is_described_only_when_a_relative_uri_needs_it() -> None:
    run = to_sarif(report_with(finding(Kind.GHOST, line=1)), artifact="docs/draft.md")["runs"][0]
    # The entry names no uri: this module never looked at the working directory,
    # and ``originalUriBaseIds`` entries may carry a description alone.
    assert "uri" not in run["originalUriBaseIds"][SRCROOT]
    assert run["originalUriBaseIds"][SRCROOT]["description"]["text"]


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        ("my draft #2.md", "my%20draft%20%232.md"),
        ("docs/my draft #2.md", "docs/my%20draft%20%232.md"),
        (PurePosixPath("/srv/my draft #2.md"), "file:///srv/my%20draft%20%232.md"),
        (PureWindowsPath(r"C:\papers\my draft.pdf"), "file:///C:/papers/my%20draft.pdf"),
        ("draft?.md", "draft%3F.md"),
    ],
)
def test_the_artifact_uri_is_percent_encoded(artifact: Path | str, expected: str) -> None:
    """A space or a ``#`` in a file name is not a URI character; a viewer that reads
    ``my draft #2.md`` literally opens ``my draft`` and looks for fragment ``2.md``."""
    payload = to_sarif(report_with(finding(Kind.GHOST, line=1)), artifact=artifact)
    location = payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] == expected


def test_a_percent_encoded_artifact_validates(schema: dict[str, Any]) -> None:
    payload = to_sarif(full_report(), artifact="docs/my draft #2.md")
    jsonschema.validate(payload, schema)


def test_an_absolute_run_declares_no_source_root() -> None:
    run = to_sarif(
        report_with(finding(Kind.GHOST, line=1)), artifact=PurePosixPath("/srv/draft.md")
    )["runs"][0]
    assert "originalUriBaseIds" not in run


def test_a_run_with_no_findings_declares_no_source_root() -> None:
    # Nothing points anywhere, so there is no base id to explain.
    assert "originalUriBaseIds" not in to_sarif(report_with(), artifact="draft.md")["runs"][0]


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        (PureWindowsPath(r"docs\papers\draft.md"), "docs/papers/draft.md"),
        (PureWindowsPath(r"C:\srv\draft.md"), "file:///C:/srv/draft.md"),
        (Path("docs") / "papers" / "draft.md", "docs/papers/draft.md"),
        ("docs/papers/draft.md", "docs/papers/draft.md"),
    ],
)
def test_the_uri_is_posix_on_every_platform(artifact: Any, expected: str) -> None:
    # A Windows path reaches SARIF with forward slashes, whichever OS wrote it, so a
    # report produced on Windows opens in a viewer anywhere.
    payload = to_sarif(report_with(finding(Kind.GHOST, line=1)), artifact=artifact)
    uri = payload["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["artifactLocation"][
        "uri"
    ]
    assert uri == expected
    assert "\\" not in uri


def test_a_pdf_page_travels_in_the_properties_not_the_region() -> None:
    # A viewer cannot open a PDF at a line, so the page is a property, not a location.
    on_page = finding(Kind.GHOST, line=112, page=4)
    off_page = finding(Kind.GHOST, line=9)
    payload = to_sarif(report_with(on_page, off_page), artifact="paper.pdf")
    pages = [result["properties"]["page"] for result in payload["runs"][0]["results"]]
    assert sorted(pages, key=lambda page: (page is None, page)) == [4, None]


# --- properties ---------------------------------------------------------------------


def test_a_result_property_bag_carries_the_whole_finding() -> None:
    item = finding(Kind.NOT_SUPPORTED, line=5, page=4)
    payload = to_sarif(report_with(item), artifact="paper.pdf")
    assert payload["runs"][0]["results"][0]["properties"] == {
        "page": 4,
        "state": STATE_WORDS[Kind.NOT_SUPPORTED],
        "tier": "high",
        "sourceId": "doi:10.1/x",
        "fetchStep": 1,
        "group": None,
        "reference": item.reference.raw if item.reference is not None else None,
        "claim": item.claim.text if item.claim is not None else None,
    }


def test_the_run_properties_state_the_coverage_of_the_run() -> None:
    payload = to_sarif(full_report(), artifact="draft.md")
    properties = payload["runs"][0]["properties"]
    assert properties["coverage"] == {
        "references": 4,
        "fulltext": 2,
        "abstract": 1,
        "unverified": 1,
        "pct": {"fulltext": 50, "abstract": 25, "unverified": 25},
        # Rule 6: a log read on its own still says *why* a source went unverified.
        "reasons": {BLOCKED_BY_ROBOTS: 1},
        "browserSkipped": 1,
        "networkDenied": False,
    }
    assert properties["models"] == {"nli": "onnx-nli", "device": "cpu"}
    assert properties["apiCalls"] == 7
    assert properties["elapsed"] == pytest.approx(38.4)
    assert properties["cancelled"] is False


def test_a_cancelled_run_is_not_reported_as_a_successful_execution() -> None:
    payload = to_sarif(full_report(cancelled=True), artifact="draft.md")
    run = payload["runs"][0]
    assert run["invocations"][0]["executionSuccessful"] is False
    assert run["properties"]["cancelled"] is True


def test_a_finished_run_reports_a_successful_execution() -> None:
    payload = to_sarif(full_report(), artifact="draft.md")
    assert payload["runs"][0]["invocations"][0]["executionSuccessful"] is True


def test_to_sarif_reads_nothing_and_writes_nothing(tmp_path: Path) -> None:
    # Pure: a path that does not exist is described, not opened.
    missing = tmp_path / "nowhere" / "draft.md"
    payload = to_sarif(full_report(), artifact=missing)
    assert payload["runs"][0]["results"]
    assert not missing.exists()


def test_a_judged_finding_still_validates_and_keeps_its_local_verdict(
    schema: dict[str, Any],
) -> None:
    """The judge adds fields to a finding; SARIF must neither crash nor repeat them
    as if they were the verdict (spec section 11.1)."""
    judged = dataclasses.replace(
        finding(Kind.NOT_SUPPORTED, line=9, page=2),
        judge=OPINION,
        detail=(judge_detail(OPINION),),
    )
    payload = to_sarif(
        dataclasses.replace(report_with(judged), coverage=COVERAGE, api_calls=3),
        artifact="draft.md",
    )
    jsonschema.validate(payload, schema)
    result = results_by_kind(payload)[Kind.NOT_SUPPORTED.value]
    assert result["level"] == "error"
    assert PASSAGE.text in result["message"]["text"]
