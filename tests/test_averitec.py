"""Offline tests for the AVeriTeC dev loader.

The dataset is never downloaded here: a tiny file with the real JSON shape is
written to ``tmp_path`` and its digest computed in the test, the way
``tests/test_scifact.py`` builds its tarball.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from proofpath.eval import averitec
from proofpath.models import Label

# Two claims: the first cites the same URL from two questions (so the loader has
# something to deduplicate), the second carries the fourth label, which maps to no
# verdict of ours.
ROWS: list[dict[str, Any]] = [
    {
        "claim": "first claim",
        "label": "Supported",
        "justification": "because the sources say so",
        "questions": [
            {
                "question": "where was it published",
                "answers": [
                    {"answer": "here", "source_url": "https://a.example/one"},
                    {"answer": "and here", "source_url": "https://b.example/two"},
                ],
            },
            {
                "question": "who said it",
                "answers": [{"answer": "again here", "source_url": "https://a.example/one"}],
            },
        ],
    },
    {
        "claim": "second claim",
        "label": "Conflicting Evidence/Cherrypicking",
        "justification": "the sources disagree",
        "questions": [
            {
                "question": "what happened",
                "answers": [{"answer": "it depends", "source_url": "https://c.example/three"}],
            }
        ],
    },
]


def _write_dataset(root: Path) -> tuple[Path, str]:
    payload = json.dumps(ROWS).encode("utf-8")
    path = root / "averitec-dev.json"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


# --- load ----------------------------------------------------------------


def test_load_yields_one_claim_per_row_numbered_by_position(tmp_path: Path) -> None:
    path, digest = _write_dataset(tmp_path)
    claims = averitec.load(path, expected_sha256=digest)
    assert [(c.id, c.text, c.label) for c in claims] == [
        (0, "first claim", "Supported"),
        (1, "second claim", "Conflicting Evidence/Cherrypicking"),
    ]


def test_evidence_urls_are_deduplicated_in_order_of_first_appearance(tmp_path: Path) -> None:
    path, digest = _write_dataset(tmp_path)
    claims = averitec.load(path, expected_sha256=digest)
    assert claims[0].source_urls == ("https://a.example/one", "https://b.example/two")
    assert claims[1].source_urls == ("https://c.example/three",)


def test_limit_takes_the_first_n_claims(tmp_path: Path) -> None:
    path, digest = _write_dataset(tmp_path)
    claims = averitec.load(path, limit=1, expected_sha256=digest)
    assert [c.id for c in claims] == [0]


def test_a_limit_beyond_the_file_returns_every_claim(tmp_path: Path) -> None:
    path, digest = _write_dataset(tmp_path)
    assert len(averitec.load(path, limit=99, expected_sha256=digest)) == len(ROWS)


def test_a_claim_with_no_questions_has_no_source_urls(tmp_path: Path) -> None:
    payload = json.dumps([{"claim": "bare", "label": "Refuted", "questions": []}]).encode("utf-8")
    path = tmp_path / "averitec-dev.json"
    path.write_bytes(payload)
    claims = averitec.load(path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert claims[0].source_urls == ()


def test_an_answer_without_a_source_url_is_skipped(tmp_path: Path) -> None:
    rows = [
        {
            "claim": "partly sourced",
            "label": "Refuted",
            "questions": [
                {
                    "question": "q",
                    "answers": [
                        {"answer": "no url here", "source_url": None},
                        {"answer": "nor here", "source_url": ""},
                        {"answer": "but here", "source_url": "https://d.example/four"},
                    ],
                }
            ],
        }
    ]
    payload = json.dumps(rows).encode("utf-8")
    path = tmp_path / "averitec-dev.json"
    path.write_bytes(payload)
    claims = averitec.load(path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert claims[0].source_urls == ("https://d.example/four",)


def test_a_source_url_that_is_not_a_url_is_counted_rather_than_fetched(tmp_path: Path) -> None:
    # 82 answers of the real dev file carry the literal string "Metadata" here. It
    # names no page, so sending it down the fetch ladder would manufacture an
    # "unreachable web source" out of an annotation (product rule 2).
    rows = [
        {
            "claim": "cited from the page's own metadata",
            "label": "Refuted",
            "questions": [
                {
                    "question": "q",
                    "answers": [
                        {"answer": "from the metadata", "source_url": "Metadata"},
                        {"answer": "from it again", "source_url": "Metadata"},
                        {"answer": "and from a page", "source_url": "https://e.example/five"},
                    ],
                }
            ],
        }
    ]
    payload = json.dumps(rows).encode("utf-8")
    path = tmp_path / "averitec-dev.json"
    path.write_bytes(payload)
    claims = averitec.load(path, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert claims[0].source_urls == ("https://e.example/five",)
    # Deduplicated the same way the URLs are: two "Metadata" answers, one value.
    assert claims[0].non_urls == 1


@pytest.mark.parametrize(
    ("value", "is_url"),
    [
        ("https://f.example/six", True),
        ("http://f.example/six", True),
        ("HTTPS://F.EXAMPLE/SIX", True),
        ("Metadata", False),
        ("ftp://f.example/six", False),
        ("www.f.example/six", False),
    ],
)
def test_only_http_and_https_values_are_treated_as_pages(
    tmp_path: Path, value: str, is_url: bool
) -> None:
    rows = [
        {
            "claim": "one source",
            "label": "Refuted",
            "questions": [{"question": "q", "answers": [{"answer": "a", "source_url": value}]}],
        }
    ]
    payload = json.dumps(rows).encode("utf-8")
    path = tmp_path / "averitec-dev.json"
    path.write_bytes(payload)
    claim = averitec.load(path, expected_sha256=hashlib.sha256(payload).hexdigest())[0]
    expected = ((value,), 0) if is_url else ((), 1)
    assert (claim.source_urls, claim.non_urls) == expected


def test_claims_whose_sources_are_all_real_urls_count_no_non_urls(tmp_path: Path) -> None:
    path, digest = _write_dataset(tmp_path)
    assert [c.non_urls for c in averitec.load(path, expected_sha256=digest)] == [0, 0]


def test_sha_mismatch_is_refused_naming_both_digests(tmp_path: Path) -> None:
    path, digest = _write_dataset(tmp_path)
    with pytest.raises(averitec.DatasetError) as excinfo:
        averitec.load(path, expected_sha256="0" * 64)
    message = str(excinfo.value)
    assert digest in message
    assert "0" * 64 in message


def test_pinned_digest_is_a_real_sha256() -> None:
    assert len(averitec.SHA256) == 64
    int(averitec.SHA256, 16)


# --- to_label ------------------------------------------------------------


@pytest.mark.parametrize(
    ("gold", "expected"),
    [
        ("Supported", Label.SUPPORTED),
        ("Refuted", Label.REFUTED),
        ("Not Enough Evidence", Label.NEI),
        # Conflicting evidence is not one of our three verdicts: it maps to nothing
        # rather than being folded into NEI, which would hide it in the report.
        ("Conflicting Evidence/Cherrypicking", None),
    ],
)
def test_to_label_maps_the_dataset_vocabulary(gold: str, expected: Label | None) -> None:
    assert averitec.to_label(gold) is expected


def test_an_unknown_label_maps_to_nothing() -> None:
    assert averitec.to_label("Mostly True") is None
