"""AVeriTeC (Schlichtkrull et al., 2023) dev split as claims with their web sources.

Loaded from the authors' repository rather than the Hugging Face mirrors, which are
gated behind an access request. The digest is pinned so the evaluation cannot drift
silently under us.

One row of the file is one real-world claim, annotated with the questions a fact
checker asked and the answers they found; every answer names the page it came from.
Those pages are what proofpath is measured on, so a ``Claim`` keeps the claim text,
the gold label and the source URLs, and drops the rest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from proofpath.models import Label

URL = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json"
# Recorded 2026-09-15 from a fresh download (1,785,475 bytes, 500 claims).
SHA256 = "499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300"

SUPPORTED = "Supported"
REFUTED = "Refuted"
NOT_ENOUGH_EVIDENCE = "Not Enough Evidence"
CONFLICTING = "Conflicting Evidence/Cherrypicking"

# The dataset's whole label vocabulary, in the order the report prints it.
LABELS: tuple[str, ...] = (SUPPORTED, REFUTED, NOT_ENOUGH_EVIDENCE, CONFLICTING)

# Conflicting evidence is deliberately absent: it is a fourth state, not a verdict
# we can produce, and folding it into NEI would report a disagreement between
# sources as an absence of them (product rule 2).
_TO_LABEL = {
    SUPPORTED: Label.SUPPORTED,
    REFUTED: Label.REFUTED,
    NOT_ENOUGH_EVIDENCE: Label.NEI,
}


class DatasetError(RuntimeError):
    """The dataset could not be loaded as pinned."""


# An answer's ``source_url`` is not always a URL: 82 answers of the dev split give
# the literal string "Metadata", meaning the fact checker read the page's own
# metadata rather than another page. Only these two schemes name something a fetch
# ladder could ever reach.
_URL_SCHEMES = ("http://", "https://")


@dataclass(frozen=True)
class Claim:
    """One dev claim and the pages its annotators read to settle it."""

    id: int  # the row's position in the file; the only stable handle a row has
    text: str
    label: str  # one of LABELS, verbatim from the file
    source_urls: tuple[str, ...]
    # How many of the row's distinct ``source_url`` values named no page at all.
    # Counted rather than dropped: a run has to be able to say it never tried them,
    # instead of reporting them as web sources it failed to reach (product rule 2).
    non_urls: int = 0


def to_label(label: str) -> Label | None:
    """The proofpath verdict a gold label corresponds to, or ``None``.

    ``None`` means the row cannot be scored three ways — either it is the
    Conflicting Evidence/Cherrypicking class, which proofpath has no verdict for, or
    it is a label this loader has never seen. Both are reported, never guessed at.
    """
    return _TO_LABEL.get(label)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_url(value: str) -> bool:
    """Whether a ``source_url`` value actually names a page a fetcher could try."""
    return value.lower().startswith(_URL_SCHEMES)


def _source_values(row: dict[str, Any]) -> tuple[tuple[str, ...], int]:
    """The row's fetchable URLs, and how many of its values were not URLs at all.

    Every ``source_url`` under the row's answers is deduplicated first, keeping the
    order of first use — that is the order a fetch run walks, and a resumed run has
    to line up with the one it resumes. The non-URL values are counted over the same
    deduplicated list, so the two numbers describe the same set of annotations.
    """
    values: list[str] = []
    for question in row.get("questions") or []:
        for answer in question.get("answers") or []:
            value = answer.get("source_url")
            if value:
                values.append(str(value))
    distinct = tuple(dict.fromkeys(values))
    urls = tuple(value for value in distinct if _is_url(value))
    return urls, len(distinct) - len(urls)


def load(path: Path, *, limit: int | None = None, expected_sha256: str = SHA256) -> list[Claim]:
    """Parse the dev file at ``path`` after checking its digest."""
    actual = _sha256(path)
    if actual != expected_sha256:
        raise DatasetError(f"{path}: sha256 {actual} does not match pinned {expected_sha256}")
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        rows = rows[:limit]
    claims: list[Claim] = []
    for position, row in enumerate(rows):
        urls, non_urls = _source_values(row)
        claims.append(
            Claim(
                id=position,
                text=str(row["claim"]),
                label=str(row["label"]),
                source_urls=urls,
                non_urls=non_urls,
            )
        )
    return claims


def ensure_downloaded(cache_dir: Path) -> Path:
    """Return the local dev file, downloading it once. Verified on every reuse."""
    target = cache_dir / "datasets" / "averitec-dev.json"
    if target.exists() and _sha256(target) == SHA256:
        return target
    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", URL, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return target
