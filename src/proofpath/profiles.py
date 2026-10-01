"""NLI profiles: which model, at which k, with which calibrated cuts.

A profile is the unit a run chooses, because the three travel together: thresholds
are calibrated for one model at one k, and moving any of them alone makes the tiers a
report prints mean something they were never measured to mean. Design and numbers:
docs/superpowers/specs/2026-10-01-accurate-nli-design.md.

This lives beside ``entailment`` rather than in it because a profile carries
``pipeline.Thresholds`` and ``pipeline`` already imports ``entailment``: defining the
profiles there would make the two modules import each other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from proofpath.entailment import DEFAULT_REPO, DEFAULT_REVISION, pick_onnx_file
from proofpath.pipeline import DEFAULT_THRESHOLDS, Thresholds


@dataclass(frozen=True)
class NliProfile:
    """One NLI model, pinned, with the k and cuts it was calibrated at.

    ``onnx_file`` maps ``platform.machine()`` to the export inside the repo, since a
    repo may ship one graph per CPU family. ``size_mb`` is what the consent prompt
    states before a download (product rule 5).
    """

    name: str
    repo: str
    revision: str
    onnx_file: Callable[[str], str]
    k: int
    thresholds: Thresholds
    size_mb: int


def _quantized_file(_machine: str) -> str:
    # The repo ships one dynamic-int8 export for every CPU.
    return "onnx/model_quantized.onnx"


# Today's model, values unchanged: SciFact dev, 2026-09-12, k=1
# (docs/eval/2026-09-12-tiers.md).
DEFAULT_PROFILE = NliProfile(
    name="default",
    repo=DEFAULT_REPO,
    revision=DEFAULT_REVISION,
    onnx_file=pick_onnx_file,
    k=1,
    thresholds=DEFAULT_THRESHOLDS,
    size_mb=244,
)

# The bake-off winner (docs/eval/2026-10-01-nli-bakeoff.md): SciFact dev macro-F1
# 0.697 against 0.580, at 643 MB and roughly 3x the time per pair. Opt-in, never the
# default. Its cuts come from SciFact train with rule-decided rows (score exactly 1.0)
# left out of the tier walk: they fired 4-6 times there and none was right, and kept
# in they pinned ``high`` at 1.0. They still count toward ``decide``.
ACCURATE_PROFILE = NliProfile(
    name="accurate",
    repo="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    revision="b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    onnx_file=_quantized_file,
    k=2,
    thresholds=Thresholds(decide=0.25, high=0.999142, medium=0.252721),
    size_mb=643,
)

#: By ``models.nli`` value. Read-only, so no caller can swap a profile under a run.
PROFILES: Mapping[str, NliProfile] = MappingProxyType(
    {profile.name: profile for profile in (DEFAULT_PROFILE, ACCURATE_PROFILE)}
)


def profile_installed(profile: NliProfile, cache_dir: Path, machine: str) -> bool:
    """True when every file ``OnnxNli`` would fetch for this profile is already cached.

    Reads the local Hugging Face cache only, never the network: this is what decides
    whether a run has to ask before a download, so it must be answerable offline and
    in CI. The ONNX file is the one this machine would run, so an arm64 cache does not
    count as installed on x86_64. ``try_to_load_from_cache`` returns a path string for
    a cached file and ``None`` or a "known missing" sentinel otherwise; only the
    string means the file is there.
    """
    from huggingface_hub import try_to_load_from_cache

    wanted = ("config.json", "tokenizer.json", profile.onnx_file(machine))
    return all(
        isinstance(
            try_to_load_from_cache(
                profile.repo, filename, cache_dir=str(cache_dir), revision=profile.revision
            ),
            str,
        )
        for filename in wanted
    )
