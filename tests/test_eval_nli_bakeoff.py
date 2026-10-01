"""Offline tests for scripts/eval_nli_bakeoff.py's importable parts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_nli_bakeoff.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_nli_bakeoff", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


def test_every_candidate_is_pinned_to_a_full_revision() -> None:
    assert set(script.CANDIDATES) == {"base", "large", "large-fever"}
    for candidate in script.CANDIDATES.values():
        assert len(candidate.revision) == 40
        int(candidate.revision, 16)


def test_only_the_large_models_are_marked_large() -> None:
    assert {name for name, c in script.CANDIDATES.items() if c.large} == {"large", "large-fever"}


def test_each_candidate_names_an_onnx_file_per_machine() -> None:
    for machine in ("arm64", "x86_64"):
        for candidate in script.CANDIDATES.values():
            assert candidate.onnx_file(machine).startswith("onnx/")
            assert candidate.onnx_file(machine).endswith(".onnx")


def test_the_baseline_is_base_k1_max() -> None:
    assert (script.BASELINE.model, script.BASELINE.k, script.BASELINE.aggregation) == (
        "base",
        1,
        "max",
    )


def test_coreml_is_never_handed_to_the_int8_models() -> None:
    assert "CoreMLExecutionProvider" not in script.nli_providers()
