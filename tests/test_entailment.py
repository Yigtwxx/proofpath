from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from proofpath import entailment as ent
from proofpath.models import Label


def test_label_order_is_fixed_and_documented() -> None:
    assert ent.LABEL_ORDER == (Label.SUPPORTED, Label.REFUTED, Label.NEI)


def test_label_permutation_maps_model_names_to_our_order() -> None:
    id2label = {"0": "contradiction", "1": "entailment", "2": "neutral"}
    assert ent.label_permutation(id2label) == [1, 0, 2]


def test_label_permutation_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="maybe"):
        ent.label_permutation({"0": "maybe", "1": "entailment", "2": "neutral"})


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("arm64", "onnx/model_qint8_arm64.onnx"),
        ("aarch64", "onnx/model_qint8_arm64.onnx"),
        ("x86_64", "onnx/model_quint8_avx2.onnx"),
        ("AMD64", "onnx/model_quint8_avx2.onnx"),
        ("riscv64", "onnx/model.onnx"),
    ],
)
def test_pick_onnx_file_by_cpu_architecture(machine: str, expected: str) -> None:
    assert ent.pick_onnx_file(machine) == expected


def test_softmax_rows_sum_to_one() -> None:
    probs = ent.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[1].tolist() == pytest.approx([1 / 3] * 3)


@pytest.mark.skipif(not os.environ.get("PROOFPATH_RUN_SLOW"), reason="downloads the NLI model")
def test_real_model_separates_entailment_from_contradiction() -> None:
    from proofpath.paths import models_dir

    scorer = ent.OnnxNli(cache_dir=models_dir())
    probs = scorer.score(
        [
            ("A man is eating food.", "A man is eating a meal."),
            ("A man is eating food.", "The man is sleeping."),
        ]
    )
    assert probs.shape == (2, 3)
    assert probs[0].argmax() == ent.LABEL_ORDER.index(Label.SUPPORTED)
    assert probs[1].argmax() == ent.LABEL_ORDER.index(Label.REFUTED)


def test_int8_export_prefers_cpu_over_coreml() -> None:
    # Measured 2026-09-11: CoreML takes 880 of 2,524 nodes of the int8 graph and is
    # 3x slower than CPU. CUDA still wins when present.
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert ent.providers_for("onnx/model_qint8_arm64.onnx", available) == ["CPUExecutionProvider"]
    assert ent.providers_for(
        "onnx/model_quint8_avx2.onnx", ["CUDAExecutionProvider", *available]
    ) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_fp32_export_keeps_the_cuda_coreml_cpu_order() -> None:
    available = ["CPUExecutionProvider", "CoreMLExecutionProvider"]
    assert ent.providers_for("onnx/model.onnx", available) == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_close_drops_the_session_and_is_idempotent() -> None:
    """Spec section 13.3: an ONNX session released at interpreter shutdown aborted
    the process (SIGABRT, exit 134), so the engine releases it itself."""
    scorer = object.__new__(ent.OnnxNli)
    scorer._session = object()
    scorer._tokenizer = object()
    scorer._input_names = ["input_ids"]

    scorer.close()
    assert scorer._session is None
    assert scorer._tokenizer is None
    assert scorer._input_names == []
    assert scorer.close() is None


def test_quantized_export_also_skips_coreml() -> None:
    # The accurate profile's one export is dynamic int8 under another name; without
    # this a Mac would run the large model on CoreML, measured 3x slower for int8.
    available = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert ent.providers_for("onnx/model_quantized.onnx", available) == ["CPUExecutionProvider"]
    assert ent.providers_for(
        "onnx/model_quantized.onnx", ["CUDAExecutionProvider", *available]
    ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]


# --- NLI profiles (docs/superpowers/specs/2026-10-01-accurate-nli-design.md) ------


def test_the_default_profile_is_todays_model_unchanged() -> None:
    from proofpath import profiles
    from proofpath.pipeline import DEFAULT_THRESHOLDS, Thresholds

    default = profiles.DEFAULT_PROFILE
    assert default.name == "default"
    assert default.repo == "cross-encoder/nli-deberta-v3-base" == ent.DEFAULT_REPO
    assert default.revision == "6c749ce3425cd33b46d187e45b92bbf96ee12ec7" == ent.DEFAULT_REVISION
    assert default.onnx_file is ent.pick_onnx_file
    assert default.k == 1
    assert default.thresholds == Thresholds(decide=0.45, high=0.99933, medium=0.457948)
    assert default.thresholds == DEFAULT_THRESHOLDS
    assert default.size_mb == 244


def test_the_accurate_profile_carries_the_bakeoff_winner() -> None:
    from proofpath import profiles
    from proofpath.pipeline import Thresholds

    accurate = profiles.ACCURATE_PROFILE
    assert accurate.name == "accurate"
    assert accurate.repo == "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    assert accurate.revision == "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
    for machine in ("arm64", "x86_64", "riscv64"):
        assert accurate.onnx_file(machine) == "onnx/model_quantized.onnx"
    assert accurate.k == 2
    assert accurate.thresholds == Thresholds(decide=0.25, high=0.999142, medium=0.252721)
    assert accurate.size_mb == 643


def test_profiles_are_keyed_by_name_and_match_the_config_choices() -> None:
    from proofpath import profiles
    from proofpath.config import NLI_PROFILE_NAMES

    assert dict(profiles.PROFILES) == {
        "default": profiles.DEFAULT_PROFILE,
        "accurate": profiles.ACCURATE_PROFILE,
    }
    assert set(profiles.PROFILES) == set(NLI_PROFILE_NAMES)


def test_a_profile_is_frozen() -> None:
    import dataclasses

    from proofpath import profiles

    with pytest.raises(dataclasses.FrozenInstanceError):
        profiles.ACCURATE_PROFILE.k = 1  # type: ignore[misc]


def _fake_cache(cached: set[str], seen: list[tuple[str, str, str | None, str]]) -> object:
    from huggingface_hub import _CACHED_NO_EXIST

    def lookup(
        repo_id: str,
        filename: str,
        cache_dir: str | None = None,
        revision: str | None = None,
        repo_type: str | None = None,
    ) -> object:
        seen.append((repo_id, filename, revision, str(cache_dir)))
        if filename in cached:
            return f"/snapshots/{filename}"
        # A cached "this file does not exist" is not the file being there.
        return _CACHED_NO_EXIST if filename == "config.json" else None

    return lookup


def test_profile_installed_needs_config_tokenizer_and_the_onnx_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    from proofpath import profiles

    seen: list[tuple[str, str, str | None, str]] = []
    every = {"config.json", "tokenizer.json", "onnx/model_quantized.onnx"}
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _fake_cache(every, seen))
    assert profiles.profile_installed(profiles.ACCURATE_PROFILE, tmp_path, "arm64") is True
    repo, revision = profiles.ACCURATE_PROFILE.repo, profiles.ACCURATE_PROFILE.revision
    assert {(r, rev, d) for r, _, rev, d in seen} == {(repo, revision, str(tmp_path))}
    assert {f for _, f, _, _ in seen} == every

    for missing in sorted(every):
        monkeypatch.setattr(
            huggingface_hub, "try_to_load_from_cache", _fake_cache(every - {missing}, [])
        )
        assert profiles.profile_installed(profiles.ACCURATE_PROFILE, tmp_path, "arm64") is False


def test_profile_installed_asks_for_the_file_this_machine_would_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    from proofpath import profiles

    cached = {"config.json", "tokenizer.json", "onnx/model_qint8_arm64.onnx"}
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", _fake_cache(cached, []))
    assert profiles.profile_installed(profiles.DEFAULT_PROFILE, tmp_path, "arm64") is True
    assert profiles.profile_installed(profiles.DEFAULT_PROFILE, tmp_path, "x86_64") is False
