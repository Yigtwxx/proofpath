from __future__ import annotations

import os

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
