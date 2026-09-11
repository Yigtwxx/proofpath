"""Device order is CUDA -> MPS -> CPU; ONNX equivalent CUDA -> CoreML -> CPU."""

from __future__ import annotations

import pytest

from proofpath import device


def test_onnx_providers_keep_preference_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device,
        "_available_onnx_providers",
        lambda: ["CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider"],
    )
    assert device.onnx_providers() == [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_onnx_providers_drop_unavailable_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device,
        "_available_onnx_providers",
        lambda: ["AzureExecutionProvider", "CPUExecutionProvider"],
    )
    assert device.onnx_providers() == ["CPUExecutionProvider"]


def test_onnx_providers_always_end_with_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_available_onnx_providers", lambda: [])
    assert device.onnx_providers() == ["CPUExecutionProvider"]


def test_device_name_is_short_and_human(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "_available_onnx_providers", lambda: ["CoreMLExecutionProvider"])
    assert device.onnx_device_name() == "coreml"
