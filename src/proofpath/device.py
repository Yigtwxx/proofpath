"""Compute device selection.

Preference order is fixed by the project rules: CUDA -> MPS -> CPU for torch, and
CUDA -> CoreML -> CPU for ONNX runtime. Nothing here imports torch; the ``[gpu]``
extra is optional and the base install must not pay for it.
"""

from __future__ import annotations

from collections.abc import Sequence

_ONNX_PREFERENCE = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)

_ONNX_SHORT_NAMES = {
    "CUDAExecutionProvider": "cuda",
    "CoreMLExecutionProvider": "coreml",
    "CPUExecutionProvider": "cpu",
}


def _available_onnx_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def onnx_providers_from(available: Sequence[str]) -> list[str]:
    """Order ``available`` by preference, dropping unknown ones; CPU is always last."""
    present = set(available)
    chosen = [p for p in _ONNX_PREFERENCE if p in present]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def onnx_providers() -> list[str]:
    """Execution providers to hand to ``InferenceSession``, best first, CPU always last."""
    return onnx_providers_from(_available_onnx_providers())


def onnx_device_name() -> str:
    """Short name of the provider that will actually run, for reports and logs."""
    return _ONNX_SHORT_NAMES[onnx_providers()[0]]


def torch_device() -> str:
    """CUDA -> MPS -> CPU. Only meaningful when the ``[gpu]`` extra is installed."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
