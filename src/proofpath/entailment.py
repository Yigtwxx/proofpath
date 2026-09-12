"""Local NLI: (claim, passage) -> SUPPORTED / REFUTED / NEI probabilities.

Runs a cross-encoder exported to ONNX. No torch, no transformers: tokenization is
``tokenizers``, inference is ``onnxruntime``. Model and revision are pinned so a
verdict cache keyed on ``model_id`` stays meaningful.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from proofpath.models import Label

LABEL_ORDER: tuple[Label, Label, Label] = (Label.SUPPORTED, Label.REFUTED, Label.NEI)

# Verified 2026-09-11: this revision ships ONNX exports and tokenizer.json.
DEFAULT_REPO = "cross-encoder/nli-deberta-v3-base"
DEFAULT_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"

_MODEL_LABEL_NAMES: Mapping[str, Label] = {
    "entailment": Label.SUPPORTED,
    "contradiction": Label.REFUTED,
    "neutral": Label.NEI,
}


class Scorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return ``(n, 3)`` probabilities in ``LABEL_ORDER``."""


def label_permutation(id2label: Mapping[str, str]) -> list[int]:
    """Column indices that reorder the model's logits into ``LABEL_ORDER``."""
    by_name: dict[Label, int] = {}
    for index, name in id2label.items():
        key = name.lower()
        if key not in _MODEL_LABEL_NAMES:
            raise ValueError(f"unknown NLI label {name!r} in model config")
        by_name[_MODEL_LABEL_NAMES[key]] = int(index)
    return [by_name[label] for label in LABEL_ORDER]


def pick_onnx_file(machine: str) -> str:
    """Choose the int8 export that matches the CPU; fall back to the fp32 graph."""
    lowered = machine.lower()
    if lowered in {"arm64", "aarch64"}:
        return "onnx/model_qint8_arm64.onnx"
    if lowered in {"x86_64", "amd64"}:
        return "onnx/model_quint8_avx2.onnx"
    return "onnx/model.onnx"


def providers_for(onnx_file: str, available: Sequence[str]) -> list[str]:
    """Execution providers for a given export, best first.

    The project rule is CUDA -> CoreML -> CPU. For the int8 exports CoreML is
    skipped: measured 2026-09-11 on Apple Silicon, CoreML handles 880 of the 2,524
    nodes and the partitioning makes it 3x slower than plain CPU.
    """
    from proofpath.device import onnx_providers_from

    chosen = onnx_providers_from(available)
    if "int8" in Path(onnx_file).stem:
        chosen = [p for p in chosen if p != "CoreMLExecutionProvider"]
    return chosen


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=1, keepdims=True))


class OnnxNli:
    """Cross-encoder NLI over ONNX runtime."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        repo_id: str = DEFAULT_REPO,
        revision: str = DEFAULT_REVISION,
        onnx_file: str | None = None,
        providers: Sequence[str] | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        import platform

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        from proofpath.device import _available_onnx_providers

        onnx_file = onnx_file or pick_onnx_file(platform.machine())
        providers = providers or providers_for(onnx_file, _available_onnx_providers())
        self.name = f"{repo_id}@{revision[:7]}:{Path(onnx_file).stem}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        def fetch(filename: str) -> Path:
            return Path(
                hf_hub_download(repo_id, filename, revision=revision, cache_dir=str(cache_dir))
            )

        config = json.loads(fetch("config.json").read_text(encoding="utf-8"))
        self._permutation = label_permutation(config["id2label"])
        # Typed loosely on purpose: ``close()`` drops both of these, and the session
        # and the tokenizer are only ever touched between construction and that call.
        self._tokenizer: Any = Tokenizer.from_file(str(fetch("tokenizer.json")))
        self._tokenizer.enable_truncation(max_length)
        self._tokenizer.enable_padding()
        model_path = fetch(onnx_file)
        # The ONNX export lives in onnx/; its external data file, when present,
        # sits beside it and hf_hub_download resolves it into the same snapshot.
        self._session: Any = ort.InferenceSession(str(model_path), providers=list(providers))
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._batch_size = batch_size

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def close(self) -> None:
        """Release the ONNX session and the tokenizer. Idempotent.

        onnxruntime frees the session on the C++ side when the last Python reference
        goes. Left to the garbage collector that happens during interpreter
        shutdown, where the teardown order is not ours to choose, and a run that had
        already printed its whole report died with SIGABRT (exit 134) once in three
        on 2026-09-12 — a code the exit contract of spec section 13.3 does not have.
        Dropping the references here makes the release happen while the interpreter
        is still up. The model is at the end of its life once this is called;
        ``score()`` after it is a programming error, not a supported state.
        """
        self._session = None
        self._tokenizer = None
        self._input_names = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        chunks = []
        for start in range(0, len(pairs), self._batch_size):
            chunks.append(self._score_batch(pairs[start : start + self._batch_size]))
        return np.concatenate(chunks, axis=0)

    def _score_batch(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch([(p, h) for p, h in pairs])
        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = np.array([e.ids for e in encodings], dtype=np.int64)
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        logits = self._session.run(None, feeds)[0]
        return softmax(np.asarray(logits, dtype=np.float32))[:, self._permutation]
