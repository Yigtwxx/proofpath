"""NLI bake-off harness (docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md).

Usage:
    uv run python scripts/eval_nli_bakeoff.py snapshot [--limit 100] [--sleep 1.0] [--resume]
    uv run python scripts/eval_nli_bakeoff.py score --model base|large|large-fever
        [--rebuild-items]
    uv run python scripts/eval_nli_bakeoff.py report [--out docs/eval/<date>-nli-bakeoff.md]

``snapshot`` is the only network step, apart from the one-time dataset and model
downloads. ``score`` runs one NLI model over every stored (passage, claim) pair.
``report`` is offline. Importing this module touches neither the network nor a model.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from proofpath import retrieval
from proofpath.entailment import pick_onnx_file
from proofpath.eval import averitec, bakeoff, scifact
from proofpath.eval.bakeoff_data import (
    Item,
    SnapshotClaim,
    load_items,
    load_snapshot,
    pending,
    save_items,
    save_probs,
    save_snapshot,
    source_from_fetched,
)
from proofpath.models import Passage
from proofpath.paths import cache_dir, models_dir

EMBEDDER = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Candidate:
    repo: str
    revision: str
    onnx_file: Callable[[str], str]  # platform.machine() -> file inside the repo
    large: bool


def _fever_file(_machine: str) -> str:
    # The repo ships one dynamic-int8 export for every CPU.
    return "onnx/model_quantized.onnx"


CANDIDATES: dict[str, Candidate] = {
    "base": Candidate(
        "cross-encoder/nli-deberta-v3-base",
        "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
        pick_onnx_file,
        large=False,
    ),
    "large": Candidate(
        "cross-encoder/nli-deberta-v3-large",
        "bab4bc7178836f731dcfd18c06ca9def0a137712",
        pick_onnx_file,
        large=True,
    ),
    "large-fever": Candidate(
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
        _fever_file,
        large=True,
    ),
}
BASELINE = bakeoff.Combo("base", 1, "max")
# Pairs handed to the scorer per call: several of its 16-pair batches, so progress
# can be printed without starving the batches.
SCORE_CHUNK = 400


def bakeoff_dir() -> Path:
    return cache_dir() / "datasets" / "bakeoff"


def snapshot_path() -> Path:
    return cache_dir() / "datasets" / "averitec_snapshot.json"


def items_path() -> Path:
    return bakeoff_dir() / "items.json"


def nli_providers() -> list[str]:
    """CUDA first where present, then CPU.

    CoreML is left out for every candidate. All three files are int8, and the
    product leaves CoreML out for int8 too (``entailment.providers_for``).
    """
    from proofpath.device import onnx_providers

    return [p for p in onnx_providers() if p != "CoreMLExecutionProvider"]


def _ranked(
    claim: str, passages: list[Passage], embedder: retrieval.Embedder
) -> tuple[Passage, ...]:
    hits = retrieval.rank(claim, passages, embedder, k=bakeoff.TOP_N)
    return tuple(hit.passage for hit in hits)


def build_items(embedder: retrieval.Embedder) -> list[Item]:
    """SciFact train and dev pairs, then every readable AVeriTeC source, top 5 each.

    Passages are split exactly as the product splits them: SciFact ships its
    sentences, and a page goes through ``retrieval.split_sentences`` as in
    ``eval_averitec._decide_claim``.
    """
    items: list[Item] = []
    tarball = scifact.ensure_downloaded(cache_dir())
    for split in ("train", "dev"):
        data = scifact.load(tarball, split=split)
        for pair in data.pairs():
            doc = data.corpus[pair.doc_id]
            passages = [Passage(t, str(doc.doc_id), i) for i, t in enumerate(doc.sentences)]
            key = f"scifact-{split}:{pair.claim_id}:{pair.doc_id}"
            items.append(
                Item(
                    key,
                    f"scifact-{split}",
                    key,
                    pair.claim,
                    pair.label.value,
                    _ranked(pair.claim, passages, embedder),
                    pair.rationale,
                )
            )
    for claim in load_snapshot(snapshot_path()):
        for position, source in enumerate(claim.sources):
            sentences = retrieval.split_sentences(source.text) if source.text.strip() else []
            if not sentences:
                continue
            passages = [Passage(s, source.url, i) for i, s in enumerate(sentences)]
            items.append(
                Item(
                    f"averitec:{claim.claim_id}:{position}",
                    "averitec",
                    f"averitec:{claim.claim_id}",
                    claim.claim,
                    claim.gold,
                    _ranked(claim.claim, passages, embedder),
                )
            )
    return items


def cmd_snapshot(args: argparse.Namespace) -> int:
    from proofpath.config import load_config
    from proofpath.verify import Engine

    path = snapshot_path()
    if path.exists() and not args.resume:
        print(
            f"error     {path} already holds a snapshot.\n"
            "          Pass --resume to continue it, or delete the file to start over.",
            file=sys.stderr,
        )
        return 2
    claims = averitec.load(averitec.ensure_downloaded(cache_dir()), limit=args.limit or None)
    stored = load_snapshot(path) if args.resume else []
    todo = pending(claims, stored)
    print(f"snapshot  {len(claims)} claims, {len(stored)} done, {len(todo)} to fetch")
    # Browser not permitted and no prompts: the default install, as in the published
    # AVeriTeC number (spec: Data).
    engine = Engine.default(load_config(), interactive=False, browser=False)
    try:
        for position, claim in enumerate(todo, start=1):
            sources = []
            for url in claim.source_urls:
                if args.sleep:
                    time.sleep(args.sleep)
                sources.append(source_from_fetched(url, engine.fetcher.fetch(url)))
            stored.append(
                SnapshotClaim(claim.id, claim.text, claim.label, tuple(sources), claim.non_urls)
            )
            save_snapshot(path, stored)
            read = sum(bool(s.text) for s in sources)
            print(
                f"  {position}/{len(todo)}  claim {claim.id}  read {read}/{len(sources)}",
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        print(f"\ninterrupted; {path} is usable with --resume", file=sys.stderr)
    finally:
        engine.close()
    return 0


def _items(rebuild: bool) -> list[Item]:
    if items_path().exists() and not rebuild:
        return load_items(items_path())
    embedder = retrieval.FastEmbedder(EMBEDDER, cache_dir=models_dir())
    try:
        items = build_items(embedder)
    finally:
        embedder.close()
    save_items(items_path(), items)
    print(f"items     {len(items)} written to {items_path()}")
    return items


def cmd_score(args: argparse.Namespace) -> int:
    from proofpath.entailment import OnnxNli

    if not snapshot_path().exists():
        print("error     no AVeriTeC snapshot; run `snapshot` first", file=sys.stderr)
        return 2
    items = _items(args.rebuild_items)
    candidate = CANDIDATES[args.model]
    scorer = OnnxNli(
        cache_dir=models_dir(),
        repo_id=candidate.repo,
        revision=candidate.revision,
        onnx_file=candidate.onnx_file(platform.machine()),
        providers=nli_providers(),
    )
    print(f"nli       {scorer.name}  providers={scorer.providers}")
    pairs = [(p.text, item.claim) for item in items for p in item.passages]
    try:
        started = time.perf_counter()
        blocks: list[np.ndarray] = []
        for start in range(0, len(pairs), SCORE_CHUNK):
            blocks.append(scorer.score(pairs[start : start + SCORE_CHUNK]))
            done = min(start + SCORE_CHUNK, len(pairs))
            print(
                f"  {done}/{len(pairs)} pairs  {time.perf_counter() - started:.0f}s",
                file=sys.stderr,
            )
        elapsed = time.perf_counter() - started
    finally:
        scorer.close()
    flat = np.concatenate(blocks) if blocks else np.zeros((0, 3), dtype=np.float32)
    offsets = np.cumsum([0, *(len(item.passages) for item in items)])
    probs = [flat[offsets[n] : offsets[n + 1]] for n in range(len(items))]
    ms_per_pair = elapsed / max(len(pairs), 1) * 1000
    out = bakeoff_dir() / f"{args.model}.npz"
    save_probs(out, items, probs, ms_per_pair=ms_per_pair)
    print(f"scored    {len(pairs)} pairs, {ms_per_pair:.0f} ms/pair -> {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    print("error     report is implemented in Task 6", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot", help="fetch AVeriTeC gold sources once (network)")
    snap.add_argument("--limit", type=int, default=100, help="first N dev claims")
    snap.add_argument("--sleep", type=float, default=1.0, help="seconds between fetches")
    snap.add_argument("--resume", action="store_true", help="skip claims already stored")
    score = sub.add_parser("score", help="run one NLI model over every stored pair")
    score.add_argument("--model", required=True, choices=sorted(CANDIDATES))
    score.add_argument("--rebuild-items", action="store_true", help="re-run retrieval")
    report = sub.add_parser("report", help="calibrate, evaluate and choose (offline)")
    report.add_argument("--out", default="")
    args = parser.parse_args(argv)
    commands = {"snapshot": cmd_snapshot, "score": cmd_score, "report": cmd_report}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
