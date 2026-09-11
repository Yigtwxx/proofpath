"""Phase 1 harness: measure retrieval + entailment on SciFact dev.

Usage:
    uv run python scripts/eval_scifact.py [--split dev] [--limit N] [--k 1,3,5]
        [--embedder BAAI/bge-small-en-v1.5] [--nli-file onnx/model_qint8_arm64.onnx]
        [--providers cpu] [--out docs/eval/<date>-scifact-<split>.md]

Everything runs locally. The only network use is the one-time download of the
dataset tarball and the two models.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from proofpath import numerics
from proofpath.device import onnx_providers
from proofpath.entailment import LABEL_ORDER, OnnxNli, pick_onnx_file
from proofpath.eval import metrics, scifact
from proofpath.models import Label, Passage
from proofpath.paths import cache_dir, models_dir
from proofpath.pipeline import Thresholds, aggregate
from proofpath.retrieval import FastEmbedder, Hit, rank

GRID = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
PROVIDER_NAMES = {
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
_SUPPORTED_COL = LABEL_ORDER.index(Label.SUPPORTED)
_REFUTED_COL = LABEL_ORDER.index(Label.REFUTED)


@dataclass(frozen=True)
class Scored:
    pair: scifact.Pair
    ranked_indices: list[int]  # passage indices, best first, over the whole abstract
    hits: list[Hit]
    probs: np.ndarray  # (n_hits, 3) in LABEL_ORDER


Row = tuple[Label, Label, float, int | None]


def _row(scored: Scored, k: int, thresholds: Thresholds, *, use_numerics: bool) -> Row:
    """(decided label, strongest non-NEI label, its score, passage index or None)."""
    if use_numerics:
        numeric = numerics.check(scored.pair.claim, [h.passage for h in scored.hits[:k]])
        if numeric is not None and numeric.mismatch:
            return Label.REFUTED, Label.REFUTED, 1.0, numeric.passage.index
    verdict = aggregate(scored.hits[:k], scored.probs[:k], thresholds=thresholds)
    strongest = verdict.label
    if strongest is Label.NEI and scored.hits[:k]:
        sub = scored.probs[:k, [_SUPPORTED_COL, _REFUTED_COL]]
        strongest = (Label.SUPPORTED, Label.REFUTED)[int(sub.argmax()) % 2]
    index = verdict.passage.index if verdict.passage else None
    return verdict.label, strongest, verdict.score, index


def _md_row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0, help="first N pairs only")
    parser.add_argument("--k", default="1,3,5")
    parser.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--nli-file", default=pick_onnx_file(platform.machine()))
    parser.add_argument("--providers", default="", help="comma list, e.g. cpu or coreml,cpu")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--no-numerics", action="store_true", help="skip the numeric layer (spec section 10)"
    )
    args = parser.parse_args(argv)

    ks = [int(x) for x in args.k.split(",")]
    use_numerics = not args.no_numerics
    providers = (
        [PROVIDER_NAMES[p.strip().lower()] for p in args.providers.split(",")]
        if args.providers
        else onnx_providers()
    )

    t0 = time.perf_counter()
    tarball = scifact.ensure_downloaded(cache_dir())
    dataset = scifact.load(tarball, split=args.split)
    pairs = dataset.pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"dataset   scifact/{args.split}  {len(pairs)} pairs  (sha256 {scifact.SHA256[:12]}…)")

    embedder = FastEmbedder(args.embedder, cache_dir=models_dir(), providers=providers)
    scorer = OnnxNli(cache_dir=models_dir(), onnx_file=args.nli_file, providers=providers)
    print(f"embedder  {embedder.name}  dim={embedder.dim}")
    print(f"nli       {scorer.name}  providers={scorer.providers}")
    print(f"setup     {time.perf_counter() - t0:.1f}s")

    # Score every (claim, abstract) once with k = all sentences; smaller k are prefixes.
    t1 = time.perf_counter()
    scored: list[Scored] = []
    for i, pair in enumerate(pairs, start=1):
        doc = dataset.corpus[pair.doc_id]
        passages = [Passage(text, str(doc.doc_id), idx) for idx, text in enumerate(doc.sentences)]
        hits = rank(pair.claim, passages, embedder, k=len(passages))
        probs = scorer.score([(h.passage.text, pair.claim) for h in hits])
        scored.append(Scored(pair, [h.passage.index for h in hits], hits, probs))
        if i % 50 == 0:
            print(f"  scored {i}/{len(pairs)}  {time.perf_counter() - t1:.0f}s", file=sys.stderr)
    elapsed = time.perf_counter() - t1
    ms_per_pair = elapsed / max(len(pairs), 1) * 1000
    print(f"scoring   {elapsed:.1f}s  ({ms_per_pair:.0f} ms/pair)")

    gold = [s.pair.label for s in scored]
    rationale = [s.pair.rationale for s in scored]
    ranked = [s.ranked_indices for s in scored]

    lines: list[str] = []
    lines.append(f"# SciFact {args.split} — {date.today().isoformat()}")
    lines.append("")
    lines.append(f"- pairs: {len(pairs)}  (limit={args.limit or 'none'})")
    lines.append(f"- embedder: `{embedder.name}`")
    lines.append(f"- nli: `{scorer.name}`  providers: `{', '.join(scorer.providers)}`")
    lines.append(f"- numeric layer: {'on' if use_numerics else 'off (--no-numerics)'}")
    lines.append(
        f"- machine: {platform.system()} {platform.machine()}  "
        f"scoring {elapsed:.1f}s, {ms_per_pair:.0f} ms/pair"
    )
    lines.append("")

    majority = max(set(gold), key=gold.count)
    lines.append("## Baselines")
    lines.append("")
    lines.append("| baseline | accuracy | macro-F1 |")
    lines.append("|---|---|---|")
    for name, pred in (
        (f"majority label ({majority.value})", [majority] * len(gold)),
        ("source exists → SUPPORTED", [Label.SUPPORTED] * len(gold)),
    ):
        lines.append(
            _md_row(
                name, f"{metrics.accuracy(gold, pred):.3f}", f"{metrics.macro_f1(gold, pred):.3f}"
            )
        )
    lines.append("")

    lines.append("## Retrieval (assumption 3.6)")
    lines.append("")
    lines.append("| k | recall@k (any gold sentence in top-k) |")
    lines.append("|---|---|")
    for k in ks:
        lines.append(f"| {k} | {metrics.recall_at_k(rationale, ranked, k=k):.3f} |")
    lines.append("")

    lines.append("## Entailment — threshold sweep per k")
    lines.append("")
    lines.append(
        "| k | decide | accuracy | macro-F1 | rationale F1 | high cut | medium cut "
        "| asserted w/o passage |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    best_overall: tuple[float, int, metrics.SweepResult] | None = None
    numeric_notes: list[str] = []
    for k in ks:
        loose = Thresholds(decide=0.0, high=1.0, medium=1.0)
        rows: list[metrics.Row] = []
        for s in scored:
            _, strongest, score, _ = _row(s, k, loose, use_numerics=use_numerics)
            rows.append((s.pair.label, strongest, score))
        best = metrics.sweep_decide(rows, grid=GRID)
        high, medium = metrics.tier_cutpoints(
            rows, decide=best.threshold, high_precision=0.85, medium_precision=0.70
        )
        tuned = Thresholds(decide=best.threshold, high=max(high, medium), medium=medium)
        preds, pred_rationale, missing = [], [], 0
        for s in scored:
            label, _, _, index = _row(s, k, tuned, use_numerics=use_numerics)
            preds.append(label)
            pred_rationale.append(frozenset({index}) if index is not None else frozenset())
            missing += label is not Label.NEI and index is None
        rf1 = metrics.rationale_f1(rationale, pred_rationale)
        if use_numerics:
            fired = [
                s
                for s in scored
                if (n := numerics.check(s.pair.claim, [h.passage for h in s.hits[:k]]))
                and n.mismatch
            ]
            correct = sum(s.pair.label is Label.REFUTED for s in fired)
            numeric_notes.append(f"- k={k}: numeric layer refuted {len(fired)}, {correct} correct")
        lines.append(
            _md_row(
                k,
                f"{best.threshold:.2f}",
                f"{metrics.accuracy(gold, preds):.3f}",
                f"{metrics.macro_f1(gold, preds):.3f}",
                f"{rf1:.3f}",
                f"{tuned.high:.2f}",
                f"{tuned.medium:.2f}",
                missing,
            )
        )
        if best_overall is None or best.accuracy > best_overall[0]:
            best_overall = (best.accuracy, k, best)
    lines.append("")

    if numeric_notes:
        lines.append("Numeric layer firings on the top-k passages:")
        lines.append("")
        lines.extend(numeric_notes)
        lines.append("")

    assert best_overall is not None
    _, best_k, best_sweep = best_overall
    trivial = max(
        metrics.accuracy(gold, [majority] * len(gold)),
        metrics.accuracy(gold, [Label.SUPPORTED] * len(gold)),
    )
    margin = best_sweep.accuracy - trivial
    lines.append("## Verdict on Phase 1")
    lines.append("")
    lines.append(
        f"Best: k={best_k}, decide={best_sweep.threshold:.2f}, accuracy {best_sweep.accuracy:.3f} "
        f"vs best trivial baseline {trivial:.3f} → margin **{margin:+.3f}**."
    )
    lines.append("")
    lines.append("Kill criterion: margin near zero after tuning means stop before Phase 2.")

    report = "\n".join(lines) + "\n"
    print()
    print(report)
    default_out = Path("docs/eval") / f"{date.today().isoformat()}-scifact-{args.split}.md"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"written   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
