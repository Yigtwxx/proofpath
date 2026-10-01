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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from proofpath import retrieval
from proofpath.entailment import pick_onnx_file
from proofpath.eval import averitec, bakeoff, scifact
from proofpath.eval.bakeoff_data import (
    Item,
    SnapshotClaim,
    load_items,
    load_probs,
    load_snapshot,
    pending,
    save_items,
    save_probs,
    save_snapshot,
    source_from_fetched,
)
from proofpath.models import Label, Passage
from proofpath.paths import cache_dir, models_dir
from proofpath.pipeline import Thresholds

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


def items_stale(items: Path, snapshot: Path) -> bool:
    """True when the snapshot was written after ``items`` was built."""
    return snapshot.stat().st_mtime > items.stat().st_mtime


def _items(rebuild: bool) -> list[Item]:
    if items_path().exists() and not rebuild:
        if not items_stale(items_path(), snapshot_path()):
            return load_items(items_path())
        print("items     the snapshot is newer than items.json; rebuilding")
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


@dataclass(frozen=True)
class ReportRow:
    combo: bakeoff.Combo
    large: bool
    thresholds: Thresholds
    scifact: bakeoff.SciFactResult
    averitec: bakeoff.AveritecResult


def model_size_mb(model: str) -> float | None:
    """Size of the ONNX file the run used, or None when it is not in the cache."""
    from huggingface_hub import try_to_load_from_cache

    candidate = CANDIDATES[model]
    found = try_to_load_from_cache(
        candidate.repo,
        candidate.onnx_file(platform.machine()),
        cache_dir=str(models_dir()),
        revision=candidate.revision,
    )
    return Path(found).stat().st_size / 1e6 if isinstance(found, str) else None


def _md(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _tier_cell(tier: bakeoff.TierStat) -> str:
    if tier.precision is None or tier.interval is None:
        return f"{tier.n} · too few to judge"
    low, high = tier.interval
    return f"{tier.n} · {tier.precision:.3f} (Wilson {low:.2f}–{high:.2f})"  # noqa: RUF001


def _without_passage_cell(count: int) -> str:
    return f"**{count} — rule 1 violated**" if count else "0"


def render_report(
    rows: Sequence[ReportRow],
    choice: bakeoff.Choice,
    *,
    speed: Mapping[str, float],
    sizes: Mapping[str, float | None],
    counts: Mapping[str, int],
    today: str,
    machine: str,
) -> str:
    scored = {row.combo.model for row in rows}
    skipped = [m for m in CANDIDATES if m not in scored]
    base_ave = rows[0].averitec if rows else None
    counted = base_ave.counted if base_ave else 0
    readable = base_ave.readable if base_ave else 0
    baseline_ave = rows[0].averitec.majority_baseline if rows else 0.0
    lines = [
        f"# NLI bake-off — {today}",
        "",
        "- spec: `docs/superpowers/specs/2026-10-01-nli-bakeoff-design.md`",
        f"- calibrated on SciFact train ({counts.get('scifact-train', 0)} pairs); reported on "
        f"SciFact dev ({counts.get('scifact-dev', 0)} pairs) and AVeriTeC dev "
        f"({counts.get('averitec', 0)} claims, {counted} answerable (3-way gold label), "
        f"{readable} of those with a readable source; frozen snapshot, browser not permitted)",
        f"- embedder `{EMBEDDER}`, top {bakeoff.TOP_N} passages per item, numeric layer on",
        f"- machine: {machine}",
        "- coverage: "
        + (f"not scored: {', '.join(skipped)}" if skipped else "all candidate models were scored"),
        "",
        "## Decision rule (fixed before any number was seen)",
        "",
        bakeoff.DECISION_RULE,
        "",
        "## Result",
        "",
    ]
    if choice.winner is None:
        lines.append(f"**No change.** {choice.reason}")
    else:
        lines.append(f"**Winner:** {choice.winner.label()} — {choice.reason}")
    lines.extend(
        ["", "## Models", "", "| model | repo | size (MB) | ms/pair |", "|---|---|---|---|"]
    )
    for model in dict.fromkeys(row.combo.model for row in rows):
        size = sizes.get(model)
        lines.append(
            _md(
                model,
                f"`{CANDIDATES[model].repo}`",
                "—" if size is None else f"{size:.0f}",
                f"{speed.get(model, 0.0):.0f}",
            )
        )
    lines.extend(
        [
            "",
            "## Every combination",
            "",
            f"AVeriTeC majority baseline: {baseline_ave:.3f}. Published numbers to compare: "
            "0.270 (all) and 0.361 (readable), 2026-09-16.",
            "",
            "| model | k | aggregation | decide | medium | high | dev acc | dev macro-F1 "
            "| F1 S / R / NEI | rationale F1 | AVeriTeC all | AVeriTeC readable | w/o passage |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        f1 = row.scifact.f1
        lines.append(
            _md(
                row.combo.model,
                row.combo.k,
                row.combo.aggregation,
                f"{row.thresholds.decide:.2f}",
                f"{row.thresholds.medium:.6f}",
                f"{row.thresholds.high:.6f}",
                f"{row.scifact.accuracy:.3f}",
                f"{row.scifact.macro_f1:.3f}",
                f"{f1[Label.SUPPORTED]:.2f} / {f1[Label.REFUTED]:.2f} / {f1[Label.NEI]:.2f}",
                f"{row.scifact.rationale_f1:.3f}",
                f"{row.averitec.accuracy_all:.3f}",
                f"{row.averitec.accuracy_readable:.3f}",
                _without_passage_cell(
                    row.scifact.asserted_without_passage + row.averitec.asserted_without_passage
                ),
            )
        )
    labels = sorted({label for row in rows for label in row.averitec.per_label})
    lines.extend(
        [
            "",
            "## AVeriTeC per label",
            "",
            "Cells are correct/n per gold label. `n (not scored)` marks a gold label with no "
            "3-way verdict; it is not a model failure.",
            "",
            _md("model", "k", "aggregation", *labels),
            _md(*(["---"] * (3 + len(labels)))),
        ]
    )
    for row in rows:
        cells = []
        for label in labels:
            seen, right = row.averitec.per_label.get(label, (0, 0))
            cells.append(
                f"{seen} (not scored)" if averitec.to_label(label) is None else f"{right}/{seen}"
            )
        lines.append(_md(row.combo.model, row.combo.k, row.combo.aggregation, *cells))
    lines.extend(
        [
            "",
            "## Tiers on SciFact dev",
            "",
            f"Cells are n · precision (95 % Wilson interval); n below {bakeoff.MIN_TIER_N} is "
            "too few to judge.",
            "",
            "| model | k | aggregation | high | medium | low |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        tiers = {t.tier: t for t in row.scifact.tiers}
        lines.append(
            _md(
                row.combo.model,
                row.combo.k,
                row.combo.aggregation,
                *(_tier_cell(tiers[name]) for name in ("high", "medium", "low")),
            )
        )
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    if not items_path().exists() or not snapshot_path().exists():
        print("error     run `snapshot` and `score --model base` first", file=sys.stderr)
        return 2
    if items_stale(items_path(), snapshot_path()):
        print(
            "error     the snapshot is newer than items.json, so the stored items are stale.\n"
            "          The next `score` run rebuilds items.json automatically; after that, "
            "re-score every model with `score --model <name>`.",
            file=sys.stderr,
        )
        return 2
    items = load_items(items_path())
    snapshot = load_snapshot(snapshot_path())
    rows: list[ReportRow] = []
    speed: dict[str, float] = {}
    for model, candidate in CANDIDATES.items():
        path = bakeoff_dir() / f"{model}.npz"
        if not path.exists():
            print(f"skip      {model}: not scored", file=sys.stderr)
            continue
        try:
            probs, speed[model] = load_probs(path, items)
        except ValueError as error:
            print(
                f"error     {model}: stored scores do not match the current items ({error}); "
                f"run `score --model {model}`",
                file=sys.stderr,
            )
            return 2
        scored = [bakeoff.Scored(i, p) for i, p in zip(items, probs, strict=True)]
        train = [s for s in scored if s.item.dataset == "scifact-train"]
        dev = [s for s in scored if s.item.dataset == "scifact-dev"]
        ave = [s for s in scored if s.item.dataset == "averitec"]
        for k in bakeoff.KS:
            for aggregation in bakeoff.AGGREGATIONS:
                thresholds = bakeoff.calibrate(
                    bakeoff.calibration_rows(train, k=k, aggregation=aggregation)
                )
                rows.append(
                    ReportRow(
                        bakeoff.Combo(model, k, aggregation),
                        candidate.large,
                        thresholds,
                        bakeoff.evaluate_scifact(dev, thresholds, k=k, aggregation=aggregation),
                        bakeoff.evaluate_averitec(
                            snapshot, ave, thresholds, k=k, aggregation=aggregation
                        ),
                    )
                )
    if "base" not in speed:
        print(
            "error     the baseline model is not scored; run `score --model base`", file=sys.stderr
        )
        return 2
    outcomes = [
        bakeoff.Outcome(
            r.combo,
            r.large,
            r.scifact.macro_f1,
            r.averitec.accuracy_readable,
            r.scifact.asserted_without_passage + r.averitec.asserted_without_passage,
        )
        for r in rows
    ]
    choice = bakeoff.choose(outcomes, baseline=BASELINE)
    counts = {
        "scifact-train": sum(i.dataset == "scifact-train" for i in items),
        "scifact-dev": sum(i.dataset == "scifact-dev" for i in items),
        "averitec": len(snapshot),
    }
    report = render_report(
        rows,
        choice,
        speed=speed,
        sizes={model: model_size_mb(model) for model in speed},
        counts=counts,
        today=date.today().isoformat(),
        machine=f"{platform.system()} {platform.machine()}",
    )
    out = (
        Path(args.out)
        if args.out
        else Path("docs/eval") / f"{date.today().isoformat()}-nli-bakeoff.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"written   {out}")
    return 0


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
