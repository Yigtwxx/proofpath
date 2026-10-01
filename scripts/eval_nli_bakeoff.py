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
    passage_offsets,
    pending,
    save_items,
    save_probs,
    save_snapshot,
    source_from_fetched,
    split_at,
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
            # Keys are not unique (dev repeats the pair 1245:7662395); consumers are positional.
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
    items.extend(averitec_items(load_snapshot(snapshot_path()), embedder))
    return items


def averitec_items(snapshot: Sequence[SnapshotClaim], embedder: retrieval.Embedder) -> list[Item]:
    """One item per readable source; the key carries the source's position in its claim."""
    items: list[Item] = []
    for claim in snapshot:
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


def split_blocks(flat: np.ndarray, items: Sequence[Item]) -> list[np.ndarray]:
    """Cut the flat ``(n_pairs, 3)`` scores into one block per item, in item order."""
    offsets = passage_offsets(items)
    if offsets[-1] != len(flat):
        raise ValueError(f"{len(flat)} score rows for {offsets[-1]} passages")
    return split_at(flat, offsets)


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
        return 130
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
    probs = split_blocks(flat, items)
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


@dataclass(frozen=True)
class NumericLayer:
    """Numeric-layer firings at one k, as ``(fired, correct)``; model-independent."""

    train: tuple[int, int]
    dev: tuple[int, int]


def outcomes_for(rows: Sequence[ReportRow]) -> list[bakeoff.Outcome]:
    """The numbers the decision rule reads, one per report row."""
    return [
        bakeoff.Outcome(
            r.combo,
            r.large,
            r.scifact.macro_f1,
            r.averitec.accuracy_readable,
            r.scifact.asserted_without_passage + r.averitec.asserted_without_passage,
        )
        for r in rows
    ]


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


def on_grid_floor(decide: float) -> bool:
    """True when ``decide`` is the lowest cut the sweep tried: the best may lie below."""
    return decide == bakeoff.GRID[0]


def _decide_cell(decide: float) -> str:
    return f"{decide:.2f} (grid floor)" if on_grid_floor(decide) else f"{decide:.2f}"


# The last AVeriTeC run before this one (docs/eval/2026-09-16-averitec.md): its snapshot,
# its cuts and its denominators.
PUBLISHED_AVERITEC = "2026-09-16"
PUBLISHED_ANSWERABLE = 89
PUBLISHED_READABLE = 61


def _result_lines(rows: Sequence[ReportRow], choice: bakeoff.Choice) -> list[str]:
    """The verdict of the rule, the winner named once, and its gain over the best base row."""
    if choice.winner is None:
        return [f"**No change.** {choice.reason}"]
    label = choice.winner.label()
    if choice.reason.startswith(label):
        lines = [f"**Winner:** {choice.reason}."]
    else:
        lines = [f"**Winner:** {label} — {choice.reason}."]
    outcomes = outcomes_for(rows)
    winner = next(row for row in outcomes if row.combo == choice.winner)
    best = bakeoff.best_eligible_base(outcomes, baseline=BASELINE)
    if best is None:
        lines.append("No `base` row is eligible under step 1, so the winner has no base bar.")
    elif best.combo == winner.combo:
        lines.append("The winner is itself the best eligible `base` row.")
    else:
        lines.append(
            f"Over the best eligible `base` row, {best.combo.label()} "
            f"({best.macro_f1:.3f}), the winner is {winner.macro_f1 - best.macro_f1:+.3f} "
            "macro-F1."
        )
    return [lines[0], "", *lines[1:]]


def _numeric_lines(rows: Sequence[ReportRow], numeric: Mapping[int, NumericLayer]) -> list[str]:
    """Per-k firings, and how much of the `high` tier the rule rather than a model holds."""
    lines = [
        "## Numeric layer",
        "",
        "The numeric layer (spec §10) runs on the same top-k passages before any model. A "
        "mismatch decides REFUTED by rule at score 1.0, and no model is credited with it. "
        "It depends on k only, so one row per k covers every model and aggregation. A "
        "firing is correct when the gold label is REFUTED.",
        "",
        "| k | SciFact train fired | train correct | SciFact dev fired | dev correct |",
        "|---|---|---|---|---|",
    ]
    for k in sorted(numeric):
        layer = numeric[k]
        lines.append(_md(k, layer.train[0], layer.train[1], layer.dev[0], layer.dev[1]))
    at_one = sum(row.thresholds.high >= 1.0 for row in rows)
    rule_only = 0
    for row in rows:
        high = next(t for t in row.scifact.tiers if t.tier == "high")
        fired = numeric[row.combo.k].dev[0] if row.combo.k in numeric else -1
        rule_only += high.n > 0 and high.n == fired
    lines.extend(
        [
            "",
            "Rule-decided rows score exactly 1.0, the top of the scale, so they are among "
            "the first rows the `high` cut's walk down SciFact train meets: a wrong firing "
            "lowers the precision every lower cut is measured with, and a `high` cut of "
            "1.000000 means no score below 1.0 held "
            f"{bakeoff.HIGH_TARGET:.2f} precision, so only a score of 1.0 reaches the tier. "
            f"The `high` cut is 1.000000 in {at_one} of {len(rows)} combinations. In "
            f"{rule_only} of {len(rows)}, every SciFact dev `high` verdict is a rule firing "
            "(the high-tier n equals the dev firings at that k): there the high tier on "
            "dev comes from the rule, not the model.",
        ]
    )
    return lines


def _limits_lines(rows: Sequence[ReportRow], choice: bakeoff.Choice) -> list[str]:
    outcomes = outcomes_for(rows)
    top = max((row.macro_f1 for row in outcomes), default=0.0)
    if choice.winner is None:
        chosen = f"The best row's {top:.3f}"
    else:
        won = next(row for row in outcomes if row.combo == choice.winner).macro_f1
        chosen = f"The winner's {won:.3f}"
    base = next((row for row in rows if row.combo == BASELINE), None)
    base_acc = f"dev accuracy {base.scifact.accuracy:.3f}" if base else "a lower dev accuracy"
    counted = rows[0].averitec.counted if rows else 0
    readable = rows[0].averitec.readable if rows else 0
    floor = sum(on_grid_floor(row.thresholds.decide) for row in rows)
    if floor:
        floor_line = (
            f"- **Grid floor.** {floor} of {len(rows)} combinations put `decide` on the grid "
            f"floor ({bakeoff.GRID[0]:.2f}), marked `(grid floor)` above: SciFact train "
            "accuracy was best at the lowest cut tried, so the best cut may lie below the grid."
        )
    else:
        floor_line = (
            f"- **Grid floor.** No combination puts `decide` on the grid floor "
            f"({bakeoff.GRID[0]:.2f})."
        )
    return [
        "## Limits",
        "",
        "- **Dev chooses and reports.** SciFact dev both picks the winner and reports it. "
        f"{chosen} macro-F1 comes from the split that chose it, out of {len(rows)} "
        f"combinations whose highest dev macro-F1 is {top:.3f}, so it is optimistic; a "
        "held-out split would likely read lower.",
        "- **int8 scores depend on the batch.** The candidate files are dynamically "
        "quantised to int8, so a pair's probabilities depend on what else is in its batch. "
        "Re-scoring 48 SciFact dev top-1 pairs one at a time moved probabilities by up to "
        "0.197 (mean 0.008) and flipped 1 argmax (measured 2026-10-01 by the final review). "
        f"That likely explains why `base` × k=1 × `max` reaches {base_acc} here "  # noqa: RUF001
        "(0.597 even at decide 0.45) against the 0.609 published on 2026-09-12 with "
        "dev-fitted cuts.",
        "- **The AVeriTeC snapshot is new.** It is not the "
        f"{PUBLISHED_AVERITEC} run's: coverage improved since (then {PUBLISHED_READABLE} of "
        f"{PUBLISHED_ANSWERABLE} answerable claims were readable, now {readable} of "
        f"{counted}), so AVeriTeC numbers here are not comparable with that run's.",
        floor_line,
    ]


def render_report(
    rows: Sequence[ReportRow],
    choice: bakeoff.Choice,
    *,
    speed: Mapping[str, float],
    sizes: Mapping[str, float | None],
    counts: Mapping[str, int],
    numeric: Mapping[int, NumericLayer],
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
    lines.extend(_result_lines(rows, choice))
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
            f"AVeriTeC majority baseline: {baseline_ave:.3f}. The {PUBLISHED_AVERITEC} run "
            "reported 0.270 (all) and 0.361 (readable), but on a different snapshot with "
            f"different cuts, over {PUBLISHED_ANSWERABLE} answerable / {PUBLISHED_READABLE} "
            f"readable claims; this run counts {counted} answerable / {readable} readable. "
            "A gap between the two runs is partly a coverage change, not only a model gain.",
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
                _decide_cell(row.thresholds.decide),
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
    lines.extend(["", *_numeric_lines(rows, numeric), "", *_limits_lines(rows, choice)])
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
    numeric: dict[int, NumericLayer] = {}
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
            # The rule reads the items only, so every model gives the same firings.
            if k not in numeric:
                numeric[k] = NumericLayer(
                    bakeoff.numeric_firings(train, k=k), bakeoff.numeric_firings(dev, k=k)
                )
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
    choice = bakeoff.choose(outcomes_for(rows), baseline=BASELINE)
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
        numeric=numeric,
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
