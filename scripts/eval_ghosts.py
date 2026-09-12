"""Phase 3 harness: reference resolution on the hand-built ghost set.

Runs live against Crossref, OpenAlex and arXiv (no key needed) and caches every
result under the cache dir so a re-run costs no requests. Reports the state
distribution per kind and the two gates from spec section 14: false-ghost rate
on real references (must be ~0) and ghost recall on fabricated ones.

Usage: uv run python scripts/eval_ghosts.py [--limit N] [--refresh] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from proofpath import resolve as rs
from proofpath.config import load_config
from proofpath.paths import cache_dir

SET = Path("tests/data/ghost_set.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="ignore cached results")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in SET.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]
    store = cache_dir() / "datasets" / "ghost_results.json"
    cached: dict[str, dict[str, object]] = (
        json.loads(store.read_text(encoding="utf-8")) if store.exists() and not args.refresh else {}
    )
    resolver = rs.Resolver(contact_email=load_config().contact.email)

    started = time.perf_counter()
    for i, row in enumerate(rows, start=1):
        raw = row["raw"]
        if raw in cached:
            continue
        result = resolver.resolve(raw)
        cached[raw] = {
            "state": result.state.name,
            "best": None if result.best is None else result.best.__dict__,
            "notes": result.notes,
            "match": None if result.match is None else result.match.__dict__,
            "n_candidates": len(result.candidates),
        }
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}  {time.perf_counter() - started:.0f}s", file=sys.stderr)
            store.parent.mkdir(parents=True, exist_ok=True)
            store.write_text(json.dumps(cached, indent=1), encoding="utf-8")
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(cached, indent=1), encoding="utf-8")

    # A "marked" row is an existing row carrying the bibliography marker the document
    # printed ("[7] ", "7. "); it counts as whatever it is underneath (task 7.3).
    def kind_of(row: dict[str, object]) -> str:
        return str(row.get("base") or row["kind"])

    by_kind: dict[str, Counter[str]] = {}
    for row in rows:
        by_kind.setdefault(kind_of(row), Counter())[str(cached[row["raw"]]["state"])] += 1
    marked = [row for row in rows if row["kind"] == "marked"]

    lines = [f"# Ghost set — {date.today().isoformat()}", ""]
    kinds = ", ".join(f"{k} {sum(v.values())}" for k, v in by_kind.items())
    lines.append(f"- references: {len(rows)}  ({kinds})")
    lines.append(
        f"- of those, {len(marked)} carry a printed bibliography marker "
        "(`[7] `, `7. `) and are counted as the kind underneath"
    )
    lines.append(
        "- providers: Crossref bibliographic → OpenAlex search → arXiv (before any ghost call)"
    )
    lines.append(
        f"- thresholds: strong title {rs.STRONG_TITLE}, weak title {rs.WEAK_TITLE}, "
        f"year ±{rs.YEAR_TOLERANCE}"
    )
    lines.append("")
    lines.append("## States per kind")
    lines.append("")
    states = [s.name for s in rs.State]
    lines.append("| kind | " + " | ".join(states) + " |")
    lines.append("|---|" + "---|" * len(states))
    for kind, counter in by_kind.items():
        total = sum(counter.values())
        lines.append(
            f"| {kind} | "
            + " | ".join(f"{counter.get(s, 0)} ({counter.get(s, 0) / total:.0%})" for s in states)
            + " |"
        )
    lines.append("")

    real = by_kind.get("real", Counter())
    fab = by_kind.get("fabricated", Counter())
    mut = by_kind.get("mutated", Counter())
    n_real, n_fab, n_mut = sum(real.values()), sum(fab.values()), sum(mut.values())
    false_ghost = real.get("GHOST", 0) / n_real if n_real else 0.0
    ghost_recall = fab.get("GHOST", 0) / n_fab if n_fab else 0.0
    fab_passed = (fab.get("RESOLVED", 0) + fab.get("RESOLVED_LOW", 0)) / n_fab if n_fab else 0.0
    mut_resolved = mut.get("RESOLVED", 0) / n_mut if n_mut else 0.0
    lines.append("## Gates")
    lines.append("")
    lines.append(
        f"- **false-ghost rate (real → GHOST): {false_ghost:.1%}** — release gate: near zero"
    )
    lines.append(f"- ghost recall (fabricated → GHOST): {ghost_recall:.1%}")
    lines.append(f"- fabricated accepted as RESOLVED/RESOLVED_LOW: {fab_passed:.1%}")
    lines.append(
        f"- mutated real (title kept, author+year wrong) called RESOLVED: {mut_resolved:.1%}; "
        f"called GHOST: {mut.get('GHOST', 0)}"
    )
    lines.append("")

    lines.append("## Real references not resolved")
    lines.append("")
    for row in rows:
        r = cached[row["raw"]]
        if kind_of(row) == "real" and r["state"] not in ("RESOLVED", "RESOLVED_LOW"):
            lines.append(
                f"- `{r['state']}` {row['raw'][:110]}  \n  notes: {'; '.join(r['notes']) or '—'}"
            )  # type: ignore[arg-type]
    lines.append("")
    lines.append("## Fabricated references not called ghost")
    lines.append("")
    for row in rows:
        r = cached[row["raw"]]
        if kind_of(row) == "fabricated" and r["state"] != "GHOST":
            best = r["best"]
            hit = (
                f" → {best['title'][:60]} ({best['year']}, {best['first_author']})"
                if isinstance(best, dict)
                else ""
            )
            lines.append(f"- `{r['state']}` {row['raw'][:100]}{hit}")
    report = "\n".join(lines) + "\n"
    print(report)
    name = f"{date.today().isoformat()}-ghosts.md"
    out = Path(args.out) if args.out else Path("docs/eval") / name
    if out.is_dir():
        out = out / name
    out.write_text(report, encoding="utf-8")
    print(f"written   {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
