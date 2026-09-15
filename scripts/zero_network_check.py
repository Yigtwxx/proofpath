"""Prove that a warm `proofpath check` reads no source over the network (task 7.2).

Both sockets the run can use -- `httpx.Client.send` and `curl_cffi.requests.get`,
which every provider and every rung of the fetch ladder goes through -- are replaced
by a guard that counts the attempt and fails it the way a pulled cable would. The
same document is then verified again with a fresh `Engine.default`, and the counts
say which stages still needed the wire.

What the counts say is which stages still need the wire. Since task 8.7 cached the
resolution and the retraction check as well as the source text (schema v3), a warm
run of an unchanged document reaches `blocked 0` and every network-bound stage is
attributed to `cache` -- a warm run is now an offline run.

One state is deliberately never cached, and it is the one that can still put a
number on that line: `UNVERIFIED (provider unavailable)`. An outage says nothing
about the reference (product rule 2), so a run that hit one asks again, and this
script will count that attempt. Re-run the normal check until nothing is unavailable
before reading the count as a regression.

Run it after a normal run of the same document, which is what fills the cache.

Usage: uv run python scripts/zero_network_check.py [PATH]
"""

from __future__ import annotations

import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

import curl_cffi.requests
import httpx

from proofpath import verify
from proofpath.config import load_config

DEFAULT_TARGET = Path("tests/data/draft-live.md")


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET
    blocked: Counter[str] = Counter()

    def guard(*_args: Any, **_kwargs: Any) -> NoReturn:
        """Fail like an unplugged cable, and record who pulled on it.

        A transport error rather than a bespoke exception on purpose: the providers
        and the fetch ladder already turn one into a reported honesty state, so the
        run finishes and its coverage block stays readable instead of the first
        attempt aborting everything.
        """
        # The nearest proofpath frame, not the nearest frame: httpx calls its own
        # `Client.send` from `Client.get`, so the immediate caller is always httpx.
        # What the count has to name is *which stage* is not offline yet.
        package = Path(verify.__file__).parent
        proofpath_frames = [
            frame for frame in traceback.extract_stack() if Path(frame.filename).parent == package
        ]
        blocked[Path(proofpath_frames[-1].filename).stem if proofpath_frames else "?"] += 1
        raise httpx.ConnectError("network disabled by zero_network_check")

    httpx.Client.send = guard  # type: ignore[method-assign]
    curl_cffi.requests.get = guard  # type: ignore[assignment]

    with verify.Engine.default(load_config(), interactive=False) as engine:
        report = verify.verify(target, engine)

    full, abstract, unverified = report.coverage.pct()
    print(f"target       {target}")
    print(f"exit code    {report.exit_code()}")
    print(f"coverage     fulltext {full}% / abstract {abstract}% / unverified {unverified}%")
    for stage in report.stages:
        print(f"stage        {stage.name:<12} {stage.by:<28} {stage.elapsed:.1f}s")
    print(f"blocked      {sum(blocked.values())} attempt(s) by module: {dict(blocked) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
