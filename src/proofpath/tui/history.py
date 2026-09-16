"""The input bar's command history: a file of lines and a shell-style walk over them.

Pure: no Textual, no app. The bar asks :meth:`History.previous` and
:meth:`History.next` and shows what comes back; the file is the user's own typing
and nothing else (spec ``2026-09-16-tui-conveniences-design.md`` section 2.1).
"""

from __future__ import annotations

import logging
from pathlib import Path

from proofpath.paths import history_path

logger = logging.getLogger(__name__)

#: How many lines the file keeps. Older ones are dropped on save.
LIMIT = 500


class History:
    """Submitted lines, newest last, and a cursor for walking back through them.

    The walk has shell semantics: ``previous`` moves towards the oldest entry and
    stops there; ``next`` moves towards the newest and, one step past it, hands back
    the *draft* -- whatever the bar held when the walk began. ``add`` and ``reset``
    end a walk; the bar calls ``reset`` on any edit.
    """

    def __init__(self, path: Path | None = None, *, limit: int = LIMIT) -> None:
        self._path = path if path is not None else history_path()
        self._limit = limit
        self._entries: list[str] = []
        #: Index into ``_entries`` while walking; ``None`` when not walking.
        self._cursor: int | None = None
        self._draft = ""

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def load(self) -> None:
        """Read the file. A missing or unreadable file is an empty history, not an error.

        ``UnicodeDecodeError`` is caught alongside ``OSError``: a crash mid-write can
        leave a truncated or non-UTF-8 byte sequence on disk, and the file is only
        history, so treating it as empty (rather than raising and crashing the TUI at
        startup) is the right answer.
        """
        # A stale cursor from a previous load must never outlive that list.
        self.reset()
        try:
            text = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self._entries = []
            return
        lines = [line for line in text.splitlines() if line.strip()]
        self._entries = lines[-self._limit :]

    def add(self, line: str) -> None:
        """Record a submitted line, cap the list, write the file, end any walk."""
        self.reset()
        if not line.strip():
            return
        if self._entries and self._entries[-1] == line:
            return
        self._entries.append(line)
        del self._entries[: -self._limit]
        self._save()

    def previous(self, draft: str) -> str | None:
        """One step towards the oldest entry, or ``None`` at the end (or when empty).

        ``draft`` is remembered on the first step so ``next`` can bring it back.
        """
        if not self._entries:
            return None
        if self._cursor is None:
            self._draft = draft
            self._cursor = len(self._entries) - 1
        elif self._cursor == 0:
            return None
        else:
            self._cursor -= 1
        return self._entries[self._cursor]

    def next(self) -> str | None:
        """One step towards the newest entry; one past it is the draft; then ``None``."""
        if self._cursor is None:
            return None
        self._cursor += 1
        if self._cursor >= len(self._entries):
            self._cursor = None
            return self._draft
        return self._entries[self._cursor]

    def reset(self) -> None:
        """End a walk. The next ``previous`` starts again from the newest entry."""
        self._cursor = None
        self._draft = ""

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("".join(f"{line}\n" for line in self._entries), encoding="utf-8")
            # Shell histories are 0600 and this one is a log of URLs, which can carry
            # tokens. Best effort: on Windows ``chmod`` only toggles the read-only bit.
            self._path.chmod(0o600)
        except OSError as exc:
            # The session keeps its in-memory history; only persistence is lost.
            logger.debug("could not write %s: %s", self._path, exc)
