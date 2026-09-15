"""Fixtures every test module gets.

``polite.SHARED_THROTTLE`` is process-wide state — one host's last call, seen by
every :class:`~proofpath.polite.PoliteClient` in the process (spec section 13.1) —
so without this the throttling assertions of one module would depend on what an
earlier module had already fetched, and on whose fake clock.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from proofpath.polite import SHARED_THROTTLE


@pytest.fixture(autouse=True)
def reset_shared_throttle() -> Iterator[None]:
    """Every test starts, and leaves, the shared throttle with nothing recorded."""
    SHARED_THROTTLE.reset()
    yield
    SHARED_THROTTLE.reset()
