"""The two ``config set`` spellings a "not permitted" line points at, named once.

A leaf: it imports nothing from the package, so the fetch ladder, the report and the
output layer can all read the same two strings without a cycle (``report`` imports
``fetch``, and ``fetch`` needs ``NETWORK_SETTING`` for its denied note). Each surface
puts its own command in front: ``/config set`` in the TUI, ``proofpath config set``
on the command line and in the report. The fact a line states comes first; the
setting is appended after it (permission hints, 2026-09-17).
"""

from __future__ import annotations

#: ``permissions.install_browser`` set to ``ask``: the browser step of the fetch
#: ladder asks before it installs and runs Chromium.
BROWSER_SETTING = "permissions.install_browser ask"
#: ``permissions.network`` set to ``allow``: every network-bound stage runs.
NETWORK_SETTING = "permissions.network allow"
