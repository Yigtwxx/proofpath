"""The ``config set`` spellings a "not permitted" line points at, named once.

A leaf: it imports nothing from the package, so the fetch ladder, the report and the
output layer can all read the same strings without a cycle (``report`` imports
``fetch``, and ``fetch`` needs ``NETWORK_SETTING`` for its denied note). Each surface
puts its own command in front: ``/config set`` in the TUI, ``proofpath config set``
on the command line and in the report. The fact a line states comes first; the
setting is appended after it (permission hints, 2026-09-17).
"""

from __future__ import annotations

#: ``permissions.install_browser`` set to ``ask``: the browser step of the fetch
#: ladder asks before it installs and runs Chromium.
BROWSER_SETTING = "permissions.install_browser ask"
#: ``permissions.install_model`` set to ``ask``: a run that wants the accurate NLI
#: profile asks before it downloads the model, instead of falling back to the default.
MODEL_SETTING = "permissions.install_model ask"
#: ``permissions.install_model`` set to ``allow``: the download goes ahead without a
#: question, which is the only way it can where there is no terminal to ask on (CI).
MODEL_ALLOW_SETTING = "permissions.install_model allow"
#: ``permissions.network`` set to ``allow``: every network-bound stage runs.
NETWORK_SETTING = "permissions.network allow"
#: ``search.provider`` set to ``tavily``: a text that cites nothing is searched with
#: the user's own Tavily key (OPEN-ITEMS 17.1a).
SEARCH_SETTING = "search.provider tavily"
#: ``search.provider`` set to ``searxng``: the other v1 provider, the user's own
#: SearXNG instance addressed by ``search.base_url`` instead of a key.
SEARXNG_SETTING = "search.provider searxng"
