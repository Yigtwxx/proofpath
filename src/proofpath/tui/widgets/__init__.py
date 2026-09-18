"""The TUI's widgets (spec section 5 of the v2 design).

No logic moves in here: every widget renders a ``Report``, ``Finding``, ``Stage`` or
``Progress`` it is handed. The modules are split by what they draw — a run's block,
a finding, the docked footer, the prompt — and share layout helpers through
:mod:`~proofpath.tui.widgets._shared`.
"""

from __future__ import annotations

from proofpath.tui.widgets._shared import PANEL_FLOOR, accent_for, panelled, textual_colour
from proofpath.tui.widgets.banner import Banner
from proofpath.tui.widgets.config_panel import ConfigPanel
from proofpath.tui.widgets.finding import FindingLine, FindingsRule, Line
from proofpath.tui.widgets.footer import CoverageFooter
from proofpath.tui.widgets.prompt import (
    ANSWER_ID,
    ANSWER_LABELS,
    FrameEdge,
    PermissionPrompt,
    Prompt,
    PromptFrame,
)
from proofpath.tui.widgets.run_block import (
    CommandBlock,
    CoverageLine,
    KvLine,
    NoteLine,
    RunBlock,
    RunHeader,
    StageLine,
)
from proofpath.tui.widgets.suggestions import Suggestions

__all__ = [
    "ANSWER_ID",
    "ANSWER_LABELS",
    "PANEL_FLOOR",
    "Banner",
    "CommandBlock",
    "ConfigPanel",
    "CoverageFooter",
    "CoverageLine",
    "FindingLine",
    "FindingsRule",
    "FrameEdge",
    "KvLine",
    "Line",
    "NoteLine",
    "PermissionPrompt",
    "Prompt",
    "PromptFrame",
    "RunBlock",
    "RunHeader",
    "StageLine",
    "Suggestions",
    "accent_for",
    "panelled",
    "textual_colour",
]
