"""The consent gate in front of the accurate NLI profile's download.

The accurate profile's model is 643 MB, fetched onto the user's machine, so it is
never downloaded silently (product rule 5) and never asked about without a terminal
(product rule 4). ``ModelGate`` mirrors ``browser.ConsentGate``: the same permission
values, the same four answers, "always" and "never" saved to the config. It differs
in one place on purpose: a denied or failed download is not a skipped source but a
fallback to the default profile, which the caller reports (product rule 6), so the
gate's job ends at returning a ``Decision`` whose reason a report can print.

Design: docs/superpowers/specs/2026-10-01-accurate-nli-design.md, "User-facing
behaviour".
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer

from proofpath.browser import _CHOICE_ANSWERS, PROMPT_CHOICES, TERMINAL_ANSWERS, Answer
from proofpath.config import ConfigError, Decision, Permission, resolve_permission, set_value
from proofpath.profiles import PROFILES, NliProfile

#: The key "always" and "never" are saved under.
PERMISSION_KEY = "permissions.install_model"

# What a model question passes as the ``host`` argument of a gate prompt. The prompt
# callable is the browser gate's, shared so the TUI answers both through one widget;
# a real host never starts with this (a ``host:port`` port is numeric), so a prompt
# can tell the two questions apart with ``subject_profile``.
_SUBJECT_PREFIX = "model:"


def prompt_subject(profile: NliProfile) -> str:
    """The first argument the gate passes to an injected prompt for ``profile``."""
    return f"{_SUBJECT_PREFIX}{profile.name}"


def subject_profile(subject: str) -> NliProfile | None:
    """The profile a prompt subject names, or ``None`` for a blocked host.

    Lets a shared prompt (the TUI's) render ``model_prompt_text`` instead of the
    browser question when the gate asking is this one.
    """
    if not subject.startswith(_SUBJECT_PREFIX):
        return None
    return PROFILES.get(subject[len(_SUBJECT_PREFIX) :])


def model_prompt_text(profile: NliProfile, *, answers: str = TERMINAL_ANSWERS) -> str:
    """The consent block for one profile, laid out like ``browser.prompt_text``: what
    is wanted, what it costs, then the answers, separated by blank lines.

    ``answers`` is the last line: the terminal's keys by default, or ``TUI_ANSWERS``.
    The size is stated before anything is fetched, because rule 5 is about the user
    knowing what they agree to, not only about being asked.
    """
    lines = [
        f"  ⚠ The {profile.name} NLI profile needs a model that is not downloaded yet.",
        "",
        "    proofpath can fetch it once into its model cache:",
        "",
        f"      {profile.repo}",
        f"      revision {profile.revision[:7]}    {profile.size_mb} MB",
        "",
        "    Without it this run uses the default profile and says so.",
        "",
        answers,
    ]
    return "\n".join(lines)


def ask_model_terminal(profile: NliProfile) -> Answer:
    """The default prompt. ``ModelGate`` never calls this without a TTY.

    Same keys and the same stderr channel as ``browser.ask_terminal`` (spec section
    13.3: ``--format json`` on stdout stays one document); only the question differs,
    which is why this is not ``ask_terminal`` itself.
    """
    typer.echo(model_prompt_text(profile), err=True)
    while True:
        raw = typer.prompt("Allow?", default="n", err=True).strip().lower()
        if raw in _CHOICE_ANSWERS:
            return _CHOICE_ANSWERS[raw]
        typer.echo(f"Please answer one of: {', '.join(PROMPT_CHOICES)}", err=True)


def download_profile(profile: NliProfile, cache_dir: Path, machine: str) -> None:
    """Fetch the three files ``OnnxNli`` loads for ``profile`` on ``machine``.

    Exactly the files ``profiles.profile_installed`` checks, at the pinned revision
    and into the same cache, so a successful download is what makes the next run's
    check say "installed" and skip the question. Imported lazily, like everywhere
    else ``huggingface_hub`` is used, to keep it off the CLI's import path.
    """
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer.json", profile.onnx_file(machine)):
        hf_hub_download(profile.repo, filename, revision=profile.revision, cache_dir=str(cache_dir))


def _failure_reason(exc: Exception) -> str:
    """A short, safe description of a failed download.

    ``str(exc)`` is never used: Hugging Face HTTP errors carry the server's response
    text, and that is not something a report should repeat. The type name, plus the
    HTTP status when there is one, says enough to act on.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return f"{type(exc).__name__} (HTTP {status})"
    return type(exc).__name__


class ModelGate:
    """Decides, once per profile per run, whether a profile's model may be used.

    ``ensure`` answers "allow" when the files are already cached, without asking.
    Otherwise it resolves the stored permission, asks at most once when the
    permission is ``ask`` and there is a terminal, and downloads on "allow". Every
    decision, and the download itself, is written to ``log`` for the report.

    ``override`` records that ``--accurate`` chose the profile. It is not consent:
    the flag says which model, not that 643 MB may be fetched, so it never changes
    the decision.
    """

    def __init__(
        self,
        permission: Permission,
        *,
        interactive: bool,
        override: bool | None = None,
        prompt: Callable[[str, int | None], Answer] | None = None,
        config_path: Path | None = None,
        download: Callable[[NliProfile], None],
        installed: Callable[[NliProfile], bool],
    ) -> None:
        self._permission = permission
        self._interactive = interactive
        self.override = override
        self._prompt = prompt
        self._config_path = config_path
        self._download = download
        self._installed = installed
        self.log: list[str] = []
        # True once the user was shown the question this run.
        self.asked = False
        # By profile name: a second ``ensure`` neither asks nor downloads again, and
        # a failed download is not retried within the run.
        self._decided: dict[str, Decision] = {}

    def _ask(self, profile: NliProfile) -> Answer:
        if self._prompt is None:
            return ask_model_terminal(profile)
        return self._prompt(prompt_subject(profile), None)

    def _persist(self, value: str) -> None:
        """Save an "always"/"never" answer. A failure must not undo the answer already
        given for this run, so it is logged, not raised."""
        try:
            set_value(PERMISSION_KEY, value, self._config_path)
        except (ConfigError, OSError) as exc:
            self.log.append(f"could not save permission: {exc}")

    def _consent(self, profile: NliProfile) -> Decision:
        decision = resolve_permission(self._permission, interactive=self._interactive)
        if decision.outcome != "prompt":
            return decision
        self.asked = True
        answer = self._ask(profile)
        if answer == "once":
            return Decision("allow", "user answered once")
        if answer == "always":
            self._persist("allow")
            return Decision("allow", "user answered always")
        if answer == "no":
            return Decision("deny", "user answered no")
        self._persist("deny")  # "never"
        return Decision("deny", "user answered never")

    def ensure(self, profile: NliProfile) -> Decision:
        """Make ``profile``'s model usable, or say why it is not."""
        if profile.name in self._decided:
            return self._decided[profile.name]
        decision = self._ensure(profile)
        self._decided[profile.name] = decision
        return decision

    def _ensure(self, profile: NliProfile) -> Decision:
        if self._installed(profile):
            self.log.append(f"model {profile.name}: installed")
            return Decision("allow", "installed")

        decision = self._consent(profile)
        if decision.outcome != "allow":
            self.log.append(f"model {profile.name}: not downloaded ({decision.reason})")
            return decision

        self.log.append(
            f"model {profile.name}: downloading {profile.repo} ({profile.size_mb} MB;"
            f" {decision.reason})"
        )
        try:
            self._download(profile)
        except Exception as exc:  # a failed download falls back, it never crashes
            failed = Decision("deny", f"model download failed: {_failure_reason(exc)}")
            self.log.append(f"model {profile.name}: {failed.reason}")
            return failed
        self.log.append(f"model {profile.name}: downloaded")
        return decision
