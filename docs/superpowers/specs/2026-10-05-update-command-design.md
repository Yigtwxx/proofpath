# `proofpath update` — design (2026-10-05)

The user asked for an update command like `claude update`. This spec records what they
decided and how it works. OPEN-ITEMS §21 points here.

## 1. Decisions (user, 2026-10-05)

| # | Question | Decision |
|---|---|---|
| 1 | Where | `proofpath update [--check]` in the CLI and `/update` in the TUI. Both call one function in `proofpath.commands`. There is no update check at startup. |
| 2 | Which installs | uv tool, pipx and pip installs are all updated. An editable (development) install is never touched: the command says how to update it. |
| 3 | Extras | The extras that are installed are written into the install spec (`proofpath[browser]`). That way the tool's own record keeps them, and a later plain `uv tool upgrade` keeps them too. |
| 4 | Behaviour | Running the command is the consent: it updates straight away. `--check` only reports. |

### Why extras need handling (measured 2026-10-05)

On consent, the browser extra is installed with `pip install` into the running
environment (`browser.install_commands`). A spike in an isolated `UV_TOOL_DIR` ran
three steps: it installed a tool, added `six` to its environment with
`uv pip install --python`, then ran `uv tool upgrade`. **`six` was gone after the
upgrade.** A plain `uv tool upgrade proofpath` would therefore remove the browser
wheels the user had consented to. The next blocked page would ask, and download, all
over again. The browser binary is not affected: it lives in the `ms-playwright` cache,
outside the environment. Neither is the accurate NLI model, which sits in the cache
directory.

## 2. Behaviour

1. **Current version.** `proofpath.__version__`.
2. **Latest version.** `GET https://pypi.org/pypi/proofpath/json` with `httpx` and a
   10 s timeout, reading `info.version`. The request carries no contact address and no
   key. `permissions.network` is decided by `fetch.network_permission`, the fetch
   ladder's own rule, not a copy of it. `deny` sends nothing, and neither does `ask`
   without a terminal (product rule 4): the command says why. `ask` with a terminal
   goes ahead, and a note in the result says so. The CLI passes `is_interactive()`;
   the TUI passes `True`. Running the command is consent to *install* (decision 4),
   but it is not an answer to `ask` where no terminal could have been asked. A refused
   or failed request fails rather than reporting "up to date": absence of evidence is
   not evidence (product rule 2).
3. **Comparison.** `packaging.version.Version`. Add `packaging` to the declared
   dependencies: it is already installed through other packages, but the code imports
   it directly. There are three outcomes:
   - up to date
   - an update is available
   - this build is newer than PyPI's (a development build). In that case nothing is run.
4. **Install detection**, in this order:
   1. *editable*: the distribution's `direct_url.json` has `dir_info.editable: true`
   2. *ephemeral*: a temporary environment, judged from `sys.prefix` alone. A `uvx` /
      `uv tool run` environment lives in uv's cache, so an `archive-v0` path component
      means uv. `pipx install` keeps every venv in `<pipx home>/venvs/<package>`, and
      `pipx run` keeps its venvs in pipx's cache instead. That cache is
      `~/Library/Caches/pipx/<hash>` on macOS and `...\AppData\Local\pipx\pipx\Cache\<hash>`
      on Windows; with `PIPX_HOME` set it is `$PIPX_HOME/.cache/<hash>`. Both kinds
      write `pipx_metadata.json`. So a prefix with `pipx_metadata.json` whose parent
      directory is not named `venvs` is a `pipx run` venv. This check comes before
      *pipx*.
   3. *uv tool*: `Path(sys.prefix) / "uv-receipt.toml"` exists
   4. *pipx*: `Path(sys.prefix) / "pipx_metadata.json"` exists
   5. *pip*: anything else, run as `sys.executable`
5. **Extras detection.**
   - `browser`: `scrapling` and `patchright` are both importable (`find_spec`).
   - `gpu`: `torch` and `sentence_transformers` are both importable.
   - For a uv tool install, the extras already listed for `proofpath` in the receipt's
     `requirements` are added to these (their union).
   - `dev` is never carried over.
6. **The command.** The spec is `proofpath[a,b]`, or `proofpath` with no extras. It is
   **never version-pinned**: a pin would freeze the uv receipt or pipx metadata, and a
   later upgrade would not move.
   - uv tool: `uv tool install --upgrade <spec>`. Needs `uv` on PATH.
   - pipx: `pipx install --force <spec>`. Needs `pipx` on PATH.
   - pip: `<sys.executable> -m pip install --upgrade <spec>`. When that fails and `uv`
     is on PATH, it falls back to `uv pip install --python <sys.executable> --upgrade <spec>`,
     the same fallback `browser.install` uses.
   - editable: nothing runs. The message names the source directory and says to update
     it with `git pull` and `uv sync`.
   - ephemeral: nothing runs. Updating a cached environment in place would be undone
     when the cache is pruned. The message says this is a temporary environment and
     that `uvx proofpath@latest` (uv) or `pipx run --no-cache proofpath` (pipx) already
     fetches the newest. `pipx run` reuses its cached venv for days unless told not to,
     hence `--no-cache`.
   - A required tool that is missing is a failure. The message prints the command to
     run by hand.
7. **Running it.** The runner is injected, as in `browser.install`. Output is captured,
   never inherited: the TUI owns the terminal. Each command that runs is logged as
   `$ <cmd>`. The command meant for pasting has no `$`: what `--check` would run, or
   what to run by hand after a failure. Both are joined by `shell.display_command`:
   `subprocess.list2cmdline` on Windows, `shlex.join` elsewhere. `shlex.join` would
   wrap `proofpath[gpu]` in single quotes, and `cmd.exe` keeps them. `shell` is a
   leaf module that `browser.install` logs through as well, so the browser install's
   log pastes on Windows too.
8. **Verification.** After a zero exit, the installed version is read in a fresh
   process:
   - uv tool and pipx: `<venv python> -I -c "import importlib.metadata as m; print(m.version('proofpath'))"`
   - pip: `sys.executable`, with the same `-I`

   `-I` runs the interpreter isolated: no `PYTHON*` variables, no working directory on
   `sys.path`. A proofpath checkout the user is standing in cannot answer for the
   install.

   The code imported in the running process is still the old version and cannot answer
   this. If the new version is not the latest (for example, the resolver held it back),
   the result is a failure that says which version was installed.
9. **Windows.** A running `proofpath.exe` can be locked while it is being replaced. A
   failed command on Windows adds this line: "close proofpath and run the command
   above". CI cannot test this live, so a unit test covers the message.
10. **Exit codes.** `0` means up to date, updated, or (with `--check`) an update is
    available. It also means ahead of PyPI, editable, and ephemeral: in those three
    cases nothing went wrong, and there was nothing for this command to run. `1` means
    the check or the update failed. Usage errors keep typer's `2`.
11. **TUI.** `/update` and `/update --check` are mirrored verbs: in `MIRRORED` and
    `VERBS`, and in the `/help` list. They run on the mirror worker like `/fetch`.
    After a successful update the block says "restart proofpath to use vX".
    Nothing prompts. A bare `/update` is refused while anything is still running:
    an unfinished run, or a mirrored verb whose worker is still out. Its install
    replaces the venv under the code those are executing, so the block says so in one
    line. `/update --check` installs nothing and is never refused. The guard works
    the other way too. While a bare `/update` is installing, `/check` (implicit or
    typed) and every mirrored verb are refused with "an update is installing; wait
    for it to finish". That includes a second `/update` and `/update --check`.
    `update` is last
    in `VERBS`: at 80x24 the `/` list holds ten of the eleven rows. The cap then hides
    `/quit`, which the banner's hint line names anyway, and `/update`. It never
    hides `/help`.

## 3. Tests (no network)

- PyPI responses come from `respx`. Install kinds are faked with tmp dirs and
  monkeypatched `sys.prefix`/`sys.executable`. Extras use monkeypatched `find_spec`.
  The runner is a fake that records commands.
- Each install kind builds the right command. Extras are carried over, including the
  union with the receipt, and `dev` never is. The spec is never pinned.
- An editable install never runs anything. `network=deny` sends no request.
- A PyPI failure exits `1` with the reason. An up-to-date or ahead build runs nothing.
  `--check` runs nothing.
- When a command fails, the output says so and prints the command, with the Windows
  line on Windows. When verification sees an old version, that is a failure.
- The CLI is tested through `CliRunner`. In the TUI, `/update --check` renders in a
  command block.

## 4. Out of scope

- An update check at startup (declined, decision 1).
- Keeping uv receipt options such as a pinned `--python` across `uv tool install
  --upgrade`. This is a known limit, written in the README.
