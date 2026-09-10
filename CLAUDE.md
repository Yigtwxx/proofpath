# proofpath — project instructions

`proofpath` verifies that the sources behind a claim actually say what the claim
says. Read `docs/superpowers/specs/2026-09-10-proofpath-design.md` before making
design decisions; it is the source of truth and records *why* things are the way
they are.

## Non-negotiable product rules

These are not style preferences. Breaking one breaks the product's reason to exist.

1. **Never assert without a passage.** A `SUPPORTED` or `REFUTED` verdict must carry
   the quoted evidence it is based on. No passage, no verdict.
2. **Never present absence of evidence as evidence of absence.** Unreachable,
   blocked, paywalled and ambiguous are distinct reported states (spec §15), never
   silently collapsed into a verdict.
3. **Never call a real reference a ghost.** Uncertainty resolves to `AMBIGUOUS`, not
   `GHOST REFERENCE` (spec §8). The false-ghost rate is a release gate.
4. **Never prompt without a TTY.** In CI or when piped, an `ask` permission is
   treated as `deny` and reported (spec §7.1).
5. **Never install anything large without explicit consent** (spec §7.1).
6. **Every report states its own coverage.** A low-coverage run must not look like a
   clean one.

## Stack

| Concern | Choice | Why |
|---|---|---|
| CLI / TUI | `typer` + `textual` | bare `proofpath` opens the TUI via `invoke_without_command=True` |
| HTTP | `httpx`, then `curl_cffi`, then browser on consent | fetch ladder, spec §7 |
| PDF | `pymupdf` | pure Python, no Java/Docker |
| Vector store | `sqlite-vec` | per-document corpora are small; no server |
| Embeddings / NLI | ONNX runtime by default, `torch` only via the `[gpu]` extra | keeps base install small |
| Config / cache paths | `platformdirs` | Windows/macOS/Linux parity |

## Conventions

- Python 3.10+. **Type annotations are mandatory** on every function signature.
- `ruff check` and `ruff format --check` must be clean.
- `pytest` must pass on Linux, macOS and Windows.
- Device preference order is always **CUDA → MPS → CPU** (and the ONNX equivalent:
  CUDA → CoreML → CPU).
- `pathlib` everywhere. Every file read/write passes `encoding="utf-8"` explicitly.
- Code, identifiers and comments in English.

## Hard constraints

- **No Docker or Java requirement** in any default path. GROBID stays opt-in.
- **No hardcoded secrets, tokens or emails.** The Crossref/OpenAlex contact address
  comes from config or `PROOFPATH_CONTACT_EMAIL` and is optional.
- No network access in unit tests. Provider responses are fixtures.
- `cli` and `tui` contain no logic; both call the same `verify()` entry point.
