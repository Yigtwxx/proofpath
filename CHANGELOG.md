# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-09-10

First release. The verification pipeline is not implemented; this reserves the name
and establishes the interface, packaging and CI that later phases build on.

### Added
- Design specification with measured source-access data (spec §6), the fetch ladder
  and its permission model (§7), and corrected reference resolution (§8).
- Phased implementation plan, ordered by risk retired rather than user-visible
  progress.
- `proofpath` command. A bare invocation is a first-class entry point rather than a
  help screen, which is where the TUI will attach. Exit codes are fixed: `0` clean,
  `1` findings, `2` the run itself failed.
- Cross-platform CI on Linux, macOS and Windows, and PyPI publishing through trusted
  publishing rather than a stored API token.
