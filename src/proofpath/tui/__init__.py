"""The streaming TUI (spec section 13.1).

Nothing here holds pipeline logic: the widgets call the same ``verify()`` entry
point the CLI calls. The pure modules in this package — :mod:`wordmark` (with
:mod:`banner`'s shared helpers) and :mod:`commands` — draw the banner and read the
slash-command line, and none imports ``textual``, so all are testable as plain strings.
"""
