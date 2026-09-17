"""The streaming TUI (spec section 13.1).

Nothing here holds pipeline logic: the widgets call the same ``verify()`` entry
point the CLI calls. The two pure modules in this package — :mod:`banner` and
:mod:`commands` — draw the raven and read the slash-command line, and neither
imports ``textual``, so both are testable as plain strings.
"""
