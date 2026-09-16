"""Where proofpath keeps its files.

All locations come from ``platformdirs`` so Windows, macOS and Linux behave the
same, and every one of them can be overridden with an environment variable so
tests never touch the real user directories.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_state_dir

APP_NAME = "proofpath"

CONFIG_DIR_ENV = "PROOFPATH_CONFIG_DIR"
CACHE_DIR_ENV = "PROOFPATH_CACHE_DIR"
STATE_DIR_ENV = "PROOFPATH_STATE_DIR"


def config_dir() -> Path:
    """Directory holding ``config.toml``."""
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override) if override else Path(user_config_dir(APP_NAME))


def config_path() -> Path:
    """Full path of the config file. It may not exist yet."""
    return config_dir() / "config.toml"


def cache_dir() -> Path:
    """Root for everything re-downloadable: models, datasets, the SQLite cache."""
    override = os.environ.get(CACHE_DIR_ENV)
    return Path(override) if override else Path(user_cache_dir(APP_NAME))


def models_dir() -> Path:
    """Where ONNX models are downloaded to."""
    return cache_dir() / "models"


def state_dir() -> Path:
    """Where session state lives: things that are neither config nor re-downloadable.

    Today that is the prompt's command history. It is not under the cache root on
    purpose: ``cache clear`` must never forget what the user typed.
    """
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else Path(user_state_dir(APP_NAME))


def history_path() -> Path:
    """The TUI's command history, one submitted line per row. It may not exist yet."""
    return state_dir() / "history"
