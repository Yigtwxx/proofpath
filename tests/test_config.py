"""Config and permissions: spec section 7.1."""

from __future__ import annotations

from pathlib import Path

import pytest

from proofpath import config as cfg
from proofpath.paths import cache_dir, config_path, models_dir


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    monkeypatch.setenv("PROOFPATH_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "conf"


def test_config_path_honours_env_override(config_dir: Path) -> None:
    assert config_path() == config_dir / "config.toml"


def test_cache_and_models_dirs_honour_env_override(tmp_path: Path, config_dir: Path) -> None:
    assert cache_dir() == tmp_path / "cache"
    assert models_dir() == tmp_path / "cache" / "models"


def test_missing_config_file_yields_defaults(config_dir: Path) -> None:
    loaded = cfg.load_config()
    assert loaded == cfg.Config()
    assert loaded.permissions.install_browser == "ask"
    assert loaded.permissions.network == "allow"
    assert loaded.fetch.respect_robots is True
    assert loaded.contact.email == ""


def test_save_then_load_round_trips(config_dir: Path) -> None:
    original = cfg.Config(
        permissions=cfg.Permissions(install_browser="deny", network="allow"),
        fetch=cfg.FetchConfig(respect_robots=False),
        contact=cfg.Contact(email="someone@example.org"),
    )
    path = cfg.save_config(original)
    assert path == config_dir / "config.toml"
    assert path.read_text(encoding="utf-8").startswith("# proofpath configuration")
    assert cfg.load_config() == original


def test_invalid_permission_value_is_rejected_with_key_and_path(config_dir: Path) -> None:
    config_dir.mkdir(parents=True)
    path = config_dir / "config.toml"
    path.write_text('[permissions]\ninstall_browser = "maybe"\n', encoding="utf-8")
    with pytest.raises(cfg.ConfigError) as excinfo:
        cfg.load_config()
    message = str(excinfo.value)
    assert "permissions.install_browser" in message
    assert "maybe" in message
    assert str(path) in message


def test_unknown_key_is_rejected(config_dir: Path) -> None:
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("[permissions]\nfoo = 1\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match=r"permissions\.foo"):
        cfg.load_config()


def test_malformed_toml_is_a_config_error(config_dir: Path) -> None:
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("[permissions\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError):
        cfg.load_config()


@pytest.mark.parametrize(
    ("value", "interactive", "expected"),
    [
        ("allow", True, "allow"),
        ("allow", False, "allow"),
        ("deny", True, "deny"),
        ("deny", False, "deny"),
        ("ask", True, "prompt"),
        # Spec 7.1: no TTY means never prompt; "ask" is treated as "deny".
        ("ask", False, "deny"),
    ],
)
def test_resolve_permission(value: str, interactive: bool, expected: str) -> None:
    decision = cfg.resolve_permission(value, interactive=interactive)  # type: ignore[arg-type]
    assert decision.outcome == expected


def test_ask_without_tty_carries_a_reportable_reason() -> None:
    decision = cfg.resolve_permission("ask", interactive=False)
    assert decision.outcome == "deny"
    assert "no interactive terminal" in decision.reason


def test_set_value_updates_one_key_and_persists(config_dir: Path) -> None:
    cfg.set_value("permissions.install_browser", "allow")
    assert cfg.load_config().permissions.install_browser == "allow"
    cfg.set_value("fetch.respect_robots", "false")
    assert cfg.load_config().fetch.respect_robots is False
    cfg.set_value("contact.email", "a@b.c")
    assert cfg.load_config().contact.email == "a@b.c"


def test_set_value_rejects_unknown_key_and_bad_value(config_dir: Path) -> None:
    with pytest.raises(cfg.ConfigError, match=r"nope\.key"):
        cfg.set_value("nope.key", "x")
    with pytest.raises(cfg.ConfigError, match="network"):
        cfg.set_value("permissions.network", "sometimes")
