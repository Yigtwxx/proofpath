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


# --- search (OPEN-ITEMS 17.1a) -----------------------------------------------------


def test_search_is_off_by_default_and_web_search_is_allowed() -> None:
    config = cfg.Config()
    assert config.search == cfg.SearchConfig()
    assert config.search.provider == "off"
    assert config.search.api_key_env == "TAVILY_API_KEY"
    assert (config.search.max_claims, config.search.results_per_claim) == (5, 3)
    assert config.permissions.web_search == "allow"


def test_the_search_section_loads_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[search]\nprovider = "searxng"\nbase_url = "http://localhost:8888"\nmax_claims = 2\n',
        encoding="utf-8",
    )
    config = cfg.load_config(path)
    assert config.search.provider == "searxng"
    assert config.search.base_url == "http://localhost:8888"
    assert config.search.max_claims == 2


@pytest.mark.parametrize(
    "body",
    ['provider = "bing"', "max_claims = 0", "max_claims = true", 'max_claims = "5"'],
)
def test_bad_search_values_are_refused_by_name(tmp_path: Path, body: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[search]\n{body}\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match=r"search\."):
        cfg.load_config(path)


def test_set_value_parses_an_integer_and_round_trips_it(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    updated = cfg.set_value("search.max_claims", "3", path)
    assert updated.search.max_claims == 3
    assert "max_claims = 3" in path.read_text(encoding="utf-8")
    assert cfg.load_config(path).search.max_claims == 3
    with pytest.raises(cfg.ConfigError, match="positive integer"):
        cfg.set_value("search.max_claims", "many", path)


def test_set_value_takes_a_search_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    assert cfg.set_value("search.provider", "tavily", path).search.provider == "tavily"
    with pytest.raises(cfg.ConfigError, match=r"search\.provider"):
        cfg.set_value("search.provider", "bing", path)


# --- the accurate NLI profile (docs/superpowers/specs/2026-10-01-accurate-nli-design.md)


def test_the_default_model_is_the_default_profile_and_its_install_asks() -> None:
    config = cfg.Config()
    assert config.models == cfg.ModelsConfig()
    assert config.models.nli == "default"
    assert cfg.NLI_PROFILE_NAMES == ("default", "accurate")
    # Rule 5: a 643 MB download is never made without consent.
    assert config.permissions.install_model == "ask"


def test_the_models_section_and_the_install_permission_load_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[models]\nnli = "accurate"\n\n[permissions]\ninstall_model = "allow"\n',
        encoding="utf-8",
    )
    config = cfg.load_config(path)
    assert config.models.nli == "accurate"
    assert config.permissions.install_model == "allow"


@pytest.mark.parametrize("body", ['nli = "fast"', "nli = 1", 'nli = "Accurate"'])
def test_an_unknown_nli_profile_is_refused_by_name(tmp_path: Path, body: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[models]\n{body}\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match=r"models\.nli"):
        cfg.load_config(path)


def test_a_bad_install_model_permission_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[permissions]\ninstall_model = "sometimes"\n', encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match=r"permissions\.install_model"):
        cfg.load_config(path)


def test_set_value_takes_the_nli_profile_and_round_trips_it(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    updated = cfg.set_value("models.nli", "accurate", path)
    assert updated.models.nli == "accurate"
    text = path.read_text(encoding="utf-8")
    assert "[models]" in text.splitlines()
    assert 'nli = "accurate"' in text.splitlines()
    assert text == cfg.render_config(updated)
    assert cfg.load_config(path) == updated
    with pytest.raises(cfg.ConfigError, match=r"models\.nli"):
        cfg.set_value("models.nli", "fast", path)
    assert cfg.load_config(path).models.nli == "accurate"


def test_set_value_takes_the_install_model_permission(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    assert cfg.set_value("permissions.install_model", "deny", path).permissions == (
        cfg.Permissions(install_model="deny")
    )
