"""Writing a key into a ``.env``: what the ``/config`` panel's key row relies on.

Everything here is a file under ``tmp_path``; nothing touches the environment or the
user's own config dir.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from proofpath.secrets import (
    SecretValueError,
    read_dotenv,
    remove_dotenv_value,
    save_dotenv_value,
    user_dotenv_path,
)

NAME = "TAVILY_API_KEY"
KEY = "tvly-dev-abc123XYZ"


def test_saving_into_a_file_that_does_not_exist_creates_it_and_its_folder(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conf" / "nested" / ".env"
    save_dotenv_value(path, NAME, KEY)
    assert path.read_text(encoding="utf-8") == f"{NAME}={KEY}\n"
    assert read_dotenv(path) == {NAME: KEY}


def test_saving_into_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    save_dotenv_value(path, NAME, KEY)
    assert path.read_text(encoding="utf-8") == f"{NAME}={KEY}\n"


def test_surrounding_whitespace_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    save_dotenv_value(path, NAME, f"  {KEY}\n")
    assert read_dotenv(path) == {NAME: KEY}


def test_an_existing_line_is_replaced_in_place_and_every_other_line_is_kept(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# my keys\nGROQ_API_KEY=groq-1\n\nTAVILY_API_KEY=old-key\n# trailing comment\nOTHER=x\n",
        encoding="utf-8",
    )
    save_dotenv_value(path, NAME, KEY)
    assert path.read_text(encoding="utf-8") == (
        f"# my keys\nGROQ_API_KEY=groq-1\n\n{NAME}={KEY}\n# trailing comment\nOTHER=x\n"
    )


def test_an_export_line_is_replaced_and_keeps_its_export(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("export TAVILY_API_KEY=old-key\nA=b", encoding="utf-8")
    save_dotenv_value(path, NAME, KEY)
    assert path.read_text(encoding="utf-8") == f"export {NAME}={KEY}\nA=b\n"


def test_a_duplicate_later_line_is_dropped_so_it_cannot_win_the_read(tmp_path: Path) -> None:
    # ``read_dotenv`` keeps the *last* value; a stale duplicate below the replaced
    # line would silently override the key just saved.
    path = tmp_path / ".env"
    path.write_text("TAVILY_API_KEY=one\nA=b\nTAVILY_API_KEY=two\n", encoding="utf-8")
    save_dotenv_value(path, NAME, KEY)
    assert path.read_text(encoding="utf-8") == f"{NAME}={KEY}\nA=b\n"


def test_a_similar_name_is_not_touched(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("TAVILY_API_KEY_OLD=keep\n", encoding="utf-8")
    save_dotenv_value(path, NAME, KEY)
    assert read_dotenv(path) == {"TAVILY_API_KEY_OLD": "keep", NAME: KEY}


@pytest.mark.parametrize(
    "value",
    ["", "   ", "tvly abc", "tvly\tabc", "tvly\nabc", 'tvly"abc', "tvly'abc", "tvly#abc"],
)
def test_a_value_that_would_not_read_back_is_refused_without_echoing_it(
    tmp_path: Path, value: str
) -> None:
    path = tmp_path / ".env"
    path.write_text("A=b\n", encoding="utf-8")
    with pytest.raises(SecretValueError) as info:
        save_dotenv_value(path, NAME, value)
    message = str(info.value)
    assert message
    for part in value.split():
        assert part not in message
    assert path.read_text(encoding="utf-8") == "A=b\n"  # nothing was written


def test_a_bad_variable_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SecretValueError):
        save_dotenv_value(tmp_path / ".env", "NOT A NAME", KEY)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_file_is_readable_by_its_owner_only(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("A=b\n", encoding="utf-8")
    path.chmod(0o644)
    save_dotenv_value(path, NAME, KEY)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    remove_dotenv_value(path, NAME)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_removing_a_saved_key_keeps_the_rest(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(f"# keys\nexport {NAME}={KEY}\nA=b\n", encoding="utf-8")
    assert remove_dotenv_value(path, NAME) is True
    assert path.read_text(encoding="utf-8") == "# keys\nA=b\n"
    assert remove_dotenv_value(path, NAME) is False


def test_removing_from_a_missing_file_is_false_and_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "nope" / ".env"
    assert remove_dotenv_value(path, NAME) is False
    assert not path.exists()


def test_the_user_dotenv_is_the_config_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROOFPATH_CONFIG_DIR", str(tmp_path / "conf"))
    assert user_dotenv_path() == tmp_path / "conf" / ".env"


def test_a_crlf_file_stays_crlf(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_bytes(b"# keys\r\nA=b\r\nTAVILY_API_KEY=old\r\n")
    save_dotenv_value(path, NAME, KEY)
    assert path.read_bytes() == f"# keys\r\nA=b\r\n{NAME}={KEY}\r\n".encode()
    assert remove_dotenv_value(path, NAME) is True
    assert path.read_bytes() == b"# keys\r\nA=b\r\n"


def test_a_value_with_equals_signs_round_trips(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    save_dotenv_value(path, NAME, "abc=def==")
    assert read_dotenv(path) == {NAME: "abc=def=="}


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_symlinked_env_is_written_through_and_the_link_survives(tmp_path: Path) -> None:
    real = tmp_path / "dotfiles" / "env"
    real.parent.mkdir()
    real.write_text("A=b\n", encoding="utf-8")
    link = tmp_path / "conf" / ".env"
    link.parent.mkdir()
    link.symlink_to(real)
    save_dotenv_value(link, NAME, KEY)
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == f"A=b\n{NAME}={KEY}\n"


def test_a_failed_replace_leaves_the_file_whole_and_no_temporary_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".env"
    path.write_text("A=b\n", encoding="utf-8")

    def refuse(self: Path, target: Path) -> Path:
        raise OSError("disk said no")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="disk said no"):
        save_dotenv_value(path, NAME, KEY)
    assert path.read_text(encoding="utf-8") == "A=b\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env"]


def test_a_stale_temporary_of_the_old_name_is_never_reused(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    stale = tmp_path / ".env.tmp"
    stale.write_text("planted\n", encoding="utf-8")
    save_dotenv_value(path, NAME, KEY)
    assert stale.read_text(encoding="utf-8") == "planted\n"
    assert read_dotenv(path) == {NAME: KEY}
