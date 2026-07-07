"""Credential seam (core/credentials.py) — .env loading and the get_key contract.

Pinned behaviors: an unset key is None (never a crash), an empty template line
counts as unset, a missing .env is a silent no-op — keyless backends must run
with nothing configured at all.
"""
import os

import core.credentials as credentials


def test_get_key_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("JARVIS_TEST_KEY", raising=False)

    assert credentials.get_key("JARVIS_TEST_KEY") is None


def test_get_key_reads_the_environment(monkeypatch):
    monkeypatch.setenv("JARVIS_TEST_KEY", "abc123")

    assert credentials.get_key("JARVIS_TEST_KEY") == "abc123"


def test_empty_value_counts_as_unset(monkeypatch):
    # A copied template line like `TAVILY_API_KEY=` must not read as configured.
    monkeypatch.setenv("JARVIS_TEST_KEY", "")

    assert credentials.get_key("JARVIS_TEST_KEY") is None


def test_load_credentials_reads_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("JARVIS_FILE_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setattr(credentials, "ENV_PATH", env_file)
    monkeypatch.delenv("JARVIS_FILE_KEY", raising=False)
    try:
        credentials.load_credentials()

        assert credentials.get_key("JARVIS_FILE_KEY") == "from-file"
    finally:
        # load_dotenv writes os.environ directly, outside monkeypatch's undo.
        os.environ.pop("JARVIS_FILE_KEY", None)


def test_load_credentials_missing_file_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(credentials, "ENV_PATH", tmp_path / "absent.env")

    credentials.load_credentials()  # must not raise
