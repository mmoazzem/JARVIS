"""Config round-trip invariants (setup/config.py).

CONFIG_PATH is monkeypatched to a pytest tmp_path so these tests NEVER read or
write the real config/config.yaml.
"""
import pytest

from setup import config as cfg
from setup.config import JarvisConfig


@pytest.fixture
def tmp_config_path(tmp_path, monkeypatch):
    """Redirect the module's CONFIG_PATH to an isolated temp file."""
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(cfg, "CONFIG_PATH", path)
    return path


def test_save_then_load_round_trips_all_fields(tmp_config_path):
    original = JarvisConfig(
        mode="local",
        primary_model="qwen3:14b",
        temperature=0.55,
        ollama_keep_alive="10m",
        enable_thinking=True,
        created_at="2026-07-02T00:00:00+00:00",
    )

    cfg.save(original)
    loaded = cfg.load()

    assert loaded == original


def test_round_trip_preserves_locked_token_budgets(tmp_config_path):
    cfg.save(JarvisConfig())
    loaded = cfg.load()

    # The locked M5 budgets: reasoning shares max_tokens, so it must stay large,
    # and the context budget must stay below it.
    assert loaded.max_tokens == 18000
    assert loaded.context_token_budget == 8000


def test_save_stamps_created_at_on_first_write_only(tmp_config_path):
    cfg.save(JarvisConfig(created_at=None))
    first = cfg.load()
    assert first.created_at is not None  # stamped on first write

    cfg.save(first)
    assert cfg.load().created_at == first.created_at  # preserved on re-save


def test_load_missing_config_raises_file_not_found(tmp_config_path):
    # The documented contract: absent config.yaml → FileNotFoundError, so the
    # caller can route the user to the wizard.
    with pytest.raises(FileNotFoundError):
        cfg.load()


def test_load_ignores_unknown_keys(tmp_config_path):
    cfg.save(JarvisConfig())
    tmp_config_path.write_text(
        tmp_config_path.read_text(encoding="utf-8") + "\nfuture_unknown_key: 42\n",
        encoding="utf-8",
    )

    loaded = cfg.load()  # extra="ignore" — old installs survive schema growth

    assert not hasattr(loaded, "future_unknown_key")
