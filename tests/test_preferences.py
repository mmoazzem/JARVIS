"""Preference store invariants (core/preferences.py) — no network, no Ollama.

Pinned behavior: a valid write persists and is readable back; an invalid one is
refused WITHOUT touching the stored file. The second half is the point — this
file is reachable from the WebSocket, so a rejected write that still corrupted
the store would turn a typo into a broken boot.
"""
import json

import pytest

from core.constants import DEFAULT_PREFERENCES
from core.preferences import PreferenceError, Preferences


def _store(tmp_path) -> Preferences:
    return Preferences(tmp_path / "preferences.json")


def test_valid_write_persists_and_bad_writes_leave_the_file_untouched(tmp_path):
    prefs = _store(tmp_path)

    # A write round-trips through disk, not through memory: a second reader
    # (the weather service) must see what the settings write stored.
    prefs.set("units.temperature", "celsius")
    assert prefs.units()["temperature"] == "celsius"
    assert _store(tmp_path).units()["temperature"] == "celsius"

    # Per dimension, not one imperial/metric flag: changing temperature must
    # leave wind alone, because fahrenheit-with-knots is a real combination.
    assert prefs.units()["wind_speed"] == DEFAULT_PREFERENCES["units"]["wind_speed"]

    stored = prefs.path.read_text(encoding="utf-8")

    with pytest.raises(PreferenceError):
        prefs.set("units.altitude", "feet")        # unknown key
    with pytest.raises(PreferenceError):
        prefs.set("units.temperature", "kelvin")   # out-of-range value
    with pytest.raises(PreferenceError):
        prefs.set("model.primary", "gpt-4o")       # outside the one namespace

    assert prefs.path.read_text(encoding="utf-8") == stored


def test_missing_file_seeds_defaults_and_a_torn_file_degrades_to_them(tmp_path):
    prefs = _store(tmp_path)

    assert prefs.load() == DEFAULT_PREFERENCES
    assert json.loads(prefs.path.read_text(encoding="utf-8")) == DEFAULT_PREFERENCES

    # Truncated mid-object, as a crashed write would leave it. A bad preference
    # file must cost the preference, never the boot.
    prefs.path.write_text('{"units": {"temperature": "cel', encoding="utf-8")
    assert prefs.load() == DEFAULT_PREFERENCES

    # One bad value costs only its own dimension — every other stored choice
    # still applies.
    prefs.path.write_text(
        json.dumps({"units": {"temperature": "kelvin", "wind_speed": "mph"}}),
        encoding="utf-8",
    )
    units = prefs.units()
    assert units["temperature"] == DEFAULT_PREFERENCES["units"]["temperature"]
    assert units["wind_speed"] == "mph"
