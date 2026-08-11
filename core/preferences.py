"""
User preferences — what the user chose, as opposed to how the install is shaped.

`config.yaml` is the WIZARD's file: it records what this machine is (model,
endpoint, budgets) and is written once at setup. Preferences are the APP's file:
rewritten at runtime whenever a setting changes, by a settings UI rather than by
a person editing YAML. Different owner, different write path, different lifetime
— so a different file, and a settings write can never clobber the wizard's.

Stored values are Open-Meteo's own vocabulary verbatim ("fahrenheit", "kn",
"inch"), so nothing translates between the stored choice and the wire. A
translation layer here would be one more pair of things that can drift apart,
which is the exact bug this slice exists to remove.

Reads are forgiving and writes are strict: a malformed or half-written file
degrades to defaults with a warning (a bad preference must never fail a boot),
while a write is validated against UNIT_CHOICES before anything touches disk, so
a rejected write leaves the stored file byte-for-byte unchanged.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from core.constants import (
    DEFAULT_PREFERENCES,
    LOGGER_PREFERENCES,
    PREFERENCE_KEY_SEPARATOR,
    PREFERENCE_UNITS,
    PREFERENCES_PATH,
    UNIT_CHOICES,
)
from core.memory.digest import atomic_write_text

logger = logging.getLogger(LOGGER_PREFERENCES)


class PreferenceError(ValueError):
    """A write the store refuses — unknown key or a value outside its choices.

    Carries a user-facing message: it is what the settings surface shows and
    what the WebSocket returns in its error frame.
    """


class Preferences:
    """The preference file, read on demand and written all-or-nothing.

    Deliberately NOT cached in memory: a read is a few hundred bytes, and a
    cached copy would be a second source of truth that a settings write in one
    process could leave stale in another. Every caller sees the file.
    """

    def __init__(self, path: Path = PREFERENCES_PATH) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict:
        """Current preferences, always complete and always valid.

        A missing file is seeded with the defaults (first run). An unreadable or
        torn one falls back to defaults IN MEMORY and is left on disk untouched
        — overwriting it would destroy whatever the user still had in there,
        and the next successful write repairs it anyway.
        """
        if not self._path.exists():
            logger.info("no preferences file at %s — writing defaults", self._path)
            self._write(copy.deepcopy(DEFAULT_PREFERENCES))
            return copy.deepcopy(DEFAULT_PREFERENCES)
        try:
            stored = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "preferences file unreadable (%s) — falling back to defaults", exc
            )
            return copy.deepcopy(DEFAULT_PREFERENCES)
        return self._sanitized(stored)

    def units(self) -> dict[str, str]:
        """The unit choice per dimension, e.g. {"temperature": "fahrenheit"}."""
        return self.load()[PREFERENCE_UNITS]

    def set(self, key: str, value) -> dict:
        """Apply one dotted write ("units.temperature") and persist it.

        Validation happens BEFORE the read-modify-write, so a rejected write
        never opens the file at all. Returns the full stored preferences so a
        caller can echo back what is now true rather than what it hoped.
        """
        dimension = self._validated(key, value)
        preferences = self.load()
        preferences[PREFERENCE_UNITS][dimension] = value
        self._write(preferences)
        logger.info("preference %s set to %r", key, value)
        return preferences

    def _validated(self, key: str, value) -> str:
        """Resolve a dotted key to its unit dimension, or raise PreferenceError.

        `units` is the only namespace that exists. Anything else is refused
        rather than stored, so this file cannot become an arbitrary key-value
        dump that later code has to guess the shape of.
        """
        namespace, _, dimension = str(key).partition(PREFERENCE_KEY_SEPARATOR)
        if namespace != PREFERENCE_UNITS or dimension not in UNIT_CHOICES:
            raise PreferenceError(
                f"unknown preference key {key!r} — valid keys: "
                f"{', '.join(f'{PREFERENCE_UNITS}.{d}' for d in UNIT_CHOICES)}"
            )
        if value not in UNIT_CHOICES[dimension]:
            raise PreferenceError(
                f"invalid value {value!r} for {key} — valid values: "
                f"{', '.join(UNIT_CHOICES[dimension])}"
            )
        return dimension

    def _sanitized(self, stored) -> dict:
        """Merge the stored file over the defaults, dropping anything invalid.

        Per key, not per file: one bad unit costs that one dimension, and every
        other stored choice still applies.
        """
        preferences = copy.deepcopy(DEFAULT_PREFERENCES)
        if not isinstance(stored, dict):
            logger.warning("preferences file is not an object — using defaults")
            return preferences
        units = stored.get(PREFERENCE_UNITS)
        if not isinstance(units, dict):
            if units is not None:
                logger.warning("preferences `units` is not an object — using defaults")
            return preferences
        for dimension, value in units.items():
            if dimension not in UNIT_CHOICES:
                logger.warning("ignoring unknown unit dimension %r", dimension)
            elif value not in UNIT_CHOICES[dimension]:
                logger.warning(
                    "ignoring invalid %s unit %r — keeping %r",
                    dimension, value, preferences[PREFERENCE_UNITS][dimension],
                )
            else:
                preferences[PREFERENCE_UNITS][dimension] = value
        return preferences

    def _write(self, preferences: dict) -> None:
        # Same all-or-nothing write as the profile: a torn preferences file
        # would be read as defaults on the next boot, silently reverting a
        # setting the user believes they changed.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self._path, json.dumps(preferences, indent=2) + "\n")
