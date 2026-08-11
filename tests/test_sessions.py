"""Slice 8 — the three failure shapes, and nothing else.

Deliberately narrow. The render path, the arm/confirm interaction and the panel
state machine are unchanged and already covered; a test asserting that a frame
gets emitted would only restate the code that emits it. What is tested here is
what can silently go wrong:

  * the compass bucket (a wrong bucket is a plausible-looking wrong answer, and
    the model got exactly this wrong when it tried to do the conversion),
  * timestamp parsing over the known-hostile values on disk (mixing naive and
    aware datetimes RAISES; a string sort puts "third_turn" first),
  * dismiss, which must hide a session and touch no record.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from core.memory.sessions_view import dismiss, sessions_event

DASHBOARD = Path(__file__).resolve().parents[1] / "web" / "dashboard.html"


# --- 1. the compass conversion ------------------------------------------------

def _compass_points(degrees: list[float]) -> list[str]:
    """Run the dashboard's OWN compassPoint() under node.

    The function is extracted from dashboard.html rather than reimplemented
    here: a copy of the table would agree with itself forever while the page
    drifted, which is the one thing this test exists to catch.
    """
    if shutil.which("node") is None:  # pragma: no cover - environment guard
        pytest.skip("node is not available to run the frontend function")
    source = DASHBOARD.read_text(encoding="utf-8")
    table = re.search(r"const COMPASS = \[[^\]]*\];", source, re.S)
    func = re.search(r"function compassPoint\(deg\) \{.*?\n\}", source, re.S)
    assert table and func, "compassPoint/COMPASS not found in dashboard.html"
    script = (
        f"{table.group(0)}\n{func.group(0)}\n"
        f"console.log(JSON.stringify({json.dumps(degrees)}.map(compassPoint)));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)


def test_compass_buckets_including_the_boundaries():
    # The full circle at every point, plus the degrees where landing in the
    # neighbouring bucket still looks like a reasonable answer.
    cases = {
        0: "N", 11: "N", 12: "NNE", 22: "NNE", 45: "NE", 90: "E", 135: "SE",
        180: "S", 199: "SSW", 225: "SW", 257: "WSW", 270: "W", 315: "NW",
        349: "N", 350: "N", 348: "NNW", 359: "N", 360: "N",
    }
    assert _compass_points(list(cases)) == list(cases.values())


def test_compass_rejects_what_is_not_a_bearing():
    # An absent wind_direction must render "—", so the converter has to answer
    # null rather than "N" — which would be a fabricated direction.
    assert _compass_points([None, "", "calm"]) == [None, None, None]


# --- 2. the hostile timestamps ------------------------------------------------

HOSTILE = [
    # naive (no offset) — mixing this with an aware value in one sort raises
    {"session_id": "naive", "role": "exchange", "user": "one", "ts": "2026-07-08T00:00:00"},
    # Z-suffixed, which fromisoformat rejects before 3.11 and we normalise
    {"session_id": "zulu", "role": "exchange", "user": "two", "ts": "2026-07-09T00:00:00Z"},
    # not a timestamp at all — an early extractor really wrote these
    {"session_id": "words", "role": "exchange", "user": "three", "ts": "third_turn"},
    # no ts key whatsoever
    {"session_id": "absent", "role": "exchange", "user": "four"},
]


def _log_dir(tmp_path: Path, records: list[dict]) -> Path:
    log_dir = tmp_path / "events"
    log_dir.mkdir()
    (log_dir / "events_2026-07-08.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return log_dir


def test_hostile_timestamps_sort_without_raising_and_land_last(tmp_path):
    event = sessions_event(log_dir=_log_dir(tmp_path, HOSTILE),
                           index_path=tmp_path / "index.json")
    order = [row["id"] for row in event["sessions"]]

    # Parseable ones first, newest first among them...
    assert order[:2] == ["zulu", "naive"]
    # ...and everything unparseable after them, never at the top.
    assert set(order[2:]) == {"words", "absent"}

    by_id = {row["id"]: row for row in event["sessions"]}
    # Z-suffixed on the way out whatever it looked like on disk.
    assert by_id["naive"]["started_at"] == "2026-07-08T00:00:00Z"
    assert by_id["zulu"]["started_at"] == "2026-07-09T00:00:00Z"
    # Unparseable is None — the panel renders UNKNOWN, never a guessed date.
    assert by_id["words"]["started_at"] is None
    assert by_id["absent"]["started_at"] is None


# --- 3. dismiss is not delete -------------------------------------------------

def test_dismiss_hides_the_session_and_writes_no_record(tmp_path):
    log_dir = _log_dir(tmp_path, HOSTILE)
    day_file = log_dir / "events_2026-07-08.jsonl"
    index = tmp_path / "sessions_index.json"
    before = day_file.read_bytes()

    assert "zulu" in {row["id"] for row in
                      sessions_event(log_dir=log_dir, index_path=index)["sessions"]}

    dismiss("zulu", path=index)

    after = sessions_event(log_dir=log_dir, index_path=index)["sessions"]
    assert "zulu" not in {row["id"] for row in after}
    # The other sessions in the same file are untouched by one dismissal.
    assert {"naive", "words", "absent"} <= {row["id"] for row in after}

    # The whole point: the source is byte-for-byte what it was.
    assert day_file.read_bytes() == before

    stored = json.loads(index.read_text(encoding="utf-8"))["dismissed"]
    assert "zulu" in stored
    # hidden_at is written now although nothing reads it yet — an ARCHIVED view
    # is planned, and this makes it a pure frontend addition later.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stored["zulu"])
