"""System-scan degradation invariants (setup/system_scan.py).

Every probe is mocked at the boundary (binary lookup, HTTP client) — these tests
run with no GPU, no subprocess, and no network. The key regression lock is the
M4 stale-scan bug: an unreachable daemon must report pulled_models = None
(UNKNOWN — could not ask), never [] (which claims "queried, none pulled").
"""
import httpx
import pytest

from setup import system_scan
from setup.system_scan import _probe_gpu, _probe_ollama, scan_system


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient: returns a canned response or raises."""

    response: _FakeResponse | None = None  # None → simulate daemon unreachable

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        if self.response is None:
            raise httpx.ConnectError("connection refused")
        return self.response


@pytest.fixture
def daemon_down(monkeypatch):
    monkeypatch.setattr(_FakeAsyncClient, "response", None)
    monkeypatch.setattr(system_scan.httpx, "AsyncClient", _FakeAsyncClient)


@pytest.fixture
def no_binaries(monkeypatch):
    monkeypatch.setattr(system_scan.shutil, "which", lambda name: None)


async def test_missing_nvidia_smi_degrades_to_none_without_raising(no_binaries):
    notes: list[str] = []

    name, vram = await _probe_gpu(notes)

    assert name is None
    assert vram is None
    assert any("nvidia-smi" in n for n in notes)  # the gap is explained, not silent


async def test_ollama_down_reports_models_unknown_not_empty(daemon_down, no_binaries):
    notes: list[str] = []

    installed, running, models = await _probe_ollama("http://localhost:11434", notes)

    assert running is False
    # THE stale-scan regression lock: None means "could not query", [] would
    # falsely mean "queried, none pulled". These must never be conflated.
    assert models is None
    assert models != []


async def test_ollama_up_with_no_models_reports_empty_list(monkeypatch):
    # The complementary case: daemon reachable, zero models → [] (queried, none).
    monkeypatch.setattr(
        _FakeAsyncClient, "response", _FakeResponse(200, {"models": []})
    )
    monkeypatch.setattr(system_scan.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(system_scan.shutil, "which", lambda name: "/usr/bin/ollama")

    installed, running, models = await _probe_ollama("http://localhost:11434", [])

    assert running is True
    assert models == []


async def test_ollama_up_lists_pulled_models(monkeypatch):
    monkeypatch.setattr(
        _FakeAsyncClient,
        "response",
        _FakeResponse(200, {"models": [{"name": "qwen3:14b"}, {"name": "other:7b"}]}),
    )
    monkeypatch.setattr(system_scan.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(system_scan.shutil, "which", lambda name: "/usr/bin/ollama")

    installed, running, models = await _probe_ollama("http://localhost:11434", [])

    assert models == ["qwen3:14b", "other:7b"]


async def test_full_scan_on_bare_machine_never_raises(daemon_down, no_binaries):
    report = await scan_system("http://localhost:11434")

    assert report.gpu_name is None
    assert report.gpu_vram_gb is None
    assert report.ollama_installed is False
    assert report.ollama_running is False
    assert report.pulled_models is None  # UNKNOWN, not []
    assert report.notes  # every gap left a human-readable reason
