"""Live integration tier — the smoke tests, made permanent.

Everything here talks to a REAL Ollama daemon and the configured primary model.
Marked `integration`: skipped by a bare `pytest`, run via `pytest -m integration`
(make test-live). If the daemon is unreachable or the model isn't pulled, the
whole module SKIPS (never errors) so the suite is safe on any machine.
"""
import httpx
import pytest

from core.constants import FALLBACK_MESSAGE, OLLAMA_ENDPOINT_TAGS
from core.orchestrator.agent import Agent
from core.orchestrator.personality import build_system_prompt
from models.ollama_model import OllamaModel
from setup import config as cfg
from setup.config import JarvisConfig

pytestmark = pytest.mark.integration

# The hard prompt from the definition-of-done smoke test: forces real reasoning,
# so it exercises the reasoning-vs-content budget split end to end.
ZEBRA_PUZZLE = (
    "Five houses in a row are each a different color and house owners of different "
    "nationalities, with different pets, drinks, and cigarette brands. "
    "The Englishman lives in the red house. The Spaniard owns the dog. "
    "Coffee is drunk in the green house. The Ukrainian drinks tea. "
    "The green house is immediately to the right of the ivory house. "
    "The Old Gold smoker owns snails. Kools are smoked in the yellow house. "
    "Milk is drunk in the middle house. The Norwegian lives in the first house. "
    "The man who smokes Chesterfields lives next to the man with the fox. "
    "Kools are smoked next to the house where the horse is kept. "
    "The Lucky Strike smoker drinks orange juice. The Japanese smokes Parliaments. "
    "The Norwegian lives next to the blue house. "
    "Who drinks water? Who owns the zebra?"
)


def _load_config() -> JarvisConfig:
    """Use the real install's config when present, else committed defaults."""
    try:
        return cfg.load()
    except FileNotFoundError:
        return JarvisConfig()


@pytest.fixture(scope="module")
def config() -> JarvisConfig:
    return _load_config()


@pytest.fixture(scope="module")
def live_ollama(config) -> JarvisConfig:
    """Skip (not fail) unless the daemon answers AND the primary model is pulled."""
    try:
        resp = httpx.get(
            f"{config.ollama_base_url.rstrip('/')}{OLLAMA_ENDPOINT_TAGS}", timeout=3.0
        )
        resp.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama not reachable at {config.ollama_base_url}: {exc}")
    pulled = [m["name"] for m in resp.json().get("models", [])]
    if config.primary_model not in pulled:
        pytest.skip(f"{config.primary_model} not pulled (have: {pulled or 'none'})")
    return config


@pytest.fixture
def model(live_ollama) -> OllamaModel:
    c = live_ollama
    return OllamaModel(
        model_id=c.primary_model,
        base_url=c.ollama_base_url,
        max_tokens=c.max_tokens,
        temperature=c.temperature,
        keep_alive=c.ollama_keep_alive,
        timeout=c.ollama_request_timeout,
    )


def _messages(config, user_text: str) -> list[dict]:
    """Mirror the app: live identity + /no_think per the configured thinking flag."""
    system = build_system_prompt(
        cfg.load_identity(), {}, enable_thinking=config.enable_thinking
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]


async def test_health_check_returns_true_against_running_daemon(model):
    assert await model.health_check() is True


async def test_stream_say_hello_yields_visible_content(model, live_ollama):
    tokens = [t async for t in model.stream(_messages(live_ollama, "say hello"))]

    assert tokens, "stream produced no content tokens"
    assert "".join(tokens).strip(), "stream produced only whitespace"


async def test_zebra_puzzle_streams_a_real_answer_through_the_agent(model, live_ollama):
    # The zebra puzzle regularly reasons past the whole max_tokens budget at the
    # raw stream() layer (a real zero-content turn — gotcha #2). The app's answer
    # to that is the Agent's one-shot recovery, so this smoke test drives the
    # actual response path: Agent.respond() over the live model.
    agent = Agent(model, live_ollama, cfg.load_identity())

    events = [e async for e in agent.respond(ZEBRA_PUZZLE)]

    errors = [e for e in events if e["type"] == "error"]
    assert not errors, f"model stream errored: {errors}"
    answer = "".join(e["content"] for e in events if e["type"] == "token").strip()
    assert answer, "turn ended with no visible content"
    # The honest fallback would mean recovery ALSO produced nothing — for the
    # definition-of-done smoke test a real answer has to come out.
    assert answer != FALLBACK_MESSAGE, "recovery failed — only the fallback was emitted"
    assert events[-1] == {"type": "done"}


async def test_warmup_succeeds_and_reports_elapsed_time(model):
    result = await model.warmup()

    assert result.success is True
    assert result.model_id == model.model_id
    assert result.elapsed_s > 0
    assert result.error is None
