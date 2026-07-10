"""
JarvisConfig schema and load/save helpers.

config.yaml is gitignored and wizard-written — this module reads and writes it
but never creates it from scratch (that is the wizard's job).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict

from core.constants import CONFIG_PATH, DEFAULTS_PATH, IDENTITY_PATH


class JarvisConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "local"
    provider: str = "ollama"
    primary_model: str = "qwen3:14b"
    # reserved/unused — future reasoning specialist; requires unload/load swap before use
    reasoning_model: str = "deepseek-r1:14b"
    ollama_base_url: str = "http://localhost:11434"
    context_token_budget: int = 8000
    max_tokens: int = 18000
    temperature: float = 0.7
    ollama_keep_alive: str = "5m"
    ollama_request_timeout: float = 30.0
    enable_thinking: bool = False
    # Voice output (Pass 1: preset voice). Off by default; /voice on toggles at runtime.
    tts_enabled: bool = False
    tts_engine: str = "piper"
    tts_voice: str = "en_GB-alan-medium"
    # Silence prepended to the first clip a playback stream opens with: some sinks
    # swallow the first ~200-400ms played into a cold sink (WSLg's RDP bridge does),
    # clipping the opening word mid-syllable. Environment-dependent, so it's config
    # not code: 0 = no pad (native stacks), ~300 covers this WSL install.
    tts_preroll_ms: int = 0
    # Raw interaction capture for the future memory layer (one JSONL record per turn).
    event_log_enabled: bool = True
    # Native tool-calling (Tools Pass 1: time + weather). The model decides when
    # to call; default_location is where "what's the weather?" points with no city.
    tools_enabled: bool = True
    default_location: str = "Buffalo, NY"
    # Web search backend behind BaseSearch (Tools Pass 2). duckduckgo is
    # keyless; keyed backends (tavily/brave) become valid once their keys
    # land in .env and their implementations exist.
    search_backend: str = "duckduckgo"
    # Fetch-and-read (Tools Pass 3): search fetches the top N result pages so
    # the model reasons over real page text, not snippets. Slower but precise;
    # 0 = snippet-only, the old fast behavior. fetch_max_chars caps the text
    # kept per page so multiple sources don't blow the context budget.
    search_fetch_count: int = 3
    fetch_max_chars: int = 6000
    fetch_timeout: float = 10.0
    # Memory Layer 2 (/digest): the fact-extraction model. Empty = reuse the
    # resident primary model — two 14B models cannot co-reside in 16 GB VRAM
    # (CLAUDE.md), so naming a different model here implies Ollama's
    # unload/load swap on each digest run.
    digest_model: str = ""
    # Extraction is ONE non-streaming completion over a whole day's log, so it
    # legitimately runs far longer than a chat turn's ollama_request_timeout.
    digest_timeout: float = 180.0
    # WHICH facts one extraction pass captures varies run to run, and digests
    # are run-once caches — a miss would freeze forever. N passes are unioned
    # so recall stabilizes; higher = better recall, N× the LLM cost.
    digest_passes: int = 3
    created_at: Optional[str] = None


def load() -> JarvisConfig:
    """Read config.yaml and return a validated JarvisConfig.

    Raises FileNotFoundError if config.yaml is absent — caller should direct the
    user to run the setup wizard.
    """
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return JarvisConfig(**data)


def save(config: JarvisConfig) -> None:
    """Persist config to config.yaml, stamping created_at on first write."""
    data = config.model_dump()
    if data.get("created_at") is None:
        data["created_at"] = datetime.now(timezone.utc).isoformat()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.dump(data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_identity() -> dict:
    """Read identity.yaml LIVE and return it as a dict.

    The persona is loaded fresh every boot and used verbatim — never copied into
    config.yaml (see CLAUDE.md). Edit identity.yaml + restart to apply changes.
    """
    return yaml.safe_load(IDENTITY_PATH.read_text(encoding="utf-8")) or {}


def load_defaults_raw() -> dict:
    """Read defaults.yaml as a raw dict (used for setup-time keys not in the schema,
    e.g. `local_vram_floor_gb`, which the wizard reads before config.yaml exists)."""
    return yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8")) or {}


def load_defaults() -> JarvisConfig:
    """Build a JarvisConfig seeded from defaults.yaml (used by the wizard)."""
    data = load_defaults_raw()
    local = data.get("local_models", {})
    return JarvisConfig(
        primary_model=local.get("primary", "qwen3:14b"),
        reasoning_model=local.get("reasoning", "deepseek-r1:14b"),
    )
