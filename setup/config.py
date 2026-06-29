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

from core.constants import CONFIG_PATH, DEFAULTS_PATH


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


def load_defaults() -> JarvisConfig:
    """Build a JarvisConfig seeded from defaults.yaml (used by the wizard)."""
    data = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8")) or {}
    local = data.get("local_models", {})
    return JarvisConfig(
        primary_model=local.get("primary", "qwen3:14b"),
        reasoning_model=local.get("reasoning", "deepseek-r1:14b"),
    )
