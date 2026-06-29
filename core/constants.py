"""
Developer-fixed literals — one sectioned home.

These are values FIXED BY DEVELOPERS, not tuned by users (user-tunable runtime
values live in setup/config.py and config.yaml). Centralising them means a literal
is defined ONCE and imported everywhere it's used, with no scattered duplicates.

This module imports nothing from the rest of the app — pure data, safe to import
from anywhere without a cycle.
"""
from pathlib import Path

# === PATHS ===

_PROJECT_ROOT = Path(__file__).parent.parent  # core/ → project root

CONFIG_DIR = _PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
DEFAULTS_PATH = CONFIG_DIR / "defaults.yaml"
IDENTITY_PATH = CONFIG_DIR / "identity.yaml"
LOGS_DIR = _PROJECT_ROOT / "logs"


# === MODEL / API ===

# Heuristic: assumed characters per token, used only to ESTIMATE history size when
# deciding whether to trim. A dev heuristic (no tokenizer dependency), NOT a
# user-tunable knob — which is why it lives here, not in config.
CHARS_PER_TOKEN = 4

# Qwen soft-switch appended to the system prompt to suppress the reasoning trace.
NO_THINK_DIRECTIVE = "/no_think"

# Ollama default base URL — the config field overrides this at runtime.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

# Ollama exposes an OpenAI-compatible API under this path; the key is required by
# the OpenAI client but ignored by Ollama.
OLLAMA_OPENAI_SUFFIX = "/v1"
OLLAMA_API_KEY_PLACEHOLDER = "ollama"

# Ollama native REST endpoints (used for health checks and non-OpenAI calls).
OLLAMA_ENDPOINT_CHAT = "/api/chat"
OLLAMA_ENDPOINT_GENERATE = "/api/generate"

# Wire field names in a model response chunk. Ollama returns the answer in
# `content` and chain-of-thought in `reasoning` (NOT <think> tags — see CLAUDE.md).
CONTENT_FIELD = "content"
REASONING_FIELDS = ("reasoning", "reasoning_content")

# OpenAI/Ollama chat-message roles.
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


# === PROVIDERS ===

# Cloud provider -> the secret name under which its API key is stored. Shared by
# the wizard (collecting keys) and main (checking them) so the mapping is defined
# exactly once.
PROVIDER_KEYS = {
    "claude": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "google_api_key",
}


# === LOGGING ===

LOGGER_ROOT = "jarvis"
LOGGER_ORCHESTRATOR = "jarvis.orchestrator"
LOGGER_MODEL = "jarvis.model"

LOG_FILE_FORMAT = "%(asctime)s  %(name)-20s %(levelname)-7s %(message)s"
LOG_CONSOLE_FORMAT = "%(levelname)s: %(message)s"
LOG_TIME_FORMAT = "%H:%M:%S"
LOG_DATE_FORMAT = "%Y-%m-%d"
LOG_FILE_NAME_FORMAT = "jarvis_%Y-%m-%d.log"  # one file per calendar day

# Written to the log at every startup so sessions are visually distinct in the file.
LOG_SESSION_BOUNDARY = "=" * 20 + " SESSION START " + "=" * 20


# === UI / PRESENTATION ===
# The terminal is a temporary test harness (a React frontend will replace it), so
# only the genuinely reusable UI literals live here.

BANNER_TEXT = "J A R V I S"
TYPEWRITER_DELAY_S = 0.012
ASSISTANT_DISPLAY_NAME = "Jarvis"
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "bye"}


# === MESSAGES ===

# Emitted only if a generation yields nothing — the turn must never be silent.
FALLBACK_MESSAGE = "I wasn't able to produce a response just now, sir. Please try again."
GOODBYE_MESSAGE = "Goodbye, sir."
