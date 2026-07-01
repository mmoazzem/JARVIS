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


# === SYSTEM SCAN ===
# Probes used by setup/system_scan.py. Each is run defensively; a missing binary or
# non-zero exit becomes a None/empty field with a noted reason, never an exception.

NVIDIA_SMI_BINARY = "nvidia-smi"
# Ask nvidia-smi for just the name and total VRAM, machine-readable (no header/units).
NVIDIA_SMI_QUERY_ARGS = (
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
)
NVIDIA_SMI_TIMEOUT_S = 5.0
MB_PER_GB = 1024  # nvidia-smi reports VRAM in MiB; divide to get GB
BYTES_PER_GB = 1024**3  # psutil reports RAM in bytes


# === RUNTIME (Ollama process management) ===
# Used by core/runtime/ollama_manager.py to DETECT and REMEDIATE runtime state
# (start the daemon, pull the model). It never INSTALLS the binary — that is the
# bootstrap bash script's job.

OLLAMA_BINARY = "ollama"
OLLAMA_SERVE_ARGS = ("ollama", "serve")
OLLAMA_SYSTEMCTL_START_ARGS = ("systemctl", "start", OLLAMA_BINARY)
# Used only to STOP a daemon Jarvis itself started via systemctl (ownership follows
# creation — see core/runtime/ollama_manager.stop_owned_daemon).
OLLAMA_SYSTEMCTL_STOP_ARGS = ("systemctl", "stop", OLLAMA_BINARY)

# Native REST endpoints used for runtime management (the chat path uses the OpenAI
# surface — see OLLAMA_OPENAI_SUFFIX above).
OLLAMA_ENDPOINT_TAGS = "/api/tags"  # lists pulled models; also a readiness probe
OLLAMA_ENDPOINT_PULL = "/api/pull"  # streams model-download progress

# Readiness polling after we start the daemon ourselves.
OLLAMA_START_TIMEOUT_S = 10.0  # total time to wait for the daemon to come up
OLLAMA_POLL_INTERVAL_S = 0.5  # gap between readiness probes
OLLAMA_PROBE_TIMEOUT_S = 2.0  # per-probe HTTP timeout (a hung probe must not block)

# Returned verbatim (and asserted in tests) when the binary is absent. The manager
# must NOT try to install it — it points the user at the bootstrap script instead.
OLLAMA_NOT_INSTALLED_DETAIL = (
    "Ollama not installed — run the setup bash script (e.g. ./setup.sh)"
)
# Factual reason the daemon couldn't be brought up (the interface adds the next step).
OLLAMA_DAEMON_FAILED_DETAIL = (
    "Ollama installed but the daemon did not become ready in time"
)


# === BOOT STATUS (structured transitions; the interface renders them) ===
# The boot sequence REPORTS state as data (BootEvent.stage), exactly like respond()'s
# event stream — the CLI now, a frontend later, decides the wording. These stage names
# are the shared vocabulary between the manager (emits) and the interface (renders).

STAGE_NOT_INSTALLED = "not_installed"
STAGE_STARTING_DAEMON = "starting_daemon"
STAGE_DAEMON_READY = "daemon_ready"
STAGE_DAEMON_FAILED = "daemon_failed"
STAGE_MODEL_MISSING = "model_missing"
STAGE_PULLING_MODEL = "pulling_model"
STAGE_PULL_FAILED = "pull_failed"
STAGE_PULL_DECLINED = "pull_declined"
STAGE_MODEL_READY = "model_ready"
STAGE_WARMING = "warming"
STAGE_WARMUP_READY = "warmup_ready"
STAGE_WARMUP_FAILED = "warmup_failed"

# Fixed lines the interface prints for the daemon-start transition (DoD-specified).
OLLAMA_STARTING_MSG = "Ollama not running. Starting Ollama…"
OLLAMA_READY_MSG = "Ollama ready."


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

# The input prompt shown once per turn before any model output.
CLI_PROMPT = "You: "


# === MESSAGES ===

# Emitted only if a generation yields nothing — the turn must never be silent.
FALLBACK_MESSAGE = "I wasn't able to produce a response just now, sir. Please try again."
GOODBYE_MESSAGE = "Goodbye, sir."

# Shown at boot when a required precondition is unmet (model absent and the user
# declined the pull, a pull failed, or Ollama is missing). An unmet precondition is
# a TERMINAL boot state: print this, then exit non-zero — never fall through into a
# model-less chat loop. Formatted with the CONFIGURED primary model, never a
# hardcoded model name (CLAUDE.md: no model names in logic).
MODEL_UNAVAILABLE_NEXT_STEP = (
    "Jarvis needs {model} to run. Pull it with `ollama pull {model}` "
    "(or run ./setup.sh), then start Jarvis again."
)

# Appended on the single zero-content recovery attempt (gotcha #2): reasoning ate
# the budget and left no answer, so we re-ask for a direct answer with no further
# reasoning to leave room in the budget for content.
RECOVERY_INSTRUCTION = (
    "Provide your final answer now, directly and concisely, with no further reasoning."
)
