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

# Downloaded TTS voice models (gitignored; fetched via piper's downloader).
VOICES_DIR = _PROJECT_ROOT / "models" / "voices"

# Locally-extracted PulseAudio client libs — the no-root WSL bootstrap fallback.
# Used only when the system has no libpulse installed (see interface/audio.py).
VENDOR_PULSE_LIB_DIR = _PROJECT_ROOT / "vendor" / "pulse" / "usr" / "lib" / "x86_64-linux-gnu"


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


# === TTS / SPEECH ===

# Engine registry keys (config.tts_engine). One value today; kokoro may join later.
TTS_ENGINE_PIPER = "piper"

# Speech events emitted by the speech subscriber into the same structured-event
# vocabulary as respond()'s stream — a future frontend renders them; the CLI ignores.
EVENT_SPEAKING_STARTED = "speaking_started"
EVENT_SPEECH_INTERRUPTED = "speech_interrupted"
EVENT_SPEECH_DONE = "speech_done"

# A spoken sentence ends at ./!/?/:/; (plus trailing quotes/brackets) followed by
# whitespace, or at a newline. Digit-dot-digit ("3.5") never matches — no space.
SENTENCE_END_PATTERN = r"[.!?:;][\"'\)\]]*\s|\n"

# Characters stripped from text before synthesis so markdown markup isn't read aloud.
SPEECH_STRIP_CHARS_PATTERN = r"[*_`#]"

# Playback is written to PulseAudio in small chunks so an interrupt lands between
# chunks — this bounds how long Enter can lag before speech actually stops.
PLAYBACK_CHUNK_MS = 100
# Server-side buffer target requested from PulseAudio. Kept small so the buffered
# tail (which an interrupt must flush) never holds more than this much audio.
PLAYBACK_BUFFER_MS = 300

# Runtime voice toggle, parsed by the CLI: "/voice on" | "/voice off".
VOICE_COMMAND = "/voice"


# === TOOLS (core/tools/) ===

# Open-Meteo: free, keyless (fits the no-secrets rule), swappable behind the
# Tool interface. Geocoding resolves a city name to coordinates; forecast
# returns current conditions + daily outlook.
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_S = 10.0
WEATHER_FORECAST_DAYS = 3
# Imperial units on the wire; the tool's field names carry them (_f, _mph, _in)
# so the model always states values with the right unit.
OPEN_METEO_UNITS = {
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch",
}

# Credentials: keys for keyed tool backends live in a gitignored .env at the
# project root, loaded once at boot (core/credentials.py). Variable names are
# documented in the committed .env.example. A missing file or unset key is
# fine — every current backend is keyless.
ENV_PATH = _PROJECT_ROOT / ".env"

# Search backend registry keys (config.search_backend). DuckDuckGo is the
# keyless default; keyed backends (Tavily/Brave) become valid values behind
# BaseSearch once their keys land in .env.
SEARCH_BACKEND_DUCKDUCKGO = "duckduckgo"
SEARCH_MAX_RESULTS = 5
SEARCH_TIMEOUT_S = 10.0

# Appended when a fetched page is cut at fetch_max_chars, so the model knows
# text is missing rather than the page ending there.
FETCH_TRUNCATION_MARKER = "\n[…truncated]"

# Wikipedia REST API (keyless). Search resolves a fuzzy topic to an article
# key; summary returns the lead extract. Wikimedia asks clients for a
# descriptive User-Agent.
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{key}"
WIKIPEDIA_TIMEOUT_S = 10.0
# Wikimedia's edge rejects (403) User-Agents without URL- or email-shaped
# contact info, per their UA policy — the mailto placeholder satisfies it.
HTTP_USER_AGENT = "Jarvis/0.1 (local personal assistant; mailto:user@example.org)"

# WMO weather-interpretation codes → human-readable conditions (Open-Meteo's
# `weather_code` field). Fixed vocabulary from the WMO standard.
WMO_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorm", 96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


# === EVENT LOG (core/memory/event_log.py) ===

EVENTS_LOG_DIR = LOGS_DIR / "events"
EVENT_LOG_FILE_FORMAT = "events_%Y-%m-%d.jsonl"  # one JSONL file per calendar day
EVENT_LOG_GLOB = "events_*.jsonl"  # bulk digest enumerates day-files by this


# === MEMORY: LAYER-2 DIGEST (core/memory/digest.py) ===

# Day-digests are generated user state (like config.yaml), gitignored — a cache
# of the per-day LLM extraction, consumed by merge without re-extraction.
DIGESTS_DIR = _PROJECT_ROOT / "data" / "digests"
DIGEST_FILE_FORMAT = "digest_%Y-%m-%d.json"  # one digest per event day-file

# Fact-source trust vocabulary, ordered HIGHEST trust first — merge resolves
# same-subject overrides by this order (user beats tool beats bare assistant).
FACT_SOURCE_USER = "user_asserted"
FACT_SOURCE_TOOL = "tool_derived"
FACT_SOURCE_ASSISTANT = "assistant_claimed"
FACT_SOURCES = (FACT_SOURCE_USER, FACT_SOURCE_TOOL, FACT_SOURCE_ASSISTANT)

# Fixed category enum — free-text categories drift across runs (same failure
# shape as field-name drift), and merge's drop-list can't match a moving
# vocabulary. Extraction is a pure classifier over these nine; whether a
# category is durable or ephemeral is decided at merge-read, never here.
CATEGORY_PERSONAL_FACT = "personal_fact"  # user's life, location, identity, relationships
CATEGORY_USER_PREFERENCE = "user_preference"  # likes, dislikes, habits, working style
CATEGORY_USER_GOAL = "user_goal"  # projects, intentions, things being learned/pursued
CATEGORY_WORLD_FACT = "world_fact"  # durable external facts (event dates, stable knowledge)
CATEGORY_PROJECT_FACT = "project_fact"  # the user's work, repos, systems
CATEGORY_CURRENT_STATE = "current_state"  # time-of-day, momentary status
CATEGORY_WEATHER_LOOKUP = "weather_lookup"  # one-off weather current/forecast queries
CATEGORY_REFERENCE_LOOKUP = "reference_lookup"  # definitions, general explanations
CATEGORY_PUZZLE_OR_TASK = "puzzle_or_task"  # answers to puzzles, one-shot task outputs
FACT_CATEGORIES = (
    CATEGORY_PERSONAL_FACT,
    CATEGORY_USER_PREFERENCE,
    CATEGORY_USER_GOAL,
    CATEGORY_WORLD_FACT,
    CATEGORY_PROJECT_FACT,
    CATEGORY_CURRENT_STATE,
    CATEGORY_WEATHER_LOOKUP,
    CATEGORY_REFERENCE_LOOKUP,
    CATEGORY_PUZZLE_OR_TASK,
)
# Off-enum model output floors to a DURABLE default: under-filtering is the
# chosen failure mode — a kept borderline fact is visible and rule-fixable,
# a dropped real one is invisible.
CATEGORY_FALLBACK = CATEGORY_WORLD_FACT

# Extraction wants reproducible parsing, not creativity — near-zero temperature
# (a dev-fixed extraction detail, unlike the user-tunable chat temperature).
DIGEST_TEMPERATURE = 0.1

# Runtime digest trigger, parsed by the CLI:
# "/digest [YYYY-MM-DD] [--all] [--force]" (default day: today).
DIGEST_COMMAND = "/digest"
DIGEST_FLAG_ALL = "--all"  # digest every day-file, skipping existing digests
DIGEST_FLAG_FORCE = "--force"  # re-digest despite an existing cache


# === MEMORY: LAYER-2 MERGE (core/memory/merge.py) ===

# The drop-list: categories skipped at MERGE-READ. This is the ONE tunable
# place for what counts as noise — raw digests keep everything, so adding a
# category here and re-merging is free and reversible, no re-extraction.
# Deliberately exactly these four (under-filter bias): widen only when real
# noise is observed surviving into the profile.
EPHEMERAL_CATEGORIES = (
    CATEGORY_CURRENT_STATE,
    CATEGORY_WEATHER_LOOKUP,
    CATEGORY_REFERENCE_LOOKUP,
    CATEGORY_PUZZLE_OR_TASK,
)

# Assistant self-reports ("the assistant is functioning well") carry no memory
# value, but the extractor emits them under DURABLE categories (observed live:
# personal_fact), so the category drop-list can't catch them. The subject the
# model assigns them is reliably assistant_-prefixed, so merge-read drops any
# such fact the USER didn't assert — same reversibility as the category rule:
# raw digests keep everything.
ASSISTANT_SELF_SUBJECT_PREFIX = "assistant_"

# Render-time cap per fact in the Layer-3 profile block (system-prompt view
# only — profile.json always stores the full fact). Guards the prompt against
# one huge extracted fact; 500 chars keeps any real fact intact while bounding
# the worst case (found in the adversarial sweep, item 8).
PROFILE_FACT_RENDER_MAX = 500

# Merge enumerates cached day digests by this glob; names that don't parse as
# DIGEST_FILE_FORMAT (e.g. *.rejected.json dumps) are skipped.
DIGEST_FILE_GLOB = "digest_*.json"

# Generated user state (like config.yaml), gitignored — profile.example.json
# is the committed schema/seed.
PROFILE_PATH = _PROJECT_ROOT / "data" / "profile.json"

# Runtime merge trigger, parsed by the CLI: "/merge" (no arguments).
MERGE_COMMAND = "/merge"


# === WEB (server.py — the WebSocket chat surface) ===

# Bind-all so a browser on another machine can reach the demo; the frontend
# carries the endpoint as DATA (an editable field), never baked into logic.
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# The one message key a client sends; events stream back verbatim as JSON.
WS_TEXT_KEY = "text"


# === LOGGING ===

LOGGER_ROOT = "jarvis"
LOGGER_ORCHESTRATOR = "jarvis.orchestrator"
LOGGER_MODEL = "jarvis.model"
LOGGER_SPEECH = "jarvis.speech"
LOGGER_MEMORY = "jarvis.memory"
LOGGER_TOOLS = "jarvis.tools"

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

# How a `delegation` event renders in the terminal (status comes from the tool).
DELEGATION_LINE_FORMAT = "[{status}…]"


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
