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
# App-written user choices, NOT install shape: config.yaml is the wizard's file
# and describes this machine; preferences are rewritten at runtime whenever the
# user changes a setting. Different owner and write path, so a different file.
PREFERENCES_PATH = CONFIG_DIR / "preferences.json"
LOGS_DIR = _PROJECT_ROOT / "logs"

# Downloaded TTS voice models (gitignored; fetched via piper's downloader).
VOICES_DIR = _PROJECT_ROOT / "models" / "voices"


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
# vocabulary as respond()'s stream — the frontend renders them; the CLI ignores.
EVENT_SPEAKING_STARTED = "speaking_started"
EVENT_SPEECH_INTERRUPTED = "speech_interrupted"
EVENT_SPEECH_DONE = "speech_done"
# Metering, not an occurrence: one RMS reading of the audio currently audible.
EVENT_SPEECH_LEVEL = "speech_level"

# A spoken sentence ends at ./!/?/:/; (plus trailing quotes/brackets) followed by
# whitespace, or at a newline. Digit-dot-digit ("3.5") never matches — no space.
SENTENCE_END_PATTERN = r"[.!?:;][\"'\)\]]*\s|\n"

# Characters stripped from text before synthesis so markdown markup isn't read aloud.
SPEECH_STRIP_CHARS_PATTERN = r"[*_`#]"

# --- sentence-split guards (interface/speech_text.py) ---
# A "." after one of these is an abbreviation, not the end of a sentence. Without
# this "Dr. Chen" becomes two clips with an audible synthesis gap between them.
# Compared with dots removed, so "e.g" matches "eg". Words that are also
# ordinary English ("no", "us", "am") are deliberately absent: guarding them
# would swallow the full stop in "the answer is no. Then...", which is a worse
# failure than the fragment it would prevent.
SPEECH_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "mt",
    "vs", "etc", "eg", "ie", "approx", "fig", "al",
    "inc", "ltd", "co", "dept", "univ", "ave", "blvd", "rd",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun",
})
# A ":" closing one of these is a URL scheme, never a sentence end.
SPEECH_URL_SCHEMES = frozenset({"http", "https", "ws", "wss", "ftp", "file", "mailto"})
# The word immediately before a candidate break, for the two checks above.
SPEECH_LAST_WORD_PATTERN = r"[A-Za-z][A-Za-z.]*$"

# --- speech normalization (interface/speech_text.py) ---
# Written form and spoken form are different jobs. The transcript and the tile
# keep their symbols; these substitutions apply ONLY to what Piper receives.
# Observed live with en_GB-alan-medium: "°F" reads as "degree F" and "kn" as
# "kay an". Percentages, decimals, "24/7" and "3:30 PM" already read correctly
# and are deliberately absent — normalizing what works only adds risk.
SPEECH_DEGREE_UNITS = (("°F", " degrees Fahrenheit"), ("°C", " degrees Celsius"))
SPEECH_DEGREE_BARE = ("°", " degrees")
# Only a unit that follows a number. A bare "kn" is not a wind speed.
SPEECH_KNOTS_PATTERN = r"(?<=\d)\s*kn\b"
SPEECH_KNOTS_REPLACEMENT = " knots"
# "re:" is read as the letters. Requires the colon, so the word "re" is untouched.
SPEECH_RE_PATTERN = r"\bre:"
SPEECH_RE_REPLACEMENT = "regarding"
# Capitalized abbreviations only — lowercase "sat"/"mar" are ordinary words.
# "May"/"June"/"July" are already whole words and are left out on purpose.
SPEECH_MONTH_WORDS = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September",
    "Sept": "September", "Oct": "October", "Nov": "November", "Dec": "December",
}
SPEECH_DAY_WORDS = {
    "Mon": "Monday", "Tue": "Tuesday", "Tues": "Tuesday", "Wed": "Wednesday",
    "Thu": "Thursday", "Thur": "Thursday", "Thurs": "Thursday",
    "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday",
}

# Playback is written to PulseAudio in small chunks so an interrupt lands between
# chunks — this bounds how long Enter can lag before speech actually stops.
PLAYBACK_CHUNK_MS = 100
# Server-side buffer target requested from PulseAudio. Kept small so the buffered
# tail (which an interrupt must flush) never holds more than this much audio.
PLAYBACK_BUFFER_MS = 300

# Metering cadence. With a meter attached the write loop paces itself at this
# interval instead of PLAYBACK_CHUNK_MS, so the meter IS the playback clock —
# no second timer to drift against the sound. Divides PLAYBACK_CHUNK_MS exactly
# so the drain loop's plateau comparison keeps its original 100 ms spacing.
SPEECH_LEVEL_INTERVAL_MS = 50
# Window each reading averages over. One reading of ~a syllable's worth of audio;
# shorter reads as flicker, longer smears the loud passages into the quiet ones.
SPEECH_LEVEL_WINDOW_MS = 50
# Meter calibration: the RMS that reads as full scale. Speech is peaky and its
# RMS sits well below digital full scale, so raw values would leave the meter
# barely off its floor — this is a reference level like any VU meter's, NOT a
# synthesized envelope. Measured over 50 ms windows of piper en_GB-alan-medium:
# median 0.11-0.16, p90 0.30, peak 0.45. This puts an ordinary syllable near
# half scale and leaves the loudest ones room to still read as louder.
SPEECH_LEVEL_REFERENCE_RMS = 0.35

# Runtime voice toggle, parsed by the CLI: "/voice on" | "/voice off".
VOICE_COMMAND = "/voice"


# === TOOLS (core/tools/) ===

# Open-Meteo: free, keyless (fits the no-secrets rule), swappable behind the
# Tool interface. Geocoding resolves a city name to coordinates; forecast
# returns current conditions + daily outlook.
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_S = 10.0
# Ten days cost the same request as three, so the planned detail view becomes a
# rendering change rather than a wire change.
WEATHER_FORECAST_DAYS = 10

# How long one fetched reading stands in for the next call. Upstream refreshes
# every 15 minutes ("interval": 900 in the response), so a faster fetch buys an
# identical value and a much slower one serves a reading the API has replaced.
# Developer-fixed on purpose: this is a property of Open-Meteo's clock, not a
# user preference.
WEATHER_READING_TTL_S = 600.0

# Requested measurements. Both lists go on the wire as comma-joined strings.
OPEN_METEO_CURRENT_FIELDS = (
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "weather_code", "wind_speed_10m", "wind_direction_10m",
)
OPEN_METEO_DAILY_FIELDS = (
    "weather_code", "temperature_2m_max", "temperature_2m_min",
    "precipitation_probability_max",
)

# Response field -> the neutral key the tool returns it under. The unit USED to
# live in these names (temperature_f); it is now a value in the `units` map, so
# a unit change cannot leave the label disagreeing with the number.
OPEN_METEO_CURRENT_KEYS = {
    "temperature": "temperature_2m",
    "feels_like": "apparent_temperature",
    "humidity": "relative_humidity_2m",
    "precipitation": "precipitation",
    "wind_speed": "wind_speed_10m",
    "wind_direction": "wind_direction_10m",
}
OPEN_METEO_DAILY_KEYS = {
    "high": "temperature_2m_max",
    "low": "temperature_2m_min",
    "precip_chance": "precipitation_probability_max",
}
# Which response field's declared unit answers for each dimension, read out of
# the response's own `current_units` (with a `daily_units` fallback for a
# current-less payload) — never inferred from what we asked for.
OPEN_METEO_UNIT_FIELDS = {
    "temperature": "temperature_2m",
    "wind_speed": "wind_speed_10m",
    "precipitation": "precipitation",
}
OPEN_METEO_DAILY_UNIT_FIELDS = {"temperature": "temperature_2m_max"}

# Preference dimension -> the query param that asks for it. Preference VALUES
# are Open-Meteo's own vocabulary (see UNIT_CHOICES), so there is no translation
# layer between the stored choice and the wire.
OPEN_METEO_UNIT_PARAMS = {
    "temperature": "temperature_unit",
    "wind_speed": "wind_speed_unit",
    "precipitation": "precipitation_unit",
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
# The date a day-file covers, recovered from its own name — the only thing that
# can group records written before session ids existed.
EVENT_LOG_DATE_PATTERN = r"events_(\d{4}-\d{2}-\d{2})\.jsonl$"


# === SESSIONS PANEL FEED (core/memory/sessions_view.py -> the frontend) ===
# Conversations as the SESSIONS panel sees them, grouped by the session_id the
# event log now writes on every record.

SESSIONS_EVENT_TYPE = "sessions"

# Records written before session ids existed are grouped one-session-per-day.
# They are NEVER rewritten to add an id: the JSONL is append-only ground truth,
# and Layer 2 has already digested part of it.
LEGACY_SESSION_PREFIX = "legacy:"
LEGACY_SESSION_TITLE = "Archived — {date}"

# A title is the session's first user message, cut on a word boundary. Long
# enough to identify the conversation, short enough for one panel row.
SESSION_TITLE_MAX = 52
SESSION_UNTITLED = "Untitled"

# DISMISSED, not deleted: this records what the panel hides, and nothing else.
# Actual retention (purging turns, retracting facts) is a memory-layer decision
# that belongs with the digest, not with a panel button — removing source lines
# would strand derived facts in profile.json with no provenance.
SESSIONS_INDEX_PATH = LOGS_DIR / "sessions_index.json"
SESSIONS_DISMISSED_KEY = "dismissed"
SESSIONS_HIDDEN_AT_KEY = "hidden_at"


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


# === PREFERENCES (core/preferences.py) ===

# Units are PER DIMENSION, not one imperial/metric flag: Fahrenheit with knots
# is a real combination a single flag cannot express, and Open-Meteo supports
# each axis independently. Values ARE Open-Meteo's param vocabulary verbatim.
UNIT_CHOICES = {
    "temperature": ("fahrenheit", "celsius"),
    "wind_speed": ("kn", "mph", "kmh", "ms"),
    "precipitation": ("inch", "mm"),
}

# Seeded on first read, and the fallback for anything the stored file gets
# wrong. A malformed preference must degrade to this, never fail a boot.
DEFAULT_PREFERENCES = {
    "units": {
        "temperature": "fahrenheit",
        "wind_speed": "kn",
        "precipitation": "inch",
    },
}

# Preferences are addressed by dotted path ("units.temperature") so every
# dimension rides one write op instead of one op per setting.
PREFERENCE_KEY_SEPARATOR = "."
PREFERENCE_UNITS = "units"


# === WEB (server.py — the WebSocket chat surface) ===

# Bind-all so a browser on another machine can reach the demo; the frontend
# carries the endpoint as DATA (an editable field), never baked into logic.
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# The chat message key a client sends; events stream back verbatim as JSON.
WS_TEXT_KEY = "text"

# The second thing a client may send: an OPERATION, not chat. Routing is on the
# presence of this key, so an op can never be mistaken for something to answer.
WS_OP_KEY = "op"
WS_OP_SET_PREFERENCE = "set_preference"
WS_OP_DISMISS_SESSION = "dismiss_session"
WS_OP_KEY_FIELD = "key"
WS_OP_VALUE_FIELD = "value"
WS_OP_ID_FIELD = "id"
# Ack for an applied preference write — the client learns the stored value,
# which may differ from what it believes it sent.
WS_PREFERENCE_EVENT_TYPE = "preference"


# === TELEMETRY (ambient host meters — core/runtime/telemetry.py) ===
# Pushed OUTSIDE a turn, unlike every other event, so the meters stay alive
# while nothing is being asked.

TELEMETRY_EVENT_TYPE = "telemetry"

# Push cadence per connection. ~1 Hz reads as live without flooding the socket.
TELEMETRY_INTERVAL_S = 1.0

# One hardware reading serves every caller inside this window: N tabs cost one
# nvidia-smi, and psutil's CPU delta keeps a real interval to measure against.
TELEMETRY_CACHE_S = 0.5

# Utilisation and VRAM for the meters, machine-readable (no header/units).
NVIDIA_SMI_TELEMETRY_ARGS = (
    NVIDIA_SMI_BINARY,
    "--query-gpu=utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)


# === MEMORY PANEL FEED (core/memory/profile_view.py -> the frontend) ===
# The profile as the MEMORY panel sees it. Ambient like telemetry, but
# EDGE-triggered: pushed on connect and again only when the digest rewrites
# profile.json, because memory changes on a human clock, not a 1 Hz one.

MEMORY_EVENT_TYPE = "memory"

# How often the file is STAT-ed (not read) for a change. A digest run is a
# deliberate act, so a couple of seconds of lag is invisible; the read itself
# happens only when the stat says the file moved.
MEMORY_POLL_INTERVAL_S = 2.0

# RECENTLY LEARNED is a glance, not an archive — the panel scrolls, but a
# profile with thousands of facts must not put all of them on the wire.
MEMORY_RECENT_MAX = 12


# === WEATHER TILE FEED (core/runtime/weather_feed.py -> the frontend) ===
# The held reading as the tile sees it. Ambient like telemetry, but on the
# UPSTREAM's clock rather than one of ours.

WEATHER_EVENT_TYPE = "weather"

# Poll cadence, and separate from WEATHER_READING_TTL_S on purpose even though
# both are ten minutes: the TTL says how long a reading may be SERVED, this says
# how often we go and look. Open-Meteo refreshes every 15 minutes, so anything
# faster spends a request to be told the same number.
WEATHER_POLL_INTERVAL_S = 600.0

# Rounded to whole units on the way to the TILE only — the held reading and the
# tool's return value keep their decimals. Temperature alone: humidity and
# precipitation carry meaning below the unit (0.02 in is not 0 in).
WEATHER_DISPLAY_ROUNDED_FIELDS = ("temperature", "feels_like")

# Clock time for a recent fact, in the viewer's local zone (stored ts is UTC).
MEMORY_TIME_FORMAT = "%H:%M"


# === SPEECH ON THE WIRE (interface/speech_session.py -> the frontend) ===
# Playback stays on the HOST — the browser is told about speech, it does not
# produce it. These are the four frames that reporting needs.

# Sent at turn start when this turn has claimed the audio device. It is the
# answer to "is speech expected?", which `done` cannot carry: for a short reply
# the last sentence is still being synthesized when `done` lands, so a client
# that rested on `done` would flicker idle->speaking->idle.
WS_SPEECH_PENDING = "speech_pending"
# Actual playback, never the first token: emitted when the first non-silent
# sample is AUDIBLE (past the sink's buffer), not when a clip is handed over.
WS_SPEECH_START = "speech_start"
# Closes every bracket a speech_pending opened — after normal completion, after
# an interruption, and after a playback failure. Fires even if nothing sounded.
WS_SPEECH_END = "speech_end"
# Real RMS of what is being heard, 0..1 against SPEECH_LEVEL_REFERENCE_RMS.
WS_AUDIO_LEVEL = "audio_level"


# === LOGGING ===

LOGGER_ROOT = "jarvis"
LOGGER_ORCHESTRATOR = "jarvis.orchestrator"
LOGGER_MODEL = "jarvis.model"
LOGGER_SPEECH = "jarvis.speech"
LOGGER_MEMORY = "jarvis.memory"
LOGGER_TOOLS = "jarvis.tools"
LOGGER_PREFERENCES = "jarvis.preferences"

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
