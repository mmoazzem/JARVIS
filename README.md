# Jarvis

Jarvis is a local-first personal AI assistant. Qwen3 14B, served by Ollama on
your own GPU, is the brain and answers everything; no cloud account is needed
to run it. The codebase is deliberately structured — a model-agnostic
`BaseModel` interface, an agent abstraction (model + role-prompt + tools +
run-loop), and structured event streams instead of prints — so it can grow
toward memory, voice, and a graphical frontend without rewrites.

## Requirements

- Ubuntu 24.04 (native or WSL2). Development happens on Windows; Jarvis runs
  in the Linux environment.
- Python 3.11+.
- [Ollama](https://ollama.com) installed and on `PATH`.
- A CUDA GPU. The first-run wizard recommends local mode at **10 GB VRAM or
  more**; the reference machine is an RTX 5080 with 16 GB. (Below the floor
  the wizard recommends cloud mode, but cloud setup is a future phase — it
  still configures local today.)
- The primary model: `qwen3:14b` (~9.3 GB download). `deepseek-r1:14b` is
  reserved in config for a future reasoning specialist — nothing loads it
  today, and two 14B models cannot co-reside in 16 GB anyway.

## Setup on a new machine

```bash
# 1. Install Ollama (Jarvis manages the daemon and models, never the binary)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the primary model (optional — Jarvis offers to pull it at boot)
ollama pull qwen3:14b

# 3. Install Python dependencies (Ubuntu 24.04 is externally-managed —
#    use a virtual environment)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Run
python main.py
```

The first boot runs the setup wizard: it scans the machine (OS, CPU, RAM, GPU
VRAM, Ollama state, pulled models), recommends local vs cloud mode, and builds
the runtime config in memory. Only after the whole boot succeeds — daemon up,
model present — is `config/config.yaml` written, so a failed first run leaves
nothing behind. Later boots load that file and skip the wizard.

## How it runs

Boot is a gated pipeline; each stage is a precondition for the next:

1. **Scan / config** — first run scans the system and builds a config via the
   wizard; later runs load `config/config.yaml`.
2. **Ensure Ollama** — if the daemon isn't answering, Jarvis starts it (via
   `systemctl`, falling back to `ollama serve`). If the configured model isn't
   pulled, Jarvis offers to pull it with streamed progress — never silently.
3. **Commit** — on a first run only, `config.yaml` is written now, at the one
   point where setup has truly succeeded.
4. **Warmup** — a 1-token request forces the model into VRAM so the first real
   question isn't a cold load.
5. **Chat** — a terminal loop with streaming answers. Multi-line pastes arrive
   as one message (bracketed paste). Exit with `/exit`, `/quit`, `exit`,
   `quit`, or `bye`.

**Daemon ownership follows creation:** if Jarvis started the Ollama daemon, it
stops it on every exit path; if the daemon was already running, it belongs to
the system and is left alone. For a fast development loop, start Ollama yourself
(`systemctl start ollama`) before running Jarvis — it then persists across runs
and the model stays warm, so you skip the daemon start and cold warmup each
time.

Two model-behavior gotchas are handled in the core: Qwen's reasoning arrives in
a separate `reasoning` field (never `<think>` tags) and is kept out of the
answer stream, and reasoning shares the token budget with the answer — a turn
whose reasoning consumes the whole budget is detected and retried once with a
direct-answer instruction, falling back to an honest visible message. A turn is
never silently empty.

## The config model

Four files, four jobs — kept strictly apart:

| File | Committed? | Holds |
| --- | --- | --- |
| `core/constants.py` | yes | Developer-fixed literals (paths, endpoints, stage names, messages). Not user-tunable. |
| `config/defaults.yaml` | yes | Seed values the wizard uses to build the config (model stack, VRAM floor). |
| `config/config.yaml` | no (gitignored) | How THIS install is set up — wizard-written, user-tunable runtime: model, token budgets, temperature, keep_alive, timeouts. |
| `config/identity.yaml` | yes | The persona. Loaded live at every boot, never copied into runtime config — edit it and restart to apply. |

The rule: if a user would tune it, it goes in config; if it's a fixed code
detail, it's a constant.

## Project layout

```
main.py                 entry point — the gated boot ceremony
core/
  constants.py          developer-fixed literals, sectioned
  orchestrator/         the first Agent: run-loop, conversation history,
                        system-prompt assembly (identity + runtime state)
  runtime/              Ollama process management: detect, start, pull, stop
models/
  base.py               BaseModel interface (local/cloud interchangeable)
  ollama_model.py       one class for every Ollama model; model_id is config
setup/
  wizard.py             first-run scan + recommendation + config build
  system_scan.py        host probe (OS, CPU, RAM, GPU, Ollama state)
  config.py             JarvisConfig schema, load/save, identity loading
  logging_setup.py      daily append-mode log file, quiet console
interface/
  cli.py                terminal chat surface — presentation only
config/                 defaults.yaml, identity.yaml (+ config.yaml, generated)
logs/                   jarvis_YYYY-MM-DD.log, one file per day (gitignored)
```

Logging: everything at INFO+ (including httpx request lines) goes to the daily
file; the console shows WARNING+ only, so the chat surface stays clean. Each
startup writes a session-boundary line into the same day's file.

## Make targets

- `make clean` — remove Python/tool caches.
- `make clean-logs` — remove daily log files (keeps `logs/`).
- `make clean-config` — remove `config/config.yaml`; the wizard runs on next boot.
- `make clean-all` — all of the above: full reset to first-run.