# CLAUDE.md — Jarvis

A local-first personal AI assistant. Qwen3 14B is the brain; the system is built
to grow toward memory, voice, and a frontend without rewrites.

## Runtime

- Dev on Windows, run on Ubuntu 24.04 / WSL2.
- RTX 5080 (16 GB VRAM), 64 GB RAM, Python 3.11+.
- Models via Ollama. Primary: `qwen3:14b`. `deepseek-r1:14b` is reserved in
  config for a future specialist — not loaded today.

## Architecture

- One model answers everything (Qwen-only). An "agent" is a reusable unit:
  model + role-prompt + tools + run-loop. The orchestrator is the first agent.
  This seam lets future multi-agent / specialist routing layer on cleanly.
- `BaseModel` interface abstracts the model so local/cloud are interchangeable.
- Identity (persona) lives in `config/identity.yaml`, loaded live every boot —
  never copied into runtime config.

## Code principles

1. Generalize by protocol, not per model. One class serves all Ollama models;
   `model_id` is config. Never hardcode model names in core logic.
2. Separate logic from presentation. Core functions return structured data;
   interfaces (CLI now, frontend later) decide how to display. No logic reachable
   only through terminal I/O.
3. One responsibility per file. Small files over big ones.
4. Async throughout. Pydantic for structured data.
5. Comment WHY, not what.

## Config / constants / identity (keep these separate)

- `config/config.yaml` — how THIS install is set up (gitignored, wizard-written).
  User-tunable runtime values live here: model, token budgets, temperature,
  keep_alive, timeouts.
- `config/defaults.yaml` — committed seed values the wizard uses to build config.
- `core/constants.py` — developer-fixed literals, sectioned. Not user-tunable.
- `config/identity.yaml` — the persona (committed, loaded live).
- Rule: if a user would tune it → config. If it's a fixed code detail → constant.
- No magic numbers/strings in the live path. No self-tests inside source files;
  tests live in `tests/`.

## Known gotchas (do not rediscover these)

- Ollama returns reasoning in a separate `reasoning` field, NOT `<think>` tags.
  Stream/parse `content` for the answer; keep `reasoning` separate. No `<think>`
  parsing anywhere.
- Reasoning shares the `max_tokens` budget. On hard questions reasoning can
  consume the whole budget and yield zero `content` → empty output. Budget must
  leave room for content; detect zero-content-with-reasoning and recover, never
  fail silently.
- `/no_think` reduces but does not fully suppress reasoning on hard prompts.
- Two 14B models cannot co-reside in 16 GB. Any specialist requires an
  unload/load swap (verified to work) — not concurrent residence.
- Multi-line paste must arrive as one message; the prompt must not print
  mid-paste.

## Logging

- One file per day: `logs/jarvis_YYYY-MM-DD.log`, append mode (don't create a new
  file if today's exists). Session-boundary line at each startup.
- File handler captures everything (INFO+, incl. httpx). Console = WARNING+ only,
  so the chat surface stays clean. `logs/` gitignored.

## Definition of done (every change)

You run inside WSL with Ollama available — so you CAN and MUST run live tests,
not just mocked ones. A change is not done until:
- Live smoke test passes, no exceptions: `say hello` (fast, terse) AND the zebra
  puzzle (hard, streams a real answer) both produce visible output via the actual
  app, not a mock.
- You verified it by running it yourself, not by assuming mocked tests imply it
  works. Mocked tests confirm logic; the live run confirms reality. When they
  disagree, the live run wins — fix the code, not the test.
- Never declare done on green mocks alone. Run the real thing first.