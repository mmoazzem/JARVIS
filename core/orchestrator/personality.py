"""
System-prompt assembly — the three-layer persona stack.

Layer 1 is the identity, loaded LIVE from identity.yaml and used VERBATIM so its
carefully-tuned tone reaches the model exactly as authored. Layer 2 is small
runtime state the model may use but should not announce. Layer 3 is a reserved,
currently-empty profile slot (future long-term memory). The `/no_think` directive
is appended when thinking is disabled, to keep reasoning from eating the budget.
"""
from __future__ import annotations

from core.constants import NO_THINK_DIRECTIVE


def build_system_prompt(
    identity: dict,
    state: dict,
    profile: str = "",
    *,
    enable_thinking: bool = True,
) -> str:
    layers: list[str] = []

    # Layer 1 — identity, verbatim. This is where the persona enters the prompt.
    identity_text = (identity.get("identity") or "").strip()
    if identity_text:
        layers.append(identity_text)

    # Layer 2 — runtime state: known to the model, not to be volunteered.
    state_lines = [f"- {key}: {value}" for key, value in state.items() if value]
    if state_lines:
        layers.append("Current system state (for your awareness only):\n" + "\n".join(state_lines))

    # Layer 3 — reserved profile slot (future memory). Present but empty for now.
    if profile.strip():
        layers.append(profile.strip())

    prompt = "\n\n".join(layers)

    # /no_think reduces (not fully suppresses) reasoning — keeps room for content.
    if not enable_thinking:
        prompt = f"{prompt}\n\n{NO_THINK_DIRECTIVE}"

    return prompt
