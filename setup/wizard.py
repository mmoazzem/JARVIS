"""
First-run setup wizard.

Two layers, kept apart on purpose (M4 rule: no business logic reachable only
through the wizard's I/O):
  * PURE logic — `recommend_mode` and `build_config` take a SystemReport and return
    data. No printing, no prompting.
  * Presentation — `run_wizard` scans the host, prints a readable rundown, calls the
    pure functions, saves the config, and (local mode) makes Ollama ready and offers
    to pull the primary model with visible progress.

LOCAL-ONLY this phase: we recommend local vs cloud but only CONFIGURE local. No
secret/key collection happens here — cloud-key collection is a future phase.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel

from core.constants import LOGGER_ROOT
from setup import config as cfg
from setup.config import JarvisConfig
from setup.system_scan import SystemReport, scan_system

logger = logging.getLogger(LOGGER_ROOT)


class ModeRecommendation(BaseModel):
    """The wizard's local-vs-cloud advice, as data so presentation can render it."""

    mode: str  # "local" | "cloud"
    reason: str


# --- PURE logic (no I/O, no printing) ---------------------------------------


def recommend_mode(report: SystemReport, vram_floor_gb: float) -> ModeRecommendation:
    """Recommend local when there's enough VRAM, else cloud. Pure.

    The threshold (`local_vram_floor_gb`) comes from defaults.yaml — read at first
    run before config.yaml exists, which is why it lives in defaults, not the schema.
    """
    vram = report.gpu_vram_gb
    if vram is not None and vram >= vram_floor_gb:
        return ModeRecommendation(
            mode="local",
            reason=f"{vram:.0f} GB VRAM meets the {vram_floor_gb:.0f} GB floor for local models",
        )
    if vram is None:
        return ModeRecommendation(
            mode="cloud",
            reason="no GPU detected — local 14B models need a CUDA GPU",
        )
    return ModeRecommendation(
        mode="cloud",
        reason=f"{vram:.0f} GB VRAM is below the {vram_floor_gb:.0f} GB floor for local models",
    )


def build_config(report: SystemReport, defaults: JarvisConfig) -> JarvisConfig:
    """Build the JarvisConfig to persist. Pure.

    This phase configures LOCAL only — even when the recommendation is cloud we write
    a local config (cloud-mode setup, including key collection, is a future phase).
    `defaults` is the defaults.yaml-seeded config, so the model stack stays editable
    in defaults.yaml rather than hardcoded here.
    """
    return defaults.model_copy(update={"mode": "local", "provider": "ollama"})


# --- Presentation (thin wrapper over the pure logic above) ------------------


def _print_rundown(report: SystemReport) -> None:
    """Show the scan as a readable rundown. Presentation only."""
    print("\n  System scan")
    print("  " + "-" * 40)
    print(f"  OS         : {report.platform_str}")
    print(f"  Python     : {report.python_version}")
    print(f"  CPU cores  : {report.cpu_cores}")
    print(f"  RAM        : {report.ram_total_gb} GB")
    if report.gpu_name:
        print(f"  GPU        : {report.gpu_name} ({report.gpu_vram_gb} GB VRAM)")
    else:
        print("  GPU        : none detected")
    ollama_state = (
        "installed, running" if report.ollama_running
        else "installed, not running" if report.ollama_installed
        else "not installed"
    )
    print(f"  Ollama     : {ollama_state}")
    # None = couldn't query a dead daemon (unknown), distinct from [] = none pulled.
    if report.pulled_models is None:
        models_str = "unknown (Ollama not running)"
    else:
        models_str = ", ".join(report.pulled_models) or "none pulled"
    print(f"  Models     : {models_str}")
    print()


async def run_wizard() -> JarvisConfig:
    """Run the first-run wizard: scan, recommend, and BUILD the JarvisConfig in memory.

    It deliberately does NOT touch Ollama and does NOT write config.yaml. Readiness
    (daemon + model) is the shared boot ladder's job in main; config.yaml is committed
    there only after full success — so a failed first run leaves no config behind.
    """
    print("\nFirst run — setting Jarvis up.\n")

    defaults_cfg = cfg.load_defaults()
    floor = cfg.load_defaults_raw().get("local_vram_floor_gb", 10.0)

    report = await scan_system(defaults_cfg.ollama_base_url)
    _print_rundown(report)

    rec = recommend_mode(report, floor)
    print(f"  Recommended mode: {rec.mode} — {rec.reason}")
    if rec.mode == "cloud":
        # We surface the recommendation but only configure local in this phase.
        print("  (Cloud setup arrives in a later phase; configuring local for now.)")

    return build_config(report, defaults_cfg)
