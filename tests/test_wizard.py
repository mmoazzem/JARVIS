"""Wizard pure-logic invariants (setup/wizard.py) — no I/O, no printing.

`recommend_mode` and `build_config` are the deliberately-pure layer of the wizard;
these tests drive them with hand-built SystemReports and never touch the host.
"""
from setup.config import JarvisConfig
from setup.system_scan import SystemReport
from setup.wizard import build_config, recommend_mode

VRAM_FLOOR_GB = 10.0


def _report(vram_gb=None, gpu_name=None) -> SystemReport:
    return SystemReport(
        os_system="Linux",
        os_release="6.0",
        platform_str="Linux-test",
        python_version="3.14.0",
        gpu_name=gpu_name,
        gpu_vram_gb=vram_gb,
    )


def test_recommends_local_above_the_vram_floor():
    rec = recommend_mode(_report(vram_gb=16.0, gpu_name="RTX 5080"), VRAM_FLOOR_GB)
    assert rec.mode == "local"


def test_recommends_local_exactly_at_the_vram_floor():
    # The floor is inclusive: exactly 10.0 GB qualifies for local.
    rec = recommend_mode(_report(vram_gb=10.0, gpu_name="GPU"), VRAM_FLOOR_GB)
    assert rec.mode == "local"


def test_recommends_cloud_just_below_the_vram_floor():
    rec = recommend_mode(_report(vram_gb=9.9, gpu_name="GPU"), VRAM_FLOOR_GB)
    assert rec.mode == "cloud"


def test_recommends_cloud_when_no_gpu_detected():
    rec = recommend_mode(_report(vram_gb=None), VRAM_FLOOR_GB)
    assert rec.mode == "cloud"
    assert "no GPU" in rec.reason


def test_recommendation_reason_is_populated():
    for report in (_report(16.0, "GPU"), _report(4.0, "GPU"), _report(None)):
        assert recommend_mode(report, VRAM_FLOOR_GB).reason.strip()


def test_build_config_configures_local_with_defaults_model_stack():
    defaults = JarvisConfig(primary_model="qwen3:14b", reasoning_model="deepseek-r1:14b")

    config = build_config(_report(vram_gb=16.0, gpu_name="RTX 5080"), defaults)

    assert config.mode == "local"
    assert config.provider == "ollama"
    # The model stack comes from defaults.yaml, never hardcoded in build_config.
    assert config.primary_model == defaults.primary_model
    assert config.reasoning_model == defaults.reasoning_model


def test_build_config_is_pure_and_does_not_mutate_defaults():
    defaults = JarvisConfig(mode="cloud")
    before = defaults.model_dump()

    config = build_config(_report(vram_gb=16.0), defaults)

    assert config is not defaults
    assert defaults.model_dump() == before
