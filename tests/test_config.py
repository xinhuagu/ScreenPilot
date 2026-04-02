"""Tests for configuration loading."""

import tempfile
from pathlib import Path

from screenpilot.config import CaptureRegion, ScreenPilotConfig


def test_default_config():
    cfg = ScreenPilotConfig()
    assert cfg.capture_fps == 20
    assert cfg.retina_scale == 2.0
    assert cfg.region.width == 1920
    assert cfg.dry_run is False


def test_config_from_yaml():
    yaml_content = """
region:
  top: 50
  left: 100
  width: 800
  height: 600
capture_fps: 30
dry_run: true
mode: debug
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = ScreenPilotConfig.from_yaml(f.name)

    assert cfg.region.top == 50
    assert cfg.region.left == 100
    assert cfg.region.width == 800
    assert cfg.capture_fps == 30
    assert cfg.dry_run is True
    assert cfg.mode == "debug"
    # Defaults preserved
    assert cfg.similarity_threshold == 0.98


def test_capture_region_defaults():
    r = CaptureRegion()
    assert r.top == 0
    assert r.left == 0
    assert r.width == 1920
    assert r.height == 1080
