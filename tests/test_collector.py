"""Tests for DataCollector (no screen capture — uses synthetic frames)."""

import json
import tempfile

import numpy as np

from screenpilot.training.collector import ActionEvent, CollectorConfig, DataCollector


def test_collector_session_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        config = CollectorConfig(output_dir=tmp, pack_name="test_app")
        collector = DataCollector(config)

        session_dir = collector.start_session("test_session")
        assert session_dir.exists()
        assert (session_dir / "images").is_dir()

        # Save a synthetic frame
        frame = np.zeros((100, 200, 4), dtype=np.uint8)
        frame[:, :] = (128, 128, 128, 255)
        img_path = collector.save_frame(frame, timestamp=1000.0)
        assert img_path.exists()
        assert img_path.suffix == ".png"
        assert collector.frame_count == 1

        # Log an action
        collector.log_action(
            ActionEvent(timestamp=1000.5, action_type="click", x=100, y=50)
        )

        # Finish session
        summary = collector.finish_session(labels=["button", "input_field"])
        assert summary["frames"] == 1
        assert summary["actions"] == 1
        assert summary["pack"] == "test_app"

        # Check action log
        log_path = session_dir / "actions.jsonl"
        assert log_path.exists()
        with open(log_path) as f:
            events = [json.loads(line) for line in f]
        assert len(events) == 1
        assert events[0]["action_type"] == "click"

        # Check dataset.yaml
        assert (session_dir / "dataset.yaml").exists()


def test_collector_multiple_frames():
    with tempfile.TemporaryDirectory() as tmp:
        config = CollectorConfig(output_dir=tmp)
        collector = DataCollector(config)
        collector.start_session()

        for i in range(5):
            frame = np.full((50, 50, 4), i * 50, dtype=np.uint8)
            collector.save_frame(frame)

        assert collector.frame_count == 5
        collector.finish_session()
