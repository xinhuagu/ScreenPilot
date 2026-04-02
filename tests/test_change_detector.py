"""Tests for change detector (no screen capture needed — uses synthetic frames)."""

import numpy as np

from gazefy.capture.change_detector import ChangeDetector, ChangeLevel


def _make_frame(width: int = 200, height: int = 150, color: tuple = (128, 128, 128, 255)) -> np.ndarray:
    """Create a solid BGRA frame."""
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[:, :] = color
    return frame


def test_first_frame_is_major():
    det = ChangeDetector()
    result = det.check(_make_frame())
    assert result.changed is True
    assert result.change_level == ChangeLevel.MAJOR


def test_identical_frames_no_change():
    det = ChangeDetector()
    frame = _make_frame()
    det.check(frame)  # first frame
    result = det.check(frame.copy())  # identical
    assert result.changed is False
    assert result.change_level == ChangeLevel.NONE


def test_slight_noise_below_threshold():
    det = ChangeDetector(similarity_threshold=0.98)
    frame1 = _make_frame()
    det.check(frame1)
    # Add very slight noise (simulates VDI compression)
    frame2 = frame1.copy()
    noise = np.random.randint(-2, 3, size=frame2.shape, dtype=np.int16)
    frame2 = np.clip(frame2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    result = det.check(frame2)
    assert result.changed is False


def test_major_change_detected():
    det = ChangeDetector()
    frame1 = _make_frame(color=(50, 50, 50, 255))
    # Create a frame with random content — guaranteed different hash and pixel values
    rng = np.random.RandomState(42)
    frame2 = rng.randint(0, 256, size=(150, 200, 4), dtype=np.uint8)
    frame2[:, :, 3] = 255
    det.check(frame1)
    result = det.check(frame2)
    assert result.changed is True
    assert result.change_level == ChangeLevel.MAJOR


def test_moderate_change_detected():
    det = ChangeDetector(similarity_threshold=0.98, major_threshold=0.85)
    frame1 = _make_frame(color=(100, 100, 100, 255))
    det.check(frame1)
    # Change a portion of the frame
    frame2 = frame1.copy()
    frame2[50:100, 50:150] = (200, 200, 200, 255)
    result = det.check(frame2)
    assert result.changed is True


def test_reset_clears_state():
    det = ChangeDetector()
    det.check(_make_frame())
    det.reset()
    result = det.check(_make_frame())
    assert result.changed is True  # After reset, first frame is always MAJOR
    assert result.change_level == ChangeLevel.MAJOR
