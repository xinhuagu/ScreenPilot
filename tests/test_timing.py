"""Tests for timing utilities."""

import time

from gazefy.utils.timing import FPSCounter, Timer


def test_timer_measures_elapsed():
    with Timer() as t:
        time.sleep(0.05)
    assert t.elapsed_ms >= 40  # Allow some slack


def test_fps_counter_empty():
    c = FPSCounter()
    assert c.fps == 0.0


def test_fps_counter_single_tick():
    c = FPSCounter()
    c.tick()
    assert c.fps == 0.0  # Need at least 2 ticks


def test_fps_counter_multiple_ticks():
    c = FPSCounter(window_size=10)
    for _ in range(5):
        c.tick()
        time.sleep(0.01)
    assert c.fps > 0
