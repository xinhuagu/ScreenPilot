#!/usr/bin/env python3
"""Benchmark screen capture FPS and change detection latency.

Usage:
    python scripts/benchmark.py                          # Capture full primary monitor
    python scripts/benchmark.py --window "Citrix"        # Auto-find window by name
    python scripts/benchmark.py --region 100,50,800,600  # Manual region (left,top,w,h)
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, str(__file__).rsplit("/scripts", 1)[0])

from gazefy.capture.change_detector import ChangeDetector
from gazefy.capture.screen_capture import ScreenCapture
from gazefy.config import CaptureRegion
from gazefy.utils.timing import FPSCounter, Timer


def resolve_region(args: argparse.Namespace) -> CaptureRegion:
    if args.window:
        from gazefy.capture.window_finder import find_window

        w = find_window(args.window)
        if w is None:
            print(f"Window '{args.window}' not found. Available windows:")
            from gazefy.capture.window_finder import print_windows

            print_windows()
            sys.exit(1)
        return w.region

    if args.region:
        parts = [int(x) for x in args.region.split(",")]
        return CaptureRegion(left=parts[0], top=parts[1], width=parts[2], height=parts[3])

    # Default: primary monitor center 800x600
    return CaptureRegion(left=100, top=100, width=800, height=600)


def benchmark_capture(region: CaptureRegion, duration: float = 5.0) -> None:
    """Benchmark raw capture FPS."""
    print(f"\n{'='*60}")
    print(f"CAPTURE BENCHMARK")
    print(f"Region: ({region.left}, {region.top}) {region.width}x{region.height}")
    print(f"Duration: {duration}s")
    print(f"{'='*60}")

    cap = ScreenCapture(region, target_fps=120)  # Uncapped to measure max throughput
    fps_counter = FPSCounter(window_size=200)
    latencies: list[float] = []

    start = time.monotonic()
    while time.monotonic() - start < duration:
        with Timer() as t:
            frame = cap.grab_once()
        latencies.append(t.elapsed_ms)
        fps_counter.tick()

    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    frame_shape = frame.image.shape

    print(f"\nResults:")
    print(f"  Frames captured:  {len(latencies)}")
    print(f"  Max FPS:          {fps_counter.fps:.1f}")
    print(f"  Frame shape:      {frame_shape} (H x W x C)")
    print(f"  Avg latency:      {avg_latency:.2f} ms")
    print(f"  P50 latency:      {p50:.2f} ms")
    print(f"  P95 latency:      {p95:.2f} ms")
    print(f"  P99 latency:      {p99:.2f} ms")

    target = 20
    status = "PASS" if fps_counter.fps >= target else "FAIL"
    print(f"\n  Target ≥{target} FPS: [{status}] (actual: {fps_counter.fps:.1f})")


def benchmark_change_detection(region: CaptureRegion, num_frames: int = 200) -> None:
    """Benchmark change detection latency."""
    print(f"\n{'='*60}")
    print(f"CHANGE DETECTION BENCHMARK")
    print(f"Frames: {num_frames}")
    print(f"{'='*60}")

    cap = ScreenCapture(region, target_fps=120)
    detector = ChangeDetector()
    latencies: list[float] = []
    change_count = 0

    for _ in range(num_frames):
        frame = cap.grab_once()
        with Timer() as t:
            result = detector.check(frame.image)
        latencies.append(t.elapsed_ms)
        if result.changed:
            change_count += 1

    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"\nResults:")
    print(f"  Frames analyzed:  {num_frames}")
    print(f"  Changes detected: {change_count} ({100*change_count/num_frames:.1f}%)")
    print(f"  Avg latency:      {avg:.2f} ms")
    print(f"  P50 latency:      {p50:.2f} ms")
    print(f"  P95 latency:      {p95:.2f} ms")
    print(f"  P99 latency:      {p99:.2f} ms")

    target = 5.0
    status = "PASS" if p95 < target else "FAIL"
    print(f"\n  Target P95 < {target}ms: [{status}] (actual: {p95:.2f}ms)")


def benchmark_threaded_capture(region: CaptureRegion, duration: float = 5.0) -> None:
    """Benchmark threaded capture with actual FPS delivery."""
    print(f"\n{'='*60}")
    print(f"THREADED CAPTURE BENCHMARK (target: 20 FPS)")
    print(f"Duration: {duration}s")
    print(f"{'='*60}")

    cap = ScreenCapture(region, target_fps=20, buffer_size=200)
    cap.start()
    time.sleep(0.5)  # Let buffer warm up

    frame_count = 0
    last_frame_num = -1
    start = time.monotonic()

    while time.monotonic() - start < duration:
        f = cap.get_latest_frame()
        if f and f.frame_number != last_frame_num:
            frame_count += 1
            last_frame_num = f.frame_number
        time.sleep(0.001)

    cap.stop()
    actual_fps = frame_count / duration

    print(f"\nResults:")
    print(f"  Unique frames received: {frame_count}")
    print(f"  Effective FPS:          {actual_fps:.1f}")

    status = "PASS" if actual_fps >= 18 else "FAIL"
    print(f"\n  Target ≥18 FPS delivered: [{status}] (actual: {actual_fps:.1f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gazefy M1 Benchmark")
    parser.add_argument("--window", type=str, help="Find window by name substring")
    parser.add_argument("--region", type=str, help="Manual region: left,top,width,height")
    parser.add_argument("--list-windows", action="store_true", help="List all visible windows")
    parser.add_argument("--duration", type=float, default=5.0, help="Benchmark duration in seconds")
    args = parser.parse_args()

    if args.list_windows:
        from gazefy.capture.window_finder import print_windows

        print("Visible windows:")
        print_windows()
        return

    region = resolve_region(args)
    benchmark_capture(region, duration=args.duration)
    benchmark_change_detection(region, num_frames=200)
    benchmark_threaded_capture(region, duration=args.duration)

    print(f"\n{'='*60}")
    print("M1 BENCHMARK COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
