"""Gazefy CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gazefy: AI-driven screen automation")
    sub = parser.add_subparsers(dest="command")

    # --- collector-ui ---
    sub.add_parser("collector", help="Open the Gazefy Collector UI")
    sub.add_parser("recorder", help="Open the floating Recorder widget")

    # --- list-windows ---
    sub.add_parser("list-windows", help="List visible macOS windows")

    # --- benchmark ---
    bench_p = sub.add_parser("benchmark", help="Run capture + change detection benchmark")
    bench_p.add_argument("--window", type=str, help="Window name to benchmark")
    bench_p.add_argument("--region", type=str, help="Manual region: left,top,width,height")
    bench_p.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")

    # --- monitor ---
    monitor_p = sub.add_parser("monitor", help="Real-time cursor-to-element monitoring")
    monitor_p.add_argument("--window", type=str, help="Window name to monitor")
    monitor_p.add_argument("--region", type=str, help="Manual region: left,top,width,height")
    monitor_p.add_argument("--pack", type=str, default="", help="Force a specific pack")
    monitor_p.add_argument("--packs-dir", type=str, default="packs")
    monitor_p.add_argument("--retina-scale", type=float, default=2.0)
    monitor_p.add_argument("--record", action="store_true", help="Record cursor trajectory")
    monitor_p.add_argument("--record-dir", type=str, default="recordings")

    # --- replay ---
    replay_p = sub.add_parser("replay", help="Replay a recorded cursor trajectory")
    replay_p.add_argument("recording", help="Path to .jsonl recording file")
    replay_p.add_argument("--speed", type=float, default=1.0, help="Playback speed (2.0 = 2x)")

    # --- record-video ---
    rv_p = sub.add_parser(
        "record-video", help="Record screen as video + click events (no YOLO needed)"
    )
    rv_p.add_argument("--fps", type=int, default=10, help="Recording frame rate")
    rv_p.add_argument("--monitor", type=int, default=1, help="Monitor index (1=primary)")
    rv_p.add_argument("--output-dir", type=str, default="recordings")

    # --- annotate-video ---
    av_p = sub.add_parser(
        "annotate-video",
        help="Annotate a video session: detector finds bboxes, OCR reads text, VLM labels icons",
    )
    av_p.add_argument(
        "session_dir",
        help="Path to session directory (contains video.mp4 + events.jsonl)",
    )
    av_p.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Also annotate every N seconds in addition to click frames (0 = clicks only)",
    )
    av_p.add_argument(
        "--detector",
        choices=["grounding", "none"],
        default="grounding",
        help=(
            "grounding: GroundingDINO (precise bboxes) + EasyOCR + VLM for icons [default]. "
            "none: send full frame to VLM (no local detector required)."
        ),
    )
    av_p.add_argument(
        "--pack",
        type=str,
        default="",
        help="Use a trained YOLO pack instead of GroundingDINO (faster, higher precision)",
    )
    av_p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for GroundingDINO: cpu | mps | cuda",
    )

    # --- convert-annotations ---
    ca_p = sub.add_parser(
        "convert-annotations",
        help=(
            "Convert annotations.jsonl + video.mp4 → YOLO training dataset "
            "(images/ + labels/ + dataset.yaml)"
        ),
    )
    ca_p.add_argument(
        "session_dir",
        help="Session directory containing video.mp4 + annotations.jsonl",
    )
    ca_p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory (default: session_dir/yolo_dataset)",
    )
    ca_p.add_argument(
        "--classes",
        nargs="+",
        default=[],
        help="Custom class list (overrides default 16-class taxonomy)",
    )
    ca_p.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep elements labelled 'unknown' / 'unknown icon' (excluded by default)",
    )

    # --- learn ---
    learn_p = sub.add_parser("learn", help="Click UI elements, VLM identifies them")
    learn_p.add_argument("--window", type=str, help="Window name")
    learn_p.add_argument("--region", type=str, help="Manual region: left,top,width,height")
    learn_p.add_argument("--pack", type=str, required=True, help="Pack name (must have model)")
    learn_p.add_argument("--packs-dir", type=str, default="packs")

    # --- collect ---
    collect_p = sub.add_parser("collect", help="Collect training screenshots")
    collect_p.add_argument("--window", type=str, help="Window name to capture")
    collect_p.add_argument("--region", type=str, help="Manual region: left,top,width,height")
    collect_p.add_argument("--pack-name", type=str, default="default", help="Pack name")
    collect_p.add_argument("--output-dir", type=str, default="datasets")
    collect_p.add_argument("--interval-ms", type=int, default=500)
    collect_p.add_argument(
        "--duration", type=float, default=0, help="Stop after N seconds (0=unlimited)"
    )
    collect_p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames")

    # --- prep ---
    prep_p = sub.add_parser("prep", help="Split dataset into train/val after annotation")
    prep_p.add_argument("session_dir", help="Path to session directory")
    prep_p.add_argument("--split", type=float, default=0.8)

    # --- train ---
    train_p = sub.add_parser("train", help="Train model and package as ApplicationPack")
    train_p.add_argument("--dataset", required=True, help="Path to dataset.yaml")
    train_p.add_argument("--pack-name", required=True)
    train_p.add_argument("--base-model", default="yolov8m.pt")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--imgsz", type=int, default=1024)
    train_p.add_argument("--batch", type=int, default=8)
    train_p.add_argument("--device", default="mps")
    train_p.add_argument("--output-dir", default="packs")
    train_p.add_argument("--window-match", nargs="+", default=[])
    train_p.add_argument("--skip-train", action="store_true")
    train_p.add_argument("--model-path", default="")

    args = parser.parse_args(argv)

    if args.command == "collector":
        from gazefy.collector_ui.main_window import main as ui_main

        ui_main()

    elif args.command == "recorder":
        from gazefy.collector_ui.recorder_widget import main as rec_main

        rec_main()

    elif args.command == "list-windows":
        from gazefy.capture.window_finder import print_windows

        print("Visible windows:")
        print_windows()

    elif args.command == "benchmark":
        import importlib.util
        from pathlib import Path

        region = _resolve_region(args)
        # Load benchmark module from scripts/ regardless of cwd
        script = Path(__file__).resolve().parent.parent / "scripts" / "benchmark.py"
        spec = importlib.util.spec_from_file_location("benchmark", script)
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)

        bench.benchmark_capture(region, duration=args.duration)
        bench.benchmark_change_detection(region, num_frames=200)
        bench.benchmark_threaded_capture(region, duration=args.duration)
        print(f"\n{'=' * 60}\nBENCHMARK COMPLETE\n{'=' * 60}")

    elif args.command == "monitor":
        from gazefy.core.monitor import run_monitor

        region = _resolve_region(args)
        run_monitor(
            region=region,
            pack_name=args.pack,
            packs_dir=args.packs_dir,
            retina_scale=args.retina_scale,
            record=args.record,
            record_dir=args.record_dir,
        )

    elif args.command == "record-video":
        import datetime

        from gazefy.core.video_recorder import VideoRecorder

        rec_dir = Path(args.output_dir)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = rec_dir / f"session_{ts}"
        recorder = VideoRecorder(fps=args.fps, monitor_index=args.monitor)

        click_count = 0

        def on_click(ev: dict) -> None:
            nonlocal click_count
            click_count += 1
            print(f"  CLICK {ev['click']} at ({ev['x']}, {ev['y']})  [total: {click_count}]")

        recorder.start(session_dir, on_click=on_click)
        print(f"Recording to {session_dir}/")
        print(f"  FPS: {args.fps}   Monitor: {args.monitor}")
        print("  Press Ctrl+C to stop.\n")
        try:
            while True:
                import time

                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        recorder.stop()
        print(f"\nSaved: {session_dir}  ({click_count} clicks)")
        print(f"Annotate with: gazefy annotate-video {session_dir}")

    elif args.command == "annotate-video":
        session_dir = Path(args.session_dir)

        def on_progress(current: int, total: int, desc: str) -> None:
            print(f"  [{current}/{total}]  {desc}")

        if args.detector == "grounding":
            from gazefy.core.hybrid_annotator import HybridAnnotator

            pack_dir = Path("packs") / args.pack if args.pack else None
            annotator = HybridAnnotator(
                sample_interval=args.interval,
                grounding_device=args.device,
                pack_dir=pack_dir,
            )
            mode_desc = (
                f"YOLO ({args.pack}) + EasyOCR + VLM (icons)"
                if args.pack
                else f"GroundingDINO + EasyOCR + VLM (icons)  device={args.device}"
            )
        else:
            from gazefy.core.video_annotator import VideoAnnotator

            annotator = VideoAnnotator(sample_interval=args.interval)
            mode_desc = "full-frame VLM only (no local detector)"

        print(f"Annotating {session_dir}/")
        print(f"  Mode:     {mode_desc}")
        print(f"  Interval: {args.interval}s  (0 = clicks only)\n")

        annotations = annotator.annotate_session(session_dir, on_progress=on_progress)
        total_el = sum(len(a.elements) for a in annotations)
        print(f"\nDone: {len(annotations)} frames annotated, {total_el} elements total")
        print(f"  → {session_dir}/annotations.jsonl")
        print(f"\nNext: gazefy convert-annotations {session_dir}")

    elif args.command == "convert-annotations":
        from pathlib import Path

        from gazefy.training.annotation_converter import AnnotationConverter

        session_dir = Path(args.session_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None
        class_names = args.classes if args.classes else None

        converter = AnnotationConverter(skip_unknown=not args.keep_unknown)
        print(f"Converting annotations: {session_dir}/")
        if output_dir:
            print(f"  Output:  {output_dir}/")
        if class_names:
            print(f"  Classes: {class_names}")
        print()

        result = converter.convert_session(
            session_dir, output_dir=output_dir, class_names=class_names
        )
        print("Done:")
        print(f"  Images:   {result.n_images}")
        print(f"  Labels:   {result.n_labels}")
        print(f"  Elements: {result.n_elements}")
        if result.n_skipped:
            print(f"  Skipped:  {result.n_skipped} frames (no valid elements)")
        print(f"  Dataset:  {result.dataset_yaml}")
        print("\nNext steps:")
        print(f"  gazefy prep {result.output_dir}")
        print(f"  gazefy train --dataset {result.dataset_yaml} --pack-name <name>")

    elif args.command == "learn":
        from gazefy.core.learner import run_learn

        region = _resolve_region(args)
        run_learn(
            region=region,
            pack_name=args.pack,
            packs_dir=args.packs_dir,
        )

    elif args.command == "replay":
        from gazefy.core.monitor import run_replay

        run_replay(args.recording, speed=args.speed)

    elif args.command == "collect":
        from gazefy.training.collector import run_collect

        region = _resolve_region(args)
        run_collect(
            region=region,
            pack_name=args.pack_name,
            output_dir=args.output_dir,
            interval_ms=args.interval_ms,
            duration_s=args.duration,
            max_frames=args.max_frames,
        )

    elif args.command == "prep":
        from gazefy.training.dataset_prep import main as prep_main

        prep_main([args.session_dir, "--split", str(args.split)])

    elif args.command == "train":
        from gazefy.training.train_pack import main as train_main

        train_argv = [
            "--dataset",
            args.dataset,
            "--pack-name",
            args.pack_name,
            "--base-model",
            args.base_model,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
            "--output-dir",
            args.output_dir,
        ]
        if args.window_match:
            train_argv += ["--window-match"] + args.window_match
        if args.skip_train:
            train_argv.append("--skip-train")
        if args.model_path:
            train_argv += ["--model-path", args.model_path]
        train_main(train_argv)

    else:
        parser.print_help()
        sys.exit(1)


def _resolve_region(args: argparse.Namespace):
    """Resolve capture region from --window or --region flags."""
    from gazefy.config import CaptureRegion

    if args.window:
        from gazefy.capture.window_finder import find_window, print_windows

        w = find_window(args.window)
        if w is None:
            print(f"Window '{args.window}' not found. Available:")
            print_windows()
            sys.exit(1)
        return w.region

    if args.region:
        parts = [int(x) for x in args.region.split(",")]
        return CaptureRegion(left=parts[0], top=parts[1], width=parts[2], height=parts[3])

    # Default: primary screen 800x600
    return CaptureRegion(left=100, top=100, width=800, height=600)


if __name__ == "__main__":
    main()
