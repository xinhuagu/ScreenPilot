"""Gazefy CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gazefy: AI-driven screen automation")
    sub = parser.add_subparsers(dest="command")

    # --- list-windows ---
    sub.add_parser("list-windows", help="List visible macOS windows")

    # --- benchmark ---
    sub.add_parser("benchmark", help="Run capture + change detection benchmark")

    # --- monitor ---
    monitor_p = sub.add_parser("monitor", help="Real-time cursor-to-element monitoring")
    monitor_p.add_argument("--window", type=str, help="Window name to monitor")
    monitor_p.add_argument("--region", type=str, help="Manual region: left,top,width,height")
    monitor_p.add_argument("--pack", type=str, default="", help="Force a specific pack")
    monitor_p.add_argument("--packs-dir", type=str, default="packs")
    monitor_p.add_argument("--retina-scale", type=float, default=2.0)

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

    if args.command == "list-windows":
        from gazefy.capture.window_finder import print_windows

        print("Visible windows:")
        print_windows()

    elif args.command == "benchmark":
        # Import inline — needs platform deps
        import importlib

        bench = importlib.import_module("scripts.benchmark")
        bench.main()

    elif args.command == "monitor":
        from gazefy.core.monitor import run_monitor

        region = _resolve_region(args)
        run_monitor(
            region=region,
            pack_name=args.pack,
            packs_dir=args.packs_dir,
            retina_scale=args.retina_scale,
        )

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
