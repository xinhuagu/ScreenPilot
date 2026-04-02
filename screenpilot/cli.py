"""ScreenPilot CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ScreenPilot: AI-driven screen automation")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list-windows", help="List visible windows")
    sub.add_parser("benchmark", help="Run capture + change detection benchmark")

    collect_p = sub.add_parser("collect", help="Collect training screenshots")
    collect_p.add_argument("--window", type=str, help="Window name to capture")
    collect_p.add_argument("--interval-ms", type=int, default=500)

    train_p = sub.add_parser("train", help="Train model and package as ApplicationPack")
    train_p.add_argument("--dataset", required=True)
    train_p.add_argument("--pack-name", required=True)

    args = parser.parse_args(argv)

    if args.command == "list-windows":
        from screenpilot.capture.window_finder import print_windows

        print_windows()

    elif args.command == "benchmark":
        from scripts.benchmark import main as bench_main

        bench_main()

    elif args.command == "train":
        from screenpilot.training.train_pack import main as train_main

        train_main(
            [
                "--dataset",
                args.dataset,
                "--pack-name",
                args.pack_name,
            ]
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
