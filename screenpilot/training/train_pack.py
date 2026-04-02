#!/usr/bin/env python3
"""CLI: Train a YOLO model and package it as an ApplicationPack.

Usage:
    python -m screenpilot.training.train_pack \\
        --dataset dataset/my_app/dataset.yaml \\
        --pack-name my_app \\
        --window-match "My App" "MyApp.exe" \\
        --epochs 50 \\
        --device mps
"""

from __future__ import annotations

import argparse
import logging
import sys

from screenpilot.training.trainer import PackTrainer, TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train YOLO model and package as ApplicationPack")
    parser.add_argument("--dataset", required=True, help="Path to dataset.yaml")
    parser.add_argument("--pack-name", required=True, help="Name for the ApplicationPack")
    parser.add_argument("--base-model", default="yolov8m.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps", help="Training device: mps, cuda, cpu")
    parser.add_argument("--output-dir", default="packs", help="Where to write the pack")
    parser.add_argument(
        "--window-match", nargs="+", default=[], help="Window name substrings for routing"
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just package")
    parser.add_argument(
        "--model-path", default="", help="Pre-trained model path (use with --skip-train)"
    )
    args = parser.parse_args(argv)

    config = TrainConfig(
        dataset_yaml=args.dataset,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    trainer = PackTrainer(config)

    if args.skip_train:
        if not args.model_path:
            logger.error("--model-path required when using --skip-train")
            sys.exit(1)
        from screenpilot.training.trainer import TrainResult

        result = TrainResult(best_model_path=args.model_path)
        logger.info("Skipping training, using model: %s", args.model_path)
    else:
        logger.info("Starting training...")
        result = trainer.train()
        logger.info("Training complete: %s", result.best_model_path)

    pack_dir = trainer.package_pack(
        train_result=result,
        pack_name=args.pack_name,
        output_dir=args.output_dir,
        window_match=args.window_match,
    )
    print(f"\nApplicationPack created: {pack_dir}")
    print(f"  Pack name:    {args.pack_name}")
    print(f"  Model:        {pack_dir}/model.pt")
    print(f"  Pack config:  {pack_dir}/pack.yaml")


if __name__ == "__main__":
    main()
