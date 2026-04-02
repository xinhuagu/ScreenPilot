# Gazefy

## Project Overview

Gazefy is a real-time screen monitoring and automation tool. It uses a custom-trained YOLO neural network to detect UI elements (buttons, menus, input fields, etc.) in a VDI-hosted Windows application from screen pixels alone, enabling LLM-driven precise software operation.

See [PRD.md](./PRD.md) for product requirements and [DESIGN.md](./DESIGN.md) for technical architecture.

## Language & Style

- Python 3.11+
- Use type hints on all function signatures
- Use dataclasses for data structures (not dicts)
- Format with ruff (`ruff format`), lint with ruff (`ruff check`)
- Tests with pytest
- Docstrings: one-line summary only, no verbose Google/Numpy style unless complex

## Project Structure

```
gazefy/          # Main package
  config.py           # GazefyConfig dataclass + YAML loading
  core/               # Orchestrator, event loop
  capture/            # Screen capture, window finder, change detection
  detection/          # YOLO inference, post-processing, screen classifier, element verifier
  tracker/            # UIMap, element tracker
  cursor/             # Cursor monitor
  actions/            # Action executor, coordinate transform
  llm/                # LLM interface, formatters, parsers
  knowledge/          # [Optional] Manual parser, semantic dict, workflow graph
  training/           # Data collector, trainer, augmentations, evaluator
  utils/              # geometry, timing
scripts/              # CLI scripts (benchmark, visualize, select_region)
models/               # Trained model files (.pt, .mlpackage)
dataset/              # Training data (images + labels)
configs/              # YAML configuration files
```

## Key Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run linter and formatter
ruff check gazefy/ && ruff format gazefy/

# Run tests
pytest tests/ -v

# Training
yolo detect train model=yolov8m.pt data=dataset/dataset.yaml imgsz=1024 epochs=50

# Export model
yolo export model=runs/detect/train/weights/best.pt format=coreml imgsz=1024

# Benchmark capture performance
python scripts/benchmark.py
```

## Tech Stack

- **Screen capture**: mss (fallback: ScreenCaptureKit via pyobjc)
- **Detection model**: YOLOv8 via Ultralytics
- **Inference**: CoreML on Apple Silicon (fallback: ONNX Runtime)
- **Action execution**: pyautogui
- **Window detection**: pyobjc-framework-Quartz
- **LLM**: Anthropic Claude API
- **Training**: Ultralytics CLI + albumentations for augmentation
- **Annotation**: Label Studio + GroundingDINO pre-labeling

## Important Conventions

- All coordinates use pixel space internally; convert to screen logical coords only at action execution boundary
- UIMap is the single source of truth for current screen state
- Model inference only runs when change detector fires (not every frame)
- Element IDs are stable across frames via IoU matching
- Never hardcode screen coordinates; always resolve through UIMap
