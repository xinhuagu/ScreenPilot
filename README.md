# Gazefy

[![CI](https://github.com/xinhuagu/gazefy/actions/workflows/ci.yml/badge.svg)](https://github.com/xinhuagu/gazefy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

AI-driven screen automation for any application — desktop, VDI, remote, or legacy. Train a custom YOLO neural network to perceive your application's UI from screen pixels alone, then let an LLM operate it precisely. No accessibility API, no source code, no plugins required.

## Why Gazefy?

Many applications have no automation API — enterprise software, legacy systems, VDI-hosted apps, proprietary tools. General-purpose screen agents (Anthropic Computer Use, OmniParser) work across any app but lack precision for specific software. Gazefy flips this: **train once for your application, operate with high accuracy forever.**

Works with:
- Desktop applications (Windows, macOS)
- VDI clients (Citrix, VMware Horizon, RDP)
- Browser-based applications
- Legacy systems and terminal emulators
- Any software you can see on screen

```
Train a model on YOUR app → Pack it → Gazefy operates it precisely
```

## How It Works

```
Screen Capture (20 FPS)
    ↓
Change Detector (skip unchanged frames)
    ↓
YOLO Model (custom-trained for your app)
    ↓
UIMap (structured element map: buttons, menus, inputs, etc.)
    ↓
    ├── Cursor Monitor → "cursor is on [button] Save"
    ├── LLM Reasoning  → decides what to click/type next
    └── Action Executor → precise mouse/keyboard via pyautogui
```

## Key Features

- **Application Packs** — hot-swappable per-app model + config artifacts. Train once, deploy as a pack.
- **Real-time cursor awareness** — know which UI element the mouse is hovering over at 60Hz
- **LLM-driven operation** — describe a task in natural language, Gazefy executes it
- **VDI-optimized** — handles compression artifacts, network latency, and pixel-only environments
- **Training pipeline included** — collect screenshots, annotate, train, package, deploy
- **Self-improving pipeline** — verification loop as reward signal, VLM→YOLO distillation, drift detection, and active learning continuously refine the model without human annotation

## Quick Start

### Install

```bash
# Core (CI-safe, cross-platform)
pip install -e .

# Full local development on macOS
pip install -e ".[all]"
```

### Collect Training Data

```bash
# List available windows
gazefy list-windows

# Capture screenshots from your VDI app
gazefy collect --window "Citrix" --pack-name my_erp --interval-ms 500
```

### Annotate & Train

```bash
# After annotating in Label Studio, split into train/val
gazefy prep datasets/my_erp/session_xxx --split 0.8

# Train and package as ApplicationPack
gazefy train \
  --dataset datasets/my_erp/session_xxx/dataset.yaml \
  --pack-name my_erp \
  --window-match "Citrix" "My ERP" \
  --device mps
```

### Run

```bash
# Benchmark capture performance
python scripts/benchmark.py --window "Citrix"

# Monitor mode (coming in M5)
# gazefy monitor --window "Citrix"
```

## Architecture

```
gazefy/
├── core/           Orchestrator, ApplicationPack, AppRouter, ModelRegistry
├── capture/        Screen capture, window finder, change detection
├── detection/      YOLO inference → list[Detection]
├── tracker/        UIMap maintenance, element tracking with stable IDs
├── cursor/         Real-time cursor-to-element resolution
├── actions/        Action execution, coordinate transform
├── llm/            LLM integration (formatters, parsers, provider interface)
├── training/       Data collection, dataset prep, model training, pack packaging
├── knowledge/      [V2] Optional manual-based enrichment
└── utils/          Geometry primitives, timing utilities
```

See [DESIGN.md](DESIGN.md) for full technical architecture and [PRD.md](PRD.md) for product requirements.

## ApplicationPack

Each application gets its own pack — a directory containing everything needed to detect and operate it:

```
packs/my_erp/
├── pack.yaml       # name, labels, window matching, thresholds
├── model.pt        # trained YOLO weights
└── workflows/      # optional workflow definitions
```

Packs are hot-swappable. The runtime loads the right pack based on which VDI window is active.

## Performance (M1 Benchmark)

Measured on Apple Silicon (M-series), 1728x1084 capture region:

| Metric | Result | Target |
|--------|--------|--------|
| Capture FPS | 39.9 | >= 20 |
| Change Detection P95 | 0.31ms | < 5ms |
| Threaded Delivery | 18.9 FPS | >= 18 |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (47 tests)
pytest tests/ -v

# Lint + format
ruff check gazefy/
ruff format gazefy/
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `gazefy collector` | Open GUI data collection window |
| `gazefy recorder` | Open floating semantic recorder widget |
| `gazefy collect` | CLI screenshot collection |
| `gazefy prep` | Split annotated dataset into train/val |
| `gazefy train` | Train model and package as ApplicationPack |
| `gazefy learn` | Click-to-label mode (VLM builds icon dictionary) |
| `gazefy record-video` | Record screen as MP4 + mouse events |
| `gazefy annotate-video` | Annotate video session (hybrid or VLM-only) |
| `gazefy monitor` | Real-time cursor-to-element monitoring |
| `gazefy replay` | Replay a recorded cursor trajectory |
| `gazefy benchmark` | Capture + change detection benchmark |
| `gazefy list-windows` | List visible windows |

## Self-Improving Pipeline

A key challenge with per-app YOLO models is that they degrade as UI themes, layouts, or VDI compression settings change. Gazefy addresses this with a self-improving loop that requires **no human annotation** after the initial pack is trained:

```
Live operation
    ↓
ActionExecutor verifies each click (screen_changed = True/False)
    ↓
Confirmed clicks → positive training samples
Failed clicks    → hard negative examples
    ↓
HybridAnnotator runs on fresh screenshots periodically
(GroundingDINO bboxes → EasyOCR text → Claude Vision for icons)
    ↓
Pseudo-labels → YOLO fine-tune buffer (VLM→YOLO distillation)
    ↓
Drift detector monitors mean YOLO confidence
    ↓ (drops below threshold)
Auto re-annotation → fine-tune → updated pack deployed
```

The five directions driving this:

| Direction | Mechanism | Status |
|-----------|-----------|--------|
| **Verification as reward** | `screen_changed` flag from executor seeds training buffer | Built (M4/M5) |
| **VLM→YOLO distillation** | HybridAnnotator pseudo-labels → YOLO training without human annotation | Built (M5); pipeline integration M8 |
| **Drift detection** | Monitor mean detection confidence; trigger re-annotation on drop | Planned M8 |
| **Active learning** | Low-confidence detections batched to Claude Vision → fine-tune buffer | Planned M8 |
| **LoRA app adapters** | Frozen universal base model + tiny per-app LoRA adapter | V2 research |

This is the "learn from doing" principle: the system gets better the more it operates, using its own action outcomes as the training signal.

## Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1: Capture + Change Detection | ✅ Done | Screen capture, frame diffing, benchmark |
| M2: Pack Contract + Training Pipeline | ✅ Done | ApplicationPack, collect/prep/train CLI |
| M3: UIMap + Cursor Monitor | ✅ Done | IoU element tracking, 60 Hz cursor resolution |
| M4: Action Execution + LLM | ✅ Done | pyautogui executor, Anthropic LLM interface |
| M5: Recording + Annotation | ✅ Done | Semantic recorder, video pipeline, hybrid annotator |
| M6: End-to-end Task Execution | 🔄 In progress | LLM→UIMap→Action orchestration loop |
| M7: Hardening | Planned | Error recovery, multi-provider LLM, regression suite |
| M8: Self-Improving Pipeline | Planned | Drift detection, VLM distillation, active learning loop |
| M9: LoRA Adapters (V2) | Research | Universal base model + per-app LoRA fine-tuning |

## License

[Apache License 2.0](LICENSE)
