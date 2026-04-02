# Gazefy

[![CI](https://github.com/xinhuagu/Gazefy/actions/workflows/ci.yml/badge.svg)](https://github.com/xinhuagu/Gazefy/actions/workflows/ci.yml)
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

## Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1: Capture + Change Detection | Done | Screen capture, frame diffing, benchmark validation |
| M2: Pack Contract + Training Pipeline | Done | ApplicationPack, model registry, collect/prep/train CLI |
| M3: First Pack Training | Next | Train first real model on target application |
| M4: UIMap + Cursor Monitor | Planned | Real-time element tracking and cursor awareness |
| M5: Action Execution + LLM | Planned | End-to-end task completion |
| M6: Hardening | Planned | Error recovery, regression suite |

## License

[Apache License 2.0](LICENSE)
