# Gazefy Design Document

## Context

A real-time screen monitoring and automation tool that uses custom-trained neural networks to recognize UI elements (menus, buttons, input fields, etc.) in VDI-hosted professional Windows applications. The system tracks cursor position against detected elements in real-time and enables LLM-driven precise software operation.

Secondary scenario: COBOL mainframe automation via Virtel browser terminal.

Core analogy: like Google MediaPipe processes camera frames to recognize hand gestures in real-time, Gazefy processes screen frames to recognize UI elements in real-time.

## Core Design Principle

**The model does not need to run every frame.** UI is static — the screen doesn't change unless the user acts. Therefore:

```
Capture 20 FPS → Frame diff (<1ms) → Run model only on change (~20ms) → Cache element map → Cursor lookup (<0.01ms)
```

The model is idle 99% of the time. CPU/GPU usage stays minimal.

## System Architecture

```
┌─ Thread 1: Capture ──────────────────────────┐
│  mss / CGWindowListCreateImage  ~20 FPS      │
│  → Ring Buffer (last N frames)               │
└──────────────┬───────────────────────────────┘
               │ frame
               v
┌─ Main Loop ──────────────────────────────────┐
│                                               │
│  ① Change Detector ──→ No change? Skip       │
│       │ Changed                               │
│       v                                       │
│  ② YOLO Inference (CoreML/MPS, ~15-30ms)     │
│       │ detections                            │
│       v                                       │
│  ③ Element Tracker updates UIMap              │
│       │ - IoU matching for stable element IDs │
│       │ - Parent-child hierarchy              │
│       │ - Stability filtering (≥2 frames)     │
│       v                                       │
│  ④ Output structured element map (UIMap)      │
│                                               │
└──────────────┬───────────────────────────────┘
               │ UIMap (cached in memory)
               │
    ┌──────────┼──────────┐
    │          │          │
    v          v          v
┌────────┐ ┌────────┐ ┌────────────┐
│ Cursor │ │  LLM   │ │   Data     │
│Monitor │ │Interface│ │ Collector  │
│ 60Hz   │ │        │ │ (training) │
│ lookup │ │serialize│ │            │
│ report │ │→ reason │ │ screenshot │
│        │ │→ action │ │ + labels   │
└────────┘ └───┬────┘ └────────────┘
               │ actions
               v
         ┌──────────┐
         │ Action   │
         │ Executor │
         │pyautogui │
         │+ coord   │
         │  transform│
         │+ verify  │
         └──────────┘
```

## Differentiation: Higher Precision Than General-Purpose Solutions

Gazefy's core advantage over Anthropic Computer Use, OmniParser, and similar general-purpose tools: **deep understanding of a specific application + high-precision operation**.

The system should support multiple applications, but not by forcing them into one monolithic model. The correct architecture is a shared runtime plus **hot-swappable application packs**.

### Three Layers of Differentiation

1. **Detection accuracy** — Trained specifically on the target application; recognizes custom controls that general solutions miss
2. **Semantic understanding** — Knows not just "this is a button" but "this is the Confirm Export button on the Report Export dialog"
3. **Operation reliability** — Pre-click verification (crop bbox + OCR text confirmation), post-action verification (screen classifier confirms expected page transition)

### Optional Knowledge Module (enabled when manual is available)

When the software's HTML manual is provided, the system automatically parses it for enrichment:
- **Extract screenshots** → High-quality uncompressed training data
- **Extract element semantics** → Name and function description for every button/field
- **Extract operation steps** → Auto-generate workflow graphs
- **Build LLM context** → LLM understands the business meaning of each element

Without a manual, the system operates normally using only the detection model + OCR + LLM reasoning.

## Multi-Application Strategy: Hot-Swappable Application Packs

Gazefy should not assume one global detector for all software. Each target application should ship as an independent runtime pack that can be loaded, unloaded, and upgraded without affecting other applications.

An **ApplicationPack** contains:
- detector model
- screen classifier
- element verifier rules
- label taxonomy
- semantic dictionary
- workflow definitions
- targeting rules
- app-specific config (window matching, thresholds, timeouts, OCR hints)

Recommended runtime flow:

```
Capture frame
  ↓
AppRouter identifies current application
  ↓
ModelRegistry loads matching ApplicationPack
  ↓
Pack-specific detector/classifier/verifier run
  ↓
UIMap + action planning + execution use that pack's rules
```

This keeps the system modular:
- one app can be retrained without touching others
- each app can use its own taxonomy and verification logic
- memory usage stays controlled because packs are loaded on demand
- deployment becomes safer because a bad model update is isolated to one app

Recommended pack directory layout:

```text
app_packs/
  sap_gui/
    detector.mlpackage
    screen_classifier.mlpackage
    labels.yaml
    semantics.yaml
    workflows.yaml
    targeting.yaml
    verifier.yaml
    app_config.yaml
  virtel_terminal/
    detector.mlpackage
    screen_classifier.mlpackage
    labels.yaml
    semantics.yaml
    workflows.yaml
    targeting.yaml
    verifier.yaml
    app_config.yaml
```

The hot-swappable boundary is the full application capability pack, not just the detection model.

### Enhanced UIElement Data Structure

```
id:           "btn_0042"           # Stable ID (persists across frames)
class_name:   "button"
semantic_id:  "confirm_export"     # App-specific semantic ID (populated by knowledge module)
bbox:         (100, 50, 200, 80)  # Pixel coordinates
center:       (150, 65)
confidence:   0.92
parent_id:    "dialog_001"
text:         "OK"                # OCR-extracted text
description:  "Confirm report export"  # From manual (optional)
context:      "Report Export Dialog"    # Current page/context
stability:    5                   # Consecutive frames confirmed
```

## Ten Modules

### 1. Screen Capture (`gazefy/capture/screen_capture.py`)
- Capture VDI window region using `mss` library, ~30 FPS on M-series
- Dedicated thread with ring buffer for last N frames
- Auto-detect VDI window position via `pyobjc-framework-Quartz`
- Fallback: manual region selection

### 2. Change Detector (`gazefy/capture/change_detector.py`)
- Three-tier filtering, progressively more precise:
  1. **Perceptual hash** (every frame, <1ms): Downsample to 160x120, compute hash; skip if identical
  2. **SSIM similarity** (when hash differs, ~3ms): > 0.98 treated as VDI compression noise; skip
  3. **Dirty rect detection** (when SSIM below threshold): Find bounding rects of changed regions
- Output: `changed: bool` + `change_level: NONE/MINOR/MAJOR` + `dirty_rects`
- MAJOR = menu opened, dialog appeared, page transition

### 3. UI Detector (`gazefy/detection/detector.py`)
- Pack-specific custom-trained YOLOv8 model, exported to CoreML for Apple Silicon ANE
- Input 640x640 (letterbox scaled), output bbox + class + confidence
- Class set defined by the active ApplicationPack; typical packs use ~8-15 classes such as button, menu_bar, menu_item, input_field, checkbox, radio_button, dropdown, dialog, toolbar, label, tab, scrollbar, icon
- Inference ~15-30ms on M1/M2/M3
- Post-processing: coordinate mapping to original resolution, hierarchical containment computation

### 3.5. App Router + Model Registry (`gazefy/runtime/`)
- `app_router.py` — Identify current application from window title, bundle/process metadata, lightweight screenshot classifier, or configured capture region
- `model_registry.py` — Load, cache, unload, and version ApplicationPacks
- `application_pack.py` — Typed definition of the pack contract (models, taxonomies, workflows, verifier rules, thresholds)
- Supports pack fallback policy: exact app match → compatible app family → abort if unknown
- Enables hot-swapping when the user moves between applications or VDI sessions

### 4. Element Tracker (`gazefy/tracker/element_tracker.py`)
- Maintains **UIMap** — structured table of all current screen elements
- Each element has a **stable ID** (cross-frame IoU matching, similar to video object tracking)
- Stability filtering: element must appear in ≥2 consecutive frames to be confirmed (prevents flickering false positives)
- Stale cleanup: old elements removed after MAJOR change when they disappear
- Parent-child hierarchy: dialog contains button, menu_bar contains menu_item

(UIElement data structure defined in "Enhanced UIElement" section above)

### 5. Cursor Monitor (`gazefy/cursor/cursor_monitor.py`)
- Dedicated thread, 60Hz polling via `pyautogui.position()`
- Screen coords → frame coords translation (subtract window offset, handle Retina 2x scaling)
- **Point-in-rectangle** test against UIMap (nanosecond-level)
- Overlapping elements: return the smallest (most specific) one
- Output: `CursorState { position, current_element, dwell_time }`
- Event callbacks: `on_element_enter`, `on_element_leave`

### 6. Action Executor (`gazefy/actions/executor.py`)
- Translates LLM decisions into pyautogui operations
- **Coordinate chain**: pack-specific target hotspot (pixels) / Retina scale + window offset = screen logical coords
- Action types: click, double_click, right_click, type_text, press_key, hotkey, scroll
- Composite actions: select_menu_item (click menu → wait → click item), fill_field (click → clear → type)
- **Verification**: wait for change detector to report change after execution; timeout = failure
- Safety: dry_run mode logs actions without executing

### 7. LLM Interface (`gazefy/llm/`)
- Serialize UIMap into LLM-readable structured text
- Include spatial descriptions (hierarchical indentation, coordinates, cursor position annotation)
- Support attaching screenshot (base64) for vision-capable models
- Parse LLM action commands from response
- Inject ApplicationPack metadata so the LLM sees the active app, available workflows, semantic IDs, and action constraints
- Support Anthropic Claude / OpenAI / local Ollama

### 8. Training Pipeline (`gazefy/training/`)
- **Per-pack data collection mode**: auto-capture screenshots + record cursor clicks while user operates one target application
- **Model-assisted labeling**: use current model for pre-annotation, human corrects in Label Studio
- **VDI-specific augmentation**: JPEG compression (quality 20-80), blur, color shift, resolution scaling
- **Training**: Ultralytics API, supports MPS/CUDA
- **Export**: produce an ApplicationPack artifact, not just a naked model file

### 8.5. Collector UI (`gazefy/collector_ui/`)
- Lightweight desktop controller for training-sample collection; intended to stay visible while the operator works inside the VDI app
- V1 recommendation: `PySide6` desktop UI, because it integrates directly with Python runtime code and avoids introducing a separate JS/Electron shell
- UI responsibilities:
  - choose target window
  - set pack/session names
  - start/pause/stop/discard collection
  - show frame counts, dedupe/skipped counts, last capture preview, and output path
  - attach notes/tags to important captures
  - hand off finished sessions to Label Studio or Finder
- The collector UI should orchestrate existing runtime modules, not duplicate them:
  - `window_finder.py` for window selection
  - `screen_capture.py` for frame source
  - `change_detector.py` for change-triggered sampling
  - `training/collector.py` for session persistence
- The UI must remain optional. The underlying collector should still be scriptable from CLI for automation and tests.

### 9. Knowledge (optional) (`gazefy/knowledge/`)
Only enabled when software manual is provided; otherwise skipped entirely.
- `manual_parser.py` — Parse HTML manual:
  - Extract screenshots (`<img>`) + surrounding description text → supplementary training data
  - Extract element names + function descriptions → semantic dictionary
  - Extract operation steps → workflow graph
- `semantic_dict.py` — Element semantic dictionary: element_name → type, function, location
- `workflow_graph.py` — Workflow definitions: task_name → step sequence (what to click, what to type, what key to press)
- Auto-inject relevant knowledge into LLM prompt (element descriptions for current screen + current workflow position)

### 10. Screen Classifier + Element Verifier (`gazefy/detection/`)
- `screen_classifier.py` — Pack-specific classifier to identify current application page ("Main Screen" / "Order Entry" / "Report Export", etc.)
- `element_verifier.py` — Pre-click: crop target bbox, run local OCR and pack-specific rules to verify text matches expected target
- Post-action: re-capture screen + screen classifier confirms transition to expected page

## Key Design Details

### Retina Coordinate Translation
macOS Retina 2x scaling is the biggest trap for click precision:
- `mss` captures in pixel coordinates (3840x2160)
- `pyautogui` operates in logical coordinates (1920x1080)
- Translation: `screen_x = pixel_x / 2.0 + window_left`

### VDI Compression Robustness
- Frame diff SSIM threshold 0.98 absorbs compression noise
- Training augmentation with JPEG quality 20-80
- Detection confidence threshold filters false positives from compression artifacts

### Screen State Change Handling

| Scenario | Detection | Response |
|----------|-----------|----------|
| Menu opens | MAJOR change | Full re-detection, new elements added to UIMap |
| Dialog appears | MAJOR change | Detected as dialog parent + child elements |
| Page transition | MAJOR change, >70% old elements gone | Full UIMap rebuild |
| VDI lag | Sustained NONE | UIMap frozen, stale timestamp logged |
| Cursor movement | Small dirty rect | No model run, cursor position update only |

### Runtime State Machine and Recovery Policy

The orchestrator should run as an explicit state machine, not as loosely coupled callbacks. This makes failure handling predictable and measurable.

```
IDLE
  ↓ user/LLM requests action
PLAN_ACTION
  ↓ target resolved
PRECHECK_TARGET
  ├─ verify passed → EXECUTE_ACTION
  └─ verify failed → RECOVER_TARGET
EXECUTE_ACTION
  ↓ input event sent
WAIT_FOR_EFFECT
  ├─ expected change observed → VERIFY_RESULT
  ├─ timeout but screen unchanged → RETRY_ACTION
  └─ unexpected major change → REDETECT_SCREEN
VERIFY_RESULT
  ├─ expected page/element confirmed → SUCCESS
  └─ mismatch → RECOVER_TARGET
RECOVER_TARGET
  ├─ target found after re-detect → PRECHECK_TARGET
  ├─ alternate targeting strategy available → PRECHECK_TARGET
  └─ target unresolved → ABORT
```

Recovery rules must be deterministic:
- **Retry only once** for low-risk actions like opening a menu or focusing a field
- **Never blind-retry** destructive actions like submit, delete, confirm, or irreversible workflow transitions
- **Trigger re-detect** if target verification fails, screen confidence drops, or post-click state is ambiguous
- **Abort immediately** if a different dialog/page appears than expected, or if multiple candidate targets remain unresolved
- **Log every transition** with screenshot hash, target ID, action payload, verification outcome, and elapsed time

Recommended timeout policy:
- `focus/click small control`: 300-500ms for visible effect
- `menu/dialog open`: 800-1200ms
- `page transition / VDI server round trip`: 2-5s
- `classifier/OCR mismatch after action`: force re-capture before deciding failure

This state machine is the reliability boundary of the system. Detection quality alone is not enough; every action must end in `SUCCESS` or a well-explained `ABORT`.

### Interaction Targeting Strategy Tiers

Do not treat all controls as "click bbox center". Different UI element types need different targeting and verification strategies.

| Element type | Primary targeting | Secondary targeting | Verification |
|--------------|-------------------|---------------------|--------------|
| Button / toolbar button | Click geometric center with padding margin | OCR-aligned click biased toward label/icon center | OCR text or icon match; expected visual state change |
| Input field | Click left-inner padding or detected text caret zone | Use label-field spatial pairing to target nearest aligned field | Caret visible, focus ring visible, or text insertion succeeds |
| Dropdown / combo box | Click arrow zone or right edge hotspot | Click field body if arrow detection weak | Menu/list appears |
| Checkbox / radio button | Click control glyph region, not label center | Click associated label only if app behavior supports label activation | Checked state toggles |
| Tab | Click tab label centroid | Use tab bar ordering and text match | Active tab highlight changes |
| Table / grid cell | Resolve row + column first, then click cell interior | Keyboard navigation from known anchor cell | Cell highlight, editor activation, or row selection changes |
| Tree / hierarchical menu | Click expand/collapse affordance separately from label | Keyboard arrow navigation | Node expanded/collapsed state changes |
| Terminal / Virtel field | Prefer DOM / row-column addressing / keyboard | OCR grid fallback | Cursor moves to expected field; screen text updates |

Target resolution should be layered:
1. **Semantic match**: find the intended control by semantic ID, OCR text, screen context, and parent container
2. **Geometric refinement**: convert the detected bbox into an action-specific hotspot rather than raw center point
3. **Pre-action verification**: confirm text/icon/class before input
4. **Post-action verification**: confirm state transition, focus, or content update

This avoids a common failure mode in desktop agents: detection is "correct enough" for bounding boxes, but not precise enough for real clicks in dense enterprise UIs.

### End-to-End Evaluation Framework

The primary benchmark is not object detection mAP. The primary benchmark is whether the system can complete real tasks safely and repeatably.

Evaluation must happen at three levels:

1. **Perception metrics**
   - Detection mAP by class
   - OCR accuracy on cropped controls
   - Screen classifier accuracy by page/dialog
   - Stable ID tracking accuracy across frame changes

2. **Interaction metrics**
   - Target resolution accuracy: did the system choose the correct actionable element?
   - Click precision success rate: did the click land on the intended effective zone?
   - Verification precision/recall: did pre/post checks correctly accept valid actions and reject invalid ones?
   - Recovery success rate: when the first attempt fails, how often does re-detect/retry recover correctly?

3. **Task-level metrics**
   - End-to-end task success rate
   - False action rate: wrong click, wrong field, wrong menu item
   - Abort rate: task stopped safely instead of acting unsafely
   - Median / P95 task completion time
   - Human intervention rate

Suggested benchmark suite:

| Category | Example task | Success criteria |
|----------|--------------|------------------|
| Navigation | Open Export dialog from main screen | Correct dialog opens with no extra actions |
| Form fill | Enter customer ID and date range | Correct fields focused and values entered |
| Selection | Choose a menu item or dropdown option | Intended option selected only once |
| Confirmation | Click safe confirmation button | Expected transition occurs, no unintended submit/delete |
| Recovery | Handle delayed VDI repaint or stale UIMap | System re-detects and completes or aborts safely |
| Terminal | Fill a Virtel field and submit with Enter | Cursor lands in correct row/column and response appears |

Minimum acceptance gates before production use:
- `task success rate >= 95%` on a fixed regression suite of high-frequency workflows
- `false action rate <= 0.5%` on the same suite
- `unsafe destructive misfire rate = 0`
- `target resolution accuracy >= 98%` for critical controls
- `recovery success rate >= 80%` for injected lag / stale-screen scenarios

Every benchmark run should save:
- task trace
- screenshots before and after each action
- chosen target metadata
- verifier outputs
- final outcome (`success`, `recovered`, `aborted`, `unsafe_failure`)

### COBOL/Virtel Special Handling
- Prefer Playwright DOM access (100% accuracy)
- Fallback to screenshot + OCR when DOM unavailable (character grid, OCR accuracy ~99%+)
- Field positioning by character row/column, not pixel coordinates
- Operation via keyboard sequences (Tab, Enter, F-keys), not mouse clicks

## Performance Budget

| Step | Latency | Frequency |
|------|---------|-----------|
| Screen capture | ~5ms | 20 FPS |
| Frame diff check | <1ms | Every frame |
| YOLO inference | ~15-30ms | Only on change |
| Tracker update | ~1ms | Only on change |
| Cursor lookup | <0.01ms | 60 Hz |
| LLM call | 500-2000ms | Async, non-blocking |

Screen change to UIMap update: **~25-40ms**. Cursor position resolution: **real-time**.

## Performance Optimization Strategy

The main bottlenecks are expected to be:
1. screen capture
2. model inference
3. OCR
4. VDI latency

These bottlenecks should not be treated equally. The highest return usually comes from reducing inference frequency and OCR scope, not from premature low-level rewrites.

### 1. Screen Capture Optimization

Screen capture is usually an engineering bottleneck, not a modeling bottleneck.

Recommended optimizations:
- Capture only the target window/region, never the full desktop unless required
- Keep one low-latency working stream for detection and a separate high-resolution snapshot path for verification/debugging
- Avoid unnecessary image format conversions between `mss`, `numpy`, `Pillow`, and OpenCV
- Use a dedicated capture thread with a ring buffer so inference never blocks capture
- Drop old frames aggressively; the latest frame matters more than processing every frame

Upgrade path:
- V1: `mss` is acceptable for fast iteration
- V2 on macOS: evaluate `ScreenCaptureKit` for lower-overhead capture and better integration with native window/screen streams

### 2. Model Inference Optimization

Inference is the most important place to optimize because it compounds across the entire runtime.

Algorithmic optimizations:
- Run inference only when the change detector reports meaningful change
- Use dirty-rect aware policies: full-screen inference for major changes, localized inference or re-check for minor changes
- Start with a smaller detector and increase model size only if validation shows a real accuracy need
- Tune input resolution empirically; UI automation often benefits more from a well-chosen `imgsz` than from a larger backbone
- Split perception into stages if needed: lightweight detector first, higher-cost verification only for selected targets

Engineering optimizations:
- Export to `CoreML` on Apple Silicon and benchmark compute unit choices
- Keep pre-processing simple and deterministic; unnecessary resize/copy steps add real latency
- Cache screen context and semantic candidates so repeated screens do not trigger the full resolution pipeline
- Consider pack-specific thresholds instead of one global confidence threshold for all applications

### 3. OCR Optimization

OCR should be used as a verifier, not as a primary whole-screen perception method.

Recommended optimizations:
- Never OCR the full frame in the steady state
- OCR only cropped candidate regions selected by the detector or verifier
- Cache OCR results for stable elements and invalidate only when the element or its parent region changes
- Use small app-specific vocabularies or expected labels to reduce ambiguity
- Prefer template/icon matching or semantic IDs when they are more stable than text
- Run OCR after detection narrows the search space, not before

Practical rule:
- Detection answers "where is the control?"
- OCR answers "is this the exact control we intend?"

### 4. VDI Latency Handling

VDI latency is mostly not solvable through local computation. It should be handled through runtime policy.

Recommended mitigations:
- Treat remote UI response as asynchronous; never assume the next frame reflects the action immediately
- Wait for stable post-action evidence instead of fixed short sleeps
- Differentiate "screen unchanged", "screen still repainting", and "unexpected screen transition"
- Retry only safe actions and only under explicit policy
- Prefer keyboard, DOM, or structural access when available instead of visual mouse interaction
- Reduce multi-step interaction chains where possible; fewer round trips means less exposure to remote lag

The right goal is not "make VDI fast". The right goal is "make the system robust under variable VDI latency".

### Optimization Priority Order

Implementation should optimize in this order:
1. reduce inference frequency
2. reduce OCR scope
3. eliminate avoidable capture overhead
4. harden state-machine behavior under VDI lag
5. only then consider lower-level rewrites in C++/Rust for proven hotspots

### When to Drop Below Python

Python is acceptable for the orchestrator and most of the runtime because the heavy work already happens in native libraries.

Only consider moving parts to C++/Rust if profiling proves that one of these is the real bottleneck:
- high-frequency capture path
- frame diff / dirty-rect computation
- custom tracker with very high update rate
- native event injection or low-level window inspection

Do not rewrite for performance before profiling. For this system, architecture choices usually outperform language changes.

## Project Structure

```
Gazefy/
├── gazefy/
│   ├── cli.py                      # CLI entry point (collect, prep, train, etc.)
│   ├── config.py                   # GazefyConfig dataclass + YAML loading
│   │
│   ├── core/                       # System wiring
│   │   ├── orchestrator.py         # Main runtime loop (wires all modules)
│   │   ├── application_pack.py     # ApplicationPack contract + loader
│   │   ├── model_registry.py       # Scan/index/lookup packs
│   │   └── app_router.py           # Route active window → pack
│   │
│   ├── capture/                    # Screen input (M1 — implemented)
│   │   ├── screen_capture.py       # mss threaded capture + ring buffer
│   │   ├── window_finder.py        # macOS Quartz window enumeration
│   │   └── change_detector.py      # 3-tier frame diff (hash → MAD → dirty rects)
│   │
│   ├── detection/                  # Model inference
│   │   └── detector.py             # Runs pack's YOLO model → list[Detection]
│   │
│   ├── tracker/                    # Element tracking
│   │   ├── ui_map.py               # UIMap, UIElement, Detection (central types)
│   │   └── element_tracker.py      # IoU matching, stability, hierarchy → UIMap
│   │
│   ├── cursor/                     # Cursor monitoring
│   │   └── cursor_monitor.py       # 60Hz poll → resolve cursor to UIElement
│   │
│   ├── actions/                    # Action execution
│   │   ├── action_types.py         # Action, ActionResult, ActionType
│   │   ├── executor.py             # Resolve target → pyautogui → verify
│   │   └── coordinate_transform.py # Pixel ↔ screen coords (Retina-aware)
│   │
│   ├── llm/                        # LLM integration
│   │   ├── interface.py            # Send UIMap to LLM → get Actions
│   │   ├── formatters.py           # UIMap + CursorState → structured text
│   │   └── parsers.py              # LLM JSON response → list[Action]
│   │
│   ├── knowledge/                  # [V2] Optional manual-based enrichment
│   │   ├── manual_parser.py
│   │   ├── semantic_dict.py
│   │   └── workflow_graph.py
│   │
│   ├── training/                   # Model training pipeline
│   │   ├── collector.py            # Capture screenshots → dataset
│   │   ├── dataset_prep.py         # Split train/val after annotation
│   │   ├── trainer.py              # Ultralytics wrapper → pack packaging
│   │   └── train_pack.py           # CLI: train → package → ApplicationPack
│   │
│   └── utils/
│       ├── geometry.py             # Rect, Point, IoU
│       └── timing.py              # FPSCounter, Timer
│
├── packs/                          # Hot-swappable ApplicationPack artifacts
│   └── <app_name>/
│       ├── pack.yaml               # Pack metadata + labels + thresholds
│       ├── model.pt                # Trained YOLO weights
│       └── workflows/              # Optional workflow definitions
│
├── datasets/                       # Per-app training data (not committed)
│   └── <app_name>/<session>/
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       └── dataset.yaml
│
├── configs/                        # YAML configuration files
├── scripts/
│   ├── benchmark.py                # M1 performance validation
│   └── ruff-hook.sh                # Auto-format hook
├── tests/                          # Unit tests (47 tests)
├── docs/
│   └── annotation-playbook.md      # Step-by-step training guide
├── pyproject.toml
├── DESIGN.md                       # This file
└── PRD.md                          # Product requirements
```

### Data Flow (types at each boundary)

```
ScreenCapture          → CapturedFrame (np.ndarray + timestamp)
ChangeDetector         → ChangeResult (changed, level, dirty_rects)
UIDetector             → list[Detection] (class_id, class_name, confidence, bbox)
ElementTracker         → UIMap (dict[str, UIElement], immutable snapshot)
CursorMonitor          → CursorState (position, current_element, dwell_time)
LLMInterface           → list[Action] (type, target_element_id, text, keys)
ActionExecutor         → list[ActionResult] (status, executed_at, screen_changed)
```

## Dependencies

```
mss, numpy, opencv-python-headless, ultralytics, onnxruntime,
pyautogui, pyobjc-framework-Quartz, Pillow, pyyaml, httpx, anthropic
Training extras: albumentations
Knowledge extras: beautifulsoup4, lxml
```

## Training Pipeline: End-to-End Workflow

Each ApplicationPack must be trained independently. Here is the complete flow from zero to a working pack for one target application.

### Overview

```
Step 1: Screenshot Collection    (you operate one target app, system auto-captures)
         ↓
Step 2: Annotation               (draw boxes on screenshots, label each UI element)
         ↓
Step 3: Data Augmentation        (simulate VDI compression, resolution changes, expand dataset)
         ↓
Step 4: Training                 (feed to YOLO, produces pack-specific detector/classifier)
         ↓
Step 5: Validation               (run on test screenshots, check detection quality)
         ↓
Step 6: Iteration                (fix weak spots: add more screenshots, re-annotate, retrain)
         ↓
Step 7: Packaging                (export models + YAML rules into ApplicationPack)
         ↓
Step 8: Deployment               (drop pack into app_packs/ and let runtime load it)
```

### Collector UI Workflow

The preferred V1 collection experience is a compact desktop collector UI, not a terminal-only workflow.

Recommended operator flow:
1. Open Gazefy Collector
2. Select the target VDI window from the visible window list
3. Enter `pack name` and `session name`
4. Choose capture mode: interval, change-triggered, click-triggered, or hybrid
5. Start collection and operate the target software normally
6. Use `Mark Important` or `Add Note` for valuable states such as dialogs, errors, or dense forms
7. Stop collection and review the session summary
8. Open the session directly in Label Studio or Finder for annotation

Recommended V1 UI layout:

```text
┌──────────────────────────────────────────────────────────────┐
│ Gazefy Collector                           ● Recording │
├──────────────────────────────────────────────────────────────┤
│ Pack: [my_app____________]  Session: [session_001________]  │
│ Window: [Citrix Viewer - My App                     v]      │
│ Capture: [Change + Click v]  Interval: [500 ms]             │
│ Hotkey: [Cmd+Shift+S]  Dedupe: [Medium v]                   │
├──────────────────────────────────────────────────────────────┤
│ [ Start ] [ Pause ] [ Stop ] [ Discard ] [ Mark Important ] │
│ [ Add Note ] [ Open Output ] [ Review in Label Studio ]     │
├──────────────────────────────────────────────────────────────┤
│ Frames: 182   Saved: 74   Skipped: 108   Last event: click  │
│ Output: datasets/my_app/session_001                         │
├──────────────────────────────────────────────────────────────┤
│ Last capture preview                                         │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │                    screenshot thumbnail                 │ │
│ └──────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ Notes / Tags: [export dialog, table full, error state____]  │
└──────────────────────────────────────────────────────────────┘
```

This UI should optimize for three things:
- low operator friction
- obvious recording state
- fast handoff into labeling

### Step 1: Screenshot Collection

Operate one target software application normally inside VDI. The data collector runs in the background, auto-capturing screenshots at regular intervals.

```
Your action:                        System auto-captures:
Open main interface              →  screenshot_001.png
Click File menu                  →  screenshot_002.png (menu expanded)
Click Export                     →  screenshot_003.png (dialog appeared)
Fill in a form                   →  screenshot_004.png
Switch to another tab            →  screenshot_005.png
Right-click context menu         →  screenshot_006.png
...
```

**Goal**: Cover all UI states of the target application systematically:
- Every menu expanded (File, Edit, View, Tools, Help, ...)
- Every dialog/popup
- Different data volumes (empty / sparse / full)
- Every page/tab
- Right-click menus, tooltips
- Disabled button states
- Error messages, toasts

**Target**: ~100-200 screenshots for a moderately complex application pack.

### Step 2: Annotation

Draw bounding boxes on each screenshot and label every UI element with its class.

Example annotation on a screenshot:
```
┌──────────────────────────────────────────────┐
│ [menu_bar]──────────────────────────         │
│ │[menu_item]File│[menu_item]Edit│...         │
│                                              │
│ [toolbar]──────────────────────────          │
│ │[button]Save│[button]Run│[button]Stop       │
│                                              │
│ [label]Customer ID:  [input_field]________   │
│ [label]Name:         [input_field]________   │
│                                              │
│              [button]Submit                  │
└──────────────────────────────────────────────┘
```

**Semi-automatic annotation (recommended)**:

Use GroundingDINO for zero-shot pre-annotation, then correct manually:

```
1. Run GroundingDINO on each screenshot
   Input:  screenshot + text prompt "button, menu, input field, checkbox, dialog"
   Output: auto-generated bounding boxes (~80% correct)

2. Open results in Label Studio / CVAT
   Human reviews and corrects:
   - Fix misclassified boxes
   - Add missed elements
   - Adjust imprecise box boundaries

Time per image: ~1 minute (vs ~5 minutes for manual-only annotation)
```

**Output format (YOLO)**:

Each image gets a corresponding `.txt` file with normalized coordinates:
```
# screenshot_001.txt
# class_id  x_center  y_center  width  height  (normalized 0-1)
0  0.15  0.02  0.30  0.03    # button
1  0.05  0.02  0.04  0.03    # menu_item "File"
1  0.10  0.02  0.04  0.03    # menu_item "Edit"
2  0.45  0.18  0.20  0.03    # input_field
7  0.50  0.01  1.00  0.03    # menu_bar
...
```

**Class taxonomy** (~12-15 classes):
```yaml
names:
  0: button
  1: menu_item
  2: input_field
  3: checkbox
  4: radio_button
  5: dropdown
  6: dialog
  7: menu_bar
  8: toolbar
  9: label
  10: tab
  11: scrollbar
  12: icon
  13: tab_bar
  14: status_bar
```

### Step 3: Data Augmentation

200 annotated images become 2000+ training images through augmentation. Bounding box coordinates are adjusted automatically — no re-annotation needed.

```
Original → JPEG compression (quality 20-80)   → simulates VDI compression
Original → Slight Gaussian blur               → simulates network-induced softness
Original → Brightness/contrast shift (±15%)   → simulates different monitors
Original → Resolution scaling (0.5x-2.0x)     → simulates DPI variation
Original → Color jitter (±10%)                → simulates color quantization
```

**Do NOT apply**: random flips, large rotations, aggressive crops — these destroy UI layout semantics.

### Step 4: Training

```bash
yolo detect train \
  model=yolov8m.pt \       # Pre-trained weights (COCO backbone)
  data=dataset.yaml \      # Dataset config
  imgsz=1024 \             # Input resolution (UI elements need high res)
  epochs=50 \              # Training epochs
  batch=8                  # Batch size
```

`dataset.yaml`:
```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: button
  1: menu_item
  2: input_field
  3: checkbox
  4: radio_button
  5: dropdown
  6: dialog
  7: menu_bar
  8: toolbar
  9: label
  10: tab
  11: scrollbar
  12: icon
  13: tab_bar
  14: status_bar
```

**Training time**: ~200 images + augmentation, 50 epochs:
- Apple Silicon MPS: ~1-2 hours
- NVIDIA RTX 4090: ~30-60 minutes
- Google Colab T4: ~4-6 hours

### Step 5: Validation

Run the trained model on the validation set and inspect results:

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml
```

**Target metrics**:
- mAP@0.5 > 90% (achievable for single-app models)
- mAP@0.5:0.95 > 65%
- Per-class inspection: small elements (checkbox, icon) may need extra attention

Visual inspection with `scripts/visualize_detections.py`:
- Draw predicted boxes on test screenshots
- Look for: missed elements, wrong classifications, imprecise boxes

### Step 6: Iteration (Active Learning Loop)

```
Run model on new screenshots
         ↓
    Results good?
    ├── Yes → Deploy
    └── No  → What's wrong?
              ├── Certain element type often missed → Capture more screenshots of that type, annotate, retrain
              ├── Too many false positives          → Raise confidence threshold / add hard negatives
              └── Small elements detected poorly    → Increase input resolution to 1280
```

Typically **3-5 iterations**, each adding 20-50 corrected images. The model improves rapidly because it's learning a single application with consistent UI patterns.

### Step 7: Deployment

Export trained model for fast inference:
```bash
# Export to CoreML (fastest on Apple Silicon)
yolo export model=best.pt format=coreml imgsz=1024

# Or export to ONNX (cross-platform)
yolo export model=best.pt format=onnx imgsz=1024
```

Place exported model in `models/` directory. The real-time pipeline loads it automatically.

### Dataset Directory Structure

```
datasets/
└── app_a/
    ├── images/
    │   ├── train/                  # 80% of images
    │   │   ├── screenshot_001.png
    │   │   ├── screenshot_002.png
    │   │   └── ...
    │   └── val/                    # 20% of images
    │       ├── screenshot_050.png
    │       └── ...
    ├── labels/
    │   ├── train/                  # Corresponding YOLO annotations
    │   │   ├── screenshot_001.txt
    │   │   ├── screenshot_002.txt
    │   │   └── ...
    │   └── val/
    │       ├── screenshot_050.txt
    │       └── ...
    └── dataset.yaml
```

### Effort Estimate

| Step | Time | Notes |
|------|------|-------|
| Screenshot collection | 2-3 hours | Systematically cover all UI states |
| Pre-annotation (GroundingDINO) | ~30 min | Automated |
| Manual correction | 3-4 hours | ~200 images x ~1 min each |
| Training | 1-2 hours | Machine runs, you wait |
| Validation + iteration (3 rounds) | 3-5 hours | Add images + retrain each round |
| **Total** | **~2 days** | Results in the first high-accuracy ApplicationPack for one target app |

## Implementation Order

1. **Phase 1 — Capture + Change Detection**: Verify stable 20 FPS capture of VDI window
2. **Phase 2 — App Router + Pack Contract**: Define `ApplicationPack`, runtime loading, and per-app config boundaries
3. **Phase 3 — First Pack Data Collection + Training**: Collect screenshots, annotate, train the first app-specific detector/classifier
4. **Phase 4 — Detection + Tracker**: Model inference + UIMap maintenance using the active pack
5. **Phase 5 — Cursor Monitor + Targeting Rules**: Real-time reporting plus pack-specific target hotspot logic
6. **Phase 6 — Action Executor + LLM**: Precise operation + AI decision-making with pack-specific verification
7. **Phase 7 — Additional Packs / Virtel**: Add more applications by shipping new packs instead of rewriting the core runtime

## Verification

1. `scripts/benchmark.py` — Capture FPS, inference latency, end-to-end latency
2. `scripts/visualize_detections.py` — Draw detection boxes on screenshots for visual accuracy check
3. Cursor monitor terminal output — Move mouse, observe real-time element reporting
4. dry_run mode — LLM generates action commands without executing; manual review for correctness
5. Training evaluation — mAP@0.5 target >90%, mAP@0.5:0.95 target >65%
6. End-to-end regression suite — Fixed workflows measuring task success rate, false action rate, abort rate, and recovery success rate

## Self-Improving Pipeline

### Motivation

The core limitation of the per-app YOLO model is that it is static: once trained, it does not adapt
to UI theme changes, VDI compression shifts, or new UI states added by software updates. The goal
of the self-improving pipeline is to close this gap with zero human annotation after the initial
pack is trained.

The underlying principle is "learn from doing": every time the system executes an action, it
produces a signal — did the screen change? That signal is a reward. The system uses this reward,
plus a VLM-based pseudo-labeling loop, to continuously refine its own perception model.

### Architecture

```
Live operation
    ↓
ActionExecutor executes click / type / key
    ↓
_verify_change(): polls ChangeDetector at 20 Hz for up to verify_timeout_s
    ↓
ActionResult.screen_changed = True  →  positive training sample
ActionResult.screen_changed = False →  hard negative (missed click)
    ↓
TrainingBuffer accumulates (frame, bbox, label, reward) tuples
    ↓
DriftMonitor tracks rolling mean YOLO confidence
    ↓ (confidence < drift_threshold)
AutoAnnotator runs HybridAnnotator on buffered frames
    ├── GroundingDINO: zero-shot bboxes
    ├── EasyOCR: text labels for text-heavy elements
    └── Claude Vision: semantic labels for icon-only elements
    ↓
Pseudo-labels → YOLO training format (annotations.jsonl → YOLO .txt)
    ↓
Fine-tune YOLO on fine-tune buffer (LoRA or last-N-layer fine-tune)
    ↓
Updated model → hot-swap into running ApplicationPack
```

### Direction 1: Verification Loop as Reward Signal

The `ActionExecutor` already captures a before-frame and polls for screen change after every
action. The `ActionResult` already carries `screen_changed: bool` and `diff_score: float`.

These signals map directly to training labels:

| ActionResult | Training interpretation |
|---|---|
| `screen_changed=True, diff_score > 0.3` | Strong positive: element was correctly targeted |
| `screen_changed=True, diff_score < 0.1` | Weak positive: some effect but low confidence |
| `screen_changed=False` | Hard negative: click probably missed the intended target |
| `status=ABORTED` (element disappeared) | Negative: UIMap stale, model missed an element |

A `RewardBuffer` collects these results alongside the captured before-frame and the detection
metadata (element_id, bbox, confidence). This buffer drives the fine-tuning pipeline.

Implementation sketch (`gazefy/training/reward_buffer.py`):
```python
@dataclass
class RewardSample:
    frame: np.ndarray           # BGRA capture before action
    bbox: list[int]             # [x1, y1, x2, y2] in pixel coords
    class_name: str             # Detected class
    confidence: float           # YOLO confidence at detection time
    reward: float               # diff_score if screen_changed, else -1.0

class RewardBuffer:
    def add(self, result: ActionResult, frame: np.ndarray, element: UIElement) -> None: ...
    def flush_positive(self, min_reward: float = 0.1) -> list[RewardSample]: ...
    def flush_hard_negatives(self) -> list[RewardSample]: ...
```

### Direction 2: VLM→YOLO Knowledge Distillation

`HybridAnnotator` already produces `annotations.jsonl` with `source` tags (`ocr`, `vlm`,
`yolo+ocr`). The missing piece is a converter that transforms these annotations into YOLO training
format so they can drive a fine-tuning run without any human labeling.

Pipeline:

```
Fresh screenshots (collected during live operation or triggered by drift)
    ↓
HybridAnnotator.annotate_session()
    → annotations.jsonl  (label, class, bbox, source)
    ↓
AnnotationConverter.to_yolo(annotations.jsonl, image_dir)
    → labels/<image_name>.txt  (YOLO normalized coords)
    ↓
gazefy train --dataset <converted_dir> --pack <name> --fine-tune
```

The `source` field determines trust weighting:
- `ocr`: high trust (OCR text is deterministic)
- `yolo+ocr`: high trust (model + OCR agree)
- `vlm`: medium trust (VLM is confident but not ground truth)
- `yolo+vlm`: medium trust

Low-trust samples are down-weighted in the training loss or filtered below a confidence threshold.

### Direction 3: Drift Detection + Auto Re-annotation

The YOLO model's mean detection confidence is a reliable proxy for "is this model still working
correctly on the current screen?"

`DriftMonitor` runs as a lightweight sidecar in the main loop:

```python
class DriftMonitor:
    window_size: int = 200         # Rolling window of recent detections
    drift_threshold: float = 0.55  # Trigger if mean confidence drops below this
    cooldown_s: float = 3600.0     # Minimum time between re-annotation triggers

    def record(self, detections: list[Detection]) -> None: ...
    def is_drifted(self) -> bool: ...
    def mean_confidence(self) -> float: ...
```

When `is_drifted()` returns True, the orchestrator:
1. Captures a batch of representative screenshots (N=50 covering recent UIMap states)
2. Runs HybridAnnotator on the batch (GroundingDINO + EasyOCR + Claude Vision)
3. Converts output to YOLO format
4. Schedules a fine-tuning job (background process, does not interrupt live operation)
5. Hot-swaps the updated model into the running pack once training completes

The re-annotation trigger can also be set manually (`gazefy drift-check --pack my_app`).

### Direction 4: Active Learning on Uncertainty

Not all detections are equally worth retraining on. The most valuable samples for improving the
model are the ones where the model is least confident.

`ActiveLearner` filters the detection stream for high-uncertainty cases:

```python
class ActiveLearner:
    uncertainty_threshold: float = 0.45  # Detections below this are candidates
    batch_size: int = 20                  # Send to VLM in batches

    def should_query(self, detection: Detection) -> bool:
        return detection.confidence < self.uncertainty_threshold

    def query_vlm(self, frame: np.ndarray, candidates: list[Detection]) -> list[UIElement]:
        # Crop each candidate, send batch to Claude Vision
        # Returns refined labels + classes
        ...
```

The VLM query uses the same icon-labeling approach as `HybridAnnotator._label_icons_on_frame()`:
number the candidate bboxes, send one image with all boxes drawn, ask Claude to name each one.
This batches cost across many candidates per API call.

Confirmed VLM labels feed directly into the fine-tune buffer with high trust weight.

### Direction 5: Universal Base + App-Specific LoRA Adapter (V2)

For V2, the goal is to replace the "train from scratch per app" model with a frozen universal UI
detector plus a tiny per-app adapter layer (LoRA).

Architecture:
```
Universal base YOLO (trained on diverse UI screenshots from many apps)
    ↓
Frozen backbone + neck
    ↓
Per-app LoRA adapter (~1-5% of base model parameters)
    ↓
App-specific detection head
```

Benefits:
- Initial pack setup requires only LoRA fine-tuning (~50 samples vs ~200 for full training)
- Base model updates improve all packs simultaneously
- Adapters are tiny (~10 MB vs ~100 MB for full model) — fast to update and distribute

The universal base can be bootstrapped by aggregating pseudo-labels from HybridAnnotator runs
across multiple apps, creating a large diverse UI screenshot corpus without human annotation.

### Data Flow for the Self-Improving Loop

```
gazefy/
├── training/
│   ├── reward_buffer.py     RewardSample + RewardBuffer (Direction 1)
│   ├── annotation_converter.py  annotations.jsonl → YOLO .txt (Direction 2)
│   ├── drift_monitor.py     DriftMonitor: rolling confidence tracking (Direction 3)
│   ├── active_learner.py    ActiveLearner: uncertainty sampling + VLM query (Direction 4)
│   └── auto_trainer.py      Wires all four directions into one fine-tune pipeline
```

`AutoTrainer` coordinates the full cycle:
1. Consumes `RewardBuffer` and `ActiveLearner` output
2. Calls `AnnotationConverter` on `HybridAnnotator` output
3. Runs `gazefy train --fine-tune` when buffer crosses a size threshold or drift is detected
4. Notifies `ModelRegistry` to hot-swap the updated pack

### Implementation Phases

| Phase | Deliverable | Depends on |
|---|---|---|
| M8a | `RewardBuffer` + logging pipeline | ActionExecutor (done) |
| M8b | `AnnotationConverter` (annotations.jsonl → YOLO) | HybridAnnotator (done) |
| M8c | `DriftMonitor` + manual trigger CLI | UIDetector (done) |
| M8d | `ActiveLearner` + VLM uncertainty query | HybridAnnotator (done) |
| M8e | `AutoTrainer` wiring all four | M8a–M8d |
| M9 | LoRA adapter framework + universal base model | M8e + large multi-app corpus |

### Connection to Karpathy's Autoresearch Principle

The self-improving pipeline applies the same core insight from recent self-improving AI research:
**the system generates its own training signal through interaction with the world**.

- Direction 1 (verification as reward) mirrors reinforcement learning from environment feedback
- Direction 2 (VLM distillation) mirrors knowledge distillation from a larger teacher model
- Direction 3 (drift detection) mirrors distribution shift monitoring in production ML systems
- Direction 4 (active learning) mirrors focused data collection on model weaknesses
- Direction 5 (LoRA adapters) mirrors parameter-efficient fine-tuning research (LoRA, QLoRA)

The key insight specific to Gazefy: the "environment" is the target application itself. Every
time the executor clicks a button and the screen changes, the system has learned something. It
does not need a human to tell it whether the click was correct — the screen feedback is the label.
7. Failure injection — Simulate VDI lag, stale UIMap, OCR mismatch, and delayed repaint to verify state-machine recovery behavior
