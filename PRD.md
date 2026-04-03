# Gazefy — Product Requirements Document

## Product Goal

Enable AI-driven precise operation of professional software interfaces presented through screen pixels by combining a shared runtime with hot-swappable application packs — no accessibility API, no source code access, no OS-level hooks.

## Target Users

- **Primary**: Engineers/operators who need to automate repetitive workflows in a specific software application that has no API and no scripting interface, whether it runs locally, inside VDI, or through another remote display surface.
- **Secondary**: QA teams who need to regression-test the same application through its GUI.

## Core Use Cases

| # | Use Case | Priority |
|---|----------|----------|
| UC1 | **Real-time cursor awareness**: System continuously reports which UI element (by name/type) the mouse cursor is currently hovering over. | P0 — V1 |
| UC2 | **LLM-driven task execution**: User describes a task in natural language; LLM reasons over detected UI elements and executes a sequence of clicks/keystrokes to complete it. | P0 — V1 |
| UC3 | **Training data collection**: User operates the software normally; system records screenshots + cursor actions for model training. | P0 — V1 |
| UC4 | **Model training & iteration**: User annotates screenshots (with model-assisted pre-labeling), trains a YOLO model, evaluates, and iterates. | P0 — V1 |
| UC5 | **Application-pack runtime**: Load app-specific detector/classifier/verifier/workflow packs without changing the shared runtime. | P0 — V1 foundation |
| UC6 | **COBOL/Virtel terminal automation**: Automate mainframe operations through a browser-based 3270 terminal emulator. | P1 — V2 |
| UC7 | **Knowledge-enriched operation**: Parse software HTML manual to enrich LLM context with element semantics and workflow definitions. | P1 — V2 |
| UC8 | **Drift detection**: System monitors its own detection confidence and alerts the operator when the pack is degrading due to UI changes. | P1 — V2 |
| UC9 | **Automatic re-annotation**: When drift is detected (or triggered manually), the system captures fresh screenshots and re-annotates them using the HybridAnnotator pipeline without operator involvement. | P1 — V2 |
| UC10 | **Verification-driven training buffer**: Confirmed clicks (screen changed) are automatically collected as positive training samples; failed clicks become hard negatives. | P1 — V2 |
| UC11 | **Active learning on uncertainty**: Low-confidence detections are batched to Claude Vision, which provides refined labels that feed directly into the fine-tune buffer. | P2 — V2 |
| UC12 | **VLM→YOLO distillation**: HybridAnnotator pseudo-labels from raw screenshots are converted to YOLO format and used to fine-tune the pack detector without human annotation. | P2 — V2 |

## Non-Goals (explicitly out of scope)

- **General-purpose UI automation**: Gazefy is not Anthropic Computer Use. V1 ships one production-ready application pack and a runtime that can support more packs later, trading generality for precision.
- **Cross-platform agent**: V1 is macOS-only on the operator machine. The target application may run locally, inside VDI, or through another remote session, but cross-host automation parity is not a V1 goal.
- **Model training infrastructure**: No cloud training pipeline. Training happens locally or on a single GPU machine. Ultralytics CLI is sufficient.
- **Full product GUI / dashboard**: V1 does not ship a general-purpose desktop console or web dashboard. A lightweight training-sample collector UI is in scope.
- **Recording and replaying macros**: Gazefy is not a macro recorder. The LLM reasons about each step, adapting to screen state.
- **Zero-annotation cold start**: The self-improving pipeline (V2) still requires an initial bootstrapped pack (100-200 annotated screenshots). Fully zero-annotation pack creation from scratch is out of scope.

## V1 Scope

### What V1 delivers

A CLI tool with a shared runtime that supports **hot-swappable application packs**, shipping initially with **one production-ready pack for one specific target application**:

1. Captures the target application surface at ~20 FPS
2. Identifies the active application/session and loads the matching `ApplicationPack`
3. Detects screen changes and runs a custom-trained YOLO model only when needed
4. Maintains a real-time UIMap of all detected elements (bounding boxes, classes, OCR text)
5. Reports in terminal which element the cursor is hovering over
6. Accepts natural language tasks, sends UIMap + screenshot to an LLM, executes returned actions
7. Executes actions through a deterministic runtime state machine with verification, retry, re-detect, and abort behavior
8. Includes a data collection mode for building per-pack training datasets
9. Includes a lightweight collector UI for session control, sample review, and export into the labeling workflow
10. Includes a training/evaluation wrapper around Ultralytics that outputs an `ApplicationPack` artifact

### What V1 does NOT deliver

- Virtel/COBOL terminal support (V2)
- Knowledge module / manual parsing (V2)
- Multi-app onboarding workflow beyond the first production pack
- GUI / web dashboard (Future)
- Windows/Linux host support (Future)

### Launch Focus

V1 is optimized first for **VDI-hosted professional Windows applications**, because that is the hardest high-value environment and exercises the most difficult constraints: compression artifacts, remote repaint delay, lack of APIs, and coordinate precision. The same runtime architecture is intended to support local desktop applications and other screen-presented software through pack-specific capture and interaction rules.

## Success Metrics

| Metric | Target | How to measure |
|--------|--------|----------------|
| **Task success rate** | ≥ 95% on fixed regression workflows | Run a fixed suite of high-frequency workflows end-to-end in a controlled environment; count successful completions. |
| **Element detection mAP@0.5** | ≥ 90% | Ultralytics val on held-out test set |
| **Element detection mAP@0.5:0.95** | ≥ 65% | Ultralytics val on held-out test set |
| **Target resolution accuracy** | ≥ 98% for critical controls | Compare chosen actionable target vs ground truth on benchmark screens |
| **False action rate** | ≤ 0.5% | Percentage of actions that click the wrong element or have no effect on the regression suite |
| **Unsafe destructive misfire rate** | 0 | Count wrong submits/deletes/confirms on the regression suite |
| **Recovery success rate** | ≥ 80% in injected lag/stale-screen cases | Run failure-injection scenarios and measure correct recover-or-abort behavior |
| **Cursor-to-element latency** | < 50ms after screen change | Benchmark script measures time from frame capture to UIMap update |
| **Click precision accuracy** | ≥ 95% effective hits on benchmark controls | Compare actual effect zone hits vs ground-truth expected hotspots |
| **Training data effort** | < 3 days from zero to working model | Clock the full cycle: collection → annotation → training → iteration |
| **Collector usability** | New user can start a capture session in < 3 minutes | Observe first-time user completing: choose window -> start session -> collect -> stop -> export |

**Primary success criterion**: no unsafe destructive misfires and task success rate ≥ 95% on the fixed regression suite. mAP is a proxy — what matters is whether the system completes tasks safely and repeatably.

## Functional Requirements

### FR1: Screen Capture
- Capture a user-defined screen region or target application window at ≥ 20 FPS
- Support auto-detection of target window by process name or window name
- Support manual region selection as fallback
- Handle Retina (2x) displays correctly

### FR2: Application Routing and Pack Loading
- Identify the active application/session from window metadata, configured region, or lightweight routing heuristics
- Load the matching `ApplicationPack` at runtime without modifying shared orchestration code
- Support pack-local assets: detector, classifier/verifier rules, labels, workflows, semantics, thresholds, targeting rules
- Fail closed if no safe pack match is available

### FR3: Change Detection
- Detect whether the screen content has changed between frames
- Classify change magnitude: NONE / MINOR / MAJOR
- Skip model inference when screen is unchanged
- Tolerate VDI compression noise (not trigger on compression artifacts alone)

### FR4: UI Element Detection
- Run the active pack's custom-trained YOLO model on changed frames
- Detect elements with bounding box + class + confidence
- Support pack-defined UI element taxonomies, with V1 pack targeting at least 12 common classes (button, menu_item, input_field, checkbox, radio_button, dropdown, dialog, menu_bar, toolbar, label, tab, scrollbar)
- Map model output coordinates back to screen coordinates

### FR5: Element Tracking (UIMap)
- Maintain a persistent map of all detected elements
- Assign stable IDs to elements across frames (IoU matching)
- Build parent-child hierarchy (dialog contains buttons)
- Filter flickering detections (require ≥ 2 frames stability)
- Clear stale elements after major screen changes

### FR6: OCR and Verification Signals
- Run OCR on cropped candidate regions, not the full frame in steady state
- Use OCR for: element naming in UIMap, pre-click verification, LLM context, and post-action validation
- Cache OCR results for stable elements until their region changes
- Must handle VDI compression artifacts gracefully

### FR7: Cursor Monitoring
- Track mouse cursor position at ≥ 30 Hz
- Resolve cursor position to the UIMap element it overlaps
- Handle overlapping elements (return smallest/most specific)
- Output current element info to terminal in real-time

### FR8: Action Execution and Recovery
- Translate element IDs to screen coordinates and execute mouse/keyboard actions
- Support: click, double_click, right_click, type_text, press_key, hotkey, scroll
- Support composite actions: select_menu_item, fill_field
- Resolve action-specific hotspots instead of always clicking raw bbox center
- Run through an explicit state machine: `PLAN_ACTION -> PRECHECK_TARGET -> EXECUTE_ACTION -> WAIT_FOR_EFFECT -> VERIFY_RESULT -> RECOVER_TARGET/ABORT`
- Verify action effect via change detection, OCR/classifier signals, and expected state transitions
- Retry only safe actions under explicit policy; never blind-retry destructive actions
- dry_run mode: log intended actions without executing

### FR9: LLM Integration
- Serialize UIMap into structured text for LLM consumption
- Optionally include screenshot as base64 image
- Parse LLM response into action sequence
- Inject active pack metadata such as workflows, semantic IDs, and action constraints
- Support at least one cloud LLM provider with a provider abstraction

### FR10: Training Data Collection
- Background mode: auto-capture screenshots while user operates one target application
- Log cursor position and click events with timestamps
- Export per-pack dataset in YOLO format (images/ + labels/ + dataset.yaml)
- Configurable capture interval

### FR11: Collector UI
- Provide a lightweight local UI for training-sample collection, optimized for use while the operator is actively working inside the target application
- Let the user select the target window, pack name, and session name before collection starts
- Provide `Start`, `Pause`, `Resume`, `Stop`, and `Discard` controls
- Show live collection status: active window, capture mode, frame count, dedupe/skipped count, last-captured thumbnail, output path
- Support multiple collection triggers: fixed interval, screen change, click-following capture, and manual hotkey capture
- Let the user mark important screens, add short notes/tags, and flag sessions for immediate annotation
- Offer one-click handoff to the labeling workflow by opening the output folder or launching Label Studio against the exported session
- Stay small, always-on-top optional, and non-disruptive to active software operation

### FR12: Model Training and Pack Packaging
- Wrap Ultralytics training API with VDI-specific augmentations
- Support training on MPS (Apple Silicon)
- Export trained model to CoreML
- Package the trained detector together with labels, targeting rules, verifier rules, and config as an `ApplicationPack`
- Evaluation script with per-class mAP breakdown and end-to-end regression reporting

## Collector UI Concept

V1 should ship a **small desktop collector controller**, not a full product shell. The UI exists to make data collection fast, safe, and obvious for non-developer operators across VDI-hosted and local applications.

### Primary user flow

1. Choose target window
2. Enter pack name and session name
3. Pick capture strategy
4. Start collection
5. Operate the target software normally
6. Pause/mark/note important screens when needed
7. Stop collection and review summary
8. Open the exported session in Label Studio or Finder

### UI layout sketch

```text
┌──────────────────────────────────────────────────────────────┐
│ Gazefy Collector                           ● Recording │
├──────────────────────────────────────────────────────────────┤
│ Pack: [sap_export_________]  Session: [2025-04-02-am_____]  │
│ Window: [Citrix Viewer - SAP Export Screen          v]      │
│ Capture: [Change + Click v]  Interval: [500 ms]             │
│ Hotkey: [Cmd+Shift+S]  Dedupe: [Medium v]                   │
├──────────────────────────────────────────────────────────────┤
│ [ Start ] [ Pause ] [ Stop ] [ Discard ] [ Mark Important ] │
│ [ Add Note ] [ Open Output ] [ Review in Label Studio ]     │
├──────────────────────────────────────────────────────────────┤
│ Frames: 182   Saved: 74   Skipped: 108   Last event: click  │
│ Output: datasets/sap_export/session_2025_04_02_01           │
├──────────────────────────────────────────────────────────────┤
│ Last capture preview                                         │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │                    screenshot thumbnail                 │ │
│ └──────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│ Notes / Tags: [export dialog, error state, full table____]  │
└──────────────────────────────────────────────────────────────┘
```

### Interaction principles

- The UI must be understandable at a glance while the user is focused on the VDI app
- The window should stay compact and movable; a minimal mode is preferred over a large dashboard
- Recording state must always be obvious
- Dangerous actions like `Discard` should require confirmation
- Collection should continue even if the preview panel is hidden or minimized
- The UI should optimize for throughput, not aesthetics

## Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Host OS | macOS 13+ (Apple Silicon) |
| Python version | 3.11+ |
| Frame-to-UIMap latency | < 50ms (when model runs) |
| Cursor lookup latency | < 1ms |
| Memory usage | < 2 GB resident (excluding model loading) |
| Model inference | < 30ms per frame on M1/M2/M3 |
| No internet required | For inference and cursor monitoring (LLM calls are the exception) |
| Single-process | No Docker, no server, no database. One Python process. |
| Failure mode | Safe abort preferred over unsafe action |
| Runtime traceability | Every action attempt logged with target, verifier outcome, and final status |

## Technical Stack Decisions

### Decided

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.11+ | ML ecosystem (Ultralytics, PyTorch, ONNX), rapid iteration |
| Detection model | YOLOv8 (Ultralytics) | Best ecosystem, training CLI, export pipeline, proven for UI detection |
| Training framework | Ultralytics API | Wraps PyTorch; handles augmentation, training, validation, export in one CLI |
| Annotation tool | Label Studio | Open-source, supports model-assisted pre-labeling, YOLO export |
| Collector UI toolkit | PySide6 | Collector UI is a local desktop control surface tightly coupled to Python capture/training code; avoiding a JS/Electron shell keeps the system single-runtime and simpler to ship |
| Runtime packaging | `ApplicationPack` directory artifact | Clean boundary for hot-swappable per-app models, labels, workflows, and verifier rules |
| Window detection | Quartz CGWindowList | Native macOS window enumeration; reliable foundation for routing/capture across local and remote app surfaces |
| Action execution baseline | pyautogui | Simple baseline for V1; upgrade path remains open if precision requires native events |
| Configuration | YAML files | Human-readable, standard |
| Dataset format | YOLO images/labels + `dataset.yaml` per pack | Matches Ultralytics training pipeline and keeps pack training isolated |

### Leaning toward (to be validated in Phase 1)

| Decision | Leaning | Alternative | Validation |
|----------|---------|-------------|------------|
| Screen capture | `mss` | ScreenCaptureKit via pyobjc | Benchmark both in Phase 1. mss is simpler; SCK is faster but harder from Python. Go with mss unless it can't hit 20 FPS for the target app window/region. |
| Inference runtime | CoreML export | ONNX Runtime + CoreML EP | Export model to both formats, benchmark inference time. CoreML should be faster on ANE, but ONNX is more portable. |
| OCR engine | PaddleOCR | EasyOCR, Tesseract | Test all three on VDI-compressed screenshots. Pick the one with best accuracy on small UI text. |
| Pre-labeling | GroundingDINO | Manual-only or other zero-shot detectors | Validate pre-label quality and correction time on the first pack before treating it as standard workflow. |
| LLM provider (V1) | Anthropic Claude | OpenAI vision models | Benchmark reasoning quality, latency, and cost on fixed workflows. |
| Coordinate input | pyautogui | CGEvent via pyobjc | Keep pyautogui for first implementation. If precision or reliability is insufficient, upgrade to CGEvent. |

### Not decided (defer to V2+)

| Decision | Options | When to decide |
|----------|---------|----------------|
| Virtel/COBOL approach | Playwright DOM vs screenshot+OCR | When V2 starts; depends on Virtel's DOM structure |
| Knowledge module storage | SQLite vs flat YAML vs in-memory | When manual parsing is implemented |
| Screen/page classifier architecture | Separate classifier vs verifier-only vs YOLO classification head | After V1 pack is stable; needs labeled screen-state data |
| Local LLM | Qwen2.5-VL-7B via Ollama vs llama.cpp | When offline operation becomes a requirement |
| Advanced multi-pack routing | Window metadata only vs screenshot classifier vs hybrid router | When second production pack is onboarded |

## Architecture Summary

(Full details in [DESIGN.md](./DESIGN.md))

```
Capture (20 FPS)
  ↓
App Router
  ↓
Model Registry loads active ApplicationPack
  ↓
Change Detect (<1ms) → Pack-specific YOLO (on change, ~20ms) → UIMap (cached)
                                                               │
                              ┌────────────────────────────────┼──────────┐
                              │                                │          │
                        Cursor Monitor (60Hz)             LLM Interface   Data Collector
                        "cursor is on [button] Save"     → reason        → per-pack dataset
                                                         → act
                                                         → verify / recover / abort
```

Key architectural properties:
- **Single process, multi-threaded**: capture thread + cursor thread + main loop
- **Hot-swappable application packs**: app-specific models and rules load at runtime behind a stable pack contract
- **Event-driven model inference**: model only runs when screen changes (not every frame)
- **Cached UIMap**: detection results persist until next screen change; cursor lookup is pure coordinate math
- **Deterministic action state machine**: actions either succeed, recover safely, or abort with traceable evidence
- **Stateless LLM calls**: each call gets full UIMap + screenshot + pack metadata; no conversation memory needed for action execution

## Milestones

### M1: Capture + Change Detection (Week 1)
**Deliverable**: CLI tool that captures the target application window/region at 20+ FPS, prints "CHANGED" / "UNCHANGED" per frame.
- Screen capture module with mss
- Window finder (auto-detect target app window)
- Change detector (perceptual hash + SSIM)
- Benchmark script proving ≥ 20 FPS capture, < 5ms change detection
- **Exit criterion**: benchmark passes on the first target application surface, with VDI as the initial focus environment

### M2: Runtime Pack Contract (Week 2)
**Deliverable**: Shared runtime can identify the active app and load an `ApplicationPack` contract.
- `ApplicationPack` schema
- App router
- Model registry / pack loader
- Pack-local config conventions
- **Exit criterion**: runtime loads a dummy pack and routes execution without shared-code changes

### M3: First Pack Data Pipeline (Week 3)
**Deliverable**: Data collector that records screenshots + clicks for the first target app; annotation workflow documented and tested.
- Data collector (background capture + click logging)
- Collector UI with session controls and output handoff
- YOLO dataset exporter (`datasets/<app>/images`, `labels`, `dataset.yaml`)
- GroundingDINO pre-annotation evaluation script
- VDI augmentation transforms
- **Exit criterion**: 50 annotated screenshots produced in under 2 hours (collection + pre-label + correction)

### M4: First Trained Pack + Detection (Week 4)
**Deliverable**: Trained application pack that detects UI elements on target application screenshots with mAP@0.5 > 90%.
- Training wrapper (Ultralytics + augmentations)
- Model evaluation script (per-class mAP)
- Detection module (inference + coordinate mapping)
- Pack packaging script
- Visualization script (draw boxes on screenshots)
- **Exit criterion**: mAP@0.5 > 90% on validation set; visual inspection shows correct boxes

### M5: UIMap + Cursor Monitor (Week 5)
**Deliverable**: Real-time terminal output showing "cursor is on [button] Save" as you move the mouse over the target application surface.
- UIMap data structures
- Element tracker (IoU matching, stability, hierarchy)
- OCR integration for element text
- Cursor monitor (60 Hz polling, hit testing)
- Coordinate transform (Retina-aware)
- **Exit criterion**: cursor monitor correctly identifies element under cursor ≥ 90% of the time during manual testing

### M6: Action Execution + LLM (Week 6-7)
**Deliverable**: End-to-end: describe a task → LLM reasons → actions execute → task completes or aborts safely.
- Action executor (pyautogui + coordinate translation)
- Action verification (post-action change detection + OCR/verifier signals)
- Runtime state machine (`precheck`, `execute`, `verify`, `recover`, `abort`)
- LLM interface (UIMap serialization, response parsing)
- Orchestrator (capture → detect → reason → act loop)
- dry_run mode
- **Exit criterion**: no unsafe destructive misfires and ≥ 95% success on the fixed regression suite

### M7: Hardening + Documentation (Week 8)
**Deliverable**: Stable, documented V1 ready for daily use.
- Error recovery (retry logic, stale UIMap handling, VDI lag tolerance)
- Configuration documentation
- Training playbook (step-by-step guide for new application packs)
- Performance tuning guide
- Failure-injection regression suite
- **Exit criterion**: system runs stable for 1 hour of continuous operation without crashes and passes lag/stale-screen recovery tests

## Risks and Open Questions

| Risk | Impact | Mitigation |
|------|--------|------------|
| VDI compression degrades detection accuracy below target | High | Aggressive augmentation during training; validate on real VDI screenshots early (M3/M4) |
| Retina coordinate translation introduces systematic click offset | High | Build coordinate validation test in M5; compare expected vs actual click positions |
| mss capture can't hit 20 FPS for the target window/region on macOS | Medium | Benchmark in M1; fallback to ScreenCaptureKit if needed |
| GroundingDINO pre-annotation quality too low for target app | Medium | Test in M3; fallback to manual annotation |
| Small UI elements (checkboxes, 12px icons) undetectable at 640px input | Medium | Train at 1024px input resolution; if still insufficient, go to 1280px |
| LLM hallucinates actions or misidentifies elements | Medium | Pre-click verification, constrained action schema, dry_run testing before live execution |
| Wrong pack routed for the active screen | Medium | Conservative router rules; fail closed unless pack match is confident |
| OCR cost inflates latency | Medium | OCR only on cropped candidates; cache OCR for stable elements |
| pyautogui click doesn't register reliably in the target application surface | Low | Test in M1 on the initial target environment. Fallback: CGEvent |
| Model overfits to current application theme/data | Low | Augmentation + periodic retraining as application updates |
| VLM pseudo-labels contain errors that corrupt the fine-tune dataset | Medium | Use source trust weighting; validate on held-out set before hot-swapping the model |
| Drift monitor triggers false re-annotation during normal UI transitions | Low | Enforce cooldown period and minimum window size before triggering |

### Open Questions

1. **What OCR engine performs best on VDI-compressed UI text?** — Test PaddleOCR, EasyOCR, Tesseract on real VDI screenshots in M3.
2. **Is the first-pack routing logic adequately robust with window metadata only, or do we need lightweight screenshot-based routing?** — Validate in M2 before onboarding a second pack.
3. **How to handle application updates that change UI layout?** — Retrain model with updated screenshots. Assess how much layout change requires full retraining vs fine-tuning.
4. **Is pyautogui's Retina handling sufficient or do we need CGEvent?** — Validate in M5 with click accuracy test.
5. **What is the minimum fine-tune buffer size before retraining is worthwhile?** — Empirically measure on the first pack: how many new samples are needed to produce a measurable mAP improvement?
6. **Can LoRA adapters for UI detection match full-model accuracy at 5% of parameters?** — Validate in M9 research phase using a multi-app dataset bootstrapped from HybridAnnotator.

## V2 Roadmap: Self-Improving Pipeline

V2 extends Gazefy from a "train once, deploy" model to a continuously improving system. The system generates its own training signal through live operation and VLM-based annotation, reducing the long-term maintenance burden to near zero.

### V2 Use Cases

| Use Case | Mechanism |
|----------|-----------|
| Pack confidence degrades after UI update | DriftMonitor detects drop → auto re-annotate → fine-tune → hot-swap |
| New UI state never seen during initial training | ActiveLearner captures low-confidence detections → VLM labels them → fine-tune |
| Operator wants to add a new application pack with minimal effort | HybridAnnotator runs on 50 screenshots → VLM pseudo-labels → LoRA fine-tune in ~30 min |
| System learns which clicks reliably produce screen changes | RewardBuffer accumulates confirmed-click frames → positive training set grows automatically |

### V2 Milestones

| Milestone | Description |
|-----------|-------------|
| M8a | `RewardBuffer`: log confirmed/failed clicks with frames for training |
| M8b | `AnnotationConverter`: transform annotations.jsonl → YOLO training format |
| M8c | `DriftMonitor`: rolling confidence tracking + manual and automatic trigger |
| M8d | `ActiveLearner`: uncertainty threshold sampling + batched VLM labeling |
| M8e | `AutoTrainer`: wire all four into an end-to-end fine-tune pipeline |
| M9 | LoRA adapter framework: universal base model + per-app adapter, bootstrapped from multi-app HybridAnnotator corpus |
