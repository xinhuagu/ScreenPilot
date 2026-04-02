"""Monitor mode: real-time terminal display of cursor-to-element state.

Usage:
    gazefy monitor --window "Citrix"
    gazefy monitor --pack my_erp --window "Citrix"

Prints a live status line showing what UI element the cursor is hovering over,
plus a periodic summary of all detected elements.
"""

from __future__ import annotations

import logging
import sys
import time

from gazefy.config import CaptureRegion, GazefyConfig
from gazefy.core.orchestrator import Orchestrator
from gazefy.utils.timing import FPSCounter

logger = logging.getLogger(__name__)


def run_monitor(
    region: CaptureRegion,
    pack_name: str = "",
    packs_dir: str = "packs",
    retina_scale: float = 2.0,
    show_all_interval: float = 5.0,
) -> None:
    """Run monitor mode: live cursor-to-element tracking in terminal.

    Args:
        region: Screen region to capture.
        pack_name: Force a specific pack (empty = auto-route or no model).
        packs_dir: Directory containing ApplicationPack artifacts.
        retina_scale: Retina display scale factor.
        show_all_interval: Seconds between full element list dumps.
    """
    config = GazefyConfig(
        region=region,
        retina_scale=retina_scale,
        mode="monitor",
    )

    orch = Orchestrator(config)
    orch.registry = _make_registry(packs_dir)

    # Force pack if specified
    if pack_name:
        orch.registry.scan()
        pack = orch.router.force_pack(pack_name)
        if pack is None:
            print(f"Pack '{pack_name}' not found in {packs_dir}/")
            available = list(orch.registry.packs.keys())
            if available:
                print(f"Available packs: {', '.join(available)}")
            else:
                print("No packs found. Run 'gazefy train' first.")
            sys.exit(1)

    orch.setup()

    ui_map = orch.tracker.current_map
    pack = orch.router.active_pack
    pack_label = pack.metadata.name if pack else "(no pack)"
    has_model = orch.detector is not None and orch.detector.is_loaded

    print("Gazefy Monitor")
    print(f"  Region: ({region.left}, {region.top}) {region.width}x{region.height}")
    print(f"  Pack: {pack_label}")
    print(f"  Model: {'loaded' if has_model else 'none (cursor tracking only)'}")
    print("  Press Ctrl+C to stop.\n")

    fps = FPSCounter()
    last_element_id = ""
    last_full_dump = 0.0
    detect_count = 0

    try:
        while True:
            # --- Run one pipeline step ---
            frame = orch.capture.get_latest_frame()
            if frame is not None:
                change = orch.change_detector.check(frame.image)
                if change.changed and has_model:
                    detections = orch.detector.detect(frame.image)
                    h, w = frame.image.shape[:2]
                    orch.tracker.update(detections, change, frame_width=w, frame_height=h)
                    orch.cursor.set_ui_map(orch.tracker.current_map)
                    detect_count += 1
                fps.tick()

            # --- Read cursor state ---
            state = orch.cursor.state
            ui_map = orch.tracker.current_map
            el = state.current_element

            # --- Live status line ---
            if el:
                text_part = f' "{el.text}"' if el.text else ""
                line = (
                    f"\r  Cursor: [{el.class_name}]{text_part} "
                    f"id={el.id} conf={el.confidence:.2f} "
                    f"dwell={state.dwell_time_ms:.0f}ms"
                )
                # Notify on element change
                if el.id != last_element_id:
                    last_element_id = el.id
                    # Print on new line when element changes
                    sys.stdout.write(f"\n  → [{el.class_name}]{text_part} id={el.id}")
                    sys.stdout.flush()
            else:
                line = (
                    f"\r  Cursor: (no element) "
                    f"pos=({state.screen_position.x:.0f},{state.screen_position.y:.0f})"
                )
                if last_element_id:
                    last_element_id = ""
                    sys.stdout.write("\n  → (left element)")
                    sys.stdout.flush()

            # Overwrite status line
            sys.stdout.write(f"{line:<80}")
            sys.stdout.flush()

            # --- Periodic full dump ---
            now = time.monotonic()
            if show_all_interval and (now - last_full_dump) >= show_all_interval:
                last_full_dump = now
                n = ui_map.element_count
                sys.stdout.write(
                    f"\n  --- [{n} elements | gen={ui_map.generation} | "
                    f"detections={detect_count} | fps={fps.fps:.1f}] ---\n"
                )
                if n > 0:
                    for e in sorted(ui_map.elements.values(), key=lambda x: (x.bbox.y1, x.bbox.x1)):
                        text = f' "{e.text}"' if e.text else ""
                        sys.stdout.write(
                            f"    {e.id:12s} {e.class_name:15s}{text:20s} "
                            f"({e.bbox.x1:.0f},{e.bbox.y1:.0f})-"
                            f"({e.bbox.x2:.0f},{e.bbox.y2:.0f}) "
                            f"conf={e.confidence:.2f} stab={e.stability}\n"
                        )
                sys.stdout.flush()

            time.sleep(0.05)  # 20Hz display refresh

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        orch.shutdown()
        print(f"Session: {detect_count} detection cycles, {ui_map.element_count} final elements")


def _make_registry(packs_dir: str):
    from gazefy.core.model_registry import ModelRegistry

    return ModelRegistry(packs_dir=packs_dir)
