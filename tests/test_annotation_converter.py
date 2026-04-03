"""Tests for AnnotationConverter: annotations.jsonl + video → YOLO dataset."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from gazefy.training.annotation_converter import (
    DEFAULT_CLASSES,
    AnnotationConverter,
    ConvertResult,
    _element_to_yolo,
    _normalise_class,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRAME = np.zeros((100, 200, 3), dtype=np.uint8)  # 200w × 100h black frame


def _make_annotation(t: float, elements: list[dict], action: str | None = None) -> dict:
    return {
        "t": t,
        "mouse_x": 10,
        "mouse_y": 10,
        "action": action,
        "elements": elements,
    }


def _make_element(
    label: str = "OK",
    cls: str = "button",
    bbox: list[int] | None = None,
    source: str = "ocr",
) -> dict:
    return {
        "label": label,
        "class": cls,
        "bbox": bbox or [10, 10, 50, 30],
        "source": source,
    }


def _write_annotations(path: Path, entries: list[dict]) -> None:
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _mock_cap(frame: np.ndarray = _FRAME, total_frames: int = 30, fps: float = 10.0):
    """Return a mock cv2.VideoCapture that always returns the given frame."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        3: float(frame.shape[1]),  # CAP_PROP_FRAME_WIDTH
        4: float(frame.shape[0]),  # CAP_PROP_FRAME_HEIGHT
        7: float(total_frames),  # CAP_PROP_FRAME_COUNT
        5: fps,  # CAP_PROP_FPS
    }.get(prop, 0.0)
    cap.read.return_value = (True, frame.copy())
    return cap


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_normalise_class_alias():
    assert _normalise_class("menu_item") == "menu"
    assert _normalise_class("input_field") == "input"
    assert _normalise_class("combobox") == "dropdown"


def test_normalise_class_passthrough():
    assert _normalise_class("button") == "button"
    assert _normalise_class("icon") == "icon"


def test_element_to_yolo_basic():
    class_index = {c: i for i, c in enumerate(DEFAULT_CLASSES)}
    el = _make_element(label="Save", cls="button", bbox=[0, 0, 100, 50])
    line = _element_to_yolo(
        el, img_w=200, img_h=100, class_index=class_index, min_bbox_px=4, skip_unknown=True
    )
    assert line is not None
    parts = line.split()
    assert len(parts) == 5
    class_id = int(parts[0])
    assert class_id == class_index["button"]
    cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    assert abs(cx - 0.25) < 1e-4  # (0+100)/2 / 200
    assert abs(cy - 0.25) < 1e-4  # (0+50)/2 / 100
    assert abs(nw - 0.5) < 1e-4  # 100/200
    assert abs(nh - 0.5) < 1e-4  # 50/100


def test_element_to_yolo_skips_unknown():
    class_index = {c: i for i, c in enumerate(DEFAULT_CLASSES)}
    el = _make_element(label="unknown icon", cls="icon")
    assert _element_to_yolo(el, 200, 100, class_index, 4, skip_unknown=True) is None


def test_element_to_yolo_keeps_unknown_when_disabled():
    class_index = {c: i for i, c in enumerate(DEFAULT_CLASSES)}
    el = _make_element(label="unknown icon", cls="icon")
    line = _element_to_yolo(el, 200, 100, class_index, 4, skip_unknown=False)
    assert line is not None


def test_element_to_yolo_skips_tiny_bbox():
    class_index = {c: i for i, c in enumerate(DEFAULT_CLASSES)}
    el = _make_element(bbox=[10, 10, 12, 12])  # 2×2 px
    assert _element_to_yolo(el, 200, 100, class_index, min_bbox_px=4, skip_unknown=True) is None


def test_element_to_yolo_unknown_class_maps_to_other():
    class_index = {c: i for i, c in enumerate(DEFAULT_CLASSES)}
    el = _make_element(label="something", cls="totally_custom_class")
    line = _element_to_yolo(el, 200, 100, class_index, 4, skip_unknown=True)
    assert line is not None
    assert line.startswith(str(class_index["other"]))


# ---------------------------------------------------------------------------
# Integration tests: convert_session
# ---------------------------------------------------------------------------


@patch("gazefy.training.annotation_converter.cv2.VideoCapture")
def test_convert_session_basic(mock_vc_cls, tmp_path):
    mock_vc_cls.return_value = _mock_cap()

    session = tmp_path / "session"
    session.mkdir()
    _write_annotations(
        session / "annotations.jsonl",
        [
            _make_annotation(
                1.0,
                [
                    _make_element("Save", "button", [10, 10, 60, 40]),
                    _make_element("Cancel", "button", [70, 10, 120, 40]),
                ],
                action="click_left",
            ),
            _make_annotation(
                2.0,
                [_make_element("File", "menu", [0, 0, 40, 20])],
            ),
        ],
    )
    (session / "video.mp4").write_bytes(b"fake")

    result = AnnotationConverter().convert_session(session)

    assert isinstance(result, ConvertResult)
    assert result.n_images == 2
    assert result.n_labels == 2
    assert result.n_elements == 3
    assert result.n_skipped == 0
    assert (result.output_dir / "images").exists()
    assert (result.output_dir / "labels").exists()
    assert len(list((result.output_dir / "images").glob("*.png"))) == 2
    assert len(list((result.output_dir / "labels").glob("*.txt"))) == 2


@patch("gazefy.training.annotation_converter.cv2.VideoCapture")
def test_convert_session_dataset_yaml(mock_vc_cls, tmp_path):
    mock_vc_cls.return_value = _mock_cap()

    session = tmp_path / "session"
    session.mkdir()
    _write_annotations(
        session / "annotations.jsonl",
        [_make_annotation(0.5, [_make_element()])],
    )
    (session / "video.mp4").write_bytes(b"fake")

    result = AnnotationConverter().convert_session(session)

    ds = yaml.safe_load(result.dataset_yaml.read_text())
    assert "names" in ds
    assert ds["train"] == "images/train"
    assert ds["val"] == "images/val"
    assert 0 in ds["names"]
    assert ds["names"][0] == "button"


@patch("gazefy.training.annotation_converter.cv2.VideoCapture")
def test_convert_session_skips_unknown_elements(mock_vc_cls, tmp_path):
    mock_vc_cls.return_value = _mock_cap()

    session = tmp_path / "session"
    session.mkdir()
    _write_annotations(
        session / "annotations.jsonl",
        [
            _make_annotation(
                1.0,
                [
                    _make_element("unknown icon", "icon"),  # should be skipped
                    _make_element("Submit", "button"),  # kept
                ],
            )
        ],
    )
    (session / "video.mp4").write_bytes(b"fake")

    result = AnnotationConverter(skip_unknown=True).convert_session(session)
    assert result.n_elements == 1


@patch("gazefy.training.annotation_converter.cv2.VideoCapture")
def test_convert_session_empty_annotations(mock_vc_cls, tmp_path):
    mock_vc_cls.return_value = _mock_cap()

    session = tmp_path / "session"
    session.mkdir()
    (session / "annotations.jsonl").write_text("")
    (session / "video.mp4").write_bytes(b"fake")

    result = AnnotationConverter().convert_session(session)
    assert result.n_images == 0


@patch("gazefy.training.annotation_converter.cv2.VideoCapture")
def test_convert_session_custom_output_dir(mock_vc_cls, tmp_path):
    mock_vc_cls.return_value = _mock_cap()

    session = tmp_path / "session"
    session.mkdir()
    out = tmp_path / "my_dataset"
    _write_annotations(
        session / "annotations.jsonl",
        [_make_annotation(0.5, [_make_element()])],
    )
    (session / "video.mp4").write_bytes(b"fake")

    result = AnnotationConverter().convert_session(session, output_dir=out)
    assert result.output_dir == out
    assert out.exists()


@patch("gazefy.training.annotation_converter.cv2.VideoCapture")
def test_convert_session_custom_classes(mock_vc_cls, tmp_path):
    mock_vc_cls.return_value = _mock_cap()

    session = tmp_path / "session"
    session.mkdir()
    _write_annotations(
        session / "annotations.jsonl",
        [_make_annotation(0.5, [_make_element("Widget", "myclass")])],
    )
    (session / "video.mp4").write_bytes(b"fake")

    result = AnnotationConverter().convert_session(session, class_names=["myclass", "other"])
    assert result.classes == ["myclass", "other"]
    lbl_file = next((result.output_dir / "labels").glob("*.txt"))
    # "myclass" → class_id 0
    assert lbl_file.read_text().startswith("0 ")


def test_convert_session_missing_annotations(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "video.mp4").write_bytes(b"fake")
    with pytest.raises(FileNotFoundError, match="annotations.jsonl"):
        AnnotationConverter().convert_session(session)


def test_convert_session_missing_video(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "annotations.jsonl").write_text("")
    with pytest.raises(FileNotFoundError, match="video.mp4"):
        AnnotationConverter().convert_session(session)
