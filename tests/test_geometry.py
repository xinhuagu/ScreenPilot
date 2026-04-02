"""Tests for geometry primitives."""

from screenpilot.utils.geometry import Point, Rect, iou


def test_point_basic():
    p = Point(10.0, 20.0)
    assert p.x == 10.0
    assert p.y == 20.0


def test_point_scaled():
    p = Point(100.0, 200.0)
    s = p.scaled(0.5)
    assert s.x == 50.0
    assert s.y == 100.0


def test_point_offset():
    p = Point(10.0, 20.0)
    o = p.offset(5.0, -3.0)
    assert o.x == 15.0
    assert o.y == 17.0


def test_rect_properties():
    r = Rect(10, 20, 110, 70)
    assert r.width == 100
    assert r.height == 50
    assert r.area == 5000
    assert r.center == Point(60, 45)


def test_rect_contains_point():
    r = Rect(0, 0, 100, 100)
    assert r.contains_point(Point(50, 50))
    assert r.contains_point(Point(0, 0))  # boundary
    assert r.contains_point(Point(100, 100))  # boundary
    assert not r.contains_point(Point(101, 50))
    assert not r.contains_point(Point(-1, 50))


def test_rect_contains_rect():
    outer = Rect(0, 0, 200, 200)
    inner = Rect(10, 10, 50, 50)
    assert outer.contains_rect(inner)
    assert not inner.contains_rect(outer)


def test_rect_intersection():
    a = Rect(0, 0, 100, 100)
    b = Rect(50, 50, 150, 150)
    inter = a.intersection(b)
    assert inter == Rect(50, 50, 100, 100)
    assert inter.area == 2500


def test_rect_no_intersection():
    a = Rect(0, 0, 10, 10)
    b = Rect(20, 20, 30, 30)
    inter = a.intersection(b)
    assert inter.area == 0


def test_iou_identical():
    r = Rect(0, 0, 100, 100)
    assert iou(r, r) == 1.0


def test_iou_no_overlap():
    a = Rect(0, 0, 10, 10)
    b = Rect(20, 20, 30, 30)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = Rect(0, 0, 100, 100)
    b = Rect(50, 0, 150, 100)
    # intersection = 50*100 = 5000, union = 10000 + 10000 - 5000 = 15000
    assert abs(iou(a, b) - 5000 / 15000) < 1e-6


def test_rect_scaled():
    r = Rect(10, 20, 30, 40)
    s = r.scaled(2.0)
    assert s == Rect(20, 40, 60, 80)


def test_rect_offset():
    r = Rect(10, 20, 30, 40)
    o = r.offset(5, -5)
    assert o == Rect(15, 15, 35, 35)
