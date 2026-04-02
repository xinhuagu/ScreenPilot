"""Geometric primitives and utilities for UI element coordinate math."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def scaled(self, factor: float) -> Point:
        return Point(self.x * factor, self.y * factor)

    def offset(self, dx: float, dy: float) -> Point:
        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True, slots=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.width) * max(0, self.height)

    def contains_point(self, p: Point) -> bool:
        return self.x1 <= p.x <= self.x2 and self.y1 <= p.y <= self.y2

    def contains_rect(self, other: Rect) -> bool:
        return (
            self.x1 <= other.x1
            and self.y1 <= other.y1
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )

    def intersection(self, other: Rect) -> Rect:
        return Rect(
            max(self.x1, other.x1),
            max(self.y1, other.y1),
            min(self.x2, other.x2),
            min(self.y2, other.y2),
        )

    def scaled(self, factor: float) -> Rect:
        return Rect(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor)

    def offset(self, dx: float, dy: float) -> Rect:
        return Rect(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)


def iou(a: Rect, b: Rect) -> float:
    """Intersection over Union between two rectangles."""
    inter = a.intersection(b)
    inter_area = inter.area
    if inter_area <= 0:
        return 0.0
    union_area = a.area + b.area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
