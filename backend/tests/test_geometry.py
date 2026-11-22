import pytest

from brutus.vision.geometry import intersection_over_area, iou, to_pixels
from brutus.vision.models import BoundingBox


def box(left: float, top: float, width: float, height: float) -> BoundingBox:
    return BoundingBox(left=left, top=top, width=width, height=height)


class TestIou:
    def test_given_boxes_overlapping_by_half_when_iou_then_one_third(self) -> None:
        assert iou(box(0, 0, 0.2, 0.2), box(0.1, 0, 0.2, 0.2)) == pytest.approx(1 / 3)

    def test_given_disjoint_boxes_when_iou_then_zero(self) -> None:
        assert iou(box(0, 0, 0.1, 0.1), box(0.5, 0.5, 0.1, 0.1)) == 0.0

    def test_given_identical_boxes_when_iou_then_one(self) -> None:
        assert iou(box(0.2, 0.2, 0.3, 0.3), box(0.2, 0.2, 0.3, 0.3)) == pytest.approx(
            1.0
        )


class TestIntersectionOverArea:
    def test_given_face_inside_person_when_measured_then_full_overlap(self) -> None:
        face, person = box(0.3, 0.1, 0.1, 0.1), box(0.2, 0.0, 0.4, 0.9)
        assert intersection_over_area(face, person) == pytest.approx(1.0)

    def test_given_face_half_outside_when_measured_then_half(self) -> None:
        face, person = box(0.5, 0.0, 0.2, 0.2), box(0.0, 0.0, 0.6, 1.0)
        assert intersection_over_area(face, person) == pytest.approx(0.5)


class TestToPixels:
    def test_given_ratio_box_when_converted_then_pixel_corners(self) -> None:
        assert to_pixels(box(0.1, 0.2, 0.5, 0.5), 1000, 500) == (100, 100, 600, 350)

    def test_given_box_past_edge_when_converted_then_clamped(self) -> None:
        assert to_pixels(box(0.8, 0.8, 0.5, 0.5), 1000, 500) == (800, 400, 999, 499)
