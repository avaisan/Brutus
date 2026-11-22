"""Pure geometry on ratio-based bounding boxes."""

from brutus.vision.models import BoundingBox


def area(box: BoundingBox) -> float:
    """Area of a box."""
    return box.width * box.height


def intersection(a: BoundingBox, b: BoundingBox) -> float:
    """Area shared by two boxes."""
    width = min(a.left + a.width, b.left + b.width) - max(a.left, b.left)
    height = min(a.top + a.height, b.top + b.height) - max(a.top, b.top)
    return max(width, 0.0) * max(height, 0.0)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union, from 0 (disjoint) to 1 (identical)."""
    shared = intersection(a, b)
    union = area(a) + area(b) - shared
    return shared / union if union > 0 else 0.0


def intersection_over_area(inner: BoundingBox, outer: BoundingBox) -> float:
    """Share of `inner` that lies inside `outer`."""
    inner_area = area(inner)
    return intersection(inner, outer) / inner_area if inner_area > 0 else 0.0


def clamp(box: BoundingBox) -> BoundingBox:
    """Trim a box to the image (Rekognition may return slightly negative values)."""
    inside = (
        box.left >= 0
        and box.top >= 0
        and box.left + box.width <= 1
        and box.top + box.height <= 1
    )
    if inside:
        return box
    left, top = max(box.left, 0.0), max(box.top, 0.0)
    right = min(box.left + box.width, 1.0)
    bottom = min(box.top + box.height, 1.0)
    return BoundingBox(
        left=left, top=top, width=max(right - left, 0.0), height=max(bottom - top, 0.0)
    )


def to_pixels(box: BoundingBox, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert to (left, top, right, bottom) pixels, clamped inside the image."""

    def scale(ratio: float, size: int) -> int:
        return min(max(round(ratio * size), 0), size - 1)

    return (
        scale(box.left, width),
        scale(box.top, height),
        scale(box.left + box.width, width),
        scale(box.top + box.height, height),
    )
