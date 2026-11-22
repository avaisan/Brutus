"""Helpers that create small in-memory JPEGs for tests."""

import io

from PIL import Image

ORIENTATION_TAG = 0x0112


def jpeg_bytes(
    width: int = 200,
    height: int = 100,
    *,
    color: str = "white",
    orientation: int | None = None,
    quality: int = 90,
) -> bytes:
    image = Image.new("RGB", (width, height), color)
    exif = Image.Exif()
    if orientation is not None:
        exif[ORIENTATION_TAG] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif, quality=quality)
    return buffer.getvalue()


def noisy_jpeg_bytes(width: int, height: int) -> bytes:
    """Random noise compresses badly, which makes a large JPEG."""
    image = Image.effect_noise((width, height), 100).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=100)
    return buffer.getvalue()
