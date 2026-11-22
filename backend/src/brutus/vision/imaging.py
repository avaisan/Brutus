"""Image decoding and encoding (pure: bytes in, image or bytes out)."""

import io

from PIL import Image, ImageOps

REKOGNITION_MAX_BYTES = 5 * 1024 * 1024
JPEG_QUALITIES = (90, 80, 70, 60, 50)


def normalize(data: bytes, *, max_edge: int) -> Image.Image:
    """Decode, rotate upright from EXIF orientation, convert to RGB and downscale.

    Drawing and Rekognition both use this image, so boxes line up with the pixels.
    """
    with Image.open(io.BytesIO(data)) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image.thumbnail((max_edge, max_edge))
    return image


def _jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def encode_jpeg(image: Image.Image, *, max_bytes: int = REKOGNITION_MAX_BYTES) -> bytes:
    """Encode as JPEG at the best quality that fits in `max_bytes`."""
    encoded = (_jpeg(image, quality) for quality in JPEG_QUALITIES)
    if (fitting := next((d for d in encoded if len(d) <= max_bytes), None)) is not None:
        return fitting
    smaller = image.resize((image.width * 3 // 4, image.height * 3 // 4))
    return encode_jpeg(smaller, max_bytes=max_bytes)
