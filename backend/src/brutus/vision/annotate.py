"""Draw analysis results on a copy of the picture."""

from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from brutus.vision.geometry import to_pixels
from brutus.vision.models import Analysis, BoundingBox, DetailLevel

COLORS = {
    "recognized": "orange",
    "person": "red",
    "animal": "blue",
    "object": "green",
}

type Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


class Mark(NamedTuple):
    """One box and its caption."""

    box: BoundingBox
    caption: str
    color: str
    line_width: int


def marks(analysis: Analysis) -> tuple[Mark, ...]:
    """Everything to draw, back to front: objects, animals, then people."""
    objects = (
        tuple(
            Mark(o.box, f"O{o.number} {o.name}", COLORS["object"], 2)
            for o in analysis.objects
        )
        if analysis.detail is not DetailLevel.BASIC
        else ()
    )
    animals = tuple(
        Mark(a.box, f"A{a.number} {a.name}", COLORS["animal"], 3)
        for a in analysis.animals
    )
    people = tuple(
        Mark(p.box, f"P{p.number} {p.identity.name}", COLORS["recognized"], 5)
        if p.identity
        else Mark(p.box, f"P{p.number}", COLORS["person"], 3)
        for p in analysis.people
    )
    return objects + animals + people


def _draw(
    draw: ImageDraw.ImageDraw, size: tuple[int, int], font: Font, mark: Mark
) -> None:
    left, top, right, bottom = to_pixels(mark.box, *size)
    draw.rectangle(
        (left, top, right, bottom), outline=mark.color, width=mark.line_width
    )

    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (0, 0), mark.caption, font=font
    )
    text_width, text_height = text_right - text_left, text_bottom - text_top
    padding = 3
    x = min(left, size[0] - text_width - 2 * padding)
    above = top - text_height - 2 * padding
    y = above if above >= 0 else top
    draw.rectangle(
        (x, y, x + text_width + 2 * padding, y + text_height + 2 * padding),
        fill=mark.color,
    )
    draw.text(
        (x + padding - text_left, y + padding - text_top),
        mark.caption,
        fill="white",
        font=font,
    )


def annotate(image: Image.Image, analysis: Analysis) -> Image.Image:
    """Return a new image with boxes and captions; `image` is left unchanged."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(14, canvas.height // 30))
    for mark in marks(analysis):
        _draw(draw, canvas.size, font, mark)
    return canvas
