"""Draw analysis results on a copy of the picture."""

from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from brutus.vision.geometry import to_pixels
from brutus.vision.models import Analysis, BoundingBox, DetailLevel, Person

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
    emphasis: bool = False


def _person_mark(person: Person) -> Mark:
    """Face box when there is a face (they line up even when people overlap)."""
    box = person.face.box if person.face else person.box
    if person.identity:
        return Mark(
            box, f"{person.number} {person.identity.name}", COLORS["recognized"], True
        )
    return Mark(box, str(person.number), COLORS["person"])


def marks(analysis: Analysis) -> tuple[Mark, ...]:
    """Everything to draw, back to front: objects, animals, then people.

    Captions are numbers that match the results list; the color tells the kind.
    """
    objects = (
        tuple(Mark(o.box, str(o.number), COLORS["object"]) for o in analysis.objects)
        if analysis.detail is not DetailLevel.BASIC
        else ()
    )
    animals = tuple(
        Mark(a.box, str(a.number), COLORS["animal"]) for a in analysis.animals
    )
    return objects + animals + tuple(_person_mark(p) for p in analysis.people)


def _draw(
    draw: ImageDraw.ImageDraw, size: tuple[int, int], font: Font, mark: Mark
) -> None:
    line = max(2, min(size) // 250)
    left, top, right, bottom = to_pixels(mark.box, *size)
    draw.rectangle(
        (left, top, right, bottom),
        outline=mark.color,
        width=line * 2 if mark.emphasis else line,
    )

    # Small black tag with the caption in the box color, above the box when it fits.
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (0, 0), mark.caption, font=font
    )
    padding = max(2, line)
    width = text_right - text_left + 2 * padding
    height = text_bottom - text_top + 2 * padding
    x = min(left, size[0] - width)
    y = top - height if top >= height else top
    draw.rectangle((x, y, x + width, y + height), fill="black")
    draw.text(
        (x + padding - text_left, y + padding - text_top),
        mark.caption,
        fill=mark.color,
        font=font,
    )


def annotate(image: Image.Image, analysis: Analysis) -> Image.Image:
    """Return a new image with boxes and captions; `image` is left unchanged."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, min(canvas.size) // 30))
    for mark in marks(analysis):
        _draw(draw, canvas.size, font, mark)
    return canvas
