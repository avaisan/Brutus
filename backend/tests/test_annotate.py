from datetime import UTC, datetime

from PIL import Image

from brutus.vision.annotate import COLORS, annotate
from brutus.vision.models import (
    Analysis,
    BoundingBox,
    DetailLevel,
    Detection,
    Identity,
    Person,
)

KNOWN = BoundingBox(left=0.1, top=0.2, width=0.2, height=0.6)
UNKNOWN = BoundingBox(left=0.6, top=0.2, width=0.2, height=0.6)
CHAIR = BoundingBox(left=0.4, top=0.7, width=0.1, height=0.2)


def analysis(detail: DetailLevel) -> Analysis:
    return Analysis(
        id="test",
        created_at=datetime(2026, 11, 22, tzinfo=UTC),
        detail=detail,
        people=(
            Person(
                number=1, box=KNOWN, identity=Identity(name="Kirk", similarity=99.0)
            ),
            Person(number=2, box=UNKNOWN),
        ),
        animals=(),
        objects=(Detection(number=1, name="Chair", confidence=95.0, box=CHAIR),),
        labels=(),
    )


def rgb(name: str) -> tuple[int, int, int]:
    color = Image.new("RGB", (1, 1), COLORS[name]).getpixel((0, 0))
    assert isinstance(color, tuple)
    return color[0], color[1], color[2]


class TestAnnotate:
    def test_given_known_and_unknown_person_when_annotated_then_colored_boxes(
        self,
    ) -> None:
        image = Image.new("RGB", (1000, 500), "white")

        result = annotate(image, analysis(DetailLevel.STANDARD))

        # Right edge pixels, away from the label text at the top.
        assert result.getpixel((299, 300)) == rgb("recognized")
        assert result.getpixel((799, 300)) == rgb("person")
        assert image.getpixel((299, 300)) == (255, 255, 255), "input must not change"

    def test_given_basic_detail_when_annotated_then_objects_not_drawn(self) -> None:
        image = Image.new("RGB", (1000, 500), "white")
        chair_right_edge = (499, 400)

        basic = annotate(image, analysis(DetailLevel.BASIC))
        standard = annotate(image, analysis(DetailLevel.STANDARD))

        assert basic.getpixel(chair_right_edge) == (255, 255, 255)
        assert standard.getpixel(chair_right_edge) == rgb("object")
