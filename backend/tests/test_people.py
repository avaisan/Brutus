from collections.abc import Sequence

from brutus.vision.models import BoundingBox, Face, FaceMatch, Person
from brutus.vision.people import combine_people


def box(left: float, top: float, width: float, height: float) -> BoundingBox:
    return BoundingBox(left=left, top=top, width=width, height=height)


def face_at(b: BoundingBox) -> Face:
    return Face(box=b, confidence=99.0)


def combine(
    persons: Sequence[BoundingBox],
    faces: Sequence[Face],
    matches: Sequence[FaceMatch] = (),
) -> tuple[Person, ...]:
    return combine_people(persons, faces, matches, face_overlap=0.5, identity_iou=0.5)


LEFT, MIDDLE, RIGHT = (
    box(0.05, 0.1, 0.2, 0.8),
    box(0.4, 0.1, 0.2, 0.8),
    box(0.7, 0.1, 0.2, 0.8),
)
LEFT_FACE, MIDDLE_FACE, RIGHT_FACE = (
    box(0.1, 0.12, 0.08, 0.1),
    box(0.45, 0.12, 0.08, 0.1),
    box(0.75, 0.12, 0.08, 0.1),
)


class TestCombinePeople:
    def test_given_three_people_facing_camera_when_combined_then_numbered_left_right(
        self,
    ) -> None:
        people = combine(
            [RIGHT, LEFT, MIDDLE],
            [face_at(MIDDLE_FACE), face_at(RIGHT_FACE), face_at(LEFT_FACE)],
        )

        assert [p.number for p in people] == [1, 2, 3]
        assert [p.box for p in people] == [LEFT, MIDDLE, RIGHT]
        assert [p.face.box if p.face else None for p in people] == [
            LEFT_FACE,
            MIDDLE_FACE,
            RIGHT_FACE,
        ]

    def test_given_one_person_facing_away_when_combined_then_still_counted(
        self,
    ) -> None:
        people = combine([LEFT, RIGHT], [face_at(RIGHT_FACE)])

        assert len(people) == 2
        assert people[0].face is None
        assert people[1].face == face_at(RIGHT_FACE)

    def test_given_close_up_face_without_person_when_combined_then_one_person(
        self,
    ) -> None:
        close_up = box(0.3, 0.2, 0.4, 0.5)

        people = combine([], [face_at(close_up)])

        assert len(people) == 1
        assert people[0].box == close_up
        assert people[0].face == face_at(close_up)

    def test_given_match_box_slightly_off_when_combined_then_identity_assigned(
        self,
    ) -> None:
        shifted = box(0.101, 0.121, 0.081, 0.099)

        people = combine(
            [LEFT, RIGHT],
            [face_at(LEFT_FACE), face_at(RIGHT_FACE)],
            [FaceMatch(name="Kirk", similarity=98.0, box=shifted)],
        )

        assert people[0].identity is not None
        assert people[0].identity.name == "Kirk"
        assert people[1].identity is None

    def test_given_two_faces_matching_same_reference_when_combined_then_best_only(
        self,
    ) -> None:
        people = combine(
            [LEFT, RIGHT],
            [face_at(LEFT_FACE), face_at(RIGHT_FACE)],
            [
                FaceMatch(name="Kirk", similarity=88.0, box=LEFT_FACE),
                FaceMatch(name="Kirk", similarity=97.0, box=RIGHT_FACE),
            ],
        )

        assert people[0].identity is None
        assert people[1].identity is not None
        assert people[1].identity.similarity == 97.0

    def test_given_face_overlapping_two_persons_when_combined_then_larger_overlap_wins(
        self,
    ) -> None:
        a, b = box(0.1, 0.1, 0.3, 0.8), box(0.3, 0.1, 0.3, 0.8)
        between = box(0.32, 0.12, 0.1, 0.1)  # fully inside b, 80% inside a
        left_face = box(0.15, 0.12, 0.08, 0.1)  # only inside a

        people = combine([a, b], [face_at(between), face_at(left_face)])

        assert len(people) == 2
        assert people[0].face == face_at(left_face)
        assert people[1].face == face_at(between)
