import pytest

from brutus.vision.models import BoundingBox
from brutus.vision.parse import (
    parse_face_matches,
    parse_faces,
    parse_label_instances,
    parse_scene_labels,
)
from tests.rekognition_fixtures import (
    compare_response,
    default_face,
    face,
    faces_response,
    label,
    labels_response,
    person,
)


class TestParseLabels:
    def test_given_two_people_and_scene_label_when_parsed_then_instances_and_scene(
        self,
    ) -> None:
        response = labels_response(
            person((0.1, 0.1, 0.2, 0.8), (0.6, 0.1, 0.2, 0.8)), label("Indoors")
        )

        instances = parse_label_instances(response)

        assert [i.name for i in instances] == ["Person", "Person"]
        assert instances[1].box == BoundingBox(left=0.6, top=0.1, width=0.2, height=0.8)
        assert "Person Description" in instances[0].categories
        assert parse_scene_labels(response) == ("Indoors",)

    def test_given_box_with_negative_left_when_parsed_then_clamped(self) -> None:
        response = labels_response(label("Chair", ((-0.01, 0.2, 0.3, 0.3),)))

        (chair,) = parse_label_instances(response)

        assert chair.box.left == 0.0
        assert chair.box.width == pytest.approx(0.29)


class TestParseFaces:
    def test_given_all_attributes_when_parsed_detailed_then_attributes_without_gender(
        self,
    ) -> None:
        response = faces_response(face((0.1, 0.1, 0.1, 0.1), smile=True))

        (parsed,) = parse_faces(response, min_confidence=90, detailed=True)

        assert parsed.attributes is not None
        assert parsed.attributes.smile is True
        assert parsed.attributes.emotion == "CALM"
        assert (parsed.attributes.age_low, parsed.attributes.age_high) == (30, 40)
        assert "gender" not in parsed.attributes.model_dump()

    def test_given_default_faces_when_parsed_not_detailed_then_no_attributes(
        self,
    ) -> None:
        response = faces_response(default_face((0.1, 0.1, 0.1, 0.1)))

        (parsed,) = parse_faces(response, min_confidence=90, detailed=False)

        assert parsed.attributes is None

    def test_given_low_confidence_face_when_parsed_then_dropped(self) -> None:
        response = faces_response(
            default_face((0.1, 0.1, 0.1, 0.1), confidence=99),
            default_face((0.5, 0.5, 0.01, 0.01), confidence=60),
        )

        assert len(parse_faces(response, min_confidence=90, detailed=False)) == 1


class TestParseFaceMatches:
    def test_given_compare_response_when_parsed_then_named_matches(self) -> None:
        response = compare_response(((0.1, 0.1, 0.1, 0.1), 97.5))

        (match,) = parse_face_matches("Kirk", response)

        assert match.name == "Kirk"
        assert match.similarity == 97.5
        assert match.box.left == 0.1
