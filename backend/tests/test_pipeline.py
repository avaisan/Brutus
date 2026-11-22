from typing import Any

from botocore.stub import ANY, Stubber
from mypy_boto3_rekognition import RekognitionClient

from brutus.vision.models import DetailLevel, Thresholds
from brutus.vision.pipeline import Reference, analyze
from tests.images import jpeg_bytes
from tests.rekognition_fixtures import (
    compare_response,
    default_face,
    dog,
    face,
    faces_response,
    label,
    labels_response,
    person,
)

IMAGE = jpeg_bytes()
LEFT, RIGHT = (0.05, 0.1, 0.2, 0.8), (0.6, 0.1, 0.2, 0.8)
LEFT_FACE, RIGHT_FACE = (0.1, 0.12, 0.08, 0.1), (0.65, 0.12, 0.08, 0.1)
REFERENCES = (
    Reference(name="Kirk", jpeg=b"kirk"),
    Reference(name="Spock", jpeg=b"spock"),
)

TWO_PEOPLE_AND_DOG = labels_response(
    person(RIGHT, LEFT),
    dog((0.4, 0.6, 0.15, 0.3)),
    label("Chair", ((0.85, 0.6, 0.1, 0.3),)),
    label("Indoors"),
)


def expect_labels(stub: Stubber, response: dict[str, Any]) -> None:
    stub.add_response(
        "detect_labels", response, {"Image": {"Bytes": ANY}, "MinConfidence": 90.0}
    )


def expect_faces(stub: Stubber, response: dict[str, Any], attributes: str) -> None:
    stub.add_response(
        "detect_faces", response, {"Image": {"Bytes": ANY}, "Attributes": [attributes]}
    )


def expect_compare(stub: Stubber, source: bytes, response: dict[str, Any]) -> None:
    stub.add_response(
        "compare_faces",
        response,
        {
            "SourceImage": {"Bytes": source},
            "TargetImage": {"Bytes": ANY},
            "SimilarityThreshold": 85.0,
        },
    )


def run(client: RekognitionClient, detail: DetailLevel) -> Any:
    return analyze(IMAGE, client, REFERENCES, detail, Thresholds())


class TestAnalyze:
    def test_given_standard_detail_with_faces_when_analyzed_then_one_pass_plus_compares(
        self, rekognition: RekognitionClient, rekognition_stub: Stubber
    ) -> None:
        expect_labels(rekognition_stub, TWO_PEOPLE_AND_DOG)
        expect_faces(
            rekognition_stub,
            faces_response(default_face(LEFT_FACE), default_face(RIGHT_FACE)),
            "DEFAULT",
        )
        expect_compare(rekognition_stub, b"kirk", compare_response((RIGHT_FACE, 99.0)))
        expect_compare(rekognition_stub, b"spock", compare_response())

        result = run(rekognition, DetailLevel.STANDARD)

        assert result.people_count == 2 == len(result.people)
        assert [p.identity.name if p.identity else None for p in result.people] == [
            None,
            "Kirk",
        ]
        assert result.animal_count == 1
        assert result.object_counts == {"Chair": 1}
        assert result.labels == ("Indoors",)
        assert all(
            p.face is not None and p.face.attributes is None for p in result.people
        )

    def test_given_reference_without_face_when_analyzed_then_skipped(
        self, rekognition: RekognitionClient, rekognition_stub: Stubber
    ) -> None:
        expect_labels(rekognition_stub, labels_response(person(RIGHT)))
        expect_faces(
            rekognition_stub, faces_response(default_face(RIGHT_FACE)), "DEFAULT"
        )
        rekognition_stub.add_client_error(
            "compare_faces", service_error_code="InvalidParameterException"
        )
        expect_compare(rekognition_stub, b"spock", compare_response((RIGHT_FACE, 95.0)))

        result = run(rekognition, DetailLevel.STANDARD)

        (only,) = result.people
        assert only.identity is not None
        assert only.identity.name == "Spock"

    def test_given_basic_detail_when_analyzed_then_no_face_comparison(
        self, rekognition: RekognitionClient, rekognition_stub: Stubber
    ) -> None:
        expect_labels(rekognition_stub, TWO_PEOPLE_AND_DOG)
        expect_faces(
            rekognition_stub, faces_response(default_face(LEFT_FACE)), "DEFAULT"
        )

        result = run(rekognition, DetailLevel.BASIC)

        assert result.people_count == 2
        assert all(p.identity is None for p in result.people)

    def test_given_no_faces_when_analyzed_then_no_face_comparison(
        self, rekognition: RekognitionClient, rekognition_stub: Stubber
    ) -> None:
        expect_labels(rekognition_stub, labels_response(dog((0.4, 0.6, 0.15, 0.3))))
        expect_faces(rekognition_stub, faces_response(), "DEFAULT")

        result = run(rekognition, DetailLevel.STANDARD)

        assert result.people_count == 0
        assert result.animal_count == 1

    def test_given_detailed_when_analyzed_then_all_face_attributes(
        self, rekognition: RekognitionClient, rekognition_stub: Stubber
    ) -> None:
        expect_labels(rekognition_stub, labels_response(person(LEFT)))
        expect_faces(
            rekognition_stub, faces_response(face(LEFT_FACE, smile=True)), "ALL"
        )
        expect_compare(rekognition_stub, b"kirk", compare_response())
        expect_compare(rekognition_stub, b"spock", compare_response())

        result = run(rekognition, DetailLevel.DETAILED)

        (only,) = result.people
        assert only.face is not None and only.face.attributes is not None
        assert only.face.attributes.smile is True
        assert result.detail is DetailLevel.DETAILED
