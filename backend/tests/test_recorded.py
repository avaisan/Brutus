"""Regression tests on real Rekognition responses recorded from tests/test_data/images.

Re-record with real AWS access if Rekognition's label model changes; see
tests/fixtures/rekognition/README.md.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from botocore.stub import ANY, Stubber
from mypy_boto3_rekognition import RekognitionClient

from brutus.vision.models import DetailLevel, Thresholds
from brutus.vision.pipeline import analyze

FIXTURES = Path(__file__).parent / "fixtures" / "rekognition"


def recorded(name: str, kind: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.{kind}.json").read_text())
    return data


@pytest.mark.parametrize(
    ("image", "people", "animals", "objects"),
    [
        ("animal", 0, ["Dog"], {}),
        ("multiple_people", 3, [], {}),
        ("object", 0, [], {"Airplane": 1}),
        ("one_person", 1, [], {"Jeans": 1}),
        ("one_person_and_object", 1, [], {"Car": 1, "Wheel": 3}),
        ("person_and_dog", 1, ["Dog"], {}),
    ],
)
def test_given_recorded_responses_when_analyzed_then_expected_counts(
    rekognition: RekognitionClient,
    rekognition_stub: Stubber,
    image: str,
    people: int,
    animals: list[str],
    objects: dict[str, int],
) -> None:
    rekognition_stub.add_response(
        "detect_labels",
        recorded(image, "labels"),
        {"Image": {"Bytes": ANY}, "MinConfidence": 90.0},
    )
    rekognition_stub.add_response(
        "detect_faces",
        recorded(image, "faces"),
        {"Image": {"Bytes": ANY}, "Attributes": ["DEFAULT"]},
    )

    result = analyze(b"jpeg", rekognition, (), DetailLevel.BASIC, Thresholds())

    assert result.people_count == people
    assert [a.name for a in result.animals] == animals
    assert result.object_counts == objects
