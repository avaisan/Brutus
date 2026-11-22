"""End-to-end checks against real AWS Rekognition (run with `-m integration`)."""

from pathlib import Path

import boto3
import pytest
from mypy_boto3_rekognition import RekognitionClient

from brutus.settings import Settings
from brutus.store import load_references
from brutus.vision.imaging import encode_jpeg, normalize
from brutus.vision.models import Analysis, DetailLevel, Thresholds
from brutus.vision.pipeline import analyze

pytestmark = pytest.mark.integration

IMAGES = Path(__file__).parent / "test_data" / "images"


@pytest.fixture(scope="module")
def live_rekognition() -> RekognitionClient:
    settings = Settings()
    session = boto3.Session(
        profile_name=settings.aws_profile, region_name=settings.aws_region
    )
    return session.client("rekognition")


def analyze_file(client: RekognitionClient, name: str) -> Analysis:
    jpeg = encode_jpeg(normalize((IMAGES / name).read_bytes(), max_edge=2048))
    references = load_references(IMAGES / "reference")
    return analyze(jpeg, client, references, DetailLevel.STANDARD, Thresholds())


def test_given_multiple_people_photo_when_analyzed_then_three_people(
    live_rekognition: RekognitionClient,
) -> None:
    result = analyze_file(live_rekognition, "multiple_people.jpg")

    assert result.people_count == 3
    assert result.animal_count == 0


def test_given_person_and_dog_photo_when_analyzed_then_one_each(
    live_rekognition: RekognitionClient,
) -> None:
    result = analyze_file(live_rekognition, "person_and_dog.jpg")

    assert result.people_count == 1
    assert result.animal_count == 1


def test_given_animal_photo_when_analyzed_then_no_people(
    live_rekognition: RekognitionClient,
) -> None:
    result = analyze_file(live_rekognition, "animal.jpg")

    assert result.people_count == 0
    assert result.animal_count == 1


def test_given_reference_person_photo_when_analyzed_then_recognized(
    live_rekognition: RekognitionClient,
) -> None:
    result = analyze_file(live_rekognition, "one_person.jpg")

    assert result.people_count == 1
    assert result.people[0].identity is not None
    assert result.people[0].identity.name == "Michael Knight"
