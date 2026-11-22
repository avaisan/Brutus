from collections.abc import Iterator

import boto3
import pytest
from botocore.stub import Stubber
from mypy_boto3_polly import PollyClient
from mypy_boto3_rekognition import RekognitionClient

FAKE_CREDENTIALS = {
    "region_name": "eu-central-1",
    "aws_access_key_id": "testing",
    "aws_secret_access_key": "testing",
}


@pytest.fixture
def rekognition() -> RekognitionClient:
    return boto3.client("rekognition", **FAKE_CREDENTIALS)  # type: ignore[call-overload,no-any-return]


@pytest.fixture
def polly() -> PollyClient:
    return boto3.client("polly", **FAKE_CREDENTIALS)  # type: ignore[call-overload,no-any-return]


@pytest.fixture
def rekognition_stub(rekognition: RekognitionClient) -> Iterator[Stubber]:
    with Stubber(rekognition) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


@pytest.fixture
def polly_stub(polly: PollyClient) -> Iterator[Stubber]:
    with Stubber(polly) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()
