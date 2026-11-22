import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber
from fastapi.testclient import TestClient
from mypy_boto3_polly import PollyClient
from mypy_boto3_rekognition import RekognitionClient

from brutus.api import create_app
from brutus.settings import Settings
from tests.images import jpeg_bytes
from tests.rekognition_fixtures import (
    default_face,
    faces_response,
    labels_response,
    person,
)

IMAGE = jpeg_bytes(400, 300)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        reference_dir=tmp_path / "reference_library",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture
def client(
    settings: Settings, rekognition: RekognitionClient, polly: PollyClient
) -> Iterator[TestClient]:
    app = create_app(settings, rekognition=rekognition, polly=polly)
    with TestClient(app) as test_client:
        yield test_client


def expect_one_person(stub: Stubber, attributes: str = "DEFAULT") -> None:
    stub.add_response(
        "detect_labels",
        labels_response(person((0.1, 0.1, 0.3, 0.8))),
        {"Image": {"Bytes": ANY}, "MinConfidence": 90.0},
    )
    stub.add_response(
        "detect_faces",
        faces_response(default_face((0.15, 0.12, 0.1, 0.1))),
        {"Image": {"Bytes": ANY}, "Attributes": [attributes]},
    )


class TestHealth:
    def test_given_running_app_when_health_checked_then_healthy(
        self, client: TestClient
    ) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "Healthy"}


class TestUploadImage:
    def test_given_jpeg_when_uploaded_then_analysis_returned_and_stored(
        self, client: TestClient, rekognition_stub: Stubber, settings: Settings
    ) -> None:
        expect_one_person(rekognition_stub)

        response = client.post(
            "/images", files={"file": ("p.jpg", IMAGE, "image/jpeg")}
        )

        assert response.status_code == 201
        body = response.json()
        assert body["people_count"] == 1
        assert body["detail"] == "standard"
        assert (settings.output_dir / "analysis.json").exists()
        assert (settings.output_dir / "annotated.jpg").exists()
        assert (settings.input_dir / "latest.jpg").exists()

    def test_given_png_content_type_when_uploaded_then_unsupported_media_type(
        self, client: TestClient
    ) -> None:
        response = client.post("/images", files={"file": ("p.png", IMAGE, "image/png")})

        assert response.status_code == 415

    def test_given_rekognition_error_when_uploaded_then_bad_gateway(
        self, client: TestClient, rekognition_stub: Stubber
    ) -> None:
        rekognition_stub.add_client_error(
            "detect_labels", service_error_code="ThrottlingException"
        )

        response = client.post(
            "/images", files={"file": ("p.jpg", IMAGE, "image/jpeg")}
        )

        assert response.status_code == 502


class TestLatestAnalysis:
    def test_given_no_upload_when_latest_requested_then_not_found(
        self, client: TestClient
    ) -> None:
        assert client.get("/analyses/latest").status_code == 404
        assert client.get("/images/latest/annotated").status_code == 404

    def test_given_upload_when_latest_requested_then_analysis_with_etag(
        self, client: TestClient, rekognition_stub: Stubber
    ) -> None:
        expect_one_person(rekognition_stub)
        analysis_id = client.post(
            "/images", files={"file": ("p.jpg", IMAGE, "image/jpeg")}
        ).json()["id"]

        response = client.get("/analyses/latest")
        not_modified = client.get(
            "/analyses/latest", headers={"If-None-Match": response.headers["ETag"]}
        )

        assert response.status_code == 200
        assert response.json()["id"] == analysis_id
        assert response.headers["ETag"] == f'"{analysis_id}"'
        assert not_modified.status_code == 304

    def test_given_upload_when_annotated_image_requested_then_jpeg(
        self, client: TestClient, rekognition_stub: Stubber
    ) -> None:
        expect_one_person(rekognition_stub)
        client.post("/images", files={"file": ("p.jpg", IMAGE, "image/jpeg")})

        response = client.get("/images/latest/annotated")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content[:2] == b"\xff\xd8"


class TestReanalyze:
    def test_given_stored_image_when_reanalyzed_detailed_then_new_analysis(
        self, client: TestClient, rekognition_stub: Stubber
    ) -> None:
        expect_one_person(rekognition_stub)
        first = client.post("/images", files={"file": ("p.jpg", IMAGE, "image/jpeg")})
        expect_one_person(rekognition_stub, attributes="ALL")

        response = client.post("/analyses", params={"detail": "detailed"})

        assert response.status_code == 201
        assert response.json()["detail"] == "detailed"
        assert response.json()["id"] != first.json()["id"]

    def test_given_no_stored_image_when_reanalyzed_then_not_found(
        self, client: TestClient
    ) -> None:
        assert client.post("/analyses").status_code == 404


class TestSpeech:
    def test_given_text_when_speech_requested_then_mp3_bytes(
        self, client: TestClient, polly_stub: Stubber
    ) -> None:
        audio = b"ID3-fake"
        polly_stub.add_response(
            "synthesize_speech",
            {"AudioStream": StreamingBody(io.BytesIO(audio), len(audio))},
            {
                "Text": ANY,
                "TextType": "ssml",
                "OutputFormat": "mp3",
                "VoiceId": "Matthew",
            },
        )

        response = client.post("/speech", params={"text": "Hello"})

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == audio

    def test_given_too_long_text_when_speech_requested_then_unprocessable(
        self, client: TestClient
    ) -> None:
        assert client.post("/speech", params={"text": "a" * 1501}).status_code == 422


class TestLegacyEndpoints:
    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/image/analyze?input_path=/etc/passwd"),
            ("POST", "/image/save?output_path=/tmp/x.jpg"),
            ("GET", "/image/latest"),
            ("POST", "/speech/generate?text=hi"),
        ],
    )
    def test_given_legacy_path_endpoint_when_called_then_not_found(
        self, client: TestClient, method: str, path: str
    ) -> None:
        assert client.request(method, path).status_code in {404, 405}
