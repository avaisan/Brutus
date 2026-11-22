import io
import subprocess
from pathlib import Path
from typing import Any

import pytest
import requests
import requests_mock
from PIL import Image
from pytest_mock import MockerFixture

import brutus_pi
from brutus_pi import Settings, capture, describe, is_backend_healthy, speak, upload

BACKEND = "http://brutus.test:8000"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(backend_url=BACKEND, image_path=tmp_path / "input" / "latest.jpg")


def person(number: int, name: str | None = None) -> dict[str, Any]:
    identity = {"name": name, "similarity": 99.0} if name else None
    return {"number": number, "identity": identity}


def detection(number: int, name: str) -> dict[str, Any]:
    return {"number": number, "name": name}


def analysis(
    people: list[dict[str, Any]],
    animals: list[dict[str, Any]] | None = None,
    objects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "people": people,
        "animals": animals or [],
        "objects": objects or [],
        "people_count": len(people),
    }


class TestCapture:
    def test_given_camera_when_captured_then_rotated_and_temp_removed(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        target = tmp_path / "latest.jpg"

        def fake_camera(args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
            Image.new("RGB", (200, 100)).save(args[-1], format="JPEG")
            return subprocess.CompletedProcess(args, 0)

        run = mocker.patch("brutus_pi.subprocess.run", side_effect=fake_camera)

        result = capture(target, rotation=-90, countdown_seconds=0)

        command = run.call_args.args[0]
        assert command[:3] == ["rpicam-still", "-n", "-o"]
        assert run.call_args.kwargs["check"] is True
        assert result == target
        assert Image.open(target).size == (100, 200)
        assert not Path(command[-1]).exists()

    def test_given_camera_failure_when_captured_then_raises(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        mocker.patch(
            "brutus_pi.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "rpicam-still"),
        )

        with pytest.raises(subprocess.CalledProcessError):
            capture(tmp_path / "latest.jpg", rotation=-90, countdown_seconds=0)


class TestUpload:
    def test_given_picture_when_uploaded_then_analysis_returned(
        self, settings: Settings, requests_mock: requests_mock.Mocker
    ) -> None:
        settings.image_path.parent.mkdir(parents=True)
        Image.new("RGB", (10, 10)).save(settings.image_path, format="JPEG")
        requests_mock.post(f"{BACKEND}/images", status_code=201, json={"id": "a1"})

        result = upload(settings, settings.image_path)

        request = requests_mock.last_request
        assert request is not None
        assert result == {"id": "a1"}
        assert request.qs == {"detail": ["standard"]}
        assert b"image/jpeg" in request.body

    def test_given_backend_error_when_uploaded_then_raises(
        self, settings: Settings, requests_mock: requests_mock.Mocker, tmp_path: Path
    ) -> None:
        image = tmp_path / "x.jpg"
        image.write_bytes(b"jpeg")
        requests_mock.post(f"{BACKEND}/images", status_code=502)

        with pytest.raises(requests.HTTPError):
            upload(settings, image)


class TestDescribe:
    def test_given_known_and_unknown_people_and_things_when_described_then_sentence(
        self,
    ) -> None:
        result = analysis(
            [person(1, "Kirk"), person(2)],
            animals=[detection(1, "Dog")],
            objects=[detection(1, "Chair"), detection(2, "Chair")],
        )

        assert describe(result) == (
            "I see 2 people: Kirk and 1 unknown person, 1 dog, 2 chairs."
        )

    def test_given_unknown_people_only_when_described_then_count(self) -> None:
        assert describe(analysis([person(1), person(2), person(3)])) == (
            "I see 3 people."
        )

    def test_given_one_known_person_when_described_then_name(self) -> None:
        assert describe(analysis([person(1, "Kirk")])) == "I see 1 person: Kirk."

    def test_given_empty_picture_when_described_then_nothing(self) -> None:
        assert describe(analysis([])) == "I see nothing I recognise."


class TestHealth:
    def test_given_healthy_backend_when_checked_then_true(
        self, settings: Settings, requests_mock: requests_mock.Mocker
    ) -> None:
        requests_mock.get(f"{BACKEND}/health", json={"status": "Healthy"})

        assert is_backend_healthy(settings) is True

    def test_given_unreachable_backend_when_checked_then_false(
        self, settings: Settings, requests_mock: requests_mock.Mocker
    ) -> None:
        requests_mock.get(f"{BACKEND}/health", exc=requests.ConnectionError)

        assert is_backend_healthy(settings) is False


class TestSpeak:
    def test_given_text_when_spoken_then_backend_audio_played(
        self,
        settings: Settings,
        requests_mock: requests_mock.Mocker,
        mocker: MockerFixture,
    ) -> None:
        requests_mock.post(f"{BACKEND}/speech", content=b"ID3-mp3")
        play = mocker.patch.object(brutus_pi, "play_audio")

        speak(settings, "Hello")

        request = requests_mock.last_request
        assert request is not None
        assert request.qs == {"text": ["hello"]}
        (source,) = play.call_args.args
        assert isinstance(source, io.BytesIO)
        assert source.getvalue() == b"ID3-mp3"
