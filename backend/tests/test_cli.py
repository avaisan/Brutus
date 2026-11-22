import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.stub import ANY, Stubber
from mypy_boto3_rekognition import RekognitionClient

from brutus import cli
from tests.images import jpeg_bytes
from tests.rekognition_fixtures import (
    default_face,
    faces_response,
    labels_response,
    person,
)


def test_given_image_file_when_cli_runs_then_prints_json_and_saves_annotated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rekognition: RekognitionClient,
    rekognition_stub: Stubber,
) -> None:
    monkeypatch.setenv("REFERENCE_DIR", str(tmp_path / "no_references"))
    monkeypatch.setattr(
        cli, "aws_session", lambda _: SimpleNamespace(client=lambda _: rekognition)
    )
    rekognition_stub.add_response(
        "detect_labels",
        labels_response(person((0.1, 0.1, 0.3, 0.8))),
        {"Image": {"Bytes": ANY}, "MinConfidence": 90.0},
    )
    rekognition_stub.add_response(
        "detect_faces",
        faces_response(default_face((0.15, 0.12, 0.1, 0.1))),
        {"Image": {"Bytes": ANY}, "Attributes": ["DEFAULT"]},
    )
    image = tmp_path / "photo.jpg"
    image.write_bytes(jpeg_bytes(400, 300))
    out = tmp_path / "annotated.jpg"

    exit_code = cli.main([str(image), "--detail", "standard", "--out", str(out)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["people_count"] == 1
    assert out.read_bytes()[:2] == b"\xff\xd8"
