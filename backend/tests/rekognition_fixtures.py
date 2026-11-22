"""Builders for Rekognition responses shaped like the real service output."""

from typing import Any

type Box = tuple[float, float, float, float]  # left, top, width, height


def bbox(box: Box) -> dict[str, float]:
    left, top, width, height = box
    return {"Left": left, "Top": top, "Width": width, "Height": height}


def label(
    name: str,
    boxes: tuple[Box, ...] = (),
    *,
    confidence: float = 98.0,
    parents: tuple[str, ...] = (),
    categories: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "Name": name,
        "Confidence": confidence,
        "Instances": [
            {"BoundingBox": bbox(b), "Confidence": confidence} for b in boxes
        ],
        "Parents": [{"Name": p} for p in parents],
        "Aliases": [],
        "Categories": [{"Name": c} for c in categories],
    }


def labels_response(*labels: dict[str, Any]) -> dict[str, Any]:
    return {"Labels": list(labels), "LabelModelVersion": "3.0"}


def person(*boxes: Box) -> dict[str, Any]:
    return label("Person", boxes, categories=("Person Description",))


def dog(*boxes: Box) -> dict[str, Any]:
    return label(
        "Dog",
        boxes,
        parents=("Animal", "Canine", "Mammal", "Pet"),
        categories=("Animals and Pets",),
    )


def face(
    box: Box,
    *,
    confidence: float = 99.9,
    smile: bool = False,
    emotion: str = "CALM",
) -> dict[str, Any]:
    def flag(value: bool) -> dict[str, Any]:
        return {"Value": value, "Confidence": 95.0}

    return {
        "BoundingBox": bbox(box),
        "Confidence": confidence,
        "AgeRange": {"Low": 30, "High": 40},
        "Smile": flag(smile),
        "Eyeglasses": flag(False),
        "Sunglasses": flag(False),
        "Gender": {"Value": "Male", "Confidence": 99.0},
        "Beard": flag(False),
        "Mustache": flag(False),
        "EyesOpen": flag(True),
        "MouthOpen": flag(False),
        "Emotions": [
            {"Type": "HAPPY", "Confidence": 2.0},
            {"Type": emotion, "Confidence": 90.0},
        ],
    }


def default_face(box: Box, *, confidence: float = 99.9) -> dict[str, Any]:
    """DetectFaces with Attributes=['DEFAULT'] returns only box and confidence."""
    return {"BoundingBox": bbox(box), "Confidence": confidence}


def faces_response(*faces: dict[str, Any]) -> dict[str, Any]:
    return {"FaceDetails": list(faces)}


def compare_response(*matches: tuple[Box, float]) -> dict[str, Any]:
    return {
        "SourceImageFace": {"BoundingBox": bbox((0.1, 0.1, 0.5, 0.5))},
        "FaceMatches": [
            {"Similarity": similarity, "Face": {"BoundingBox": bbox(b)}}
            for b, similarity in matches
        ],
        "UnmatchedFaces": [],
    }
