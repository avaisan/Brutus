"""Pure conversion of Rekognition responses into analysis models."""

from collections.abc import Mapping
from typing import Any

from brutus.vision.geometry import clamp
from brutus.vision.models import (
    BoundingBox,
    Face,
    FaceAttributes,
    FaceMatch,
    LabelInstance,
)

type Response = Mapping[str, Any]


def parse_box(raw: Mapping[str, float]) -> BoundingBox:
    """Convert a Rekognition BoundingBox, trimmed to the image."""
    return clamp(
        BoundingBox(
            left=raw["Left"], top=raw["Top"], width=raw["Width"], height=raw["Height"]
        )
    )


def parse_label_instances(response: Response) -> tuple[LabelInstance, ...]:
    """Every boxed label instance from a DetectLabels response."""
    return tuple(
        LabelInstance(
            name=label["Name"],
            confidence=instance.get("Confidence", label["Confidence"]),
            box=parse_box(instance["BoundingBox"]),
            parents=frozenset(p["Name"] for p in label.get("Parents", ())),
            categories=frozenset(c["Name"] for c in label.get("Categories", ())),
        )
        for label in response.get("Labels", ())
        for instance in label.get("Instances", ())
    )


def parse_scene_labels(response: Response) -> tuple[str, ...]:
    """Names of labels without boxes, which describe the whole scene."""
    return tuple(
        label["Name"]
        for label in response.get("Labels", ())
        if not label.get("Instances")
    )


def _flag(face: Mapping[str, Any], key: str) -> bool:
    return bool(face.get(key, {}).get("Value", False))


def parse_face_attributes(face: Mapping[str, Any]) -> FaceAttributes:
    """Attributes from a DetectFaces(ALL) FaceDetail. Gender is left out on purpose."""
    age = face.get("AgeRange", {})
    emotion: Mapping[str, Any] = max(
        face.get("Emotions", ()), key=lambda e: e["Confidence"], default={}
    )
    return FaceAttributes(
        age_low=age.get("Low", 0),
        age_high=age.get("High", 0),
        emotion=emotion.get("Type"),
        emotion_confidence=emotion.get("Confidence"),
        smile=_flag(face, "Smile"),
        eyeglasses=_flag(face, "Eyeglasses"),
        sunglasses=_flag(face, "Sunglasses"),
        beard=_flag(face, "Beard"),
        mustache=_flag(face, "Mustache"),
        eyes_open=_flag(face, "EyesOpen"),
        mouth_open=_flag(face, "MouthOpen"),
    )


def parse_faces(
    response: Response, *, min_confidence: float, detailed: bool
) -> tuple[Face, ...]:
    """Faces from a DetectFaces response, dropping low-confidence detections."""
    return tuple(
        Face(
            box=parse_box(face["BoundingBox"]),
            confidence=face["Confidence"],
            attributes=parse_face_attributes(face) if detailed else None,
        )
        for face in response.get("FaceDetails", ())
        if face["Confidence"] >= min_confidence
    )


def parse_face_matches(name: str, response: Response) -> tuple[FaceMatch, ...]:
    """Faces in the target picture that matched the reference called `name`."""
    return tuple(
        FaceMatch(
            name=name,
            similarity=match["Similarity"],
            box=parse_box(match["Face"]["BoundingBox"]),
        )
        for match in response.get("FaceMatches", ())
    )
