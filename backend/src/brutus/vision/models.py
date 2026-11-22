"""Immutable data types for image analysis results."""

from collections import Counter
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, computed_field


class Model(BaseModel):
    """Base for all frozen models."""

    model_config = ConfigDict(frozen=True)


class DetailLevel(StrEnum):
    """How much Brutus reports about a picture."""

    BASIC = "basic"
    """Counts, people boxes, animals and object names."""
    STANDARD = "standard"
    """Adds identities (face recognition) and object boxes."""
    DETAILED = "detailed"
    """Adds face attributes such as age range, emotion and glasses."""


class BoundingBox(Model):
    """Box as ratios (0..1) of the image width and height."""

    left: float
    top: float
    width: float
    height: float


class LabelInstance(Model):
    """One boxed occurrence of a Rekognition label."""

    name: str
    confidence: float
    box: BoundingBox
    parents: frozenset[str] = frozenset()
    categories: frozenset[str] = frozenset()


class FaceAttributes(Model):
    """Face details reported at DetailLevel.DETAILED."""

    age_low: int
    age_high: int
    emotion: str | None
    emotion_confidence: float | None
    smile: bool
    eyeglasses: bool
    sunglasses: bool
    beard: bool
    mustache: bool
    eyes_open: bool
    mouth_open: bool


class Face(Model):
    """A detected face."""

    box: BoundingBox
    confidence: float
    attributes: FaceAttributes | None = None


class FaceMatch(Model):
    """A face in the picture that matched a reference image."""

    name: str
    similarity: float
    box: BoundingBox


class Identity(Model):
    """Who a person was recognized as."""

    name: str
    similarity: float


class Person(Model):
    """A person in the picture, with their face and identity when known."""

    number: int
    box: BoundingBox
    face: Face | None = None
    identity: Identity | None = None


class Detection(Model):
    """A numbered animal or object."""

    number: int
    name: str
    confidence: float
    box: BoundingBox


class Thresholds(Model):
    """Tunable limits used by the analysis pipeline."""

    label_min_confidence: float = 90.0
    face_min_confidence: float = 90.0
    similarity: float = 85.0
    face_overlap: float = 0.5
    """Share of a face box that must be inside a person box to belong to it."""
    identity_iou: float = 0.5
    """IoU needed to tie a recognized face to a detected face."""
    dedupe_iou: float = 0.8
    """IoU above which a parent label's box duplicates its child's box."""


class Analysis(Model):
    """The complete result of analyzing one picture."""

    id: str
    created_at: datetime
    detail: DetailLevel
    people: tuple[Person, ...]
    animals: tuple[Detection, ...]
    objects: tuple[Detection, ...]
    labels: tuple[str, ...]
    """Scene-level labels without boxes, such as "Indoors"."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def people_count(self) -> int:
        """Number of people in the picture."""
        return len(self.people)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def animal_count(self) -> int:
        """Number of animals in the picture."""
        return len(self.animals)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def object_counts(self) -> dict[str, int]:
        """Number of objects per name."""
        return dict(Counter(o.name for o in self.objects))
