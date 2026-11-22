"""One Rekognition pass per picture, composed from the pure vision functions."""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from brutus.vision.models import Analysis, DetailLevel, FaceMatch, Thresholds
from brutus.vision.objects import person_boxes, split_detections
from brutus.vision.parse import (
    parse_face_matches,
    parse_faces,
    parse_label_instances,
    parse_scene_labels,
)
from brutus.vision.people import combine_people

if TYPE_CHECKING:
    from mypy_boto3_rekognition import RekognitionClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Reference:
    """A reference face: the file name (without extension) is the person's name."""

    name: str
    jpeg: bytes


def recognize(
    client: RekognitionClient,
    jpeg: bytes,
    references: Sequence[Reference],
    similarity: float,
) -> tuple[FaceMatch, ...]:
    """Compare the picture against every reference face."""

    def compare(reference: Reference) -> tuple[FaceMatch, ...]:
        try:
            response = client.compare_faces(
                SourceImage={"Bytes": reference.jpeg},
                TargetImage={"Bytes": jpeg},
                SimilarityThreshold=similarity,
            )
        except client.exceptions.InvalidParameterException:
            logger.warning("Reference image has no usable face: %s", reference.name)
            return ()
        return parse_face_matches(reference.name, response)

    return tuple(match for reference in references for match in compare(reference))


def analyze(
    jpeg: bytes,
    client: RekognitionClient,
    references: Sequence[Reference],
    detail: DetailLevel,
    thresholds: Thresholds,
) -> Analysis:
    """Detect, recognize and combine everything in one picture."""
    detailed = detail is DetailLevel.DETAILED
    labels = client.detect_labels(
        Image={"Bytes": jpeg}, MinConfidence=thresholds.label_min_confidence
    )
    faces = parse_faces(
        client.detect_faces(
            Image={"Bytes": jpeg}, Attributes=["ALL" if detailed else "DEFAULT"]
        ),
        min_confidence=thresholds.face_min_confidence,
        detailed=detailed,
    )
    wants_identities = detail is not DetailLevel.BASIC and faces
    matches = (
        recognize(client, jpeg, references, thresholds.similarity)
        if wants_identities
        else ()
    )

    instances = parse_label_instances(labels)
    animals, objects = split_detections(
        instances,
        dedupe_iou=thresholds.dedupe_iou,
        part_inside=thresholds.part_inside,
    )
    people = combine_people(
        person_boxes(instances),
        faces,
        matches,
        face_overlap=thresholds.face_overlap,
        identity_iou=thresholds.identity_iou,
    )
    analysis = Analysis(
        id=uuid.uuid4().hex,
        created_at=datetime.now(UTC),
        detail=detail,
        people=people,
        animals=animals,
        objects=objects,
        labels=parse_scene_labels(labels),
    )
    logger.info(
        "Analysis complete",
        extra={
            "analysis_id": analysis.id,
            "detail": detail.value,
            "people": analysis.people_count,
            "animals": analysis.animal_count,
            "objects": analysis.object_counts,
        },
    )
    return analysis
