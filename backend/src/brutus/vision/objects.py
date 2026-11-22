"""Pure rules that turn label instances into numbered animals and objects."""

from collections.abc import Iterable, Sequence
from functools import reduce

from brutus.vision.geometry import iou
from brutus.vision.models import BoundingBox, Detection, LabelInstance

PERSON = "Person"
PERSON_CATEGORY = "Person Description"
ANIMAL = "Animal"
ANIMAL_CATEGORY = "Animals and Pets"


def is_person_related(instance: LabelInstance) -> bool:
    """Person itself and labels describing people (Man, Face, Adult, ...)."""
    return (
        instance.name == PERSON
        or PERSON in instance.parents
        or PERSON_CATEGORY in instance.categories
    )


def is_animal(instance: LabelInstance) -> bool:
    """Animals, by Rekognition category or by an Animal parent label."""
    return (
        ANIMAL_CATEGORY in instance.categories
        or ANIMAL in instance.parents
        or instance.name == ANIMAL
    )


def person_boxes(instances: Iterable[LabelInstance]) -> tuple[BoundingBox, ...]:
    """Boxes of the Person label, one per person Rekognition found."""
    return tuple(i.box for i in instances if i.name == PERSON)


def _duplicates(
    kept: LabelInstance, candidate: LabelInstance, threshold: float
) -> bool:
    same_or_parent = candidate.name == kept.name or candidate.name in kept.parents
    return same_or_parent and iou(kept.box, candidate.box) >= threshold


def deduplicate(
    instances: Iterable[LabelInstance], *, iou_threshold: float
) -> tuple[LabelInstance, ...]:
    """Keep the most specific label when parent labels box the same thing.

    "Dog", "Pet" and "Animal" on one box become a single "Dog".
    """
    most_specific_first = sorted(
        instances, key=lambda i: (len(i.parents), i.confidence), reverse=True
    )
    return reduce(
        lambda kept, candidate: (
            kept
            if any(_duplicates(k, candidate, iou_threshold) for k in kept)
            else (*kept, candidate)
        ),
        most_specific_first,
        (),
    )


def number(instances: Sequence[LabelInstance]) -> tuple[Detection, ...]:
    """Number detections 1..n from left to right."""
    ordered = sorted(instances, key=lambda i: (i.box.left, i.box.top))
    return tuple(
        Detection(number=n, name=i.name, confidence=i.confidence, box=i.box)
        for n, i in enumerate(ordered, start=1)
    )


def split_detections(
    instances: Iterable[LabelInstance], *, dedupe_iou: float
) -> tuple[tuple[Detection, ...], tuple[Detection, ...]]:
    """Return (animals, objects), each deduplicated and numbered."""
    unique = deduplicate(
        (i for i in instances if not is_person_related(i)), iou_threshold=dedupe_iou
    )
    return (
        number([i for i in unique if is_animal(i)]),
        number([i for i in unique if not is_animal(i)]),
    )
