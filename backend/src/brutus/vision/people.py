"""Pure rules that combine person boxes, faces and face matches into people."""

from collections.abc import Hashable, Iterable, Sequence

from brutus.vision.geometry import intersection_over_area, iou
from brutus.vision.models import BoundingBox, Face, FaceMatch, Identity, Person


def greedy_pairs[A: Hashable, B: Hashable](
    scored: Iterable[tuple[float, A, B]], *, minimum: float = 0.0
) -> dict[A, tuple[B, float]]:
    """Pair items one-to-one, best score first, ignoring scores below `minimum`."""
    pairs: dict[A, tuple[B, float]] = {}
    taken: set[B] = set()
    for score, a, b in sorted(scored, key=lambda item: item[0], reverse=True):
        if score >= minimum and a not in pairs and b not in taken:
            pairs[a] = (b, score)
            taken.add(b)
    return pairs


def match_identities(
    faces: Sequence[Face], matches: Iterable[FaceMatch], *, min_iou: float
) -> dict[int, Identity]:
    """Identity per face index. Each reference name goes to at most one face."""
    candidates = (
        (match.similarity, index, match.name)
        for match in matches
        for index, face in enumerate(faces)
        if iou(match.box, face.box) >= min_iou
    )
    return {
        index: Identity(name=name, similarity=similarity)
        for index, (name, similarity) in greedy_pairs(candidates).items()
    }


def assign_faces(
    persons: Sequence[BoundingBox], faces: Sequence[Face], *, min_overlap: float
) -> dict[int, int]:
    """Face index per person index, by how much of the face lies inside the person."""
    candidates = (
        (intersection_over_area(face.box, person), p, f)
        for p, person in enumerate(persons)
        for f, face in enumerate(faces)
    )
    return {p: f for p, (f, _) in greedy_pairs(candidates, minimum=min_overlap).items()}


def combine_people(
    persons: Sequence[BoundingBox],
    faces: Sequence[Face],
    matches: Iterable[FaceMatch],
    *,
    face_overlap: float,
    identity_iou: float,
) -> tuple[Person, ...]:
    """One Person per person box, plus one per face that no person box contains.

    People are numbered 1..n from left to right.
    """
    identities = match_identities(faces, matches, min_iou=identity_iou)
    face_of = assign_faces(persons, faces, min_overlap=face_overlap)
    unassigned = sorted(set(range(len(faces))) - set(face_of.values()))

    with_box = [(box, face_of.get(p)) for p, box in enumerate(persons)] + [
        (faces[f].box, f) for f in unassigned
    ]

    ordered = sorted(with_box, key=lambda item: (item[0].left, item[0].top))
    return tuple(
        Person(
            number=n,
            box=box,
            face=faces[f] if f is not None else None,
            identity=identities.get(f) if f is not None else None,
        )
        for n, (box, f) in enumerate(ordered, start=1)
    )
