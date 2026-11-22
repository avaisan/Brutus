"""Pure functions that turn backend Analysis JSON into display text."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

type Json = Mapping[str, Any]

ATTRIBUTE_FLAGS = {
    "smile": "Smiling",
    "eyeglasses": "Eyeglasses",
    "sunglasses": "Sunglasses",
    "beard": "Beard",
    "mustache": "Mustache",
    "mouth_open": "Mouth open",
}


def plural(count: int, word: str, plural_form: str | None = None) -> str:
    """'1 chair', '2 chairs', or a given irregular plural."""
    if count == 1:
        return f"{count} {word}"
    return f"{count} {plural_form or (word if word.endswith('s') else word + 's')}"


def _named_counts(detections: Iterable[Json]) -> list[str]:
    counts = Counter(d["name"].lower() for d in detections)
    return [plural(n, name) for name, n in counts.items()]


def describe_counts(analysis: Json) -> str:
    """One-line summary such as '2 people · 1 dog · 2 chairs'."""
    people = analysis["people_count"]
    parts = [
        *([plural(people, "person", "people")] if people else []),
        *_named_counts(analysis["animals"]),
        *_named_counts(analysis["objects"]),
    ]
    return " · ".join(parts) or "Nothing recognised"


def describe_person(person: Json) -> tuple[str, tuple[str, ...]]:
    """Title ('P1 · Kirk (98.7%)') and attribute lines for one person."""
    identity = person.get("identity")
    who = (
        f"{identity['name']} ({identity['similarity']:.1f}%)" if identity else "Unknown"
    )
    attributes = (person.get("face") or {}).get("attributes")
    if not attributes:
        return f"P{person['number']} · {who}", ()

    emotion = attributes.get("emotion")
    details = (
        f"Age {attributes['age_low']}-{attributes['age_high']}",
        *(
            [f"{emotion.capitalize()} ({attributes['emotion_confidence']:.1f}%)"]
            if emotion
            else []
        ),
        *(label for key, label in ATTRIBUTE_FLAGS.items() if attributes.get(key)),
        *([] if attributes.get("eyes_open", True) else ["Eyes closed"]),
    )
    return f"P{person['number']} · {who}", details
