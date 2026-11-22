"""An Analysis payload shaped like the backend's GET /analyses/latest."""

from typing import Any

BOX = {"left": 0.1, "top": 0.1, "width": 0.2, "height": 0.6}


def analysis(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "abc123",
        "created_at": "2026-11-22T12:00:00Z",
        "detail": "detailed",
        "people": [
            {
                "number": 1,
                "box": BOX,
                "face": {
                    "box": BOX,
                    "confidence": 99.9,
                    "attributes": {
                        "age_low": 30,
                        "age_high": 40,
                        "emotion": "HAPPY",
                        "emotion_confidence": 91.5,
                        "smile": True,
                        "eyeglasses": True,
                        "sunglasses": False,
                        "beard": False,
                        "mustache": False,
                        "eyes_open": True,
                        "mouth_open": False,
                    },
                },
                "identity": {"name": "Kirk", "similarity": 98.7},
            },
            {"number": 2, "box": BOX, "face": None, "identity": None},
        ],
        "animals": [{"number": 1, "name": "Dog", "confidence": 97.0, "box": BOX}],
        "objects": [
            {"number": 1, "name": "Chair", "confidence": 95.0, "box": BOX},
            {"number": 2, "name": "Chair", "confidence": 94.0, "box": BOX},
        ],
        "labels": ["Indoors"],
        "people_count": 2,
        "animal_count": 1,
        "object_counts": {"Chair": 2},
    }
    return payload | overrides
