"""Brutus terminal client for the Raspberry Pi: take pictures, play audio, speak."""

import io
import json
import logging
import logging.config
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
import requests
from dotenv import load_dotenv
from PIL import Image

CLIENT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("brutus_pi")

type Json = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Settings:
    """Client configuration; see .env.template for the environment variables."""

    backend_url: str = "http://localhost:8000"
    camera_rotation: int = -90
    """Degrees counter-clockwise applied after capture (-90 = clockwise)."""
    countdown_seconds: int = 5
    detail: str = "standard"
    image_path: Path = CLIENT_ROOT / "input" / "latest.jpg"
    audio_dir: Path = CLIENT_ROOT / "audio"
    timeout_seconds: float = 30.0


def load_settings(env: Mapping[str, str]) -> Settings:
    """Settings from environment variables, falling back to the defaults."""
    defaults = Settings()
    return Settings(
        backend_url=env.get("BACKEND_URL", defaults.backend_url).rstrip("/"),
        camera_rotation=int(env.get("CAMERA_ROTATION", defaults.camera_rotation)),
        countdown_seconds=int(env.get("COUNTDOWN_SECONDS", defaults.countdown_seconds)),
        detail=env.get("DETAIL", defaults.detail),
    )


# --- Backend ---------------------------------------------------------------


def is_backend_healthy(settings: Settings) -> bool:
    """True when the backend answers its health check."""
    try:
        response = requests.get(f"{settings.backend_url}/health", timeout=5)
        return response.ok and response.json().get("status") == "Healthy"
    except requests.RequestException:
        logger.warning("Backend health check failed", exc_info=True)
        return False


def upload(settings: Settings, image: Path) -> dict[str, Any]:
    """Send a picture to the backend and return its analysis."""
    with image.open("rb") as file:
        response = requests.post(
            f"{settings.backend_url}/images",
            params={"detail": settings.detail},
            files={"file": (image.name, file, "image/jpeg")},
            timeout=settings.timeout_seconds,
        )
    response.raise_for_status()
    analysis: dict[str, Any] = response.json()
    return analysis


# --- Camera ----------------------------------------------------------------


def capture(target: Path, *, rotation: int, countdown_seconds: int) -> Path:
    """Count down, take a picture with rpicam-still and save it rotated."""
    for remaining in range(countdown_seconds, 0, -1):
        print(f"Countdown: {remaining}...")
        time.sleep(1)

    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.with_name(f".{target.stem}.raw.jpg")
    try:
        subprocess.run(["rpicam-still", "-n", "-o", str(raw)], check=True, timeout=60)
        with Image.open(raw) as image:
            image.rotate(rotation, expand=True).save(target, format="JPEG")
    finally:
        raw.unlink(missing_ok=True)
    logger.info("Picture saved to %s", target)
    return target


# --- Audio -----------------------------------------------------------------


def play_audio(source: Path | io.BytesIO) -> None:
    """Play an MP3 file or MP3 bytes and wait until it finishes."""
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    if isinstance(source, Path):
        pygame.mixer.music.load(source)
    else:
        pygame.mixer.music.load(source, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)


def speak(settings: Settings, text: str) -> None:
    """Have the backend synthesize `text` and play it."""
    response = requests.post(
        f"{settings.backend_url}/speech",
        params={"text": text},
        timeout=settings.timeout_seconds,
    )
    response.raise_for_status()
    play_audio(io.BytesIO(response.content))


# --- Describing results ------------------------------------------------------


def plural(count: int, word: str, plural_form: str | None = None) -> str:
    """'1 chair', '2 chairs', or a given irregular plural."""
    if count == 1:
        return f"{count} {word}"
    return f"{count} {plural_form or (word if word.endswith('s') else word + 's')}"


def _describe_people(people: Iterable[Json]) -> list[str]:
    people = list(people)
    if not people:
        return []
    names = [p["identity"]["name"] for p in people if p.get("identity")]
    unknown = len(people) - len(names)
    who = [*names, *([plural(unknown, "unknown person", "unknown")] if unknown else [])]
    total = plural(len(people), "person", "people")
    return [f"{total}: {' and '.join(who)}" if names else total]


def _describe_things(detections: Iterable[Json]) -> list[str]:
    counts = Counter(d["name"].lower() for d in detections)
    return [plural(n, name) for name, n in counts.items()]


def describe(analysis: Json) -> str:
    """Sentence such as 'I see 2 people: Kirk and 1 unknown person, 1 dog.'."""
    parts = [
        *_describe_people(analysis["people"]),
        *_describe_things(analysis["animals"]),
        *_describe_things(analysis["objects"]),
    ]
    return f"I see {', '.join(parts)}." if parts else "I see nothing I recognise."


# --- Menu --------------------------------------------------------------------


def take_picture(settings: Settings) -> None:
    """Capture, upload and print what Brutus sees."""
    picture = capture(
        settings.image_path,
        rotation=settings.camera_rotation,
        countdown_seconds=settings.countdown_seconds,
    )
    print(describe(upload(settings, picture)))


def say_something(settings: Settings) -> None:
    """Ask for text and speak it."""
    if text := input("Enter text for Brutus to speak: ").strip():
        speak(settings, text)


type Action = Callable[[Settings], None]

MENU: dict[str, tuple[str, Action]] = {
    "1": ("Take picture and analyze", take_picture),
    "2": ("Introduce Brutus", lambda s: play_audio(s.audio_dir / "introduction.mp3")),
    "3": ("Laugh", lambda s: play_audio(s.audio_dir / "laugh.mp3")),
    "4": ("Say something", say_something),
}

RECOVERABLE = (
    requests.RequestException,
    subprocess.SubprocessError,
    OSError,
    pygame.error,
)


def run_menu(settings: Settings) -> None:
    """Show the menu until the user quits."""
    while True:
        print("\nBRUTUS TERMINAL INTERFACE\n------------------------")
        for key, (title, _) in MENU.items():
            print(f"{key}: {title}")
        print("q: Quit")

        choice = input("\nEnter your choice: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            return
        if choice not in MENU:
            print("Invalid option, please try again.")
            continue
        title, action = MENU[choice]
        try:
            action(settings)
        except RECOVERABLE as error:
            logger.exception("%s failed", title)
            print(f"Error: {error}")


def main() -> int:
    """Check the backend, greet, and run the menu."""
    load_dotenv(CLIENT_ROOT / ".env")
    settings = load_settings(os.environ)
    # Paths in logging.json are relative to this folder, as for the uvicorn services.
    os.chdir(CLIENT_ROOT)
    Path("logs").mkdir(exist_ok=True)
    logging.config.dictConfig(json.loads(Path("logging.json").read_text()))

    if not is_backend_healthy(settings):
        print(f"Backend is not available at {settings.backend_url}")
        return 1
    try:
        play_audio(settings.audio_dir / "by_your_command.mp3")
        run_menu(settings)
    finally:
        pygame.mixer.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
