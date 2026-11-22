"""Backend configuration from environment variables and backend/.env."""

from pathlib import Path

import boto3
from pydantic_settings import BaseSettings, SettingsConfigDict

from brutus.vision.models import Thresholds

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings; every field can be overridden with an env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    aws_profile: str | None = None
    aws_region: str = "eu-central-1"

    input_dir: Path = BACKEND_ROOT / "input"
    output_dir: Path = BACKEND_ROOT / "output"
    reference_dir: Path = BACKEND_ROOT / "reference_library"
    log_dir: Path = BACKEND_ROOT / "logs"

    max_image_edge: int = 2048
    max_upload_bytes: int = 20 * 1024 * 1024
    thresholds: Thresholds = Thresholds()

    polly_voice: str = "Matthew"
    speech_rate: int = 85
    max_speech_chars: int = 1500


def aws_session(settings: Settings) -> boto3.Session:
    """Session for the configured AWS SSO profile and region."""
    return boto3.Session(
        profile_name=settings.aws_profile, region_name=settings.aws_region
    )
