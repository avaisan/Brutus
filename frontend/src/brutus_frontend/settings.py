"""Frontend configuration from environment variables and frontend/.env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

FRONTEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Settings; every field can be overridden with an env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=FRONTEND_ROOT / ".env", extra="ignore", frozen=True
    )

    backend_url: str = "http://localhost:8000"
    """Backend address as seen from the frontend server (not from the browser)."""
    refresh_seconds: int = 3
    backend_timeout_seconds: float = 10.0
