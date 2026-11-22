"""JSON logging, configured once per process."""

import logging.config
from pathlib import Path


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    """Log JSON lines to `log_dir/brutus.log` (rotated daily) and warnings to stderr."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.json.JsonFormatter",
                    "fmt": "%(asctime)s %(levelname)s %(name)s %(funcName)s "
                    "%(lineno)d %(message)s",
                },
                "plain": {"format": "%(levelname)s %(name)s: %(message)s"},
            },
            "handlers": {
                "file": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "filename": str(log_dir / "brutus.log"),
                    "when": "midnight",
                    "backupCount": 14,
                    "encoding": "utf-8",
                    "formatter": "json",
                },
                "stderr": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "plain",
                },
            },
            "root": {"level": level, "handlers": ["file", "stderr"]},
        }
    )
