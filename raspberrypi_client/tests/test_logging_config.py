import json
import logging
import logging.config
from pathlib import Path

import pytest

CONFIG = Path(__file__).parents[1] / "logging.json"


def test_given_logging_json_when_applied_then_json_lines_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    try:
        logging.config.dictConfig(json.loads(CONFIG.read_text()))
        logging.getLogger("brutus.test").warning("hello", extra={"analysis_id": "a1"})
        for handler in logging.getLogger().handlers:
            handler.flush()

        (log_file,) = (tmp_path / "logs").iterdir()
        record = json.loads(log_file.read_text().splitlines()[-1])
    finally:
        logging.config.dictConfig({"version": 1, "disable_existing_loggers": False})

    assert record["message"] == "hello"
    assert record["levelname"] == "WARNING"
    assert record["name"] == "brutus.test"
    assert record["analysis_id"] == "a1"
