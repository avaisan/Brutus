"""Filesystem storage for the latest picture, its analysis and reference faces."""

import tempfile
from functools import lru_cache
from pathlib import Path

from brutus.vision.imaging import encode_jpeg, normalize
from brutus.vision.models import Analysis
from brutus.vision.pipeline import Reference

LATEST_INPUT = "latest.jpg"
ANALYSIS_FILE = "analysis.json"
ANNOTATED_FILE = "annotated.jpg"
REFERENCE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def write_atomic(path: Path, data: bytes) -> None:
    """Write so readers never see a half-written file, even with concurrent writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary.write(data)
    Path(temporary.name).replace(path)


def save_input(input_dir: Path, jpeg: bytes) -> None:
    """Keep the latest uploaded picture for re-analysis."""
    write_atomic(input_dir / LATEST_INPUT, jpeg)


def load_input(input_dir: Path) -> bytes | None:
    """The latest uploaded picture, if any."""
    path = input_dir / LATEST_INPUT
    return path.read_bytes() if path.exists() else None


def save_result(output_dir: Path, analysis: Analysis, annotated_jpeg: bytes) -> None:
    """Store the annotated image first, so a new analysis id implies its image."""
    write_atomic(output_dir / ANNOTATED_FILE, annotated_jpeg)
    write_atomic(output_dir / ANALYSIS_FILE, analysis.model_dump_json().encode())


def load_analysis(output_dir: Path) -> Analysis | None:
    """The latest analysis, if any."""
    path = output_dir / ANALYSIS_FILE
    return Analysis.model_validate_json(path.read_bytes()) if path.exists() else None


def annotated_path(output_dir: Path) -> Path:
    """Where the latest annotated image is stored."""
    return output_dir / ANNOTATED_FILE


@lru_cache(maxsize=256)
def _reference(path: Path, mtime_ns: int, size: int, max_edge: int) -> Reference:
    """Decode and re-encode one reference photo; cached until the file changes."""
    return Reference(
        name=path.stem,
        jpeg=encode_jpeg(normalize(path.read_bytes(), max_edge=max_edge)),
    )


def load_references(
    reference_dir: Path, *, max_edge: int = 2048
) -> tuple[Reference, ...]:
    """Reference faces from `<Name>.jpg|jpeg|png`, sorted by name."""
    if not reference_dir.is_dir():
        return ()
    return tuple(
        _reference(path, stat.st_mtime_ns, stat.st_size, max_edge)
        for path in sorted(reference_dir.iterdir())
        if path.suffix.lower() in REFERENCE_SUFFIXES and (stat := path.stat())
    )
