"""Analyze an image file from the command line (no Raspberry Pi needed)."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from brutus.log import configure_logging
from brutus.settings import Settings, aws_session
from brutus.store import load_references
from brutus.vision.annotate import annotate
from brutus.vision.imaging import encode_jpeg, normalize
from brutus.vision.models import DetailLevel
from brutus.vision.pipeline import analyze


def main(argv: Sequence[str] | None = None) -> int:
    """Print the analysis as JSON and optionally save the annotated image."""
    parser = argparse.ArgumentParser(prog="brutus-analyze", description=__doc__)
    parser.add_argument("image", type=Path, help="JPEG or PNG to analyze")
    parser.add_argument(
        "--detail",
        type=DetailLevel,
        choices=list(DetailLevel),
        default=DetailLevel.STANDARD,
    )
    parser.add_argument("--out", type=Path, help="where to save the annotated JPEG")
    args = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_dir)
    client = aws_session(settings).client("rekognition")
    image = normalize(args.image.read_bytes(), max_edge=settings.max_image_edge)
    references = load_references(
        settings.reference_dir, max_edge=settings.max_image_edge
    )

    analysis = analyze(
        encode_jpeg(image), client, references, args.detail, settings.thresholds
    )

    if args.out:
        args.out.write_bytes(encode_jpeg(annotate(image, analysis)))
    print(analysis.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
