"""HTTP API: analyze each picture once, then serve the stored result."""

import logging
import threading
from typing import TYPE_CHECKING, Annotated

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from brutus import store
from brutus.settings import Settings, aws_session
from brutus.speech import synthesize
from brutus.vision.annotate import annotate
from brutus.vision.imaging import encode_jpeg, normalize
from brutus.vision.models import Analysis, DetailLevel
from brutus.vision.pipeline import analyze

if TYPE_CHECKING:
    from mypy_boto3_polly import PollyClient
    from mypy_boto3_rekognition import RekognitionClient

logger = logging.getLogger(__name__)

JPEG_TYPES = frozenset({"image/jpeg", "image/jpg"})

type Detail = Annotated[DetailLevel, Query()]


def create_app(
    settings: Settings | None = None,
    *,
    rekognition: RekognitionClient | None = None,
    polly: PollyClient | None = None,
) -> FastAPI:
    """Build the app. AWS clients are created from settings unless injected."""
    settings = settings or Settings()
    if rekognition is None or polly is None:
        session = aws_session(settings)
        rekognition = rekognition or session.client("rekognition")
        polly = polly or session.client("polly")

    app = FastAPI(title="Brutus API", version="2.0.0")
    # One analysis at a time: the stored image and JSON always belong together,
    # and a single camera never needs parallel analyses.
    analysis_lock = threading.Lock()

    def run_analysis(image: Image.Image, jpeg: bytes, detail: DetailLevel) -> Analysis:
        with analysis_lock:
            references = store.load_references(
                settings.reference_dir, max_edge=settings.max_image_edge
            )
            analysis = analyze(
                jpeg, rekognition, references, detail, settings.thresholds
            )
            store.save_result(
                settings.output_dir, analysis, encode_jpeg(annotate(image, analysis))
            )
            return analysis

    @app.exception_handler(BotoCoreError)
    @app.exception_handler(ClientError)
    async def aws_failed(_: Request, error: Exception) -> JSONResponse:
        logger.error("AWS request failed", exc_info=error)
        return JSONResponse({"detail": "AWS request failed"}, status_code=502)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness check used by the Pi client and the start script."""
        return {"status": "Healthy"}

    @app.post("/images", status_code=201)
    def upload_image(
        file: Annotated[UploadFile, File()], detail: Detail = DetailLevel.STANDARD
    ) -> Analysis:
        """Store a JPEG from the camera and analyze it."""
        if file.content_type not in JPEG_TYPES:
            raise HTTPException(415, "Only JPEG images are accepted")
        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "Image is too large")
        try:
            image = normalize(data, max_edge=settings.max_image_edge)
        except (OSError, Image.DecompressionBombError) as error:
            # OSError covers unreadable (UnidentifiedImageError) and truncated files.
            raise HTTPException(422, "File is not a readable image") from error
        jpeg = encode_jpeg(image)
        store.save_input(settings.input_dir, jpeg)
        return run_analysis(image, jpeg, detail)

    @app.post("/analyses", status_code=201)
    def reanalyze(detail: Detail = DetailLevel.STANDARD) -> Analysis:
        """Analyze the latest stored picture again, for example at another detail."""
        jpeg = store.load_input(settings.input_dir)
        if jpeg is None:
            raise HTTPException(404, "No picture has been uploaded yet")
        image = normalize(jpeg, max_edge=settings.max_image_edge)
        return run_analysis(image, jpeg, detail)

    @app.get("/analyses/latest", response_model=Analysis)
    def latest_analysis(request: Request) -> Response:
        """The latest analysis; send If-None-Match to get 304 when unchanged."""
        analysis = store.load_analysis(settings.output_dir)
        if analysis is None:
            raise HTTPException(404, "No analysis yet")
        etag = f'"{analysis.id}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return JSONResponse(analysis.model_dump(mode="json"), headers={"ETag": etag})

    @app.get("/images/latest/annotated", response_class=FileResponse)
    def latest_annotated_image() -> FileResponse:
        """The latest picture with boxes and captions."""
        path = store.annotated_path(settings.output_dir)
        if not path.exists():
            raise HTTPException(404, "No analyzed image yet")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/speech", response_class=Response)
    def speech(
        text: Annotated[str, Query(min_length=1, max_length=settings.max_speech_chars)],
    ) -> Response:
        """Speak the text with Polly and return MP3 audio."""
        audio = synthesize(
            polly, text, voice=settings.polly_voice, rate=settings.speech_rate
        )
        return Response(audio, media_type="audio/mpeg")

    return app
