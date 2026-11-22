"""Server-rendered page; htmx refreshes the results panel from the backend."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from brutus_frontend.settings import Settings
from brutus_frontend.view import describe_counts, describe_person

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the app. `transport` lets tests stand in for the backend."""
    settings = settings or Settings()
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    templates.env.globals |= {
        "describe_counts": describe_counts,
        "describe_person": describe_person,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx2.AsyncClient(
            base_url=settings.backend_url,
            timeout=settings.backend_timeout_seconds,
            transport=transport,
        ) as backend:
            app.state.backend = backend
            yield

    app = FastAPI(title="B.R.U.T.U.S.", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")

    def backend(request: Request) -> httpx2.AsyncClient:
        client: httpx2.AsyncClient = request.app.state.backend
        return client

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """The page shell; the panel inside it refreshes itself."""
        return templates.TemplateResponse(
            request, "index.html", {"refresh_seconds": settings.refresh_seconds}
        )

    @app.get("/panel", response_class=HTMLResponse)
    async def panel(request: Request) -> HTMLResponse:
        """The latest analysis, or a waiting or error message."""
        try:
            response = await backend(request).get("/analyses/latest")
        except httpx2.HTTPError as error:
            # No traceback: htmx polls every few seconds while the backend is down.
            logger.warning("Backend is not reachable: %r", error)
            return templates.TemplateResponse(request, "panel.html", {"state": "down"})
        if response.status_code == 404:
            return templates.TemplateResponse(
                request, "panel.html", {"state": "waiting"}
            )
        if response.is_error:
            logger.warning("Backend returned %s", response.status_code)
            return templates.TemplateResponse(request, "panel.html", {"state": "down"})
        return templates.TemplateResponse(
            request, "panel.html", {"state": "ready", "analysis": response.json()}
        )

    @app.get("/image/{analysis_id}")
    async def image(analysis_id: str, request: Request) -> Response:
        """Annotated image; the id in the URL lets the browser cache each picture."""
        try:
            upstream = await backend(request).get("/images/latest/annotated")
        except httpx2.HTTPError as error:
            logger.warning("Backend is not reachable: %r", error)
            return Response(status_code=502)
        if upstream.is_error:
            return Response(status_code=upstream.status_code)
        return Response(
            upstream.content,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=86400"},
        )

    return app
