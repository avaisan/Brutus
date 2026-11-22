from collections.abc import Callable, Iterator

import httpx2
import pytest
from fastapi.testclient import TestClient

from brutus_frontend.app import create_app
from brutus_frontend.settings import Settings
from tests.analysis_fixture import analysis

type Handler = Callable[[httpx2.Request], httpx2.Response]


def client_for(handler: Handler) -> Iterator[TestClient]:
    app = create_app(
        Settings(backend_url="http://backend.test"),
        transport=httpx2.MockTransport(handler),
    )
    with TestClient(app) as client:
        yield client


@pytest.fixture
def backend_with_analysis() -> Iterator[TestClient]:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/analyses/latest":
            return httpx2.Response(200, json=analysis())
        if request.url.path == "/images/latest/annotated":
            return httpx2.Response(200, content=b"\xff\xd8jpeg")
        return httpx2.Response(404)

    yield from client_for(handler)


class TestIndex:
    def test_given_app_when_index_requested_then_page_polls_panel(
        self, backend_with_analysis: TestClient
    ) -> None:
        response = backend_with_analysis.get("/")

        assert response.status_code == 200
        assert 'hx-get="/panel"' in response.text
        assert "every 3s" in response.text


class TestPanel:
    def test_given_backend_analysis_when_panel_requested_then_results_rendered(
        self, backend_with_analysis: TestClient
    ) -> None:
        response = backend_with_analysis.get("/panel")

        assert response.status_code == 200
        assert "2 people · 1 dog · 2 chairs" in response.text
        assert "P1 · Kirk (98.7%)" in response.text
        assert 'src="/image/abc123"' in response.text
        assert "Indoors" in response.text

    def test_given_no_analysis_yet_when_panel_requested_then_waiting_message(
        self,
    ) -> None:
        for client in client_for(lambda _: httpx2.Response(404)):
            assert "Waiting for the first picture" in client.get("/panel").text

    def test_given_backend_down_when_panel_requested_then_unreachable_message(
        self,
    ) -> None:
        def down(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("refused", request=request)

        for client in client_for(down):
            response = client.get("/panel")
            assert response.status_code == 200
            assert "Backend is not reachable" in response.text


class TestImageProxy:
    def test_given_annotated_image_when_requested_then_proxied_jpeg(
        self, backend_with_analysis: TestClient
    ) -> None:
        response = backend_with_analysis.get("/image/abc123")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"\xff\xd8jpeg"

    def test_given_backend_down_when_image_requested_then_bad_gateway(
        self,
    ) -> None:
        def down(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("refused", request=request)

        for client in client_for(down):
            assert client.get("/image/abc123").status_code == 502
