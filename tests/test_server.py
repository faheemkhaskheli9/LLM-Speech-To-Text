"""Smoke tests for the WebSocket-proxy server (issue #1). No real OpenAI
connection or microphone is used -- these only exercise the plain HTTP
route and the "no API key" error path, which need neither."""

from __future__ import annotations

import importlib

import server
from fastapi.testclient import TestClient


def test_root_serves_index_html():
    client = TestClient(server.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Realtime Speech" in resp.text


def test_index_html_lives_next_to_the_server_module():
    assert server.INDEX_HTML.exists()
    assert server.INDEX_HTML.name == "index.html"


def test_websocket_closes_with_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(server, "OPENAI_API_KEY", None)
    client = TestClient(server.app)

    with client.websocket_connect("/ws/transcribe") as ws:
        msg = ws.receive_json()

    assert msg == {"type": "error", "message": "OPENAI_API_KEY not set"}


def test_cors_defaults_to_localhost_not_wildcard():
    origins, allow_credentials = server._resolve_cors_origins()
    assert origins != ["*"]
    assert all("localhost" in o or "127.0.0.1" in o for o in origins)
    assert allow_credentials is True


def test_cors_origin_is_configurable_via_env_var(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://example.com,https://app.example.com")
    origins, allow_credentials = server._resolve_cors_origins()
    assert origins == ["https://example.com", "https://app.example.com"]
    assert allow_credentials is True


def test_cors_wildcard_requires_explicit_opt_in_and_disables_credentials(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    origins, allow_credentials = server._resolve_cors_origins()
    assert origins == ["*"]
    assert allow_credentials is False


def test_module_wires_the_resolved_origins_into_the_app(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://example.com")
    reloaded = importlib.reload(server)
    try:
        middleware = next(
            m for m in reloaded.app.user_middleware if m.cls.__name__ == "CORSMiddleware"
        )
        assert middleware.kwargs["allow_origins"] == ["https://example.com"]
    finally:
        monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
        importlib.reload(server)
