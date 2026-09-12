"""Smoke tests for the WebSocket-proxy server (issue #1). No real OpenAI
connection or microphone is used -- these only exercise the plain HTTP
route and the "no API key" error path, which need neither."""

from __future__ import annotations

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
