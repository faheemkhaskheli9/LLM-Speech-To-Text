"""Docs stay in sync with the code they describe (issue #1)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_env_example_has_openai_api_key():
    content = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in content


def test_readme_explains_the_websocket_proxy_architecture():
    content = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "WebSocket" in content
    assert "index.html" in content
    assert "/ws/transcribe" in content
