"""
Tests for the FastAPI server endpoints.

Uses httpx.AsyncClient with the ASGI transport to test routes without
starting a real server. The LangGraph agent is mocked so no DB, network,
or API keys are required.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio
import httpx
from langchain_core.messages import AIMessage


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_graph():
    """A fake LangGraph graph that always replies with a fixed message."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="Here are some recommendations.")]}
    )
    return graph


@pytest_asyncio.fixture
async def client(mock_graph):
    """AsyncClient wired to the FastAPI app with the agent graph mocked out."""
    import agent.server as server_module

    with (
        patch("agent.server.load_youtube_tools", new=AsyncMock(return_value=([], None))),
        patch("agent.server.load_sefaria_tools", new=AsyncMock(return_value=([], None))),
        patch("agent.server.build_graph", new=AsyncMock(return_value=mock_graph)),
        patch.object(server_module, "_graph", mock_graph),
    ):
        from agent.server import app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── /chat ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_returns_reply(client):
    r = await client.post("/chat", json={"session_id": "s1", "message": "Hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "s1"
    assert "recommendations" in body["reply"].lower()


@pytest.mark.asyncio
async def test_chat_missing_fields(client):
    r = await client.post("/chat", json={"session_id": "s1"})
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_chat_empty_message(client):
    r = await client.post("/chat", json={"session_id": "s2", "message": ""})
    assert r.status_code == 200
    assert "session_id" in r.json()


@pytest.mark.asyncio
async def test_chat_graph_error_returns_500(client, mock_graph):
    mock_graph.ainvoke.side_effect = RuntimeError("DB is down")
    r = await client.post("/chat", json={"session_id": "s3", "message": "hi"})
    assert r.status_code == 500


# ── /chat/reset ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reset_clears_session(client):
    # First send a message to create a session
    await client.post("/chat", json={"session_id": "sess-reset", "message": "hi"})
    # Then reset it
    r = await client.post("/chat/reset", json={"session_id": "sess-reset"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-reset"
    assert body["status"] == "cleared"


@pytest.mark.asyncio
async def test_reset_nonexistent_session(client):
    """Resetting a session that doesn't exist should still succeed."""
    r = await client.post("/chat/reset", json={"session_id": "ghost-session"})
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"
