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


# ── Multi-graph fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def mock_multi_graph():
    """A fake multi-agent graph that always replies with a fixed message."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="Multi-agent reply.")]}
    )
    return graph


@pytest_asyncio.fixture
async def client_multi(mock_graph, mock_multi_graph):
    """AsyncClient with both simple and multi graphs mocked out."""
    import agent.server as server_module

    with (
        patch("agent.server.load_books_tools", new=AsyncMock(return_value=([], None))),
        patch("agent.server.load_youtube_tools", new=AsyncMock(return_value=([], None))),
        patch("agent.server.load_sefaria_tools", new=AsyncMock(return_value=([], None))),
        patch("agent.server.build_graph", new=AsyncMock(return_value=mock_graph)),
        patch("agent.server.build_multi_graph", new=AsyncMock(return_value=mock_multi_graph)),
        patch.object(server_module, "_graph", mock_graph),
        patch.object(server_module, "_graph_multi", mock_multi_graph),
    ):
        from agent.server import app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


# ── /chat?mode=multi ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_mode_multi_uses_multi_graph(client_multi, mock_graph, mock_multi_graph):
    r = await client_multi.post("/chat?mode=multi", json={"session_id": "m1", "message": "hi"})
    assert r.status_code == 200
    assert "Multi-agent" in r.json()["reply"]
    mock_multi_graph.ainvoke.assert_called_once()
    mock_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_chat_default_mode_is_simple(client_multi, mock_graph, mock_multi_graph):
    r = await client_multi.post("/chat", json={"session_id": "m2", "message": "hi"})
    assert r.status_code == 200
    mock_graph.ainvoke.assert_called_once()
    mock_multi_graph.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_chat_mode_multi_error_returns_500(client_multi, mock_multi_graph):
    mock_multi_graph.ainvoke.side_effect = RuntimeError("specialist failed")
    r = await client_multi.post("/chat?mode=multi", json={"session_id": "m3", "message": "hi"})
    assert r.status_code == 500


@pytest.mark.asyncio
async def test_chat_stream_mode_multi(client_multi, mock_multi_graph):
    """POST /chat/stream?mode=multi should yield a reply event."""
    # astream_events must be a plain callable returning an async generator, not an AsyncMock
    mock_multi_graph.astream_events = MagicMock(return_value=_fake_stream())
    r = await client_multi.post(
        "/chat/stream?mode=multi", json={"session_id": "m4", "message": "hi"}
    )
    assert r.status_code == 200
    assert b"reply" in r.content


@pytest.mark.asyncio
async def test_chat_stream_only_tokens_outside_tool_calls(client_multi, mock_multi_graph):
    """Chat-model stream chunks emitted while inside a tool call (e.g. a specialist
    agent's internal turn) must not become `token` events; only the top-level
    answer's chunks should."""
    mock_multi_graph.astream_events = MagicMock(return_value=_fake_stream_with_tokens())
    r = await client_multi.post(
        "/chat/stream?mode=multi", json={"session_id": "m5", "message": "hi"}
    )
    assert r.status_code == 200
    assert r.content.count(b"event: token") == 1
    assert b'data: "visible answer"' in r.content
    assert b"internal reasoning" not in r.content
    assert r.content.rstrip().endswith(b'event: reply\ndata: "final reply"')


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _fake_stream():
    """Minimal astream_events output: one on_chain_end event with a messages key."""
    from langchain_core.messages import AIMessage as AI
    yield {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"messages": [AI(content="streamed multi reply")]}},
    }


async def _fake_stream_with_tokens():
    """astream_events output with a token chunk inside a tool call (should be
    suppressed) and one outside it (should be forwarded as a `token` event)."""
    from langchain_core.messages import AIMessage as AI, AIMessageChunk

    yield {"event": "on_tool_start", "name": "consult_books", "data": {}}
    yield {
        "event": "on_chat_model_stream",
        "name": "agent",
        "data": {"chunk": AIMessageChunk(content="internal reasoning")},
    }
    yield {"event": "on_tool_end", "name": "consult_books", "data": {"output": ""}}
    yield {
        "event": "on_chat_model_stream",
        "name": "agent",
        "data": {"chunk": AIMessageChunk(content="visible answer")},
    }
    yield {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"messages": [AI(content="final reply")]}},
    }
