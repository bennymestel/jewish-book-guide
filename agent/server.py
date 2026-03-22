"""
FastAPI server exposing the Jewish book guide agent.

Endpoints:
    POST /chat        { session_id, message } -> { session_id, reply }
    POST /chat/stream { session_id, message } -> text/event-stream (tool + reply events)
    POST /chat/reset  { session_id }          -> { session_id, status }
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG if os.getenv("LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

from langchain_core.messages import HumanMessage
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent.graph import build_graph, load_youtube_tools, load_sefaria_tools

TOOL_LABELS: dict[str, str] = {
    "get_recommendations": "Finding recommendations",
    "lookup_book": "Looking up book",
    "browse_collection": "Browsing collection",
    "search_by_theme": "Searching by theme",
    "get_sefaria_passage": "Fetching passage",
    "searchVideos": "Searching YouTube",
}

_graph = None
_mcp_clients: list = []

# In-memory session store: session_id -> AgentState
_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _mcp_clients
    youtube_tools, youtube_client = await load_youtube_tools()
    sefaria_tools, sefaria_client = await load_sefaria_tools()
    _mcp_clients = [c for c in [youtube_client, sefaria_client] if c is not None]
    _graph = await build_graph(extra_tools=youtube_tools + sefaria_tools)
    yield


app = FastAPI(title="Jewish Book Guide", version="1.0", lifespan=lifespan)

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    session_id: str
    status: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    prev = _sessions.get(req.session_id, {"messages": []})
    state = {"messages": list(prev["messages"]) + [HumanMessage(content=req.message)]}

    try:
        result = await _graph.ainvoke(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _sessions[req.session_id] = result

    last = result["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    if isinstance(content, list):
        reply = "".join(
            block["text"] for block in content if isinstance(block, dict) and "text" in block
        )
    else:
        reply = content

    return ChatResponse(session_id=req.session_id, reply=reply)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    prev = _sessions.get(req.session_id, {"messages": []})
    state = {"messages": list(prev["messages"]) + [HumanMessage(content=req.message)]}

    async def event_stream():
        final_state = state
        try:
            async for event in _graph.astream_events(state, version="v2"):
                kind = event["event"]
                if kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    label = TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").capitalize())
                    yield f"event: tool\ndata: {json.dumps(label)}\n\n"
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output", state)
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
            return

        _sessions[req.session_id] = final_state
        last = final_state["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        if isinstance(content, list):
            reply = "".join(
                block["text"] for block in content if isinstance(block, dict) and "text" in block
            )
        else:
            reply = content
        yield f"event: reply\ndata: {json.dumps(reply)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/reset", response_model=ResetResponse)
async def reset(req: ResetRequest) -> ResetResponse:
    _sessions.pop(req.session_id, None)
    return ResetResponse(session_id=req.session_id, status="cleared")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
