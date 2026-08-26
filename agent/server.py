"""
FastAPI server exposing the Jewish book guide agent.

Endpoints:
    POST /chat        { session_id, message } -> { session_id, reply }
    POST /chat/stream { session_id, message } -> text/event-stream (tool + reply events)
    POST /chat/reset  { session_id }          -> { session_id, status }
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG if os.getenv("LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

from langchain_core.messages import HumanMessage
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import db
from agent.graph import build_graph, load_books_tools, load_youtube_tools, load_sefaria_tools
from agent.multi_graph import build_multi_graph

logger = logging.getLogger(__name__)

# ── Daily circuit breaker: a hard backstop on total LLM-backed requests/day. ───
# In-process only — resets on restart and does not coordinate across instances;
# that's fine for a single-instance demo deployment, not a substitute for the
# hard quota that should also be set on the Gemini API key itself.
DAILY_REQUEST_LIMIT = int(os.getenv("DAILY_REQUEST_LIMIT", "300"))
_daily_lock = asyncio.Lock()
_daily_count = 0
_daily_day = None


async def _check_daily_limit() -> None:
    global _daily_count, _daily_day
    today = datetime.now(timezone.utc).date()
    async with _daily_lock:
        if _daily_day != today:
            _daily_day = today
            _daily_count = 0
        if _daily_count >= DAILY_REQUEST_LIMIT:
            raise HTTPException(
                status_code=503,
                detail="This demo has reached its daily usage limit — please check back tomorrow, or see the README to run it locally.",
            )
        _daily_count += 1


QUOTA_MESSAGE = (
    "The demo is temporarily out of AI capacity — it runs on a limited free quota. "
    "Please try again in a little while, or see the README to run it locally."
)


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("resourceexhausted", "quota", "rate limit", "429", "exhausted"))


def _text_from_content(content) -> str | None:
    """Normalize a LangChain message/chunk `.content` (str, or list of content blocks) to text."""
    if isinstance(content, list):
        return "".join(
            block["text"] for block in content if isinstance(block, dict) and "text" in block
        )
    if isinstance(content, str):
        return content
    return None


TOOL_LABELS: dict[str, str] = {
    "get_recommendations": "Finding recommendations",
    "lookup_book": "Looking up book",
    "browse_collection": "Browsing collection",
    "search_by_theme": "Searching by theme",
    "get_text": "Fetching passage",
    "get_text_catalogue_info": "Looking up Sefaria catalogue",
    "searchVideos": "Searching YouTube",
    "consult_books": "Consulting books specialist",
    "consult_sefaria": "Consulting Sefaria specialist",
    "consult_youtube": "Consulting YouTube specialist",
}

_graph = None
_graph_multi = None
_mcp_clients: list = []
_books_tool_count = 0
_books_client = None
_ui_tools: dict[str, str] = {}  # tool name -> ui:// resource URI
_ui_resource_cache: dict[str, str] = {}  # resource URI -> HTML

# In-memory session store: session_id -> AgentState
_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _graph_multi, _mcp_clients, _books_tool_count, _books_client, _ui_tools
    (books_tools, books_client), (youtube_tools, youtube_client), (sefaria_tools, sefaria_client) = (
        await asyncio.gather(load_books_tools(), load_youtube_tools(), load_sefaria_tools())
    )
    _mcp_clients = [c for c in [books_client, youtube_client, sefaria_client] if c is not None]
    _books_client = books_client
    _books_tool_count = len(books_tools)
    if _books_tool_count == 0:
        logger.error("No books MCP tools loaded — the books server is unreachable or failed to start")
    for t in books_tools:
        uri = (t.metadata or {}).get("_meta", {}).get("ui", {}).get("resourceUri")
        if uri:
            _ui_tools[t.name] = uri
    _graph = await build_graph(tools=books_tools + youtube_tools + sefaria_tools)
    _graph_multi = await build_multi_graph(books_tools, sefaria_tools, youtube_tools)
    yield


app = FastAPI(title="Jewish Book Guide", version="1.0", lifespan=lifespan)

@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


@app.get("/ui/{path:path}")
async def ui_resource(path: str) -> HTMLResponse:
    """Serves an MCP Apps ui:// resource fetched from the books MCP server, by URI path."""
    uri = f"ui://{path}"
    if uri not in _ui_tools.values():
        raise HTTPException(status_code=404, detail="Unknown UI resource")
    if uri not in _ui_resource_cache:
        if _books_client is None:
            raise HTTPException(status_code=503, detail="Books MCP server unavailable")
        blobs = await _books_client.get_resources("books", uris=[uri])
        if not blobs:
            raise HTTPException(status_code=404, detail="Resource not found")
        _ui_resource_cache[uri] = blobs[0].as_string()
    return HTMLResponse(content=_ui_resource_cache[uri])

# The frontend is served same-origin (API_BASE="" in frontend/index.html), so no
# browser CORS grant is needed for it to work, locally or deployed. ALLOWED_ORIGINS
# defaults to empty (i.e. none) rather than "*" so a deployed instance isn't an
# open API for any other origin to call from browser JS.
_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., max_length=1000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    session_id: str
    status: str


HISTORY_LIMIT = 20  # keep last 20 messages (~10 exchanges) to cap token usage


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, mode: str = Query("simple")) -> ChatResponse:
    await _check_daily_limit()
    graph = _graph_multi if mode == "multi" else _graph
    prev = _sessions.get(req.session_id, {"messages": []})
    state = {"messages": list(prev["messages"])[-HISTORY_LIMIT:] + [HumanMessage(content=req.message)]}

    try:
        result = await graph.ainvoke(state)
    except Exception as e:
        if _is_quota_error(e):
            raise HTTPException(status_code=503, detail=QUOTA_MESSAGE)
        raise HTTPException(status_code=500, detail=str(e))

    _sessions[req.session_id] = result

    last = result["messages"][-1]
    content = last.content if hasattr(last, "content") else str(last)
    reply = _text_from_content(content)

    return ChatResponse(session_id=req.session_id, reply=reply)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, mode: str = Query("simple")) -> StreamingResponse:
    await _check_daily_limit()
    graph = _graph_multi if mode == "multi" else _graph
    prev = _sessions.get(req.session_id, {"messages": []})
    state = {"messages": list(prev["messages"])[-HISTORY_LIMIT:] + [HumanMessage(content=req.message)]}

    async def event_stream():
        final_state = None
        last_ai_text = None
        # Tracks nesting inside a tool call (incl. a multi-mode specialist agent, which
        # runs inside a consult_* tool body) so we only stream tokens for the top-level
        # answer, not internal tool-calling turns.
        tool_depth = 0
        try:
            async for event in graph.astream_events(state, version="v2"):
                kind = event["event"]
                logger.debug("[STREAM] event=%s name=%s", kind, event.get("name"))
                if kind == "on_tool_start":
                    tool_depth += 1
                    tool_name = event.get("name", "")
                    label = TOOL_LABELS.get(tool_name, tool_name.replace("_", " ").capitalize())
                    yield f"event: tool\ndata: {json.dumps(label)}\n\n"
                elif kind == "on_tool_end":
                    tool_depth = max(0, tool_depth - 1)
                    tool_name = event.get("name", "")
                    resource_uri = _ui_tools.get(tool_name)
                    if resource_uri:
                        try:
                            output = event["data"].get("output")
                            content = output.content if hasattr(output, "content") else output
                            content = _text_from_content(content)
                            payload = json.loads(content)
                            yield f"event: ui\ndata: {json.dumps({'resourceUri': resource_uri, 'toolName': tool_name, 'payload': payload})}\n\n"
                        except Exception as e:
                            logger.debug("[STREAM] skipping ui event for %s: %s", tool_name, e)
                elif kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if (
                        tool_depth == 0
                        and chunk is not None
                        and not getattr(chunk, "tool_call_chunks", None)
                        and not getattr(chunk, "tool_calls", None)
                    ):
                        text = _text_from_content(getattr(chunk, "content", None))
                        if text:
                            yield f"event: token\ndata: {json.dumps(text)}\n\n"
                elif kind == "on_chat_model_end":
                    output = event["data"].get("output")
                    if output is not None:
                        text = _text_from_content(getattr(output, "content", None))
                        if text:
                            last_ai_text = text
                            logger.debug("[STREAM] on_chat_model_end captured text len=%d", len(text))
                elif kind == "on_chain_end":
                    output = event["data"].get("output")
                    logger.debug("[STREAM] on_chain_end output type=%s keys=%s", type(output).__name__, list(output.keys()) if isinstance(output, dict) else "n/a")
                    if isinstance(output, dict) and "messages" in output:
                        final_state = output
        except Exception as e:
            logger.exception("[STREAM] exception: %s", e)
            msg = QUOTA_MESSAGE if _is_quota_error(e) else str(e)
            yield f"event: error\ndata: {json.dumps(msg)}\n\n"
            return

        if final_state is None:
            final_state = state
        _sessions[req.session_id] = final_state

        # Extract reply: prefer the final state messages, fall back to last captured AI text
        last = final_state["messages"][-1]
        content = last.content if hasattr(last, "content") else str(last)
        reply = _text_from_content(content)

        if not reply and last_ai_text:
            logger.debug("[STREAM] falling back to last_ai_text")
            reply = last_ai_text

        yield f"event: reply\ndata: {json.dumps(reply)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/reset", response_model=ResetResponse)
async def reset(req: ResetRequest) -> ResetResponse:
    _sessions.pop(req.session_id, None)
    return ResetResponse(session_id=req.session_id, status="cleared")


@app.get("/health")
async def health() -> dict:
    """Liveness: is the process up?"""
    return {"status": "ok"}


def _check_db() -> None:
    with db.connect() as conn:
        conn.execute("SELECT 1")


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness: can the app actually serve a request right now? Touches the DB (with
    the bounded timeout from db.connect), so this doubles as the keep-warm target —
    pinging it periodically keeps both this instance and DB from going idle."""
    try:
        await asyncio.to_thread(_check_db)
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "db": str(e)})
    if _books_tool_count == 0:
        return JSONResponse(status_code=503, content={"status": "error", "db": "ok", "books_tools": 0})
    return JSONResponse(status_code=200, content={"status": "ok", "db": "ok", "books_tools": _books_tool_count})
