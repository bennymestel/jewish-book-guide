# Multi-Agent Supervisor Graph — Implementation Plan

## Context

This is a learning/portfolio project: a Jewish book guide agent on LangGraph + Google Gemini. Today it runs a single flat ReAct loop (`agent/graph.py`) where one agent sees every tool (books MCP, Sefaria MCP, YouTube MCP). The goal is to add a **second graph** that demonstrates a current-best-practice multi-agent architecture, kept alongside the existing one and selectable via `?mode=simple|multi` on `/chat` and `/chat/stream`.

Two design decisions were settled with the user:

1. **No rigid "books-first" pipeline.** The entry point depends on the query — "videos on Mesilat Yesharim" goes straight to YouTube; "a quote from Vayikra" goes straight to Sefaria; "recommend a book and a passage from it" needs books *then* Sefaria. So routing must be dynamic.
2. **Supervisor / agents-as-tools pattern**, confirmed as LangChain 1.0's officially recommended approach (build the supervisor *directly via tools*, not the `langgraph-supervisor` library — that library is now only for upgrading legacy code). Source: LangChain multi-agent guide / langgraph-supervisor README.

The key correctness principle (and the thing that makes this best-practice rather than naive): **each specialist is a full agent exposed to the supervisor as a single tool.** The supervisor never sees raw MCP tools — only `consult_books`, `consult_sefaria`, `consult_youtube`. The raw domain tools live *inside* each specialist, where a focused prompt can use them well (this is how the currently-ignored Sefaria tools finally get used).

## Architecture

```
                    ┌──────────────────────────────┐
   user msg ───────▶│  SUPERVISOR (a ReAct agent)   │
                    │  tools = [consult_books,       │
                    │           consult_sefaria,     │
                    │           consult_youtube]     │
                    └───────┬───────────┬───────────┘
            (LLM picks any subset, any order; multiple
             calls in one turn run concurrently)
              │            │            │
      ┌───────▼──┐  ┌──────▼─────┐  ┌───▼────────┐
      │ Books    │  │ Sefaria    │  │ YouTube    │   each = create_react_agent
      │ agent    │  │ agent      │  │ agent      │   with its own focused prompt
      │ (4 books │  │ (ALL sefaria│ │ (searchVideos)│ + only its domain tools
      │  tools)  │  │  tools)    │  │            │
      └──────────┘  └────────────┘  └────────────┘
```

- **Dynamic entry / combinations**: the supervisor LLM decides which specialists to consult per query. Any subset, any order.
- **Parallelism is emergent**: when the supervisor emits two independent tool calls in one turn (e.g. passage + video for an already-named book), LangGraph's tool execution runs them **concurrently** because the consult tools are `async`. When there's a dependency (pick a book → then quote it), the LLM naturally sequences across turns.
- **Termination**: the supervisor stops when it returns a final answer with no tool calls (standard ReAct end condition) — encoded in its prompt ("when the request is fully addressed, produce the final answer and stop").

## New Files

### `agent/prompts_multi.py`
Four prompt constants, each short and role-scoped:
- `SUPERVISOR_PROMPT` — explains the three `consult_*` delegation tools; instructs the LLM to call only what the query needs, to issue independent calls together, to pass the relevant book title/reference into `consult_sefaria`/`consult_youtube` when one depends on a books result, and to synthesize a final answer applying the existing length rules (3 sentences max conversational; one line per book for lists). **Encodes the books→Sefaria fallback explicitly: "If `consult_books` reports a title isn't in the curated collection, call `consult_sefaria` to look it up there before telling the user it's unavailable."**
- `BOOKS_AGENT_PROMPT` — scoped to the 4 books tools; mirrors the books portion of the current `SYSTEM_PROMPT` (always `lookup_book` before naming a title; the `Title - Author - Difficulty: N - description` format). **When `lookup_book` returns "not found in the local collection", report that plainly upward — do NOT attempt to reach Sefaria (this specialist has no Sefaria tools); the supervisor owns that fallback.**
- `SEFARIA_AGENT_PROMPT` — scoped to text retrieval; **explicitly tells the agent it has the full Sefaria toolset** and to search the catalogue first when the exact reference is unknown, then fetch the passage. Returns the passage or a clean "not available".
- `YOUTUBE_AGENT_PROMPT` — scoped to `searchVideos`; max 3 introductory results as clickable `title — channel — url`.

### `agent/multi_graph.py`
```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import config
from agent.graph import build_graph            # reuse the existing ReAct builder as the supervisor
from agent.prompts_multi import (
    SUPERVISOR_PROMPT, BOOKS_AGENT_PROMPT, SEFARIA_AGENT_PROMPT, YOUTUBE_AGENT_PROMPT,
)

async def build_multi_graph(books_tools, sefaria_tools, youtube_tools):
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, ..., temperature=0.3)

    # 1. Build each specialist as a full ReAct agent (idiomatic LangGraph prebuilt).
    books_agent   = create_react_agent(llm, books_tools,   prompt=BOOKS_AGENT_PROMPT)
    sefaria_agent = create_react_agent(llm, sefaria_tools, prompt=SEFARIA_AGENT_PROMPT)
    youtube_agent = create_react_agent(llm, youtube_tools, prompt=YOUTUBE_AGENT_PROMPT)

    # 2. Wrap each specialist as ONE async tool the supervisor can call.
    #    (Async so multiple consults issued in one turn execute concurrently.)
    @tool
    async def consult_books(request: str) -> str:
        """Recommend, look up, browse, or theme-search books in the curated collection."""
        out = await books_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        return out["messages"][-1].content

    @tool
    async def consult_sefaria(request: str) -> str:
        """Fetch or search actual Jewish text passages from Sefaria. Pass the title/reference."""
        out = await sefaria_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        return out["messages"][-1].content

    @tool
    async def consult_youtube(request: str) -> str:
        """Find YouTube shiurim/lectures. Pass the book or topic to search for."""
        out = await youtube_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        return out["messages"][-1].content

    consult_tools = [consult_books, consult_sefaria, consult_youtube]
    if not youtube_tools:                 # YOUTUBE_API_KEY absent → drop the youtube specialist
        consult_tools.remove(consult_youtube)

    # 3. The supervisor IS the existing ReAct graph — its tools are now agents.
    return await build_graph(tools=consult_tools, system_prompt=SUPERVISOR_PROMPT)
```
- Returns a graph with the **same `AgentState` shape** (`messages` only) as the simple graph, so session handling is identical across modes.
- If a specialist's tool list is empty (e.g. no YouTube key), its consult tool is omitted so the supervisor won't try to call a dead specialist.

## Modified Files

### `agent/graph.py` — one small, backward-compatible change
Parametrize the supervisor prompt so `build_graph` can be reused:
```python
from agent.prompts import SYSTEM_PROMPT

async def build_graph(tools: list = [], system_prompt: str = SYSTEM_PROMPT):
    ...
    async def agent_node(state):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        ...
```
Existing call sites pass no `system_prompt` and get the current behavior unchanged.

### `agent/server.py`
1. Imports: add `from agent.multi_graph import build_multi_graph` and `Query` to the FastAPI import.
2. Global: add `_graph_multi = None`.
3. `lifespan`: keep loading the three tool groups separately (they already are), build `_graph` as today, then also:
   ```python
   _graph_multi = await build_multi_graph(books_tools, sefaria_tools, youtube_tools)
   ```
4. `/chat` and `/chat/stream`: add `mode: str = Query("simple")`; at the top select `graph = _graph_multi if mode == "multi" else _graph`. `ChatRequest` body is unchanged, so existing clients keep working and default to `simple`.
5. Session persistence: continue storing only `result["messages"]` back into `_sessions` (both graphs share the `AgentState` shape, so sessions are interchangeable between modes).
6. `TOOL_LABELS`: add labels for `consult_books` → "Consulting books specialist", `consult_sefaria` → "Consulting Sefaria specialist", `consult_youtube` → "Consulting YouTube specialist". Note: in `mode=multi`, `astream_events` surfaces tool-start events from **both** the consult wrappers and the raw MCP tools running inside specialists, so the existing labels ("Looking up book", etc.) still appear too — the stream gets richer, not broken.

### `tests/test_server.py`
Add alongside existing fixtures (no edits to existing tests), following the current `patch`/`AsyncClient` pattern:
- `mock_multi_graph` fixture (MagicMock, `ainvoke = AsyncMock(...)`).
- `client_multi` fixture patching `agent.server.build_multi_graph`, `build_graph`, and the three tool loaders; injects both `_graph` and `_graph_multi`.
- `test_chat_mode_multi_uses_multi_graph` — `POST /chat?mode=multi` invokes the multi graph.
- `test_chat_default_mode_is_simple` — `POST /chat` (no param) invokes the simple graph.
- `test_chat_mode_multi_error_returns_500` — multi graph raises → 500.
- `test_chat_stream_mode_multi` — `POST /chat/stream?mode=multi` yields a `reply` event.

Plus a unit-level check in `tests/test_pure.py` style if desired: assert `build_multi_graph([], [], [])` omits `consult_youtube` (uses a stubbed `create_react_agent`/`build_graph` so no network) — optional, only if it stays cheap.

## Implementation Order
1. `agent/prompts_multi.py` (no deps).
2. `agent/graph.py` — add the `system_prompt` parameter.
3. `agent/multi_graph.py` — specialists, consult-tool wrappers, supervisor assembly.
4. `agent/server.py` — `mode` routing + dual-graph startup + tool labels.
5. `tests/test_server.py` — new fixtures and tests.

## Verification
```bash
pytest                        # unit + server tests, no DB/network

docker compose up             # full stack
# Dynamic entry point — should NOT touch books:
curl -X POST "http://localhost:8000/chat?mode=multi" -H "Content-Type: application/json" \
  -d '{"session_id":"a","message":"find me videos on Mesilat Yesharim"}'
# Sefaria-only entry:
curl -X POST "http://localhost:8000/chat?mode=multi" -H "Content-Type: application/json" \
  -d '{"session_id":"b","message":"what does Vayikra 19:18 say?"}'
# Books→Sefaria FALLBACK (title not in DB): supervisor should consult_books → "not found" → consult_sefaria
curl -X POST "http://localhost:8000/chat?mode=multi" -H "Content-Type: application/json" \
  -d '{"session_id":"f","message":"tell me about a book not in your collection, e.g. some obscure title"}'
# Dependency: books THEN sefaria:
curl -X POST "http://localhost:8000/chat?mode=multi" -H "Content-Type: application/json" \
  -d '{"session_id":"c","message":"recommend a book on prayer and quote a related passage"}'
# Parallel: passage + video for a known book (independent → concurrent):
curl -X POST "http://localhost:8000/chat?mode=multi" -H "Content-Type: application/json" \
  -d '{"session_id":"d","message":"for Mesilat Yesharim, give me a passage and a video"}'
# Regression: simple mode unchanged:
curl -X POST "http://localhost:8000/chat?mode=simple" -H "Content-Type: application/json" \
  -d '{"session_id":"e","message":"recommend a book on prayer"}'
```
With `LANGCHAIN_TRACING_V2=true`, the LangSmith trace for `mode=multi` should show the supervisor calling `consult_*` tools, each expanding into a nested specialist run — and for the last query, two consult calls running as concurrent siblings.

## Portfolio note
Keep the simple graph (`mode=simple`) intact. Being able to demo flat-ReAct vs. supervisor side by side — and explain *why* agents-as-tools beats dumping every tool into one agent (tool-selection accuracy, context isolation, the previously-unused Sefaria tools now reachable) — is the strongest interview story.
