# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Stack

- **Agent**: LangGraph ReAct graph (`agent/graph.py`) powered by Google Gemini, connecting to three MCP servers at startup
- **API server**: FastAPI (`agent/server.py`) at `:8000` — sessions are stored in-memory
- **Books MCP server**: FastMCP (`mcp_server/server.py`) at `:8001/mcp` over streamable HTTP
- **Database**: PostgreSQL with pgvector — `pgvector/pgvector:pg17` Docker image, database `books`
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers, 384 dimensions, stored in `books.embedding`

## Environment

Set these via `.env` (copy `.env.example`) or as shell environment variables:
```
GOOGLE_API_KEY=...       # required
YOUTUBE_API_KEY=...      # optional; YouTube search is skipped if absent
DATABASE_URL=...         # defaults to postgresql://localhost/books
GEMINI_MODEL=...         # defaults to gemini-3.1-flash-lite-preview

# LangSmith observability (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...     # your LangSmith API key
LANGCHAIN_PROJECT=...     # defaults to jewish-book-guide
```

## Running the stack

Docker handles startup order (db → mcp → api):
```bash
docker compose up
```

The frontend is served at `http://localhost:8000` by the FastAPI server (`GET /`).

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run unit tests (no DB or network required)
pytest

# Run end-to-end agent evals (needs the stack running + GOOGLE_API_KEY)
python -m evals.run_evals
```

## Architecture: data flow

1. `docker compose up` starts db, then the books MCP server, then the FastAPI agent server
2. On FastAPI startup (`lifespan`), the agent connects to all three MCP servers and builds the LangGraph graph with their tools bound to Gemini
3. `POST /chat` or `POST /chat/stream` appends the user message to the session history and invokes the graph; the graph loops agent↔tools until it produces a final reply
4. The books MCP server (`mcp_server/server.py`) handles tool calls by querying PostgreSQL directly — it does **not** go through `agent/`

## Architecture: recommendation engine

`recommender/query.py` uses a two-stage pipeline called by the `get_recommendations` MCP tool:
1. **Stage 1**: pgvector cosine similarity — retrieves top-20 candidates by embedding distance, excluding seed books
2. **Stage 2**: re-rank with bonuses/penalties defined in `config.py` (`WEIGHT_*`) — same category, same subcategory, theme overlap, difficulty proximity

The embedding for each book is built by `ingestion/embed.py:build_profile`, which concatenates title, author, themes, and description into a single string before encoding.

## Key MCP tools exposed by the books server

- `lookup_book` — fuzzy title/key match, returns full metadata
- `get_recommendations` — two-stage vector + re-rank pipeline
- `browse_collection` — filtered SQL query (category, difficulty, foundational flag)
- `search_by_theme` — `unnest(themes) ILIKE` search
- Resource `books://all` — full collection dump as JSON
- Prompts: `reading_plan`, `explain_book_to_beginner`

## External MCP servers

- **Sefaria** (`https://mcp.sefaria.org/sse`) — SSE transport, used for fetching actual text passages
- **YouTube** (`npx @kirbah/mcp-youtube`) — stdio transport, only `searchVideos` tool is loaded; requires `YOUTUBE_API_KEY`

## Session state

Sessions are stored in `_sessions: dict[str, dict]` in `agent/server.py` — in-memory only, lost on restart. History is trimmed to the last 20 messages per session (`HISTORY_LIMIT`).
