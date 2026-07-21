# Jewish Book Guide

An AI agent for exploring and recommending Jewish texts, built with LangGraph, pgvector, Google Gemini, and custom MCP servers — with both a flat ReAct and a multi-agent supervisor architecture. Developed as a portfolio project demonstrating agent orchestration, RAG-based vector search, and MCP server development.

**[Try it live →](https://jewish-book-guide-887998576030.australia-southeast1.run.app)** — deployed on Google Cloud Run with a Supabase (PostgreSQL + pgvector) backend.


https://github.com/user-attachments/assets/46d31470-8a1a-4529-acd8-6373963e8a1c

*Both Simple and Multi-agent mode run the same tools — the toggle just switches the agent architecture behind the scenes. Either mode can handle any query.*


## What it does

- Recommends similar books and looks up, browses, or searches the curated ~50-book collection — RAG-based vector search, filterable by category, difficulty, or theme
- Searches the broader [Sefaria](https://www.sefaria.org) library for books and passages beyond the curated collection, and fetches text references
- Finds relevant YouTube shiurim/lectures on a book or topic
- Holds a multi-turn conversation via chat, remembering context within a session

## Architecture

The API supports two interchangeable graphs, selected per request with `POST /chat?mode=simple|multi` (default `simple`).

### `simple` — flat ReAct agent

One agent sees every tool from all three MCP servers directly.

```
User
 │
 ▼
FastAPI server (agent/server.py)
 │   session state (in-memory)
 ▼
LangGraph ReAct agent (agent/graph.py)
 │   Google Gemini
 ▼
tool calls
 ├──► Books MCP server (mcp_server/server.py, streamable HTTP :8001)
 │      Tools:    lookup_book         → PostgreSQL (exact/fuzzy match)
 │                get_recommendations → pgvector cosine sim + re-rank
 │                browse_collection   → PostgreSQL (filtered query)
 │                search_by_theme     → PostgreSQL (array search)
 │      Resource: books://all         → full collection dataset
 │      Prompts:  reading_plan, explain_book_to_beginner
 ├──► Sefaria MCP server    → https://mcp.sefaria.org (SSE)
 └──► YouTube MCP server    → npx @kirbah/mcp-youtube (stdio)
```

### `multi` — supervisor with specialist agents

A supervisor ReAct agent (`agent/multi_graph.py`) replaces the "tool calls" step above with three delegation tools — `consult_books`, `consult_sefaria`, `consult_youtube` — each backed by its own full ReAct agent scoped to one MCP server's tools:

```
User → FastAPI server → Supervisor (ReAct agent, Google Gemini)
                          │   tools = consult_books, consult_sefaria, consult_youtube
                          │
              ┌───────────┼───────────────┐
              ▼           ▼               ▼
        Books agent   Sefaria agent   YouTube agent
        (4 books      (all Sefaria    (searchVideos
         tools)         tools)         tool)
```

Agents-as-tools beats one flat agent: each specialist only sees its own tools, and independent `consult_*` calls in one turn run concurrently.

## Data pipeline

The books MCP server's `get_recommendations` tool is powered by a RAG pipeline built ahead of time, independent of the agent:

```
Sefaria API ──► ingestion/fetch_sefaria.py ──► books table (PostgreSQL)
                                                     │
                                    ingestion/embed.py: build_profile()
                                    title + author + themes + description
                                                     │
                                    sentence-transformers (all-MiniLM-L6-v2)
                                                     ▼
                                          books.embedding (pgvector, 384-dim)
```

`recommender/query.py` then serves recommendations in two stages:
1. **Vector search** — pgvector cosine similarity retrieves the top 20 candidates
2. **Re-rank** — weighted bonuses/penalties (`config.py`, `WEIGHT_*`) for category match, subcategory match, theme overlap, and difficulty proximity

## Setup

### Prerequisites
- Docker
- A Google Gemini API key
- A YouTube Data API v3 key
- A [LangSmith](https://smith.langchain.com) API key (optional — for agent tracing)

### Install

```bash
git clone https://github.com/bennymestel/jewish-book-guide
cd jewish-book-guide
cp .env.example .env
# Edit .env and fill in your API keys
```

### Run

```bash
docker compose up -d
# Open http://localhost:8000
```

This starts three services: PostgreSQL (with pgvector), the books MCP server on port 8001, and the FastAPI app on port 8000.

> **Using Claude Code?** This repo includes a custom Skill (`.claude/skills/initialize-project`) — just tell it to start or spin up the project and it'll handle setup automatically.

## Testing & evaluation

```bash
pytest                          # unit tests (no DB or network)
python -m evals.run_evals       # end-to-end agent evals, offline table (needs stack + GOOGLE_API_KEY)
python -m evals.langsmith_eval  # same evals via LangSmith dashboard (also needs LANGCHAIN_API_KEY)
```

The evals run single- and multi-turn questions through the real agent graph, checking tool usage, grounding, difficulty constraints, and LLM-as-judge quality. Run locally for a quick offline table, or via LangSmith for a tracked experiment dashboard with per-case scores and run-over-run comparison. `--mode` selects which graph to evaluate: `simple` (default, the flat ReAct graph) or `multi` (the supervisor/agents-as-tools graph); `both` is only available for `run_evals`.

## Project structure

```
agent/          LangGraph agent (graph.py, multi_graph.py supervisor, prompts, FastAPI server)
mcp_server/     Standalone MCP server exposing four tools, a resource, and two prompts
ingestion/      Data pipeline (Sefaria fetch, embedding generation)
recommender/    Two-stage recommendation engine
db/             PostgreSQL schema
frontend/       Single-page chat UI (Tailwind CSS)
config.py       Central config (DB URL, model, re-ranking weights)
db.py           Shared DB connection helper (bounded connect/statement timeouts)
cli.py          Typer CLI entry point
deploy/         Cloud Run deployment script
evals/          End-to-end eval harness (tool trajectory, grounding, difficulty checks)
tests/          Unit test suite
```

## Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph |
| LLM | Google Gemini (via LangChain) |
| Vector DB | PostgreSQL + pgvector |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Web framework | FastAPI |
| MCP (consumed) | Sefaria (SSE), YouTube (stdio) |
| MCP (built) | Books tool server (streamable HTTP) |
| Data source | Sefaria API |
| Agent Skill | Claude Code Skill (.claude/skills/) |
| Observability | LangSmith (Tracing + Datasets & Experiments) |
