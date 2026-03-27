# Jewish Book Guide

A RAG-based AI agent for exploring and recommending Jewish texts — built with LangGraph, pgvector, Google Gemini, and MCP tool integration. Developed as a portfolio project demonstrating AI agent architecture, vector search, and external API orchestration.


https://github.com/user-attachments/assets/b3179790-780c-4acc-9ed7-e36acbc2e05b


## What it does

- Maintains a curated collection of ~50 canonical Jewish texts ingested from the [Sefaria](https://www.sefaria.org) library API
- Generates vector embeddings (sentence-transformers) and stores them in PostgreSQL with [pgvector](https://github.com/pgvector/pgvector)
- Runs a **LangGraph ReAct agent** powered by Google Gemini that converses with users, retrieves book information via RAG, and uses MCP servers (Sefaria, YouTube) to query text metadata and find related lectures and shiurim

## Architecture

```
User
 │
 ▼
FastAPI server (agent/server.py)
 │   session state (in-memory)
 ▼
LangGraph ReAct agent (agent/graph.py)
 │   Google Gemini
 │   tool calls ───────────────────────────────────────────┐
 ▼                                                         │
Tools (agent/tools.py)                                     │
  lookup_book         → PostgreSQL (exact/fuzzy match)     │
  get_recommendations → pgvector cosine sim + re-rank      │
  browse_collection   → PostgreSQL (filtered query)        │
  search_by_theme     → PostgreSQL (array search)          │
  Sefaria MCP tools   → Sefaria MCP server (SSE)       ◄───┤
  YouTube MCP tools   → YouTube MCP server (stdio)     ◄───┘
```

**Recommendation engine** (recommender/query.py) uses a two-stage approach:
1. Vector cosine similarity retrieves the top 20 candidates
2. Re-ranking applies weighted bonuses for category match, subcategory match, theme overlap, and difficulty alignment

## Setup

### Prerequisites
- Docker
- A Google Gemini API key
- A YouTube Data API v3 key

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

This starts PostgreSQL with pgvector pre-installed, seeds the database with book metadata and embeddings, and serves the app on port 8000.

## Project structure

```
agent/          LangGraph agent (graph, tools, prompts, FastAPI server)
ingestion/      Data pipeline (Sefaria fetch, embedding generation)
recommender/    Two-stage recommendation engine
db/             PostgreSQL schema
frontend/       Single-page chat UI (Tailwind CSS)
config.py       Central config (DB URL, model, re-ranking weights)
cli.py          Typer CLI entry point
tests/          Test suite
```

## Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangGraph
| LLM | Google Gemini (via LangChain) |
| Vector DB | PostgreSQL + pgvector |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Web framework | FastAPI |
| Data source | Sefaria API |
| MCP integration | Sefaria, YouTube search |
