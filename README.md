# jewish-book-guide

A conversational AI guide for Jewish books — recommends texts from the Chasidut, Musar, and Jewish Thought traditions based on what you've read and enjoyed.

Built as a portfolio project to demonstrate AI agent development with LangGraph, vector search, and MCP tool integration.

<video src="jewish-book-demo.mov" controls width="100%"></video>

## What it does

- Maintains a curated collection of ~50 canonical Jewish texts ingested from the [Sefaria](https://www.sefaria.org) library API
- Generates vector embeddings (sentence-transformers) and stores them in PostgreSQL with [pgvector](https://github.com/pgvector/pgvector)
- Runs a **LangGraph ReAct agent** powered by Google Gemini that converses with users, calls tools to look up and recommend books, and fetches text passages directly from Sefaria
- Integrates YouTube search via an MCP server to find lectures and shiurim related to any book

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
 │   tool calls ──────────────────────────────────────────┐
 ▼                                                        │
Tools (agent/tools.py)                                    │
  lookup_book        → PostgreSQL (exact/fuzzy match)     │
  get_recommendations→ pgvector cosine sim + re-rank      │
  browse_collection  → PostgreSQL (filtered query)        │
  search_by_theme    → PostgreSQL (array search)          │
  Sefaria MCP tools  → Sefaria MCP server (SSE)        ◄───┤
  YouTube MCP tools  → YouTube MCP server (stdio)      ◄───┘
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
git clone https://github.com/yourusername/jewish-books-guide
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
config/         Book enrichment data (difficulty, themes)
config.py       Central config (DB URL, model, re-ranking weights)
cli.py          Typer CLI entry point
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
