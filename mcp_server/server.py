"""
Standalone MCP server exposing Jewish book guide tools over
streamable HTTP.
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import psycopg
import psycopg.rows
from mcp.server.fastmcp import FastMCP

import config
import db

logging.basicConfig(
    level=logging.DEBUG if os.getenv("LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP("jewish-books", host="0.0.0.0", port=8001)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_book(title_query: str):
    from recommender.query import find_book
    return find_book(title_query)


def _recommend(seed_titles, top_n=3, difficulty_pref=None, category_pref=None):
    from recommender.query import recommend
    return recommend(seed_titles, top_n=top_n, difficulty_pref=difficulty_pref, category_pref=category_pref)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def lookup_book(title_query: str) -> str:
    """Look up a Jewish book by title or Sefaria key in the local curated collection.
    Returns detailed metadata including author, category, difficulty, themes, and description.
    Use this to verify a book exists and get its details before recommending.
    If the book is not found locally, use the Sefaria MCP tools to look it up."""
    logger.info("[TOOL] lookup_book: title_query=%r", title_query)
    book = _find_book(title_query)
    if book is None:
        return (
            f"'{title_query}' was not found in the local collection. "
            "Use the Sefaria MCP tools (e.g. get_text_catalogue_info) to look it up."
        )

    difficulty_label = config.DIFFICULTY_LABELS.get(book.get("difficulty") or 0, "")

    result = {
        "title": book["title_en"],
        "sefaria_key": book["sefaria_key"],
        "author": book.get("author_en") or "Unknown",
        "category": book.get("category") or "—",
        "subcategory": book.get("subcategory") or None,
        "pub_date": book.get("pub_date"),
        "difficulty": book.get("difficulty"),
        "difficulty_label": difficulty_label,
        "themes": book.get("themes") or [],
        "is_foundational": book.get("is_foundational") or False,
        "description": book.get("desc_en") or book.get("desc_en_short") or "No description available.",
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def get_recommendations(
    seed_titles: list[str],
    top_n: int = 3,
    difficulty: int | None = None,
    category: str | None = None,
) -> str:
    """Get book recommendations similar to one or more seed books.
    seed_titles: list of book titles or Sefaria keys (verify they exist with lookup_book first).
    difficulty: target difficulty 1-5 (optional).
    category: filter to 'Chasidut', 'Musar', or 'Jewish Thought' (optional).
    Returns a ranked list of recommended books."""
    logger.info("[TOOL] get_recommendations: seeds=%r top_n=%d difficulty=%r category=%r", seed_titles, top_n, difficulty, category)
    validated: list[str] = []
    not_found: list[str] = []
    for title in seed_titles:
        book = _find_book(title)
        if book:
            validated.append(book["title_en"])
        else:
            not_found.append(title)

    if not_found:
        msg = f"Could not find the following books: {', '.join(not_found)}. Please verify the titles."
        if not validated:
            return msg

    results = _recommend(
        seed_titles=validated,
        top_n=top_n,
        difficulty_pref=difficulty,
        category_pref=category,
    )

    if not results:
        return "No recommendations found with those filters. Try loosening the difficulty or category constraints."

    books = []
    for r in results:
        diff = r.get("difficulty")
        diff_label = config.DIFFICULTY_LABELS.get(diff or 0, "") if diff else ""
        books.append({
            "title": r["title_en"],
            "author": r.get("author_en") or "Unknown",
            "category": r.get("category") or "—",
            "difficulty": diff,
            "difficulty_label": diff_label,
            "score": round(r.get("score", 0), 3),
            "description": r.get("desc_en_short") or "No description available.",
            "themes": r.get("themes") or [],
        })

    output = {"recommendations": books}
    if not_found:
        output["warning"] = f"Seeds not found (ignored): {', '.join(not_found)}"
    return json.dumps(output, ensure_ascii=False)


@mcp.tool()
def browse_collection(
    category: str | None = None,
    difficulty_max: int | None = None,
    foundational_only: bool = False,
    limit: int = 10,
) -> str:
    """Browse the book collection with optional filters.
    category: 'Chasidut', 'Musar', or 'Jewish Thought' (optional).
    difficulty_max: show only books at or below this difficulty level 1-5 (optional).
    foundational_only: if True, return only foundational/entry-level texts.
    Returns a list of matching books ordered by foundational status then difficulty."""
    logger.info("[TOOL] browse_collection: category=%r difficulty_max=%r foundational_only=%r limit=%d", category, difficulty_max, foundational_only, limit)
    conditions = ["1=1"]
    params: list = []

    if category:
        conditions.append("category ILIKE %s ESCAPE '\\'")
        params.append(f"%{_escape_like(category)}%")
    if difficulty_max is not None:
        conditions.append("(difficulty IS NULL OR difficulty <= %s)")
        params.append(difficulty_max)
    if foundational_only:
        conditions.append("is_foundational = TRUE")

    params.append(limit)

    sql = f"""
        SELECT title_en, author_en, category, subcategory,
               difficulty, themes, is_foundational, desc_en_short
        FROM books
        WHERE {" AND ".join(conditions)}
        ORDER BY is_foundational DESC NULLS LAST, difficulty ASC NULLS LAST
        LIMIT %s
    """

    try:
        with db.connect(row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except Exception as e:
        return f"Error querying collection: {e}"

    if not rows:
        return "No books found matching those filters."

    books = []
    for r in rows:
        diff = r.get("difficulty")
        diff_label = config.DIFFICULTY_LABELS.get(diff or 0, "") if diff else ""
        books.append({
            "title": r["title_en"],
            "author": r.get("author_en") or "Unknown",
            "category": r.get("category") or "—",
            "difficulty": diff,
            "difficulty_label": diff_label,
            "is_foundational": r.get("is_foundational") or False,
            "description": r.get("desc_en_short") or "",
            "themes": r.get("themes") or [],
        })

    return json.dumps({"books": books}, ensure_ascii=False)


@mcp.tool()
def search_by_theme(theme: str, limit: int = 8) -> str:
    """Find books related to a specific theme or topic (e.g. 'prayer', 'teshuvah', 'Kabbalah', 'love of God').
    Returns books whose themes array contains the given theme."""
    logger.info("[TOOL] search_by_theme: theme=%r limit=%d", theme, limit)
    try:
        with db.connect(row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT title_en, author_en, category, difficulty, themes, desc_en_short
                    FROM books
                    WHERE EXISTS (
                        SELECT 1 FROM unnest(themes) t WHERE t ILIKE %s ESCAPE '\\'
                    )
                    ORDER BY is_foundational DESC NULLS LAST, difficulty ASC NULLS LAST
                    LIMIT %s
                    """,
                    (f"%{_escape_like(theme)}%", limit),
                )
                rows = cur.fetchall()
    except Exception as e:
        return f"Error searching by theme: {e}"

    if not rows:
        return f"No books found with the theme '{theme}'. Try a broader term like 'prayer', 'soul', or 'ethics'."

    books = []
    for r in rows:
        diff = r.get("difficulty")
        diff_label = config.DIFFICULTY_LABELS.get(diff or 0, "") if diff else ""
        books.append({
            "title": r["title_en"],
            "author": r.get("author_en") or "Unknown",
            "category": r.get("category") or "—",
            "difficulty": diff,
            "difficulty_label": diff_label,
            "description": r.get("desc_en_short") or "",
            "themes": r.get("themes") or [],
        })

    return json.dumps({"theme": theme, "books": books}, ensure_ascii=False)


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("books://all")
def all_books() -> str:
    """The full curated Jewish book collection as JSON.
    Returns every book with title, author, category, difficulty, themes, and description.
    Fetch this once to have the complete dataset available as context."""
    try:
        with db.connect(row_factory=psycopg.rows.dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT title_en, author_en, category, subcategory,
                           difficulty, themes, is_foundational, desc_en_short
                    FROM books
                    ORDER BY category, difficulty ASC NULLS LAST
                """)
                rows = cur.fetchall()
    except Exception as e:
        return json.dumps({"error": str(e)})

    books = []
    for r in rows:
        diff = r.get("difficulty")
        books.append({
            "title": r["title_en"],
            "author": r.get("author_en") or "Unknown",
            "category": r.get("category") or "—",
            "subcategory": r.get("subcategory"),
            "difficulty": diff,
            "difficulty_label": config.DIFFICULTY_LABELS.get(diff or 0, "") if diff else "",
            "is_foundational": r.get("is_foundational") or False,
            "themes": r.get("themes") or [],
            "description": r.get("desc_en_short") or "",
        })

    return json.dumps({"total": len(books), "books": books}, ensure_ascii=False)


# ── Prompts ────────────────────────────────────────────────────────────────────

@mcp.prompt()
def reading_plan(topic: str, background: str = "none") -> str:
    """Generate a structured reading plan prompt for a given topic and background level.
    topic: the subject of interest (e.g. 'prayer', 'teshuvah', 'Kabbalah').
    background: the user's prior knowledge in their own words (e.g. 'none', 'a little', 'advanced')."""
    return f"""Create a personalised Jewish reading plan on the topic of "{topic}" for someone with {background} prior background.

Follow these steps:
1. Call search_by_theme("{topic}") to find relevant books.
2. Call browse_collection with foundational_only=True if the background suggests a beginner (e.g. none, little, some), or foundational_only=False if the background suggests experience (e.g. advanced, familiar, studied before). Use your judgement based on what the user said.
3. From the combined results, select 3–5 books that form a logical progression suited to someone with {background} prior background.
4. Present them as a list ordered by difficulty, with:
   - Bold title and author
   - Difficulty label (e.g. Introductory, Beginner)
   - One sentence explaining why this book belongs at this stage of the journey
Do not add a preamble or closing remarks."""


@mcp.prompt()
def explain_book_to_beginner(title: str) -> str:
    """Generate a prompt that instructs the agent to explain a book to a complete beginner.
    title: the name of the book to explain."""
    return f"""Explain the book "{title}" to someone who has never studied Jewish texts before.

Follow these steps:
1. Call lookup_book("{title}") to get the book's metadata (author, category, difficulty, themes).
2. Call get_text with the book's Sefaria key to fetch the opening passage (chapter 1). Use version_language="english".
3. Present your explanation in this order:
   a. One sentence on who wrote it and when.
   b. One sentence on what the book is fundamentally about.
   c. Quote 1–2 sentences from the opening passage.
   d. 2–3 sentences explaining that passage in plain everyday language, avoiding jargon.
   e. One sentence on why a modern reader might find it relevant.
Keep the total response under 200 words."""


if __name__ == "__main__":
    from recommender.query import warm_model

    logger.info("Warming embedding model...")
    warm_model()
    logger.info("Embedding model ready.")

    mcp.run(transport="streamable-http")
