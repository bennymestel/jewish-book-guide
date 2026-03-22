"""
LangGraph tool definitions for the Jewish book guide agent.
"""
from __future__ import annotations

import json
import logging

import psycopg
import psycopg.rows
from langchain_core.tools import tool

import config

logger = logging.getLogger(__name__)


def _find_book(title_query: str):
    from recommender.query import find_book
    return find_book(title_query)


def _recommend(seed_titles, top_n=3, difficulty_pref=None, category_pref=None):
    from recommender.query import recommend
    return recommend(seed_titles, top_n=top_n, difficulty_pref=difficulty_pref, category_pref=category_pref)


def _escape_like(value: str) -> str:
    """Escape ILIKE special characters so user input is treated as a literal string."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
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


@tool
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


@tool
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
        with psycopg.connect(config.DB_URL, row_factory=psycopg.rows.dict_row) as conn:
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


@tool
def search_by_theme(theme: str, limit: int = 8) -> str:
    """Find books related to a specific theme or topic (e.g. 'prayer', 'teshuvah', 'Kabbalah', 'love of God').
    Returns books whose themes array contains the given theme."""
    logger.info("[TOOL] search_by_theme: theme=%r limit=%d", theme, limit)
    try:
        with psycopg.connect(config.DB_URL, row_factory=psycopg.rows.dict_row) as conn:
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


ALL_TOOLS = [lookup_book, get_recommendations, browse_collection, search_by_theme]
