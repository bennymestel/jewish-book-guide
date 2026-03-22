"""
Two-stage book recommendation:
  Stage 1 — vector cosine similarity via pgvector
  Stage 2 — re-rank with category / difficulty / theme bonuses

Public API:
    recommend(seed_titles, top_n, difficulty_pref, category_pref) -> list[dict]
    find_book(title_query) -> dict | None
"""
from __future__ import annotations

import psycopg
import psycopg.rows

import config
from ingestion.embed import build_profile

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _era_midpoint(row: dict) -> int | None:
    s = row.get("comp_date_start")
    e = row.get("comp_date_end")
    if s is not None and e is not None:
        return (s + e) // 2
    return s or e


def _score(candidate: dict, seed: dict, difficulty_pref: int | None) -> float:
    score: float = candidate["cosine_sim"]

    if candidate["category"] == seed["category"]:
        score += config.WEIGHT_SAME_CATEGORY
    if candidate.get("subcategory") and candidate["subcategory"] == seed.get("subcategory"):
        score += config.WEIGHT_SAME_SUBCATEGORY

    # Difficulty penalty
    if difficulty_pref is not None and candidate.get("difficulty"):
        gap = abs(candidate["difficulty"] - difficulty_pref)
        score -= gap * config.WEIGHT_PER_DIFFICULTY

    # Theme overlap bonus
    c_themes = set(candidate.get("themes") or [])
    s_themes = set(seed.get("themes") or [])
    overlap = len(c_themes & s_themes)
    score += overlap * config.WEIGHT_PER_THEME

    return score


def _query_vector(conn: psycopg.Connection, vector: list[float], exclude_ids: list[int], top_k: int = 20) -> list[dict]:
    placeholders = ", ".join(["%s"] * len(exclude_ids)) if exclude_ids else "NULL"
    sql = f"""
        SELECT
            id, sefaria_key, title_en,
            author_en, category, subcategory,
            difficulty, themes, is_foundational,
            desc_en_short,
            1 - (embedding <=> %s::vector) AS cosine_sim
        FROM books
        WHERE embedding IS NOT NULL
          AND id NOT IN ({placeholders if exclude_ids else 'SELECT NULL'})
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params: list = [vector]
    if exclude_ids:
        params.extend(exclude_ids)
    params.extend([vector, top_k])

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _get_books_by_titles(conn: psycopg.Connection, titles: list[str]) -> list[dict]:
    results = []
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        for title in titles:
            cur.execute(
                """
                SELECT id, sefaria_key, title_en,
                       author_en, category, subcategory,
                       difficulty, themes, is_foundational,
                       desc_en_short, embedding
                FROM books
                WHERE lower(title_en) = lower(%s)
                   OR lower(sefaria_key) = lower(%s)
                LIMIT 1
                """,
                (title, title),
            )
            row = cur.fetchone()
            if row:
                results.append(row)
    return results


def find_book(title_query: str) -> dict | None:
    with psycopg.connect(config.DB_URL, row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, sefaria_key, title_en, author_en,
                       category, subcategory, difficulty, themes,
                       is_foundational, desc_en_short, desc_en,
                       pub_date
                FROM books
                WHERE lower(title_en) ILIKE %s
                   OR lower(sefaria_key) ILIKE %s
                ORDER BY
                    CASE WHEN lower(title_en) = lower(%s) THEN 0 ELSE 1 END
                LIMIT 1
                """,
                (f"%{title_query.lower()}%", f"%{title_query.lower()}%", title_query),
            )
            return cur.fetchone()


def recommend(
    seed_titles: list[str],
    top_n: int = 3,
    difficulty_pref: int | None = None,
    category_pref: str | None = None,
) -> list[dict]:
    """
    Returns up to top_n recommended books similar to the seed_titles.
    seed_titles: list of title_en or sefaria_key values
    difficulty_pref: 1–5 target difficulty level (optional)
    category_pref: filter candidates to this category (optional)
    """
    model = _get_model()

    with psycopg.connect(config.DB_URL, row_factory=psycopg.rows.dict_row) as conn:
        seeds = _get_books_by_titles(conn, seed_titles)
        if not seeds:
            return []

        seed_ids = [s["id"] for s in seeds]

        # Build query vector: average embeddings of all seed books
        seed_vectors = []
        for seed in seeds:
            if seed.get("embedding") is not None:
                seed_vectors.append(seed["embedding"])
            else:
                # Fallback: encode a profile on the fly
                vec = model.encode([build_profile(seed)])[0]
                seed_vectors.append(vec.tolist())

        if len(seed_vectors) == 1:
            query_vector = seed_vectors[0]
        else:
            import statistics
            query_vector = [
                statistics.mean(v[i] for v in seed_vectors)
                for i in range(config.EMBEDDING_DIM)
            ]

        # Stage 1: vector retrieval
        candidates = _query_vector(conn, query_vector, seed_ids, top_k=20)

        if category_pref:
            candidates = [c for c in candidates if c.get("category") == category_pref]

        if not candidates:
            return []

        # Stage 2: re-rank using the first (or primary) seed as reference
        primary_seed = seeds[0]
        scored = [
            {**c, "score": _score(c, primary_seed, difficulty_pref)}
            for c in candidates
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored[:top_n]
