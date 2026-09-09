"""
Hybrid search: dense (pgvector cosine) + lexical (pg_trgm) retrieval,
fused with Reciprocal Rank Fusion, then cross-encoder re-ranked.

Kept separate from query.py so this stays torch-free and importable without
a DB or model — rrf_fuse is pure and unit-tested directly.
"""
from __future__ import annotations

import psycopg
import psycopg.rows
import torch

import config


def rrf_fuse(ranked_lists: list[list[int]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """Merge several ranked id lists into one ranking: score(id) = sum of 1/(k+rank)
    across lists. Fuses ranks, not raw scores, since cosine and trigram similarity
    aren't on the same scale. Returns (id, score) sorted by score descending."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def lexical_search(
    conn: psycopg.Connection,
    query_text: str,
    exclude_ids: list[int] | None = None,
    top_k: int = config.HYBRID_ARM_TOP_K,
    min_similarity: float = config.TRIGRAM_MIN_SIMILARITY,
) -> list[dict]:
    """Top-k books by trigram similarity of title/author/key/themes to query_text."""
    exclude_ids = exclude_ids or []
    exclude_clause = f"AND id NOT IN ({', '.join(str(i) for i in exclude_ids)})" if exclude_ids else ""
    sql = f"""
        SELECT * FROM (
            SELECT
                id, sefaria_key, title_en, author_en, category, subcategory,
                difficulty, themes, is_foundational, desc_en_short,
                GREATEST(
                    similarity(title_en, %(q)s),
                    similarity(coalesce(sefaria_key, ''), %(q)s),
                    0.9 * coalesce((SELECT max(similarity(t, %(q)s)) FROM unnest(themes) t), 0),
                    0.8 * similarity(coalesce(author_en, ''), %(q)s)
                ) AS lex_sim
            FROM books
            WHERE true {exclude_clause}
        ) scored
        WHERE lex_sim >= %(min_sim)s
        ORDER BY lex_sim DESC
        LIMIT %(top_k)s
    """
    params: dict = {"q": query_text, "min_sim": min_similarity, "top_k": top_k}

    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def merge_rows(*row_lists: list[dict]) -> dict[int, dict]:
    """Union rows from multiple arms by id, keeping the first-seen copy of each."""
    merged: dict[int, dict] = {}
    for rows in row_lists:
        for row in rows:
            merged.setdefault(row["id"], row)
    return merged


def hybrid_search(
    conn: psycopg.Connection,
    query_text: str,
    limit: int = 8,
    rerank: bool = True,
) -> list[dict]:
    """Dense + lexical retrieval, RRF-fused, then cross-encoder re-ranked.
    Returns up to `limit` books, each tagged with how it was matched."""
    from recommender.query import _get_model, _get_cross_encoder, _query_vector
    from ingestion.embed import build_profile

    model = _get_model()
    vector = model.encode([query_text])[0].tolist()

    dense_rows = _query_vector(conn, vector, exclude_ids=[], top_k=config.HYBRID_ARM_TOP_K, min_cosine=config.HYBRID_MIN_COSINE)
    lexical_rows = lexical_search(conn, query_text)

    dense_ids = [r["id"] for r in dense_rows]
    lexical_ids = [r["id"] for r in lexical_rows]
    fused = rrf_fuse([dense_ids, lexical_ids])

    rows_by_id = merge_rows(dense_rows, lexical_rows)
    dense_id_set, lexical_id_set = set(dense_ids), set(lexical_ids)

    candidates = []
    for item_id, rrf_score in fused:
        row = dict(rows_by_id[item_id])
        row["rrf_score"] = rrf_score
        in_dense, in_lexical = item_id in dense_id_set, item_id in lexical_id_set
        row["match"] = "both" if (in_dense and in_lexical) else ("semantic" if in_dense else "keyword")
        candidates.append(row)

    if rerank and candidates:
        encoder = _get_cross_encoder()
        pairs = [(query_text, build_profile(c)) for c in candidates]
        cross_scores = list(encoder.predict(pairs, activation_fn=torch.nn.Identity()))
        for c, s in zip(candidates, cross_scores):
            c["cross_score"] = float(s)
        candidates.sort(key=lambda c: c["cross_score"], reverse=True)

    return candidates[:limit]
