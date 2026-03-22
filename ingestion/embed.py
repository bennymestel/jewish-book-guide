"""
Generates vector embeddings for all books that don't have one yet
and stores them in the books.embedding column.

Each book is embedded using a composite "profile" string so
that structurally similar books cluster together in embedding space.
"""
from __future__ import annotations

import psycopg
from rich.console import Console
from rich.progress import track

import config

console = Console()


def difficulty_label(difficulty: int | None) -> str:
    return config.DIFFICULTY_LABELS.get(difficulty or 0, "")


def build_profile(row: dict) -> str:
    title = row["title_en"]
    author = f"by {row['author_en']}" if row.get("author_en") else ""
    difficulty = difficulty_label(row.get("difficulty"))
    themes_str = ", ".join(row["themes"]) if row.get("themes") else ""
    desc_long = row.get("desc_en") or ""
    desc_short = row.get("desc_en_short") or ""

    parts = [
        title,
        author,
        themes_str,
        desc_long,
    ]
    return " | ".join(p for p in parts if p)


def run_embedding(force: bool = False) -> None:
    from sentence_transformers import SentenceTransformer
    console.print(f"Loading model [cyan]{config.EMBEDDING_MODEL}[/cyan]...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    with psycopg.connect(config.DB_URL, row_factory=psycopg.rows.dict_row) as conn:
        if force:
            rows = conn.execute(
                "SELECT id, title_en, author_en, category, subcategory, "
                "difficulty, themes, desc_en, desc_en_short FROM books"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title_en, author_en, category, subcategory, "
                "difficulty, themes, desc_en, desc_en_short FROM books "
                "WHERE embedding IS NULL"
            ).fetchall()


        if not rows:
            console.print("[green]All books already have embeddings.[/green]")
            return

        console.print(f"Embedding {len(rows)} book(s)...")
        profiles = [build_profile(r) for r in rows]
        vectors = model.encode(profiles, show_progress_bar=False)

        for row, vector in track(
            zip(rows, vectors), total=len(rows), description="Storing embeddings..."
        ):
            conn.execute(
                "UPDATE books SET embedding = %s WHERE id = %s",
                (vector.tolist(), row["id"]),
            )

        conn.commit()

    console.print(f"[green]Done.[/green] Embedded {len(rows)} book(s).")
