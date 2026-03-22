"""
jewish-book-guide CLI

Commands:
    ingest          Fetch book metadata from Sefaria and store in DB
    embed           Generate / refresh vector embeddings
    recommend       Suggest books similar to a given title
    search          Look up a book by title
    stats           Show database statistics
"""
from __future__ import annotations

from typing import Optional
import typer
import psycopg
import psycopg.rows
from rich.console import Console
from rich.table import Table
from rich import box

import config

app = typer.Typer(help="Jewish book recommendation system powered by Sefaria + pgvector.")
console = Console()


@app.command()
def ingest(
    slug: Optional[list[str]] = typer.Option(
        None, "--slug", "-s", help="Specific Sefaria slug(s) to ingest (default: all canonical books)"
    ),
) -> None:
    """Fetch book metadata from Sefaria and upsert into the database."""
    from ingestion.fetch_sefaria import run_ingestion
    run_ingestion(slugs=list(slug) if slug else None)


@app.command()
def embed(
    force: bool = typer.Option(False, "--force", "-f", help="Re-embed all books, even those with existing embeddings"),
) -> None:
    """Generate vector embeddings for books that don't have one yet."""
    from ingestion.embed import run_embedding
    run_embedding(force=force)


@app.command()
def recommend(
    title: list[str] = typer.Argument(..., help="Title(s) of books you enjoyed"),
    top: int = typer.Option(5, "--top", "-n", help="Number of recommendations to return"),
    difficulty: Optional[int] = typer.Option(
        None, "--difficulty", "-d",
        help="Your preferred difficulty level 1–5 (1=accessible, 5=scholarly)",
        min=1, max=5,
    ),
    category: Optional[str] = typer.Option(
        None, "--category", "-c",
        help="Limit recommendations to a specific category (e.g. 'Chasidut', 'Musar')",
    ),
) -> None:
    """Suggest books similar to one or more books you enjoyed."""
    from recommender.query import recommend as _recommend

    console.print(f"\nFinding books similar to: [cyan]{', '.join(title)}[/cyan]\n")
    results = _recommend(
        seed_titles=list(title),
        top_n=top,
        difficulty_pref=difficulty,
        category_pref=category,
    )

    if not results:
        console.print("[yellow]No recommendations found. Make sure the book title matches exactly and embeddings have been generated.[/yellow]")
        raise typer.Exit(1)

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", min_width=28)
    table.add_column("Author", min_width=18)
    table.add_column("Category", min_width=14)
    table.add_column("Difficulty", width=10)
    table.add_column("Score", width=7)
    table.add_column("Short Description", min_width=36)

    difficulty_stars = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}

    for i, r in enumerate(results, 1):
        diff = difficulty_stars.get(r.get("difficulty") or 0, "—")
        desc = (r.get("desc_en_short") or "")[:80]
        if len(r.get("desc_en_short") or "") > 80:
            desc += "…"
        table.add_row(
            str(i),
            r["title_en"],
            r.get("author_en") or "—",
            r.get("category") or "—",
            diff,
            f"{r['score']:.3f}",
            desc,
        )

    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Book title or Sefaria key to look up"),
) -> None:
    """Look up a book by title and show its details."""
    from recommender.query import find_book

    book = find_book(query)
    if not book:
        console.print(f"[yellow]No book found matching '{query}'[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{book['title_en']}[/bold]")
    if book.get("author_en"):
        console.print(f"Author: {book['author_en']}")
    console.print(f"Category: {book.get('category')} / {book.get('subcategory') or '—'}")
    if book.get("difficulty"):
        console.print(f"Difficulty: {book['difficulty']}/5 — {config.DIFFICULTY_LABELS.get(book['difficulty'], '')}")
    if book.get("themes"):
        console.print(f"Themes: {', '.join(book['themes'])}")
    if book.get("desc_en"):
        console.print(f"\n{book['desc_en']}")


@app.command()
def stats() -> None:
    """Show database statistics."""
    with psycopg.connect(config.DB_URL, row_factory=psycopg.rows.dict_row) as conn:
        totals = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(embedding) AS with_embedding,
                COUNT(*) FILTER (WHERE is_foundational) AS foundational
            FROM books
            """
        ).fetchone()

        by_category = conn.execute(
            """
            SELECT category, COUNT(*) AS n
            FROM books
            GROUP BY category
            ORDER BY n DESC
            """
        ).fetchall()

    console.print(f"\n[bold]Database stats[/bold]")
    console.print(f"  Total books:      {totals['total']}")
    console.print(f"  With embeddings:  {totals['with_embedding']}")
    console.print(f"  Foundational:     {totals['foundational']}")
    console.print()

    table = Table(box=box.SIMPLE)
    table.add_column("Category")
    table.add_column("Books", justify="right")
    for row in by_category:
        table.add_row(row["category"], str(row["n"]))
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    debug: bool = typer.Option(False, "--debug", help="Enable DEBUG logging"),
) -> None:
    """Start the conversational book guide as a FastAPI server."""
    import os, uvicorn
    if debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
    uvicorn.run("agent.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
