"""
Fetches book metadata from Sefaria's /api/v2/raw/index/<slug> endpoint,
resolves author names via /api/topics/<slug>, and upserts into the books table.
"""
from __future__ import annotations

import json
import time

import httpx
import psycopg
from rich.console import Console
from rich.progress import track

import config
from ingestion.models import SefariaIndexResponse

console = Console()

SEFARIA_INDEX_URL  = "https://www.sefaria.org/api/v2/raw/index/{slug}"
SEFARIA_AUTHOR_URL = "https://www.sefaria.org/api/topics/{slug}"
REQUEST_DELAY = 0.4

CANONICAL_BOOKS: list[str] = [
    # ── Chasidut ──────────────────────────────────────────────────────────────
    "Tanya",
    "Likutei_Moharan",
    "Noam_Elimelekh",
    "Me'or_Einayim",
    "Kedushat_Levi",
    "Degel_Machaneh_Ephraim",
    "Tzava'at_HaRivash",
    "Keter_Shem_Tov",
    "Ohr_HaMeir",
    "Maor_VaShemesh",
    "Sefat_Emet",
    "Shem_MiShmuel",
    "Likutei_Halakhot",
    "Likkutei_Etzot",
    "Mevo_HaShearim",
    "Nefesh_HaChayim",
    "Torah_Ohr",
    "Tzidkat_HaTzadik",
    "Peri_HaAretz",
    "Divrei_Emet",

    # ── Musar ─────────────────────────────────────────────────────────────────
    "Mesillat_Yesharim",
    "Chovot_HaLevavot",
    "Orchot_Tzadikim",
    "Sha'arei_Teshuvah",
    "Shaarei_Kedusha",
    "Ohr_Yisrael",
    "Cheshbon_HaNefesh",
    "Sefer_HaYashar",
    "Pele_Yoetz",
    "Shevet_Musar",
    "Yesod_VeShoresh_HaAvodah",
    "Tomer_Devorah",
    "Reshit_Chokhmah",
    "Sefer_HaMiddot",
    "Kav_HaYashar",
    "Ben_Ish_Hai",

    # ── Jewish Thought / Medieval Philosophy ──────────────────────────────────
    "Moreh_Nevukhim",
    "Kuzari",
    "Sefer_HaIkkarim",
    "HaEmunot_veHaDeot",
    "Ohr_Hashem",
    "Derashot_HaRan",
    "Sefer_HaChinuch",
    "Akeidat_Yitzchak",
    "Menorat_HaMaor",
    "Chovat_HaTalmidim",
]

# Cache so we don't re-fetch the same author twice
_author_cache: dict[str, str | None] = {}


def _load_enrichment() -> dict:
    if config.ENRICHMENT_FILE.exists():
        return json.loads(config.ENRICHMENT_FILE.read_text())
    return {}


def _resolve_author_name(client: httpx.Client, slug: str) -> str | None:
    if slug in _author_cache:
        return _author_cache[slug]
    try:
        r = client.get(SEFARIA_AUTHOR_URL.format(slug=slug), timeout=10)
        r.raise_for_status()
        name = r.json().get("primaryTitle", {}).get("en") or None
    except Exception:
        name = None
    _author_cache[slug] = name
    time.sleep(REQUEST_DELAY)
    return name


def fetch_book(client: httpx.Client, slug: str) -> SefariaIndexResponse | None:
    try:
        resp = client.get(SEFARIA_INDEX_URL.format(slug=slug), timeout=15)
        resp.raise_for_status()
        return SefariaIndexResponse.model_validate(resp.json())
    except httpx.HTTPStatusError as e:
        console.print(f"[yellow]HTTP {e.response.status_code} for {slug}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error fetching {slug}: {e}[/red]")
        return None


def upsert_book(
    conn: psycopg.Connection,
    client: httpx.Client,
    slug: str,
    book: SefariaIndexResponse,
    enrichment: dict,
) -> None:
    author_slug = book.author_slugs[0] if book.author_slugs else None
    author_en = _resolve_author_name(client, author_slug) if author_slug else None

    extra = enrichment.get(slug, {})

    conn.execute(
        """
        INSERT INTO books (
            sefaria_key, title_en,
            author_en, author_slug,
            category, subcategory, era,
            comp_date_start, comp_date_end, comp_place,
            pub_date, pub_place,
            desc_en, desc_en_short,
            is_cited,
            difficulty, themes, is_foundational,
            updated_at
        ) VALUES (
            %s, %s,
            %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s,
            %s, %s, %s,
            now()
        )
        ON CONFLICT (sefaria_key) DO UPDATE SET
            title_en        = EXCLUDED.title_en,
            author_en       = EXCLUDED.author_en,
            author_slug     = EXCLUDED.author_slug,
            category        = EXCLUDED.category,
            subcategory     = EXCLUDED.subcategory,
            era             = EXCLUDED.era,
            comp_date_start = EXCLUDED.comp_date_start,
            comp_date_end   = EXCLUDED.comp_date_end,
            comp_place      = EXCLUDED.comp_place,
            pub_date        = EXCLUDED.pub_date,
            pub_place       = EXCLUDED.pub_place,
            desc_en         = EXCLUDED.desc_en,
            desc_en_short   = EXCLUDED.desc_en_short,
            is_cited        = EXCLUDED.is_cited,
            difficulty      = EXCLUDED.difficulty,
            themes          = EXCLUDED.themes,
            is_foundational = EXCLUDED.is_foundational,
            updated_at      = now()
        """,
        (
            slug, book.title,
            author_en, author_slug,
            book.category, book.subcategory, book.era,
            book.comp_date_start, book.comp_date_end, book.compPlace,
            book.pub_date, book.pubPlace,
            book.enDesc, book.enShortDesc,
            book.is_cited,
            extra.get("difficulty"), extra.get("themes"), extra.get("is_foundational", False),
        ),
    )


def run_ingestion(slugs: list[str] | None = None) -> None:
    slugs = slugs or CANONICAL_BOOKS
    enrichment = _load_enrichment()
    ok = fail = 0

    with httpx.Client(headers={"User-Agent": "jewish-book-guide/1.0"}) as client:
        with psycopg.connect(config.DB_URL) as conn:
            for slug in track(slugs, description="Fetching from Sefaria..."):
                book = fetch_book(client, slug)
                if book is None:
                    fail += 1
                    continue
                upsert_book(conn, client, slug, book, enrichment)
                conn.commit()
                ok += 1
                time.sleep(REQUEST_DELAY)

    console.print(f"\n[green]Done.[/green] Inserted/updated: {ok}  Failed: {fail}")
