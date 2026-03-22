"""
Central configuration: DB connection, embedding model, canonical book list, re-ranking weights.
"""
import os
from pathlib import Path

# ── Database ──────────────────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost/books")

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Re-ranking weights ────────────────────────────────────────────────────────
WEIGHT_SAME_CATEGORY    = 0.15
WEIGHT_SAME_SUBCATEGORY = 0.10
WEIGHT_PER_DIFFICULTY   = 0.10   # penalty per difficulty level of mismatch
WEIGHT_PER_THEME        = 0.05   # bonus per overlapping theme
WEIGHT_FOUNDATIONAL     = 0.05   # bonus for foundational books (new users)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
ENRICHMENT_FILE = PROJECT_ROOT / "config"

# ── Difficulty labels ─────────────────────────────────────────────────────────
DIFFICULTY_LABELS: dict[int, str] = {
    1: "Introductory — accessible to any reader",
    2: "Beginner — some familiarity with Jewish concepts",
    3: "Intermediate — regular Torah study background",
    4: "Advanced — significant textual knowledge required",
    5: "Scholar — deep expertise in rabbinic literature",
}
