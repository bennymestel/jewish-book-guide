CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS books (
    -- Identity
    id              SERIAL PRIMARY KEY,
    sefaria_key     TEXT NOT NULL UNIQUE,
    title_en        TEXT NOT NULL,

    -- Authorship
    author_en       TEXT,
    author_slug     TEXT,

    -- Categorization
    category        TEXT NOT NULL,       -- 'Chasidut' | 'Musar' | 'Jewish Thought'
    subcategory     TEXT,                -- e.g. 'Chabad', 'Breslov', 'Rishonim'

    -- Descriptions (from Sefaria API)
    desc_en         TEXT,
    desc_en_short   TEXT,

    -- Publication info
    pub_date        INTEGER,

    -- Curated enrichment (from config)
    difficulty      SMALLINT CHECK (difficulty BETWEEN 1 AND 5),
    themes          TEXT[],
    is_foundational BOOLEAN DEFAULT FALSE,

    -- Embedding (all-MiniLM-L6-v2, 384 dimensions)
    embedding       vector(384),

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_books_category   ON books (category);
CREATE INDEX IF NOT EXISTS idx_books_difficulty ON books (difficulty);
