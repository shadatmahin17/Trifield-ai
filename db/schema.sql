-- ============================================================
-- TriField AI — Supabase schema
-- Run once in Supabase → SQL Editor → New Query → Run
-- ============================================================

-- ── 1. Search analytics ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS search_events (
    id            BIGSERIAL PRIMARY KEY,
    query         TEXT        NOT NULL,
    discipline    TEXT        NOT NULL DEFAULT 'all',
    intent        TEXT        NOT NULL DEFAULT 'general',
    result_count  INTEGER     NOT NULL DEFAULT 0,
    latency_ms    FLOAT       NOT NULL DEFAULT 0,
    success       BOOLEAN     NOT NULL DEFAULT TRUE,
    source        TEXT        NOT NULL DEFAULT 'api',   -- 'api' | 'stream'
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_events_created_at  ON search_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_events_discipline  ON search_events (discipline);
CREATE INDEX IF NOT EXISTS idx_search_events_intent      ON search_events (intent);
CREATE INDEX IF NOT EXISTS idx_search_events_success     ON search_events (success);

-- ── 2. PDF sessions ──────────────────────────────────────────
-- Each uploaded PDF gets one row; session_id ties it to Qdrant + Storage.
CREATE TABLE IF NOT EXISTS pdf_sessions (
    session_id    TEXT        PRIMARY KEY,
    filename      TEXT        NOT NULL,
    storage_path  TEXT,                    -- supabase storage object path
    size_bytes    INTEGER,
    chunk_count   INTEGER     NOT NULL DEFAULT 0,
    latency_ms    FLOAT       NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_sessions_created_at ON pdf_sessions (created_at DESC);

-- ── 3. Chat history ──────────────────────────────────────────
-- Persistent per-session conversation, replaces the in-memory dict.
CREATE TABLE IF NOT EXISTS chat_messages (
    id            BIGSERIAL   PRIMARY KEY,
    session_id    TEXT        NOT NULL REFERENCES pdf_sessions(session_id) ON DELETE CASCADE,
    role          TEXT        NOT NULL CHECK (role IN ('user','assistant')),
    content       TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages (session_id, created_at ASC);

-- ── 4. Row-Level Security (public API — no user auth yet) ────
-- Enable RLS on every table so future auth can be layered on.
ALTER TABLE search_events  ENABLE ROW LEVEL SECURITY;
ALTER TABLE pdf_sessions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages  ENABLE ROW LEVEL SECURITY;

-- Service-role key (used by the backend) bypasses RLS automatically.
-- Anon key (public) gets no access by default — add policies if needed.

-- ── 5. Storage bucket ───────────────────────────────────────
-- Create via Supabase dashboard: Storage → New Bucket → "pdfs" (private)
-- OR run:
--   insert into storage.buckets (id, name, public)
--   values ('pdfs', 'pdfs', false);
