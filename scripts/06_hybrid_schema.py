"""
Step 6 (custom feature): Hybrid Search Schema
==============================================
Adds a second, talk-level "concept" index and keyword (full-text) search,
then fuses them with sentence-level semantic search in one SQL function.

  1. talk_concepts       — one vector per talk: its big ideas (see 07_build_talk_concepts.py)
  2. full-text index     — exact-word / quoted-phrase lookup on every sentence
  3. hybrid_search()     — ranks talks by all three signals with Reciprocal Rank Fusion

Usage:
    python scripts/06_hybrid_schema.py
"""

import json
import sys

import requests

with open('config.secret.json', 'r') as f:
    secrets = json.load(f)

SCHEMA_SQL = """
-- ============================================
-- 1. Talk-level concept vectors (one row per talk)
-- ============================================
CREATE TABLE IF NOT EXISTS talk_concepts (
    talk_id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    speaker TEXT,
    calling TEXT,
    year INTEGER,
    season TEXT,
    url TEXT,
    embedding vector(1536) NOT NULL
);

-- Same access rule as sentence_embeddings: logged-in users read, only service role writes
ALTER TABLE talk_concepts ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow authenticated users to read" ON talk_concepts;
CREATE POLICY "Allow authenticated users to read"
ON talk_concepts FOR SELECT
TO authenticated
USING (true);

-- ============================================
-- 2. Full-text (keyword) index on sentences
-- ============================================
-- An expression index, not a stored column: adding a column would rewrite the
-- whole table (and its 200 MB vector index), which doesn't fit the free tier.
CREATE INDEX IF NOT EXISTS sentence_embeddings_fts_idx
ON sentence_embeddings USING gin (to_tsvector('english', text));

-- ============================================
-- 3. Hybrid search: concepts + sentences + keywords, fused by rank
-- ============================================
-- Reciprocal Rank Fusion: each signal ranks talks; a talk's score is the sum of
-- 1 / (60 + rank) over the signals that found it. Ranks (not raw scores) are
-- combined because cosine similarity and ts_rank aren't on comparable scales.
CREATE OR REPLACE FUNCTION hybrid_search(
  query_text text,
  query_embedding vector(1536),
  match_count int DEFAULT 3
)
RETURNS TABLE (
  talk_id uuid,
  title text,
  speaker text,
  url text,
  score float,
  concept_rank int,
  sentence_rank int,
  keyword_rank int,
  best_similarity float,
  best_sentence text
)
LANGUAGE sql STABLE
SET hnsw.ef_search = 100
AS $$
  WITH
  -- Signal 1: which talks are ABOUT this question overall?
  concept AS (
    SELECT tc.talk_id, row_number() OVER (ORDER BY tc.embedding <=> query_embedding) AS r
    FROM talk_concepts tc
    ORDER BY tc.embedding <=> query_embedding
    LIMIT 20
  ),
  -- Signal 2: which talks contain a sentence that MEANS what was asked?
  sent AS (
    SELECT s.talk_id, s.text, s.embedding <=> query_embedding AS dist
    FROM sentence_embeddings s
    ORDER BY s.embedding <=> query_embedding
    LIMIT 50
  ),
  sent_talk AS (
    SELECT sent.talk_id, row_number() OVER (ORDER BY min(sent.dist)) AS r
    FROM sent GROUP BY sent.talk_id
  ),
  -- Signal 3: which talks contain the actual WORDS? A "quoted phrase" must match
  -- exactly; otherwise any of the (stemmed, stop-word-free) words can match.
  q AS (
    SELECT CASE
      WHEN query_text LIKE '%"%' THEN websearch_to_tsquery('english', query_text)
      ELSE to_tsquery('english', NULLIF(replace(plainto_tsquery('english', query_text)::text, '&', '|'), ''))
    END AS tsq
  ),
  kw AS (
    SELECT s.talk_id, max(ts_rank(to_tsvector('english', s.text), q.tsq)) AS rk
    FROM sentence_embeddings s, q
    WHERE q.tsq IS NOT NULL AND to_tsvector('english', s.text) @@ q.tsq
    GROUP BY s.talk_id
    ORDER BY rk DESC
    LIMIT 20
  ),
  kw_talk AS (
    SELECT kw.talk_id, row_number() OVER (ORDER BY kw.rk DESC) AS r FROM kw
  ),
  ranks AS (
    SELECT concept.talk_id, 'c' AS src, concept.r FROM concept
    UNION ALL SELECT sent_talk.talk_id, 's', sent_talk.r FROM sent_talk
    UNION ALL SELECT kw_talk.talk_id, 'k', kw_talk.r FROM kw_talk
  ),
  fused AS (
    SELECT ranks.talk_id,
           sum(1.0 / (60 + ranks.r)) AS score,
           min(ranks.r) FILTER (WHERE src = 'c') AS cr,
           min(ranks.r) FILTER (WHERE src = 's') AS sr,
           min(ranks.r) FILTER (WHERE src = 'k') AS kr
    FROM ranks
    GROUP BY ranks.talk_id
    ORDER BY score DESC
    LIMIT match_count
  )
  SELECT f.talk_id, tc.title, tc.speaker, tc.url, f.score::float,
         f.cr::int, f.sr::int, f.kr::int,
         (SELECT 1 - min(sent.dist) FROM sent WHERE sent.talk_id = f.talk_id)::float,
         (SELECT sent.text FROM sent WHERE sent.talk_id = f.talk_id ORDER BY sent.dist LIMIT 1)
  FROM fused f
  JOIN talk_concepts tc ON tc.talk_id = f.talk_id
  ORDER BY f.score DESC;
$$;
"""


def main():
    url = f"https://api.supabase.com/v1/projects/{secrets['SUPABASE_PROJECT_REF']}/database/query"
    headers = {
        "Authorization": f"Bearer {secrets['SUPABASE_ACCESS_TOKEN']}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, headers=headers, json={"query": SCHEMA_SQL}, timeout=120)
    if resp.status_code not in (200, 201):
        print(f"❌ Schema update failed: {resp.status_code}")
        print(resp.text[:500])
        sys.exit(1)
    print("✅ Hybrid search schema ready: talk_concepts, full-text index, hybrid_search()")
    print("\nNext: python scripts/07_build_talk_concepts.py")


if __name__ == '__main__':
    main()
