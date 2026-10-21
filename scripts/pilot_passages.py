"""
Pilot: compact storage format for the full 1971–2026 archive
=============================================================
Builds the proposed format for 2020–2025 BESIDE the live tables, so the site
keeps working:

  talks     — one row per talk: metadata stored once + concept vector (halfvec)
  passages  — ~8-sentence chunks pointing at talks, 16-bit vectors (halfvec)

halfvec stores each number in 2 bytes instead of 4 (measured: same top-20 results).
Passages instead of sentences cut row count ~8x; normalizing talk metadata stops
title/speaker/url from repeating on every row.

Usage:
    python scripts/pilot_passages.py

Cost: ~3,300 passage embeddings (~$0.02). Talk concept vectors are reused from
scripts/output/talk_vectors.npz (free).
"""

import json
import os
import sys

import numpy as np
import requests
from openai import OpenAI
from supabase import create_client
from tqdm import tqdm

OUT = os.path.join('scripts', 'output')
SENTENCES_PER_PASSAGE = 8
ALPHA = 0.7  # same talk-concept blend as 07_build_talk_concepts.py

with open('config.public.json') as f:
    public_config = json.load(f)
with open('config.secret.json') as f:
    secrets = json.load(f)

SCHEMA_SQL = """
SELECT '[1,1]'::vector <=> '[1,1]'::vector;  -- load pgvector so hnsw.* settings exist

CREATE TABLE IF NOT EXISTS talks (
    talk_id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    speaker TEXT,
    calling TEXT,
    year SMALLINT,
    season TEXT,
    url TEXT,
    concept halfvec(1536)
);

CREATE TABLE IF NOT EXISTS passages (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    talk_id UUID NOT NULL REFERENCES talks(talk_id) ON DELETE CASCADE,
    passage_num SMALLINT NOT NULL,
    text TEXT NOT NULL,
    embedding halfvec(1536)
);

-- Indexes created on the empty table: Postgres maintains them as rows arrive,
-- avoiding one big HNSW build that the free tier's memory can't hold
CREATE INDEX IF NOT EXISTS passages_talk_idx ON passages (talk_id, passage_num);
-- The HNSW index holds a copy of every vector it indexes, so index only the first
-- 512 numbers (text-embedding-3 vectors still work when shortened). Search uses it to
-- pick 100 candidates, then re-ranks them with the full 1,536-number vector in the
-- table. Measured: keeps 99.5% of exact top-20 results at a third of the index size.
DROP INDEX IF EXISTS passages_embedding_hnsw_idx;
CREATE INDEX IF NOT EXISTS passages_embedding_512_hnsw_idx
ON passages USING hnsw ((subvector(embedding, 1, 512)::halfvec(512)) halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS passages_fts_idx ON passages USING gin (to_tsvector('english', text));

ALTER TABLE talks ENABLE ROW LEVEL SECURITY;
ALTER TABLE passages ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow authenticated users to read" ON talks;
CREATE POLICY "Allow authenticated users to read" ON talks FOR SELECT TO authenticated USING (true);
DROP POLICY IF EXISTS "Allow authenticated users to read" ON passages;
CREATE POLICY "Allow authenticated users to read" ON passages FOR SELECT TO authenticated USING (true);

-- Same three-signal hybrid search as hybrid_search(), over the compact tables.
-- The question arrives as a full vector and is cast to halfvec to use the index.
CREATE OR REPLACE FUNCTION hybrid_search_passages(
  query_text text,
  query_embedding vector(1536),
  match_count int DEFAULT 3
)
RETURNS TABLE (
  talk_id uuid, title text, speaker text, url text, score float,
  concept_rank int, passage_rank int, keyword_rank int,
  best_similarity float, best_passage text
)
LANGUAGE sql STABLE
SET hnsw.ef_search = 100
AS $$
  WITH
  qv AS (SELECT query_embedding::halfvec(1536) AS v),
  concept AS (
    SELECT t.talk_id, row_number() OVER (ORDER BY t.concept <=> qv.v) AS r
    FROM talks t, qv ORDER BY t.concept <=> qv.v LIMIT 20
  ),
  -- Fast pass on the short index, then precise re-rank on the full vectors
  candidates AS (
    SELECT p.talk_id, p.text, p.embedding
    FROM passages p
    ORDER BY subvector(p.embedding, 1, 512)::halfvec(512) <=> (SELECT subvector(v, 1, 512)::halfvec(512) FROM qv)
    LIMIT 100
  ),
  pas AS (
    SELECT c.talk_id, c.text, c.embedding <=> (SELECT v FROM qv) AS dist
    FROM candidates c ORDER BY dist LIMIT 30
  ),
  pas_talk AS (
    SELECT pas.talk_id, row_number() OVER (ORDER BY min(pas.dist)) AS r FROM pas GROUP BY pas.talk_id
  ),
  q AS (
    SELECT CASE
      WHEN query_text LIKE '%"%' THEN websearch_to_tsquery('english', query_text)
      ELSE to_tsquery('english', NULLIF(replace(plainto_tsquery('english', query_text)::text, '&', '|'), ''))
    END AS tsq
  ),
  kw AS (
    SELECT p.talk_id, max(ts_rank(to_tsvector('english', p.text), q.tsq)) AS rk
    FROM passages p, q
    WHERE q.tsq IS NOT NULL AND to_tsvector('english', p.text) @@ q.tsq
    GROUP BY p.talk_id ORDER BY rk DESC LIMIT 20
  ),
  kw_talk AS (SELECT kw.talk_id, row_number() OVER (ORDER BY kw.rk DESC) AS r FROM kw),
  ranks AS (
    SELECT concept.talk_id, 'c' AS src, concept.r FROM concept
    UNION ALL SELECT pas_talk.talk_id, 'p', pas_talk.r FROM pas_talk
    UNION ALL SELECT kw_talk.talk_id, 'k', kw_talk.r FROM kw_talk
  ),
  -- A "quoted phrase" asks for exact text, so its keyword match counts 3x:
  -- one exact sentence gets diluted inside an 8-sentence passage's vector
  fused AS (
    SELECT ranks.talk_id,
           sum(CASE WHEN src = 'k' AND query_text LIKE '%"%' THEN 3.0 ELSE 1.0 END / (60 + ranks.r)) AS score,
           min(ranks.r) FILTER (WHERE src = 'c') AS cr,
           min(ranks.r) FILTER (WHERE src = 'p') AS pr,
           min(ranks.r) FILTER (WHERE src = 'k') AS kr
    FROM ranks GROUP BY ranks.talk_id ORDER BY score DESC LIMIT match_count
  )
  SELECT f.talk_id, t.title, t.speaker, t.url, f.score::float, f.cr::int, f.pr::int, f.kr::int,
         (SELECT 1 - min(pas.dist) FROM pas WHERE pas.talk_id = f.talk_id)::float,
         (SELECT pas.text FROM pas WHERE pas.talk_id = f.talk_id ORDER BY pas.dist LIMIT 1)
  FROM fused f JOIN talks t ON t.talk_id = f.talk_id
  ORDER BY f.score DESC;
$$;
"""


def run_sql(sql):
    resp = requests.post(
        f"https://api.supabase.com/v1/projects/{secrets['SUPABASE_PROJECT_REF']}/database/query",
        headers={"Authorization": f"Bearer {secrets['SUPABASE_ACCESS_TOKEN']}", "Content-Type": "application/json"},
        json={"query": sql}, timeout=120)
    if resp.status_code not in (200, 201):
        print(f"❌ SQL failed: {resp.status_code}\n{resp.text[:500]}")
        sys.exit(1)
    return resp.json()


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    run_sql(SCHEMA_SQL)
    print("✅ Pilot tables and hybrid_search_passages() ready")

    # Rebuild talks from the imported sentences so talk_ids match the live data
    with open(os.path.join(OUT, 'sentences.json'), encoding='utf-8') as f:
        sentences = json.load(f)
    talks, by_talk = {}, {}
    for s in sorted(sentences, key=lambda s: (s['talk_id'], s['sentence_num'])):
        talks.setdefault(s['talk_id'], {k: s[k] for k in ('talk_id', 'title', 'speaker', 'calling', 'year', 'season', 'url')})
        by_talk.setdefault(s['talk_id'], []).append(s['text'])

    passages = []
    for talk_id, sents in by_talk.items():
        for n, start in enumerate(range(0, len(sents), SENTENCES_PER_PASSAGE), 1):
            passages.append({'talk_id': talk_id, 'passage_num': n,
                             'text': ' '.join(sents[start:start + SENTENCES_PER_PASSAGE])})
    print(f"   {len(talks)} talks → {len(passages):,} passages of up to {SENTENCES_PER_PASSAGE} sentences "
          f"(was {len(sentences):,} sentence rows)")

    # Embed passages (cached so re-runs are free)
    cache = os.path.join(OUT, 'passage_embeddings.npy')
    if os.path.exists(cache):
        emb = np.load(cache)
    else:
        client = OpenAI(api_key=secrets['OPENAI_API_KEY'])
        out = []
        for start in tqdm(range(0, len(passages), 100), desc="Embedding passages"):
            resp = client.embeddings.create(model='text-embedding-3-small',
                                            input=[p['text'] for p in passages[start:start + 100]])
            out.extend(d.embedding for d in resp.data)
        emb = np.array(out, dtype=np.float32)
        np.save(cache, emb)

    vectors = np.load(os.path.join(OUT, 'talk_vectors.npz'), allow_pickle=True)
    concept = dict(zip(vectors['ids'], unit(ALPHA * vectors['content'] + (1 - ALPHA) * vectors['full'])))

    db = create_client(public_config['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])
    db.table('passages').delete().gte('id', 0).execute()
    talk_rows = [{**t, 'concept': np.round(concept[t['talk_id']], 5).tolist()} for t in talks.values()]
    for start in tqdm(range(0, len(talk_rows), 50), desc="Uploading talks"):
        db.table('talks').upsert(talk_rows[start:start + 50]).execute()
    rows = [{**p, 'embedding': np.round(emb[n], 5).tolist()} for n, p in enumerate(passages)]
    for start in tqdm(range(0, len(rows), 100), desc="Uploading passages"):
        db.table('passages').insert(rows[start:start + 100]).execute()

    print(run_sql("""
        SELECT (SELECT count(*) FROM talks) AS talks, (SELECT count(*) FROM passages) AS passages,
               pg_size_pretty(pg_total_relation_size('talks')) AS talks_size,
               pg_size_pretty(pg_total_relation_size('passages')) AS passages_total,
               pg_size_pretty(pg_relation_size('passages_embedding_hnsw_idx')) AS hnsw,
               pg_size_pretty(pg_relation_size('passages_fts_idx')) AS fts,
               pg_total_relation_size('passages') / greatest((SELECT count(*) FROM passages), 1) AS bytes_per_passage,
               pg_size_pretty(pg_database_size(current_database())) AS db
    """))


if __name__ == '__main__':
    main()
