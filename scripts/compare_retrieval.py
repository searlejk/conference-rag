"""
Compare retrieval: original (average sentence similarity) vs hybrid_search().
Retrieval only — no GPT-4o calls. Question embeddings are cached in scripts/output/.

Usage:
    python scripts/compare_retrieval.py
"""

import json
import os
import time

from openai import OpenAI
from supabase import create_client

QUESTIONS = [
    'What have church leaders taught about prayer?',
    'How can I find peace during hard times?',
    '"peace and joy even in hard times"',
    'How should parents teach their children the gospel?',
    'Why do we make covenants in the temple?',
    'How can someone overcome addiction?',
    'What did leaders say during the COVID pandemic?',
]
CACHE = os.path.join('scripts', 'output', 'question_embeddings.json')

with open('config.public.json') as f:
    public_config = json.load(f)
with open('config.secret.json') as f:
    secrets = json.load(f)
db = create_client(public_config['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
missing = [q for q in QUESTIONS if q not in cache]
if missing:
    ai = OpenAI(api_key=secrets['OPENAI_API_KEY'])
    for q, d in zip(missing, ai.embeddings.create(model='text-embedding-3-small', input=missing).data):
        cache[q] = d.embedding
    json.dump(cache, open(CACHE, 'w'))


def original(emb):
    """What app.js did: top 20 sentences, grouped by talk, ranked by AVERAGE similarity."""
    hits = db.rpc('match_sentences', {'query_embedding': emb, 'match_count': 20}).execute().data
    talks = {}
    for h in hits:
        talks.setdefault(h['talk_id'], {'title': h['title'], 'speaker': h['speaker'], 'sims': []})['sims'].append(h['similarity'])
    ranked = sorted(talks.values(), key=lambda t: -sum(t['sims']) / len(t['sims']))[:3]
    return [f"{t['title']} — {t['speaker'][3:]} ({len(t['sims'])} hit{'s' if len(t['sims']) > 1 else ''})" for t in ranked]


def hybrid(q, emb):
    t0 = time.time()
    rows = db.rpc('hybrid_search', {'query_text': q, 'query_embedding': emb, 'match_count': 3}).execute().data
    ms = (time.time() - t0) * 1000
    fmt = lambda r: f"c{r['concept_rank'] or '-'} s{r['sentence_rank'] or '-'} k{r['keyword_rank'] or '-'}"
    return [f"{r['title']} — {r['speaker'][3:]} [{fmt(r)}]" for r in rows], ms


def pilot(q, emb):
    """Same hybrid search over the compact format: 8-sentence passages, halfvec vectors."""
    t0 = time.time()
    rows = db.rpc('hybrid_search_passages', {'query_text': q, 'query_embedding': emb, 'match_count': 3}).execute().data
    ms = (time.time() - t0) * 1000
    fmt = lambda r: f"c{r['concept_rank'] or '-'} p{r['passage_rank'] or '-'} k{r['keyword_rank'] or '-'}"
    return [f"{r['title']} — {r['speaker'][3:]} [{fmt(r)}]" for r in rows], ms


for q in QUESTIONS:
    emb = cache[q]
    h, ms = hybrid(q, emb)
    p, pms = pilot(q, emb)
    print(f"\nQ: {q}")
    print("  ORIGINAL:"); [print(f"    {i}. {t}") for i, t in enumerate(original(emb), 1)]
    print(f"  HYBRID ({ms:.0f} ms):"); [print(f"    {i}. {t}") for i, t in enumerate(h, 1)]
    print(f"  PILOT passages+halfvec ({pms:.0f} ms):"); [print(f"    {i}. {t}") for i, t in enumerate(p, 1)]
    overlap = len({t.split(' [')[0] for t in h} & {t.split(' [')[0] for t in p})
    print(f"  → pilot shares {overlap}/3 talks with hybrid")
print("\n[c/s/p/k = rank from concept / sentence / passage / keyword signal; '-' = not found by that signal]")
