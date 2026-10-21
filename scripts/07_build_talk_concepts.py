"""
Step 7 (custom feature): Build Talk Concept Vectors
====================================================
One vector per talk capturing its big ideas, built as a weighted blend:

    concept = ALPHA * embed(nouns + verbs + adjectives)  +  (1 - ALPHA) * mean(sentence embeddings)

The content-word half emphasizes WHAT the talk is about; the full-text half keeps
the small words ("not", "without", "even") in the meaning, just with less weight.
The full-text half is free: it averages sentence embeddings we already paid for.

Usage:
    python scripts/07_build_talk_concepts.py            # build + upload (ALPHA = 0.7)
    python scripts/07_build_talk_concepts.py --alpha 0.5

Input:
    scripts/output/sentences_with_embeddings.json  — from Step 4/5
Output (cached, so re-runs cost nothing):
    scripts/output/talk_content_words.json         — content words per talk
    scripts/output/talk_vectors.npz                — content + full-text vectors
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import spacy
from openai import OpenAI
from supabase import create_client
from tqdm import tqdm

OUT = os.path.join('scripts', 'output')
SENTENCES_FILE = os.path.join(OUT, 'sentences_with_embeddings.json')
WORDS_FILE = os.path.join(OUT, 'talk_content_words.json')
VECTORS_FILE = os.path.join(OUT, 'talk_vectors.npz')
CONTENT_POS = {'NOUN', 'PROPN', 'VERB', 'ADJ'}
MAX_CHARS = 24_000  # stays under the embedding model's 8,191-token input limit


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_talks():
    """Group the embedded sentences back into talks (keeping import-time talk_ids)."""
    print("Loading sentence embeddings (large file)...")
    with open(SENTENCES_FILE, 'r', encoding='utf-8') as f:
        rows = json.load(f)
    talks = {}
    vecs = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r['talk_id'], r['sentence_num'])):
        t = talks.setdefault(r['talk_id'], {k: r[k] for k in ('talk_id', 'title', 'speaker', 'calling', 'year', 'season', 'url')})
        t.setdefault('sentences', []).append(r['text'])
        vecs[r['talk_id']].append(r['embedding'])
    ids = sorted(talks)
    full = unit(np.array([np.mean(np.array(vecs[i], dtype=np.float32), axis=0) for i in ids]))
    print(f"   {len(ids)} talks, full-text vectors averaged from {len(rows):,} sentences (free)")
    return ids, talks, full


def content_words(ids, talks):
    if os.path.exists(WORDS_FILE):
        with open(WORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    texts = [' '.join(talks[i]['sentences']) for i in ids]
    words = {}
    for i, doc in tqdm(zip(ids, nlp.pipe(texts, batch_size=16)), total=len(ids), desc="Tagging parts of speech"):
        words[i] = ' '.join(tok.text for tok in doc if tok.pos_ in CONTENT_POS)
    with open(WORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(words, f, ensure_ascii=False)
    return words


def embed_content(ids, words):
    with open('config.secret.json', 'r') as f:
        client = OpenAI(api_key=json.load(f)['OPENAI_API_KEY'])
    out = []
    for start in tqdm(range(0, len(ids), 20), desc="Embedding content words"):
        batch = [words[i][:MAX_CHARS] for i in ids[start:start + 20]]
        resp = client.embeddings.create(model='text-embedding-3-small', input=batch)
        out.extend(d.embedding for d in resp.data)
    return unit(np.array(out, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7, help='weight of the content-word vector (0–1)')
    args = parser.parse_args()

    ids, talks, full = load_talks()
    words = content_words(ids, talks)
    sample = ids[0]
    print(f"\n   Example — {talks[sample]['title']}:")
    print(f"   full text:     {' '.join(talks[sample]['sentences'])[:110]}...")
    print(f"   content words: {words[sample][:110]}...\n")

    if os.path.exists(VECTORS_FILE):
        cached = np.load(VECTORS_FILE, allow_pickle=True)
        assert list(cached['ids']) == ids, "Talk ids changed — delete talk_vectors.npz and re-run"
        content = cached['content']
    else:
        content = embed_content(ids, words)
        np.savez(VECTORS_FILE, ids=np.array(ids), content=content, full=full)

    concept = unit(args.alpha * content + (1 - args.alpha) * full)
    print(f"Blending: {args.alpha:.0%} content words + {1 - args.alpha:.0%} full text")

    with open('config.public.json', 'r') as f:
        public_config = json.load(f)
    with open('config.secret.json', 'r') as f:
        secrets = json.load(f)
    db = create_client(public_config['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])
    rows = [{**{k: talks[i][k] for k in ('talk_id', 'title', 'speaker', 'calling', 'year', 'season', 'url')},
             'embedding': concept[n].tolist()} for n, i in enumerate(ids)]
    for start in tqdm(range(0, len(rows), 50), desc="Uploading"):
        db.table('talk_concepts').upsert(rows[start:start + 50]).execute()

    count = db.table('talk_concepts').select('talk_id', count='exact').limit(1).execute().count
    print(f"\n✅ talk_concepts has {count} talks")


if __name__ == '__main__':
    main()
