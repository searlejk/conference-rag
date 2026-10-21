"""
RLS Demo (Step 03)
==================
Shows the same anon key getting different access to two tables,
because each table has different Row Level Security policies.

Usage:
    python scripts/rls_demo.py
"""

import json
import uuid

from supabase import create_client

with open('config.public.json', 'r') as f:
    public_config = json.load(f)
with open('config.secret.json', 'r') as f:
    secrets = json.load(f)

URL = public_config['SUPABASE_URL']
anon = create_client(URL, public_config['SUPABASE_ANON_KEY'])
service = create_client(URL, secrets['SUPABASE_SERVICE_KEY'])


def attempt(label, fn):
    try:
        result = fn()
        print(f"  {label}: ✅ {len(result.data)} row(s)")
    except Exception as e:
        msg = getattr(e, 'message', None) or str(e)
        print(f"  {label}: ❌ {msg[:90]}")


# Put one row in sentence_embeddings so there is something for RLS to hide
test_id = str(uuid.uuid4())
service.table('sentence_embeddings').insert({
    'id': test_id, 'talk_id': str(uuid.uuid4()),
    'title': 'RLS test talk', 'text': 'This row exists, but can you see it?',
}).execute()

try:
    print("\n🌍 page_views (policies: SELECT + INSERT TO anon, authenticated)")
    attempt("anon insert", lambda: anon.table('page_views').insert({'page_url': 'rls_demo.py'}).execute())
    attempt("anon select", lambda: anon.table('page_views').select('*').execute())

    print("\n🔒 sentence_embeddings (policy: SELECT TO authenticated only)")
    attempt("anon select   ", lambda: anon.table('sentence_embeddings').select('id,title').execute())
    attempt("anon insert   ", lambda: anon.table('sentence_embeddings').insert({
        'talk_id': str(uuid.uuid4()), 'title': 'hacked', 'text': 'sneaky'}).execute())
    attempt("service select", lambda: service.table('sentence_embeddings').select('id,title').execute())
finally:
    service.table('sentence_embeddings').delete().eq('id', test_id).execute()
    print("\n🧹 Test row removed.")
