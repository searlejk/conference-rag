# Conference Talk RAG — Assignment Notes (from Canvas, CS 452)

Source: https://byu.instructure.com/courses/35462/assignments/1472713
Starter repo: https://github.com/byu-cs-452/conference-rag (cloned into this folder)

## Deadlines (Mountain Time)

| What | Points | Weight | Due | Submit |
|---|---|---|---|---|
| Project (Conference Talk RAG) | 3 | 10% | Thu 10/22, 11:59 PM | Canvas file upload |
| Agent Transcript | 6 | 20% | Thu 10/22, 11:59 PM | Canvas file upload (Claude Code: `/export`, or `~/.claude/projects/`) |
| Learning Log (handwritten) | 21 | 70% | In class Mon 10/26 | On paper, nothing to upload |

## The assignment (Canvas text)

> Build an interesting website that could scale to millions and use RAG using postgres (supabase).
> Actually, make correct access controls using RLS. If your agent has credentials to supabase and GIT,
> you could potentially move much faster.

Up to ~6 hours; class days Mon 10/19 and Wed 10/21.

## Project deliverables (from steps/07_make_it_yours.md)

1. GitHub repository URL (your fork)
2. Live deployment URL (GitHub Pages)
3. Screenshot of a working RAG query + answer
4. Your custom feature: what you added and why
5. Written reflection: security architecture, Edge Functions, keyword vs semantic, RAG vs fine-tuning, AI-assisted dev

## Learning log topics the professor listed

- AuthN/AuthZ
- Protecting credentials while allowing access
- Supabase: RLS, Edge Functions
- Scalable systems
- Granting agent keys for supabase, gh, etc. speeds implementation up
- Exploration of semantic search strategies
- Exploration of practical applications for improving life

Log format: handwritten, one sheet (front/back). Name, project, date at top; aha moments, key concepts
(low level and high level), time spent (0–6 hrs), questions, how the agent helped or misled you.

## Accounts / keys needed

- GitHub (fork must be public for Pages)
- Supabase (free tier): URL, anon key, service_role key, project ref, access token
- OpenAI API key (~$0.60 for embeddings)
- Supabase CLI, Python 3.9+

Secrets go in `config.secret.json` (git-ignored). Only `config.public.json` (URL + anon key) is committed.
