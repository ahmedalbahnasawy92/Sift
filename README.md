# Sift

A document assistant built with [Agno](https://github.com/agno-agi/agno), modelled on the
architecture of [Mike](https://github.com/open-legal-products/mike) (an open-source legal AI
platform), built step by step.

One agent, one tool loop, a small set of document tools, and a harness around them that
keeps it grounded, safe and cheap.

## Features

| Step | Feature | Where |
|---|---|---|
| 1 | One agent + tools loop, 16-call step limit, model retries, saved sessions, history without old tool outputs, streaming | `agno_agent.py` |
| 2 | Long-term user memory + rolling chat summary, on a cheap "housekeeping" model | `agno_agent.py` |
| 3 | Human-in-the-loop: multiple-choice (`ask_user`), free text (`get_user_input`), approval before writes (`save_note`) | `agno_agent.py` |
| 4 | Document tools: `list_documents`, `read_document`, `find_in_document`; per-user folders; page markers; capped reads; text cache | `doc_tools.py` |
| 5 | Tool guard: read-once per run, error safety net, output cap, audit log, previous-turn tool activity | `tool_guard.py` |
| 6 | Document list in the system prompt; untrusted text fenced with a per-chat code (prompt-injection defence) | `spotlight.py`, `doc_tools.py` |
| 7 | Grounding check: names, numbers, quotes and citations in each answer verified against this turn's tool results (Arabic-aware); one automatic correction if something isn't found | `grounding.py` |
| 8 | Search across all documents: `search_documents` (hybrid pgvector + Arabic-stemmed keywords, RRF), `find_in_document("all", ...)` with Arabic-tolerant matching, one-line summary per file in the documents list | `search_index.py`, `doc_tools.py` |
| 9 | `fetch_documents` (several docs in one call, shared size budget) and `read_table_cells` (xlsx/csv with sheets, ranges and cell coordinates) | `doc_tools.py` |
| 10 | Workflows: saved step-by-step procedures as `SKILL.md` folders (Agno `LocalSkills` loader), shared + per-user, `list_workflows` / `read_workflow`, `/wf` command | `workflows.py`, `workflows/` |
| – | Reply language decided in code: Arabic or English, other languages refused | `language.py` |
| – | Routing (`planner.py`): rules, then a 1.5 s `adept3o` JSON call; `chat` (small agent, no document tools, escape tool `needs_documents`), `documents` (full agent, gets Arabic + English search hints) or `overview` (file list). Every doubt goes to documents | `planner.py`, `agno_agent.py` |
| – | LiteLLM proxy: aliases `sift-local` / `sift-fast` / `sift-strong`, 16k limit for adept3o, retries, cooldowns, fallbacks between cloud models, Langfuse logging | `litellm/` |
| – | OCR for PDFs (Nanonets OCR2 via `NANO_OCR_API_BASE_URL`): every page by default, or scanned pages only (`OCR_MODE=missing`); cached per file | `doc_tools.py` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in your keys
mkdir -p docs/demo-user       # put PDF / DOCX / XLSX / TXT files here
python doc_tools.py --extract docs/demo-user   # optional: run OCR once up front
./litellm/run_proxy.sh &      # optional: LiteLLM proxy (needs LITELLM_* in .env)
python agno_agent.py
```

Chat commands: `/docs` `/reindex` `/workflows` `/wf <name> [text]` `/memories` `/summary` `/activity` `/audit` `/new`

### Search index (step 8)

```bash
# Postgres + pgvector (optional; without VECTOR_DB_URL a local SQLite index is used)
docker run -d --name sift-pg -e POSTGRES_USER=sift -e POSTGRES_PASSWORD=sift \
  -e POSTGRES_DB=sift -p 5432:5432 pgvector/pgvector:pg16

# Embedding model on vLLM (Arabic + English), OpenAI-compatible /v1/embeddings
vllm serve BAAI/bge-m3 --task embed --port 8001     # then EMBEDDING_BASE_URL=http://localhost:8001/v1

python search_index.py demo-user           # index now (also happens automatically per turn)
python search_index.py demo-user --force   # rebuild after changing embedding/summary model
```

### Workflows (step 10)

A workflow is a folder with a `SKILL.md` (Agent Skills format):

```
workflows/shared/<name>/SKILL.md        # for everyone
workflows/users/<user_id>/<name>/SKILL.md  # one user's own (overrides a shared one with the same name)
```

```markdown
---
name: gazette-appointments            # lowercase + hyphens, same as the folder name
description: When to use it, in one line (quote it if it contains ": ")
---
1. Step one ...
```

The agent sees names and descriptions, loads the steps with `read_workflow` when a request matches,
or you run one directly: `/wf gazette-appointments العدد 790`.
Included: `gazette-appointments`, `document-brief`, `compare-documents`.

## Tests (offline, no API key)

```bash
python tests/test_step5.py
python tests/test_step6.py
python tests/test_step7.py
python tests/test_step8.py                       # local SQLite index
VECTOR_DB_URL=postgresql://... python tests/test_step8.py   # against pgvector
python tests/test_step9_10.py
```

## Layout

```
agno_agent.py     agent, prompts, streaming CLI, human-in-the-loop handling
doc_tools.py      document tools + text extraction + AVAILABLE DOCUMENTS block
tool_guard.py     tool_hooks guard + post-hook tool-activity note
spotlight.py      untrusted-text fencing (nonce per chat)
grounding.py      answer-vs-evidence check (step 7)
search_index.py   chunking, embeddings, pgvector/SQLite index, search_documents, summaries (step 8)
workflows.py      saved workflows: loader, prompt block, list/read tools (step 10)
workflows/        shared/ and users/<user_id>/ workflow folders
language.py       reply-language detection
tests/            offline tests
examples/         router_agno.py (router + specialists example)
docs/<user_id>/   each user's documents (not committed)
```

## Roadmap

- Step 9: API server (AgentOS / FastAPI) + Postgres
- Step 10: document editing with versions, DOCX/XLSX generation, tabular review
