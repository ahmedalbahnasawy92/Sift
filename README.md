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
| – | Reply language decided in code: Arabic or English, other languages refused | `language.py` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in your keys
mkdir -p docs/demo-user       # put PDF / DOCX / XLSX / TXT files here
python agno_agent.py
```

Chat commands: `/docs` `/memories` `/summary` `/activity` `/audit` `/new`

## Tests (offline, no API key)

```bash
python tests/test_step5.py
python tests/test_step6.py
```

## Layout

```
agno_agent.py     agent, prompts, streaming CLI, human-in-the-loop handling
doc_tools.py      document tools + text extraction + AVAILABLE DOCUMENTS block
tool_guard.py     tool_hooks guard + post-hook tool-activity note
spotlight.py      untrusted-text fencing (nonce per chat)
language.py       reply-language detection
tests/            offline tests
examples/         router_agno.py (router + specialists example)
docs/<user_id>/   each user's documents (not committed)
```

## Roadmap

- Step 7: grounding check — verify names and quotes in answers against this turn's tool results
- Step 8: semantic search (Knowledge + pgvector) for large documents
- Step 9: API server (AgentOS / FastAPI) + Postgres
- Step 10: document editing with versions, DOCX/XLSX generation, tabular review
