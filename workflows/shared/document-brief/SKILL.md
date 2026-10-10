---
name: document-brief
description: A one-page structured brief of a single document (what it is, key points, decisions, dates, amounts, people, open questions) with page citations. Use for "summarise this file", "brief me on", "لخص", "ملخص".
---

# Document brief

## Steps

1. Pick the document from AVAILABLE DOCUMENTS (use the summaries). If it is
   unclear which one, ask with ask_user.
2. Read it with read_document. If it is truncated, use find_in_document /
   search_documents for the remaining sections (decisions, amounts, dates).
   For spreadsheets use read_table_cells instead.
3. Write the brief with these headings (in the reply language):
   - What it is: type, issuer, number/date (one line)
   - Key points: 3-7 bullets
   - Decisions / obligations: who must do what, by when
   - Numbers and dates: amounts, deadlines, effective dates
   - People and bodies named
   - Open questions: anything unclear, missing or contradictory
4. Cite each point as (doc_id, Page N). Quote exact wording for decisions.
5. Keep it under ~400 words unless the user asks for more.
