---
name: compare-documents
description: Compare two or more documents side by side (differences, changes between versions, conflicting numbers or dates) as a table with citations. Use for "compare", "what changed", "difference between", "قارن", "ما الفرق".
---

# Compare documents

## Steps

1. Identify the documents (2-5). If unclear, ask with ask_user.
2. Read them together with fetch_documents (small/medium files), or read
   each with read_document. For spreadsheets use read_table_cells.
3. Decide the comparison points from the content (e.g. parties, amounts,
   dates, obligations, members, decisions). If the user named points, use those.
4. Output a table: one row per point, one column per document, plus a
   "Difference" column. Cite every cell as (doc_id, Page N).
5. Below the table: the 3 most important differences in plain words, and
   anything present in one document but missing in the other.

## Rules
- Only compare what the tools returned; write "not stated" when a document
  doesn't mention a point.
