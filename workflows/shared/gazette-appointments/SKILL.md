---
name: gazette-appointments
description: List every appointment, committee formation or membership decision in an official gazette issue as a table (decision number, person, role, body, page). Use for "who was appointed", "list the committees", "من تم تعيينه", "أعضاء اللجان".
---

# Gazette appointments and committees

Goal: a complete, verified table of every decision that names people.

## Steps

1. Identify the gazette issue. If the user didn't name one, use the most
   recent gazette in AVAILABLE DOCUMENTS; if several could match, ask with
   ask_user (2-4 choices).
2. Find the decisions. Run these searches on that document (doc_ids=...):
   - search_documents: "تعيين", "تشكيل لجنة", "عضوية", "appointment", "committee"
   - find_in_document: "قرار رقم" to list every decision number with its page
3. For each decision that names a person, read enough context to get all
   fields: find_in_document with the decision number and context_chars 600-1000.
4. Build the table. Columns:
   | Decision no. | Person | Role (رئيس/عضو/مقرر...) | Body / committee | Page |
   One row per person. Copy names EXACTLY as written, with titles (السيد/،
   الدكتور/). Never complete or correct a name.
5. Check completeness: count the decisions you found in step 2 and say how
   many were included and which (if any) were skipped and why.
6. Offer next steps: save the table with save_note, or ask about one decision.

## Rules
- Every row must come from this turn's tool results (the grounding check
  verifies names and numbers).
- If a page is unreadable or a name is cut off, write "غير واضح / unclear"
  instead of guessing.
