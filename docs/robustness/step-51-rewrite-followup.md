# Step 51 · `rewrite_followup`

> **Part 8 · Robustness, Caching, and Chat Memory** — step 51 of 51
> Code: [`rag_pipeline/robustness.py`](../../rag_pipeline/robustness.py) · Tests: [`tests/test_robustness.py`](../../tests/test_robustness.py)
> Previous: [Step 50 · `update_chat_memory`](step-50-update-chat-memory.md)

---

## The task

```python
def rewrite_followup(followup: str, history: list) -> str: ...
```

Convert a conversational follow-up into a self-contained search query by
stitching the most recent user turn onto it. With no prior user turn, return the
cleaned follow-up. Normalise whitespace with
[`normalize_text`](../ingestion/step-04-normalize-text.md).

---

## The last step

Fifty steps have assumed each query stands alone. Real conversations do not work
that way, and this is the gap:

> **User:** Tell me about Mars.
> **Assistant:** Mars is red.
> **User:** How big is it?

Embed `"How big is it?"` and you get a vector for a question about size and an
unresolved pronoun. Nothing in the corpus is about "it". Retrieval returns
whatever is nearest to a generic size question, and the whole pipeline —
chunking, embedding, hybrid search, reranking — operates flawlessly on a query
that carries no topic.

This is **anaphora**, and it is the single most common way a working RAG system
fails the moment it is put behind a chat interface.

---

## What's happening

```python
previous = [turn["content"] for turn in history if turn.get("role") == "user"]

if not previous:
    return normalize_text(followup)

return normalize_text(f"{previous[-1]} {followup}")
```

**Filter by role, then take the last.** Both halves are the guide's named
pitfall, and they fail differently.

Taking `history[-1]` without filtering grabs the *assistant's* answer, so the
query becomes `"Mars is red. How big is it?"`. That is not absurd — it does
mention Mars — but it seeds the query with the model's own output, which
compounds any error already there and pulls retrieval toward text the model
generated rather than text the user asked about.

Taking the *first* user turn instead of the last freezes the conversation at its
opening subject. Three topic changes later, every follow-up is still being
rewritten against the original question.

`previous[-1]` after filtering is the most recent thing the *user* actually
asked, which is where the referent almost always lives.

**`.get("role")`, not `["role"]`**, so a malformed turn is skipped rather than
raising mid-conversation. Consistent with
[`track_source_chunk_ids`](../generation/step-31-track-source-chunk-ids.md)'s
tolerance of partial records.

**`normalize_text` on the way out** collapses the whitespace introduced by the
join and any the user typed. Note it does *not* lowercase — case survives, which
matters if the rewritten query is shown to a user or logged.

---

## Why concatenation is enough

Prepending the previous question is crude. It does not resolve the pronoun; it
just puts the referent in the same string, so `"Tell me about Mars. How big is
it?"` embeds near passages about Mars and about size.

For dense retrieval, that is usually enough — the vector only needs to land in
the right neighbourhood, and it does not care that the sentence is ungrammatical.
For BM25 it works too: `Mars` is now a query term.

The alternative is an LLM rewriter producing `"How big is Mars?"`, which is
genuinely better and costs a generation per turn, adds latency to the critical
path, and can hallucinate a referent that was never there. Deterministic
concatenation is free, testable, and has no failure mode worse than a slightly
noisy query.

---

## Boundaries of the contract

**One turn of context.** Only the last user turn. A reference reaching back two
questions is not resolved.

**Always concatenates.** No attempt to detect whether the follow-up *needs*
rewriting. A self-contained question like `"What is FAISS?"` still gets the
previous question glued to the front, which dilutes a query that was already
fine. Detecting anaphora — leading pronoun, very short query, no noun — would
help and is not here.

**Topic changes are polluting.** Ask about Mars, then ask about tax law, and the
tax query is rewritten as `"Tell me about Mars. What are the tax rules?"`. This
is the cost of always concatenating, and it is the main reason to add a
detection heuristic.

**Assistant turns are ignored entirely**, even when the answer introduced the
entity the follow-up refers to.

**Not wired in.** Call it before
[`cache_query_embedding`](step-49-cache-query-embedding.md) — the rewritten
string is what you want as the cache key, and what you want to embed. The
*original* follow-up is still what belongs in the
[prompt](../generation/step-30-rag-answer.md#the-original-query-goes-in-the-prompt).

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `history[-1]` without filtering by role | Grabs the assistant's answer and seeds retrieval with model output. |
| Taking the first user turn | The conversation is frozen at its opening subject. |
| Concatenating the whole history | The query becomes a wall of text; the actual question is diluted. |
| Rewriting unconditionally after a topic change | The old subject pollutes an unrelated query. |
| Putting the rewritten query in the prompt | The model answers something the user did not ask. |
| Skipping normalisation | Double spaces from the join reach the tokenizer. |

---

## Example

```python
>>> rewrite_followup('How big is it?', [])
'How big is it?'
>>> history = [{'role': 'user', 'content': 'Tell me about Mars.'},
...            {'role': 'assistant', 'content': 'Mars is red.'}]
>>> rewrite_followup('How big is it?', history)
'Tell me about Mars. How big is it?'
```

The assistant's `"Mars is red."` is skipped, which is the behaviour the role
filter buys.

---

## Where it fits

```
  follow-up ──▶ [ rewrite_followup ] ──▶ standalone query ──▶ embed ──▶ retrieve
       │                  ▲
       │            chat history
       │
       └──────────────────────────────────────────────▶ the prompt (unchanged)
```

---

## The end of the guide

Fifty-one steps: a text file becomes a document, a document becomes chunks,
chunks become vectors, vectors become search, search becomes an answer, the
answer gets measured, and the whole thing learns to hold a conversation and to
admit when it does not know.

Every piece is small enough to read. That was the point.
