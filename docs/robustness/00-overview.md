# Part 8 · Robustness, Caching, and Chat Memory

Parts 1 through 7 build a RAG pipeline and measure it. This part is the things
that turn it into something you could put in front of a user.

Five functions, three unrelated concerns, joined by being what the happy path
ignores.

## Knowing when to say nothing

[`handle_no_context`](step-47-handle-no-context.md) is the most important
function in this part. Retrieval always returns *something* — the top k chunks
exist whether or not any of them are relevant — and a language model handed
irrelevant context and a question will still answer. That combination is the
main way a RAG system produces confident nonsense.

A confidence gate turns a weak retrieval into an honest refusal. It uses the
same [`REFUSAL`](../generation/step-27-add-system-instruction.md) constant as
the prompt, so the system has exactly one way of saying it does not know.

[`deduplicate_chunks`](step-48-deduplicate-chunks.md) attacks the same budget
from a different angle: if three of your five retrieved chunks are the same
paragraph, you have effectively retrieved three. Pruning duplicates at index
time means the top-k slots hold distinct information.

## Not doing the same work twice

[`cache_query_embedding`](step-49-cache-query-embedding.md) is a dict lookup
standing in front of a transformer forward pass. In a chat loop the same query
recurs often enough for that to matter, and the embedding is a pure function of
the string, so memoising is safe.

## Making follow-ups searchable

"How big is it?" retrieves nothing useful, because "it" has no referent in the
embedding space. Chat needs state.

[`update_chat_memory`](step-50-update-chat-memory.md) keeps the conversation in
the role/content shape every chat API uses.
[`rewrite_followup`](step-51-rewrite-followup.md) uses it to stitch the previous
question onto the new one, restoring the referent before retrieval sees it.

## Steps

All live in [`rag_pipeline/robustness.py`](../../rag_pipeline/robustness.py).

| # | Function | What it does |
| --- | --- | --- |
| 47 | [`handle_no_context`](step-47-handle-no-context.md) | Abstain when nothing scored above the threshold. |
| 48 | [`deduplicate_chunks`](step-48-deduplicate-chunks.md) | Drop near-duplicates, keeping the first of each group. |
| 49 | [`cache_query_embedding`](step-49-cache-query-embedding.md) | Memoise query embeddings in a shared dict. |
| 50 | [`update_chat_memory`](step-50-update-chat-memory.md) | Append a user and assistant turn, without mutating. |
| 51 | [`rewrite_followup`](step-51-rewrite-followup.md) | Prepend the last user turn to make a follow-up standalone. |

## Where they go

```
  chat history ──▶ [ rewrite_followup ] ──▶ standalone query
                                                  │
                                                  ▼
                                   [ cache_query_embedding ] ──▶ vector
                                                  │
                                                  ▼
                                            retrieval over
                                    [ deduplicate_chunks ]'d corpus
                                                  │
                                                  ▼
                                     [ handle_no_context ]
                                          │            │
                                   abstain│            │answer
                                          ▼            ▼
                                 "I do not know"   generation
                                                       │
                                                       ▼
                                          [ update_chat_memory ]
                                                       │
                                                       └──▶ next turn
```

## A note on wiring

None of these are called by
[`rag_answer`](../generation/step-30-rag-answer.md). They are built as
standalone pieces, per the guide's step specifications, and assembling them into
a serving loop is left to you. The gate in particular belongs *before*
generation — checking after you have already paid for the tokens defeats the
purpose.
