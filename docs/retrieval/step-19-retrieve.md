# Step 19 · `retrieve`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 19 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 18 · `top_k_chunks`](step-18-top-k-chunks.md) · Next: [Step 20 · `build_faiss_index`](step-20-build-faiss-index.md)

---

## The task

```python
def retrieve(query: str, model, chunk_matrix, chunks: list, k: int) -> list: ...
```

The end-to-end dense retrieval entry point. Embed the query, score it against
every row, return the top-k chunks paired with their scores. Reuse the existing
helpers rather than recomputing anything.

---

## Why this step exists

Steps 12, 16, 17, and 18 are each useful alone and tedious to chain by hand at
every call site. This is the one function an application calls, and the point at
which "a RAG pipeline" becomes a thing you can invoke.

It is also where the naming stops being about vectors. Everything below takes
arrays; this takes a string and returns text with provenance.

---

## What's happening

```python
query_vector = embed_text(model, query)
scores = cosine_similarity_search(query_vector, chunk_matrix)

return top_k_chunks(scores, chunks, k)
```

Three lines, three steps, no new logic. That is deliberate — the value of the
function is that it is the *only* place the order of those three operations is
written down, so it cannot be got wrong differently in two places.

Reuse matters for a specific reason beyond tidiness: `embed_text` is where the
query gets its
[float32 cast and 1D shape](../embeddings/step-12-embed-text.md). Re-implementing the
embedding call here would be the natural place to forget one of those and end up
with a `(1, d)` query that broadcasts into an `(n, d)` result instead of `(n,)`
scores.

---

## Boundaries of the contract

**The model must be the one the corpus was embedded with.** Nothing checks it,
and the failure is silent. This is the last place the invariant from
[step 11](../embeddings/step-11-load-embedding-model.md) could plausibly be
enforced, and it isn't — the function takes a model, not a corpus handle that
knows its own model.

**No normalisation is applied here**, because
[`cosine_similarity_search`](step-16-cosine-similarity-search.md) normalises
internally. So `retrieve` works on an unnormalised corpus too, unlike the FAISS
path. Convenient, and slightly slower per query.

**No threshold.** The top `k` come back however weak they are, including on a
query about something the corpus has never heard of. Pair with
[`handle_no_context`](../robustness/step-47-handle-no-context.md) before feeding
the results to a generator.

**No caching.** The same query re-embeds every call. See
[step 49](../robustness/step-49-cache-query-embedding.md).

**Returns chunk dicts, not indices.** The guide names this as the pitfall, and
the reason is concrete: everything downstream reads `chunk['text']`. An index is
only meaningful next to the exact list it indexes, and that list is often not in
scope by the time an answer is being rendered.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning raw indices | Every consumer needs the chunk list to interpret them. Breaks `chunk['text']`. |
| Re-embedding the corpus per query | The corpus matrix is precomputed for exactly this reason. |
| A different model from the one that built the matrix | No error, no shape mismatch, just wrong rankings. |
| Passing the matrix and chunks from different builds | Row *i* resolves to the wrong text. |
| Treating the score as a probability | It is a cosine. Its useful range is model-specific and mostly narrower than [-1, 1]. |

---

## Example

```python
>>> hits = retrieve('q1', model, matrix, chunks, 2)
>>> [(c['chunk_id'], round(s, 4)) for c, s in hits]
[('c1', 0.8), ('c0', 0.6)]
```

---

## Where it fits

```
  query str ──▶ [ embed_text ] ──▶ [ cosine_similarity_search ] ──▶ [ top_k_chunks ]
                                                                          │
                                                                          ▼
                                                              [(chunk, score), ...]
                                                                          │
                                                                          ▼
                                                   Part 5 · Prompting and Generation
```

This is the seam between "a pile of vectors" and "a system that answers
questions". [`rag_answer`](../generation/step-30-rag-answer.md) in Part 5 wraps
it with prompt assembly and a language model.
