# Step 34 · `hyde_retrieve`

> **Part 6 · Advanced Retrieval Techniques** — step 34 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 33 · `query_rewrite`](step-33-query-rewrite.md) · Next: [Step 35 · `reciprocal_rank_fusion`](step-35-reciprocal-rank-fusion.md)

---

## The task

```python
def hyde_retrieve(query, hypothetical_answer, chunks, embeddings, model, k=5) -> list: ...
```

Dense retrieval, but embedding a *hypothetical answer* instead of the query.
Return the top-k chunk dicts ranked by cosine similarity between that embedding
and the precomputed chunk embeddings.

---

## Why this step exists

Embedding models place text by what it *is* as much as by what it is about. A
question and the passage answering it are different kinds of text — one short
and interrogative, one long and declarative — and that difference registers in
the vector. So the nearest neighbours of `"who founded Acme?"` are often other
questions, or sentences shaped like questions, rather than `"Acme was founded in
1994 by Jane Roe."`

HyDE (Hypothetical Document Embeddings) sidesteps it. Ask a language model to
*guess* an answer, then search with that. The guess is usually wrong on the
facts and right on the shape: same register, same vocabulary, same length as a
real passage. Searching with a fake passage finds real ones.

The counter-intuitive part is that the guess being factually wrong barely
matters. You are not using it as evidence, only as a probe.

---

## What's happening

```python
vector = embed_text(model, hypothetical_answer)
scores = cosine_similarity_search(vector, embeddings)

return [chunk for chunk, _ in top_k_chunks(scores, chunks, k)]
```

Identical to [`retrieve`](../retrieval/step-19-retrieve.md) except for the first
argument to `embed_text`, and that one substitution is the entire technique.

**The query is never embedded.** It is a parameter, and it is deliberately
unused in the scoring. The guide names ignoring the hypothetical as the pitfall,
and it is a quiet one: pass `query` to `embed_text` by mistake and the function
still runs, still returns plausible chunks, and is simply plain dense retrieval
wearing a different name. A test asserts the model saw the hypothetical and not
the query.

The query is kept in the signature because the caller still needs it — for the
prompt, and for logging what was actually asked.

**Chunk dicts, not tuples.** Unlike `retrieve`, scores are dropped. They are
similarities to a fabricated document, so their absolute values mean even less
than usual; using them for a threshold would be misleading.

---

## Where the hypothetical comes from

Not from here. The function takes it as a string, which keeps it testable and
free of a model dependency, but it means the expensive half is the caller's
problem:

```python
draft = generate_answer(model, tokenizer, f"Write a short passage answering: {query}")
hits = hyde_retrieve(query, draft, chunks, embeddings, embed_model, k=5)
```

That is a full generation before retrieval even starts, which is the real cost
of HyDE: every query now pays for two model calls, and the latency of the first
one is on the critical path.

---

## Boundaries of the contract

**No fallback.** If the hypothetical is empty or nonsense, retrieval is
correspondingly bad, with no comparison against the plain-query path. Running
both and fusing with
[`reciprocal_rank_fusion`](step-35-reciprocal-rank-fusion.md) is the usual
hedge.

**No averaging.** A common variant blends the query and hypothetical vectors.
The guide's warning applies if you try it: average *before* normalising and
whichever vector is longer dominates. Normalise both, then average, then
normalise again.

**Helps least where the query is already passage-like.** A long, detailed
question does not need this, and paying for a generation to produce one is
wasted.

**Can amplify a bad guess.** If the model hallucinates confidently in the wrong
domain, the probe lands in the wrong region and retrieval is worse than plain
dense search. HyDE is not a free win.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Embedding the query instead of the hypothetical | Silently reduces to plain dense retrieval. Same shapes, no error. |
| Averaging the two vectors unnormalised | The longer vector dominates the mix. |
| Putting the hypothetical in the prompt | It is fabricated. Only the retrieved chunks are evidence. |
| Ignoring the latency | A generation per query, before retrieval starts. |
| Using it for short factoid lookups on a keyword-heavy corpus | BM25 would do better and cost nothing. |

---

## Example

```python
>>> chunks = [{'chunk_id': 'c0'}, {'chunk_id': 'c1'}]
>>> emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
>>> out = hyde_retrieve('q?', 'hypo about second', chunks, emb, model, k=1)
>>> [c['chunk_id'] for c in out]
['c1']
```

---

## Where it fits

```
  query ──▶ [ LLM drafts an answer ] ──▶ hypothetical
                                              │
                                              ▼
                                       [ embed_text ]
                                              │
                                              ▼
                              [ cosine_similarity_search ] ──▶ top-k chunks
  query ──────────────────────────────────────────────────▶ the prompt
```
