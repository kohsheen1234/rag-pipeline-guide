# Step 38 · `rerank_cross_encoder`

> **Part 6 · Advanced Retrieval Techniques** — step 38 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 37 · `hybrid_search`](step-37-hybrid-search.md) · Next: [Step 39 · `maximal_marginal_relevance`](step-39-maximal-marginal-relevance.md)

---

## The task

```python
def rerank_cross_encoder(query, candidate_chunks, cross_encoder) -> list: ...
```

Score every `(query, chunk_text)` pair with a cross-encoder's `.predict(pairs)`
and return the chunks sorted from most to least relevant.

---

## Why this step exists

Everything so far has used a **bi-encoder**: the query and each chunk are
embedded separately and compared as vectors. That separation is what makes the
corpus precomputable — embed a million chunks once, then each query is a matrix
multiply.

It is also a real limit. The chunk's vector was produced with no knowledge of
the query. All the interaction between the two is compressed into a single dot
product between two fixed points.

A **cross-encoder** concatenates query and passage and runs them through a
transformer together, so every query token can attend to every passage token,
and the output is a single relevance score. It is markedly more accurate, and it
cannot precompute anything: scoring `N` chunks means `N` forward passes, at
query time.

So you use both. Retrieve 50 cheaply, rerank those 50 expensively, keep 5. The
bi-encoder needs only to get the right answer into the top 50; the cross-encoder
sorts out the order.

---

## What's happening

```python
pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
scores = cross_encoder.predict(pairs)

order = sorted(range(len(candidate_chunks)), key=lambda i: -scores[i])

return [candidate_chunks[index] for index in order]
```

**Pairs, not bare texts.** A cross-encoder's input is the pair; passing only the
chunk text asks "how relevant is this passage to nothing", which most models
will happily score. The guide names this, and it is silent — you get numbers, and
they rank something other than relevance to the query.

**One batched `predict` call**, not one per chunk. Same batching argument as
[`embed_chunks`](../embeddings/step-13-embed-chunks.md).

### Sorting indices, not tuples

The natural-looking version is:

```python
sorted(zip(scores, candidate_chunks), key=lambda p: -p[0])   # don't
```

which works until two chunks score identically. Python's tuple comparison then
falls through to the second element, compares two dicts, and raises
`TypeError: '<' not supported between instances of 'dict' and 'dict'`. Tied
scores are not rare with a cross-encoder rounding to float32.

Sorting the *indices* with an explicit key never compares the chunks at all.
`sorted` is stable, so ties keep their input order. There is a test for the tied
case specifically.

**`-scores[i]` for descending.** Same trap as
[`top_k_indices`](../retrieval/step-17-top-k-indices.md): the obvious sort is
ascending, and forgetting the minus returns the *least* relevant candidates
first, in a plausible-looking list.

---

## Boundaries of the contract

**No truncation.** All candidates come back, reordered. Slice to `k` yourself.

**Scores are dropped.** Only the reordered chunks are returned, so you cannot
threshold on cross-encoder confidence afterwards — which is a shame, because
those scores are better calibrated than cosine. Returning `(chunk, score)`
pairs, as [`top_k_chunks`](../retrieval/step-18-top-k-chunks.md) does, would be
more useful.

**Duck-typed.** Anything with `.predict(pairs)` returning a sequence of numbers
works — `sentence_transformers.CrossEncoder`, an API client, or the stub in the
tests. No import, no optional dependency.

**Cost is linear in candidates.** Reranking 1000 chunks defeats the purpose.
The useful window is roughly 20–100.

**Long pairs get truncated.** Query plus passage must fit the cross-encoder's
window, typically 512 tokens. A long chunk is silently cut.

**Requires `text`.** `KeyError` otherwise.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Passing bare texts instead of pairs | Scores something other than relevance to the query. No error. |
| Sorting `(score, chunk)` tuples | `TypeError` on tied scores, comparing dicts. |
| Sorting ascending | Returns the worst candidates first, looking entirely normal. |
| Losing score-to-chunk alignment | Right order, wrong chunks. Same class of bug as [step 18](../retrieval/step-18-top-k-chunks.md#the-alignment-trap). |
| Reranking the whole corpus | One forward pass per chunk. This is the thing two-stage retrieval exists to avoid. |
| One `predict` call per pair | Loses batching. |

---

## Example

```python
>>> ce = DummyCE({'a': 0.1, 'b': 0.9, 'c': 0.5})
>>> chunks = [{'text': 'a'}, {'text': 'b'}, {'text': 'c'}]
>>> [c['text'] for c in rerank_cross_encoder('q', chunks, ce)]
['b', 'c', 'a']
```

---

## Where it fits

```
  first stage (cheap, whole corpus)        second stage (accurate, top-N)
  ┌────────────────────────────┐           ┌──────────────────────────┐
  │ dense / bm25 / hybrid      │──top 50──▶│ rerank_cross_encoder     │──top 5──▶
  └────────────────────────────┘           └──────────────────────────┘
```
