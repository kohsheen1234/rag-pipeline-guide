# Step 37 · `hybrid_search`

> **Part 6 · Advanced Retrieval Techniques** — step 37 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 36 · `bm25_search`](step-36-bm25-search.md) · Next: [Step 38 · `rerank_cross_encoder`](step-38-rerank-cross-encoder.md)

---

## The task

```python
def hybrid_search(query, chunks, embeddings, model, alpha=0.5, k=5) -> list: ...
```

Combine dense cosine scores with BM25 lexical scores. Min-max scale each to
`[0, 1]`, mix as `alpha * dense + (1 - alpha) * bm25`, and return the top-k
`(chunk_index, combined_score)` pairs. Break ties by original chunk order.

---

## Why this step exists

The two retrievers fail in opposite directions. Dense search understands that
"car" and "automobile" are related and cannot find an error code it has never
seen. BM25 finds the error code exactly and has no idea the two words for car
are the same thing.

Running both and mixing gets you a ranking where a chunk needs to do well on at
least one axis, and does best when it does well on both.

---

## What's happening

```python
dense = cosine_similarity_search(embed_text(model, query), embeddings)

lexical = np.zeros(len(chunks), dtype=float)
for index, score in bm25_search(query, chunks, k=len(chunks)):
    lexical[index] = score

combined = alpha * _min_max(dense) + (1 - alpha) * _min_max(lexical)
order = np.argsort(-combined, kind="stable")[:k]

return [(int(index), float(combined[index])) for index in order]
```

### Filling the lexical gaps

[`bm25_search`](step-36-bm25-search.md) omits chunks with no query-term overlap,
so it returns a *sparse* list of `(index, score)` — often much shorter than the
corpus. The dense vector has one entry per chunk. Adding them requires the same
shape.

Hence the pre-allocated zero vector, scattered into by index. Chunks BM25 never
mentioned keep 0, which is the right value: no lexical evidence. Getting this
wrong — zipping the two result lists together, say — misaligns every score with
the wrong chunk, and the output still looks like a plausible ranking.

`k=len(chunks)` on the BM25 call makes sure nothing is truncated before the
scatter.

### Why normalise

Cosine sits in roughly `[-1, 1]`. BM25 is unbounded and grows with corpus size
and term rarity — scores of 8 or 15 are ordinary. Add them raw and the mix is
`0.7 + 12.3`; the BM25 term decides everything and `alpha` is decoration.

Min-max maps each onto `[0, 1]` so `alpha` means what it says.

The cost, worth knowing: min-max is **relative to this query's results**. The
best chunk always scores exactly 1.0 on each axis, however bad it is in absolute
terms. So a query that matches nothing still produces a confident-looking 1.0,
and combined scores cannot be compared across queries or thresholded. Use
[`handle_no_context`](../robustness/step-47-handle-no-context.md) on raw cosine
scores, not on these.

`_min_max` maps a flat vector to zeros rather than dividing by a zero range —
which is what happens on the lexical side whenever BM25 matched nothing.

### Ties

`kind="stable"` on the argsort means equal combined scores stay in chunk order,
matching [`top_k_indices`](../retrieval/step-17-top-k-indices.md). Deterministic
output makes the function testable.

`int(...)` and `float(...)` return Python scalars rather than numpy ones, so the
result is JSON-serialisable — same reasoning as
[`top_k_chunks`](../retrieval/step-18-top-k-chunks.md).

---

## Choosing alpha

`alpha = 1.0` is pure dense, `0.0` is pure lexical, `0.5` is an even mix.

There is no universally right value, and it depends on the corpus more than the
model. Technical documentation full of identifiers wants more lexical weight;
conversational or narrative content wants more dense. The honest answer is that
this is a parameter to *measure*, using [Part 7](../evaluation/00-overview.md) —
which is much of why Part 7 exists.

If you find yourself unable to pick, [RRF](step-35-reciprocal-rank-fusion.md)
sidesteps the question entirely.

---

## Boundaries of the contract

**Returns indices, not chunks.** Resolve them yourself.

**Two full scans per query.** Dense over the whole matrix, BM25 over the whole
corpus with an inner `df` scan. Fine for thousands of chunks.

**Scores are query-relative** and not thresholdable, as above.

**No metadata filtering.** Apply
[`filter_by_metadata`](step-40-filter-by-metadata.md) first — but note that
filtering changes the corpus BM25 computes `df` and `avgdl` over, and therefore
changes the scores.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Mixing raw cosine and raw BM25 | BM25's magnitude drowns the dense signal; `alpha` does nothing. |
| Zipping the two score lists | BM25's list is sparse. Every score attaches to the wrong chunk. |
| Truncating BM25 before the scatter | Chunks below its `k` silently lose their lexical score. |
| Thresholding the combined score | Min-max makes the top result 1.0 regardless of quality. |
| Unstable sort | Non-deterministic order among ties. |
| Returning numpy scalars | `json.dumps` raises later. |

---

## Example

```python
>>> chunks = [{'text': 'cat dog'}, {'text': 'fish bird'}, {'text': 'cat fish'}]
>>> emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
>>> hybrid_search('cat', chunks, emb, model, alpha=0.5, k=2)
[(0, 1.0), (2, 0.8535533905932737)]
```

Chunk 0 wins both axes. Chunk 2 matches lexically (`cat`) but is further away
densely, so it lands between. Chunk 1 has neither and drops out.

---

## Where it fits

```
  query ──┬──▶ [ embed_text ] ──▶ cosine ──▶ min-max ──┐
          │                                             ├──▶ α·d + (1-α)·b ──▶ top-k
          └──▶ [ bm25_search ] ──▶ scatter ──▶ min-max ─┘
```
