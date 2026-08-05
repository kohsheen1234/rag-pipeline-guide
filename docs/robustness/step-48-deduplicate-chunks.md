# Step 48 · `deduplicate_chunks`

> **Part 8 · Robustness, Caching, and Chat Memory** — step 48 of 51
> Code: [`rag_pipeline/robustness.py`](../../rag_pipeline/robustness.py) · Tests: [`tests/test_robustness.py`](../../tests/test_robustness.py)
> Previous: [Step 47 · `handle_no_context`](step-47-handle-no-context.md) · Next: [Step 49 · `cache_query_embedding`](step-49-cache-query-embedding.md)

---

## The task

```python
def deduplicate_chunks(chunks, embeddings, similarity_threshold): ...
```

Prune near-duplicates from a corpus given its L2-normalised embeddings. Keep the
first occurrence of each near-duplicate group; drop any later chunk whose cosine
similarity to an already-kept chunk strictly exceeds the threshold. Return both
the surviving chunks and the matching rows of the matrix.

---

## Why this step exists

Duplicates are not an edge case in a real corpus. They arrive from three
directions:

- **Boilerplate.** Legal footers, navigation text, and standard disclaimers
  repeat across every document.
- **Overlapping chunkers.**
  [`chunk_with_overlap`](../chunking/step-09-chunk-with-overlap.md) manufactures
  them deliberately, and its
  [redundant tail](../chunking/step-09-chunk-with-overlap.md#the-redundant-tail)
  produces chunks that are strict subsets of their neighbours.
- **Genuinely repeated content** across document versions.

The cost is the top-k budget. If three of your five retrieved chunks are the
same paragraph, the model sees three facts' worth of tokens and one fact's worth
of information. It also biases the answer: a claim that appears in the context
five times reads as more established than one appearing once, regardless of
whether the repetition is evidence or an artefact of chunking.

---

## What's happening

```python
kept_indices = []

for index in range(len(chunks)):
    vector = embeddings[index]
    duplicate = any(
        float(vector @ embeddings[kept]) > similarity_threshold
        for kept in kept_indices
    )
    if not duplicate:
        kept_indices.append(index)

return [chunks[index] for index in kept_indices], embeddings[kept_indices]
```

**Compared against the kept set, not all previous chunks.** This is what makes
"keep the first occurrence" well-defined and stops duplicates from chaining. If
A and B are near-duplicates and B and C are too, but A and C are not, then
comparing against everything seen would drop C for resembling B — a chunk that
was itself dropped. Comparing only against survivors keeps C.

It also means a chunk is never compared to itself, since its own index is not in
`kept_indices` when it is being tested. No self-similarity of 1.0 to special-case.

**`vector @ embeddings[kept]`** is a dot product, which is a cosine only because
the rows are unit length. Same assumption as
[`build_faiss_index`](../retrieval/step-20-build-faiss-index.md) and
[MMR](../advanced-retrieval/step-39-maximal-marginal-relevance.md), and unchecked
here too — unnormalised input silently compares magnitudes.

**Strictly greater than.** A chunk exactly at the threshold is kept. Consistent
with [`handle_no_context`](step-47-handle-no-context.md), and it means a
threshold of 1.0 keeps even exact duplicates, since their similarity is 1.0 and
`1.0 > 1.0` is false. Worth knowing if you reach for 1.0 expecting "drop only
identical chunks" — use 0.999.

**`any(...)` short-circuits**, so a chunk matching an early survivor stops
scanning.

**Both halves are returned**, and `embeddings[kept_indices]` uses numpy fancy
indexing to select the same rows in the same order. This is the pattern
[`filter_by_metadata`](../advanced-retrieval/step-40-filter-by-metadata.md)
notably does *not* follow — returning chunks alone would leave the matrix
misaligned, and row *i* would stop meaning chunk *i*, corrupting every
subsequent retrieval with no error.

---

## Cost

`O(n · k)` dot products, where `k` is the number of survivors, so worst case
`O(n²)` on a corpus with no duplicates — every chunk compared against every
survivor. At 100k chunks that is 5 billion dot products in a Python loop.

For anything large, the practical approach is to compute `embeddings @
embeddings.T` in one BLAS call when memory allows, or cluster first and
deduplicate within clusters. This implementation is written to be read.

Run it **once, at index time**. It is not a per-query operation.

---

## Choosing the threshold

Not calibrated across models, same as
[step 47](step-47-handle-no-context.md#choosing-the-threshold). As a starting
point with sentence-transformers models:

- **0.99+** — essentially exact duplicates only
- **0.95** — near-duplicates, minor edits, the usual default
- **0.90** — same topic, different wording; starts removing genuinely distinct
  passages

Too aggressive and you delete content that only *looks* similar in embedding
space — two different error codes in the same template phrasing, for instance,
which sit very close together and mean entirely different things. Deletion is
not recoverable at query time.

---

## Boundaries of the contract

**Order-dependent.** "First occurrence" means the earliest in the input list
survives, so reordering the corpus changes which chunk is kept. Deterministic
for a fixed order, which is one more reason the
[filename ordering](../ingestion/step-02-load-text-directory.md) from step 2
matters.

**Positions shift.** Dropping chunks renumbers everything after them, so any
[`chunk_id`](../chunking/step-10-attach-chunk-metadata.md) already assigned by
position now disagrees with the row index. Deduplicate *before* assigning ids,
or accept that ids and rows have diverged.

**Greedy, not clustering.** No attempt to pick the best representative of a
group; the first one wins whether or not it is the most complete.

**Only embeddings.** Two chunks with identical text but different metadata are
one duplicate.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning chunks without the matrix rows | Row *i* stops meaning chunk *i*. Every later retrieval is wrong. |
| Comparing against all previous chunks | Duplicates chain; distinct content gets dropped. |
| `>=` instead of `>` | Discards vectors sitting exactly at the threshold. |
| Unnormalised embeddings | Dot products are not cosines; the threshold means nothing. |
| Threshold too low | Silently deletes distinct passages that happen to be phrased alike. |
| Running it per query | It is an index-time pass. |
| Deduplicating after assigning chunk ids | Ids and row positions diverge. |

---

## Example

```python
>>> chunks = [{'chunk_id': 0}, {'chunk_id': 1}, {'chunk_id': 2}]
>>> emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
>>> kept, kept_emb = deduplicate_chunks(chunks, emb, 0.95)
>>> [c['chunk_id'] for c in kept]
[0, 2]
>>> kept_emb.tolist()
[[1.0, 0.0], [0.0, 1.0]]
```

---

## Where it fits

```
  embed corpus ──▶ [ l2_normalize ] ──▶ [ deduplicate_chunks ] ──▶ save_corpus
                                               │
                                               └── index time, once
```
