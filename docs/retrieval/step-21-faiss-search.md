# Step 21 · `faiss_search`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 21 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 20 · `build_faiss_index`](step-20-build-faiss-index.md) · Next: [Step 22 · `compare_faiss_to_numpy`](step-22-compare-faiss-to-numpy.md)

---

## The task

```python
def faiss_search(index, query: np.ndarray, k: int): ...
```

Query a FAISS-style index with a single `(d,)` embedding and return the top-k
scores and indices as flat 1D arrays. Indices must be `int64`, scores `float32`.

---

## Why this step exists

FAISS is built for batches. `index.search` takes `(nq, d)` and returns two
`(nq, k)` matrices. The rest of this pipeline works one query at a time and
expects flat vectors, the same shape contract the
[numpy retriever](step-17-top-k-indices.md) uses.

Without this wrapper every call site would add and drop the batch axis itself,
and the shapes are just similar enough to go wrong quietly: a `(1, k)` score
array broadcasts against a `(k,)` array instead of failing, so the bug surfaces
as strange numbers somewhere else.

---

## What's happening

```python
scores, indices = index.search(query.reshape(1, -1), k)

return (
    np.asarray(scores, dtype=np.float32).reshape(-1),
    np.asarray(indices, dtype=np.int64).reshape(-1),
)
```

**`query.reshape(1, -1)`** adds the batch axis: `(d,)` becomes `(1, d)`. Passing
a 1D array directly is rejected by FAISS.

**`.reshape(-1)` on both outputs** removes it again, turning `(1, k)` into
`(k,)`. Same reasoning as
[`embed_text`](../embeddings/step-12-embed-text.md#whats-happening): `reshape`
rather than `squeeze`, so a `k == 1` result stays a 1-element vector instead of
collapsing to a scalar.

**The dtype casts** pin the contract. FAISS returns float32 and int64 already,
so the casts are no-ops against real FAISS — they exist so that a stand-in index
(or a future backend) cannot quietly hand back float64 scores or int32 ids.
`int64` matters because ids index into the chunk list, and FAISS uses `-1` to
mean "fewer than k results were available".

**Returned scores are cosines**, given a normalised corpus and query, sorted
descending. No post-processing needed.

---

## Boundaries of the contract

**`-1` indices are possible.** If the index holds fewer than `k` vectors, FAISS
pads the result with `-1` and a score of `-inf`. Nothing here filters those, so
`chunks[-1]` would silently return the *last* chunk — a real bug waiting for a
small corpus. Guard with `k = min(k, index.ntotal)` or drop negative ids at the
call site.

**Query dtype is not cast.** A float64 query into a float32 index raises or
loses precision. Unlike the matrix in
[step 20](step-20-build-faiss-index.md), there is no cast here, because
`embed_text` is supposed to have already produced float32.

**Single query only.** Batching is where FAISS earns its keep; this wrapper
gives that up for shape consistency with the numpy path.

**Duck-typed.** Anything with a `.search(queries, k)` returning two 2D arrays
works, which is how the tests exercise it without `faiss` installed.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning the raw `(1, k)` arrays | Broadcasts instead of failing. Wrong numbers, no traceback. |
| Passing a `(d,)` query straight to `index.search` | FAISS rejects it; at least this one is loud. |
| `np.squeeze` instead of `reshape(-1)` | `k == 1` collapses to a 0-d scalar. |
| Not handling `-1` ids | `chunks[-1]` returns the last chunk as if it were a hit. |
| float64 query into a float32 index | Raises, or degrades precision. |
| Assuming scores need sorting | They already come back descending. |

---

## Example

```python
>>> M = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
>>> idx = build_faiss_index(M)
>>> q = np.array([1.0, 0.0], dtype=np.float32)
>>> scores, ids = faiss_search(idx, q, 2)
>>> ids.tolist()
[0, 2]
>>> [round(float(s), 4) for s in scores]
[1.0, 0.7071]
```

Identical to what [step 16](step-16-cosine-similarity-search.md) gives on the
same input, which is the point.

---

## Where it fits

```
  (d,) query ──▶ [ faiss_search ] ──▶ (k,) scores, (k,) int64 indices
                        │
                        └── indices are positions in the chunk list
```
