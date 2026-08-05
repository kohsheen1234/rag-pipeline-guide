# Step 20 · `build_faiss_index`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 20 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 19 · `retrieve`](step-19-retrieve.md) · Next: [Step 21 · `faiss_search`](step-21-faiss-search.md)

---

## The task

```python
def build_faiss_index(matrix: np.ndarray): ...
```

Construct a flat inner-product FAISS index over an `(n, d)` float32 matrix whose
rows are already L2-normalised, add all vectors, and return the populated index.

---

## Why this step exists

The numpy search is `O(n · d)` per query in Python-driven BLAS calls. That is
fine for thousands of chunks and slow for millions. FAISS does the same scan in
optimised C++ with SIMD, and is the standard tool for this.

Worth being clear about what you are and are not buying. `IndexFlatIP` is
**exact**: it compares against every vector, same as numpy, and returns the same
neighbours. The speedup is constant-factor, not asymptotic. FAISS's approximate
indexes (`IVF`, `HNSW`, `PQ`) are where the asymptotic wins live, and they trade
recall for it. Starting flat means the comparison in
[step 22](step-22-compare-faiss-to-numpy.md) is a genuine correctness check
rather than a tolerance check.

---

## What's happening

```python
import faiss

index = faiss.IndexFlatIP(matrix.shape[1])
index.add(matrix)

return index
```

**`IndexFlatIP(d)`** — *Flat* means uncompressed, every vector stored as-is.
*IP* means inner product. The dimensionality is fixed at construction and cannot
change afterwards, so it is read off the matrix rather than passed in, which
removes one way to get it wrong.

**Why IP and not L2.** There is no cosine index in FAISS, because for unit
vectors cosine *is* the inner product. That equivalence is the whole reason
[step 14](../embeddings/step-14-l2-normalize.md) exists. If the rows are not
normalised, this index still works, still returns numbers, and ranks by raw
inner product — which means longer vectors win regardless of direction. Nothing
raises. This is the sharpest edge in Part 4.

**`index.add(matrix)`** copies the vectors in. Row order is preserved, so FAISS
ids are positions in the original chunk list, and the same
`index → chunk` mapping as the numpy path applies.

**The lazy import** keeps the numpy path usable without `faiss` installed. See
the [Part 4 overview](00-overview.md#on-testing-without-faiss).

---

## Boundaries of the contract

**float32, contiguous.** FAISS is strict. float64 raises or forces a copy —
numpy's default dtype is float64, so `np.array([[1.0, 2.0]])` without an
explicit dtype will trip this. A non-contiguous array (a slice or a transpose)
may also need `np.ascontiguousarray`.

**No normalisation is applied.** The function trusts the caller. `faiss` ships
`faiss.normalize_L2(matrix)` which normalises in place if you want a belt.

**No id mapping.** Results are row positions. `IndexIDMap` exists if you need
your own ids; here positions are the ids, consistent with
[step 13](../embeddings/step-13-embed-chunks.md).

**Additive only.** You can `add` more vectors later; a flat index has no delete.
Removing a chunk means rebuilding.

**Memory is `n · d · 4` bytes.** A million 384-dim vectors is about 1.5 GB,
uncompressed and resident. That is the price of exactness.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Unnormalised vectors with `IndexFlatIP` | Returns inner products, not cosines. Long vectors outrank relevant ones, silently. |
| float64 input | Raises, or silently copies and doubles memory. |
| Mismatched `d` between index and matrix | Raises on `add`, which is at least loud. |
| Assuming FAISS is approximate | `IndexFlatIP` is exact. If results differ from numpy, something is genuinely wrong. |
| Expecting a speedup on a small corpus | Under a few thousand chunks the overhead can exceed the gain. |

---

## Example

```python
>>> M = np.eye(3, dtype=np.float32)
>>> idx = build_faiss_index(M)
>>> idx.ntotal
3
>>> idx.d
3
```

---

## Where it fits

```
  (n, d) float32, unit rows ──▶ [ build_faiss_index ] ──▶ index ──┬──▶ [ faiss_search ]
                                                                  └──▶ [ save_faiss_index ]
```
