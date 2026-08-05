# Step 16 · `cosine_similarity_search`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 16 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 15 · `save_corpus`](../embeddings/step-15-save-corpus.md) · Next: [Step 17 · `top_k_indices`](step-17-top-k-indices.md)

---

## The task

```python
def cosine_similarity_search(query: np.ndarray, matrix: np.ndarray) -> np.ndarray: ...
```

Compute the cosine similarity between one `(d,)` query vector and every row of
an `(n, d)` chunk matrix, returning `(n,)` scores. Inputs are **not**
pre-normalised, so handle the normalisation here.

---

## Why this step exists

This is the numerical core of dense retrieval. Everything before it prepares
inputs; everything after it sorts the output.

Cosine is the standard choice because it measures direction and ignores
magnitude. Embedding magnitude correlates with things you do not want to rank
on — mostly text length and token count — so a raw dot product systematically
prefers longer chunks. Cosine removes that.

---

## What's happening

```python
query_norm = np.linalg.norm(query)
row_norms = np.linalg.norm(matrix, axis=1)
denominator = row_norms * query_norm

return (matrix @ query) / np.where(denominator == 0, 1, denominator)
```

`matrix @ query` is the whole numerator: an `(n, d)` by `(d,)` product giving
`(n,)` dot products, one per chunk, in a single BLAS call.

The denominator is the outer product of the row norms and the single query
norm. `axis=1` is the part to get right — the matrix is `(n, d)`, so norms have
to be taken across the *features* of each row, giving `n` values. Reducing along
`axis=0` gives `d` values instead, which either fails to broadcast or, when
`n == d`, silently produces a wrong `(n,)` result. There is a test for the
square case specifically, because that is where the mistake survives.

`np.where(denominator == 0, 1, denominator)` handles zero vectors. A zero row
has no direction, so its cosine is undefined; dividing anyway gives `nan`, and
NaN compares false against everything, so a NaN row silently never ranks. Using
1 as the divisor leaves those rows at 0, which is the sensible answer: no
direction, no similarity.

No `keepdims` is needed here, unlike
[`l2_normalize`](../embeddings/step-14-l2-normalize.md), because the result is
`(n,)` and divides a `(n,)` numerator elementwise.

---

## Boundaries of the contract

**It re-normalises every call.** If your corpus is already unit length — and
after [step 14](../embeddings/step-14-l2-normalize.md) it should be — this
recomputes `n` norms per query for no gain. Correct, and wasteful at scale. The
FAISS path skips it precisely because it assumes normalised input.

**One query at a time.** A `(m, d)` batch would need `matrix @ query.T` and a
different denominator shape.

**Scores are in [-1, 1].** Negative means pointing away. In practice most
sentence-embedding models produce mostly positive similarities, so the useful
range is narrower than the theoretical one, and absolute thresholds need
calibrating per model.

**No `inf`/`nan` guarding.** Only the zero case is handled.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `axis=0` instead of `axis=1` | Normalises down the feature axis. Silent whenever `n == d`. |
| Dividing without a zero guard | An empty chunk puts NaN in the scores, and NaN rows never rank. |
| Using a raw dot product | Ranks by magnitude as well as direction, quietly favouring longer chunks. |
| Normalising the corpus but not the query | Every score scaled by a constant. Ranking survives, thresholds do not. |
| Looping over rows | Correct and orders of magnitude slower than one matrix multiply. |

---

## Example

```python
>>> q = np.array([1.0, 0.0])
>>> M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
>>> np.round(cosine_similarity_search(q, M), 4).tolist()
[1.0, 0.0, 0.7071]
```

Same direction, orthogonal, 45°. Note the third row has a larger magnitude than
the first and still scores lower, which is the point of cosine.

---

## Where it fits

```
  (d,) query ──┐
               ├──▶ [ cosine_similarity_search ] ──▶ (n,) scores ──▶ [ top_k_indices ]
  (n, d) corpus┘
```
