# Step 17 · `top_k_indices`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 17 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 16 · `cosine_similarity_search`](step-16-cosine-similarity-search.md) · Next: [Step 18 · `top_k_chunks`](step-18-top-k-chunks.md)

---

## The task

```python
def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray: ...
```

Return the indices of the `k` largest scores in descending order. If `k`
exceeds the number of scores, return all indices ranked by score.

---

## Why this step exists

Ranking is separated from scoring so both the numpy and FAISS paths can share
it, and so the sort can be tested without constructing embeddings. It is four
lines that are easy to get subtly wrong in three different ways.

---

## What's happening

```python
order = np.argsort(-scores, kind="stable")

return order[: min(k, len(scores))]
```

**`-scores` for descending.** `argsort` is ascending only. Negating the scores
and sorting ascending gives descending order of the originals. This is the
classic bug the guide warns about: forget the minus and you return the *worst*
matches, ranked confidently.

**`kind="stable"`** makes ties break by index. Without it numpy uses an
introsort whose tie order is unspecified, so equal scores could come back in
different orders on different runs or numpy versions. That matters more than it
sounds: [step 22](step-22-compare-faiss-to-numpy.md) compares this ranking
against FAISS, and non-deterministic tie order would make that check flaky for
no real reason.

**`min(k, len(scores))`** clamps. Slicing past the end of a numpy array is
already safe, so this is belt-and-braces, but it makes the contract explicit.

---

## Full sort vs partial selection

`argsort` is `O(n log n)` and we only need the top `k`. `np.argpartition` finds
them in `O(n)` and you sort just those:

```python
top = np.argpartition(-scores, k - 1)[:k]
return top[np.argsort(-scores[top])]
```

That is genuinely faster on a large corpus. It is not used here because it is
fiddlier at the edges — `k == 0` and `k >= n` both need special-casing, and
`argpartition` gives no stability guarantee, so ties would need handling
separately. For a corpus small enough that numpy retrieval is the right tool at
all, the full sort is not the bottleneck; for one large enough that it is, you
should be on the [FAISS path](step-20-build-faiss-index.md) anyway.

---

## Boundaries of the contract

**Returns indices, not scores.** The caller looks both up by the same index,
which is what keeps them aligned. See
[step 18](step-18-top-k-chunks.md#the-alignment-trap).

**`k = 0` returns an empty array**, not an error.

**Ties go to the lower index**, i.e. the earlier chunk. Arbitrary, but fixed.

**NaN scores sort unpredictably.** `argsort` places NaN at the end when
ascending, so negated NaN also lands at the end — but do not rely on it. Better
to not produce NaNs, which is why
[step 16](step-16-cosine-similarity-search.md) guards its division.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `np.argsort(scores)[:k]` | Returns the **worst** k. The output looks structurally right, so it passes eyeballing. |
| `np.argsort(scores)[::-1][:k]` | Descending, but reverses tie order too, so ties come back highest-index-first. |
| Forgetting to clamp `k` | Harmless with numpy slicing, an `IndexError` with a Python list. |
| Default sort kind | Unspecified tie order, which makes backend comparisons flaky. |
| Sorting the scores instead of the indices | Loses the mapping back to chunks entirely. |

---

## Example

```python
>>> scores = np.array([0.1, 0.9, 0.4, 0.7, 0.2])
>>> top_k_indices(scores, 3).tolist()
[1, 3, 2]
>>> top_k_indices(np.array([0.5, 0.5]), 5).tolist()
[0, 1]
```

The second case shows both the clamp and the tie-break: `k=5` on two scores
returns two, and the equal scores stay in index order.

---

## Where it fits

```
  (n,) scores ──▶ [ top_k_indices ] ──▶ (k,) indices ──▶ [ top_k_chunks ]
```
