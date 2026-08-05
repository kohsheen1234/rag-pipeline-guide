# Step 22 · `compare_faiss_to_numpy`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 22 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 21 · `faiss_search`](step-21-faiss-search.md) · Next: [Step 23 · `save_faiss_index`](step-23-save-faiss-index.md)

---

## The task

```python
def compare_faiss_to_numpy(query, matrix, index, k) -> bool: ...
```

Verify that FAISS and the numpy cosine search agree on the top-k indices for a
query. Return `True` if and only if both select the same **set** of indices.

---

## Why this step exists

Two backends over one corpus is two chances to build it wrong, and the ways it
goes wrong are all silent:

- the index was built from an unnormalised matrix, so it ranks by magnitude
- the index was built from a stale matrix and the chunk list has moved on
- the query was normalised on one path and not the other
- vectors were added in a different order than the chunk list

None of these raise. All of them produce a ranked list that looks fine. The only
cheap way to notice is to run both paths and check they agree — which works
precisely because `IndexFlatIP` is
[exact](step-20-build-faiss-index.md#why-this-step-exists), not approximate. If
this returns `False`, something is genuinely broken, not merely imprecise.

---

## What's happening

```python
scores = cosine_similarity_search(query, matrix)
numpy_indices = top_k_indices(scores, k)
_, faiss_indices = faiss_search(index, query, k)

return set(numpy_indices.tolist()) == set(faiss_indices.tolist())
```

Both paths are run through the same helpers the pipeline uses in production, so
the check exercises the real code rather than a reimplementation of it.

### Why sets and not lists

The guide is right that position-by-position equality is the wrong comparison,
and the reason is worth spelling out. Two chunks with genuinely equal scores may
be ordered either way, and the two backends break ties differently:
[`top_k_indices`](step-17-top-k-indices.md) sorts stably by index, FAISS orders
by whatever its scan produces.

Float precision compounds it. The numpy path computes in float64 and divides by
norms; FAISS computes in float32 and does not. Two scores that are
mathematically identical can differ in the last bit, flipping their order
without changing which chunks were selected.

Set comparison asks the question that actually matters: *did both backends pick
the same k chunks?* Ordering within a tied group is not a retrieval difference.

The cost is that a genuine ordering bug inside the top-k is invisible here. If
you care about that, sort both by `(-score, index)` and compare — but expect
occasional false alarms from float noise.

---

## Boundaries of the contract

**One query.** Real confidence needs a sweep over many queries and several `k`
values. The tests do this; a single call does not.

**Requires a normalised matrix.** The numpy path normalises internally, FAISS
does not — so on an *unnormalised* corpus the two legitimately disagree and this
returns `False`. That is arguably the most useful thing it catches, but it means
a `False` is not always an index bug.

**Boolean only.** No indication of *which* chunks differed. Fine as a build-time
assertion, thin as a debugging tool.

**Set equality hides ordering**, as above.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Comparing lists element-wise | Tie reorderings and float32-vs-float64 noise flag harmless differences as failures. |
| Comparing scores instead of indices | Different precision on each path; they will rarely be bit-equal. |
| Using it on an approximate index | `IVF`/`HNSW` are meant to disagree sometimes. This check only makes sense for a flat index. |
| Running it once and calling it verified | One query and one `k` is a smoke test. |
| Assuming `False` means FAISS is wrong | An unnormalised corpus fails this too, and that is a corpus bug. |

---

## Example

```python
>>> M = np.eye(4, dtype=np.float32)
>>> idx = faiss.IndexFlatIP(4); idx.add(M)
>>> q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
>>> compare_faiss_to_numpy(q, M, idx, k=2)
True
```

The tests also assert the negative: an index built over *different* vectors
returns `False`. A check that can only pass is not a check.

---

## Where it fits

```
            ┌──▶ [ cosine_similarity_search ] ──▶ [ top_k_indices ] ──┐
  query ────┤                                                         ├──▶ set == set?
            └──▶ [ faiss_search ] ─────────────────────────────────── ┘
```

Run it once after building an index, before serving anything from it.
