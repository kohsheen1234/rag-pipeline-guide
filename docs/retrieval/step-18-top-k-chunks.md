# Step 18 · `top_k_chunks`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 18 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 17 · `top_k_indices`](step-17-top-k-indices.md) · Next: [Step 19 · `retrieve`](step-19-retrieve.md)

---

## The task

```python
def top_k_chunks(scores: np.ndarray, chunks: list, k: int) -> list: ...
```

Given scores, the parallel list of chunks, and `k`, return `(chunk, score)`
tuples sorted by descending score, truncated to `min(k, len(chunks))`. Reuse
[`top_k_indices`](step-17-top-k-indices.md). Each score should be a plain
Python `float`.

---

## Why this step exists

Consumers need both halves. The chunk text goes into the prompt; the score is
used to threshold, to display, to rerank, or to decide to
[abstain](../robustness/step-47-handle-no-context.md). Returning only indices
pushes that join onto every caller, and the join is exactly where the bug is.

---

## What's happening

```python
indices = top_k_indices(scores, min(k, len(chunks)))

return [(chunks[index], float(scores[index])) for index in indices]
```

### The alignment trap

The guide names it and it is worth being precise. The failure mode is:

```python
ranked = np.sort(scores)[::-1][:k]          # scores, now sorted
indices = top_k_indices(scores, k)           # positions in the ORIGINAL order
list(zip([chunks[i] for i in indices], ranked))
```

Every chunk is now paired with a score that belongs to a different chunk. The
*ranking* is still correct, and the scores are still descending, so the output
looks entirely plausible. What is wrong is the association, and nothing about
the shape of the result reveals it.

The fix is the one-line discipline in the implementation: pull both from the
same `index`, in the same expression. There is no intermediate sorted-scores
array to get out of step with anything.

### `float(...)`

`scores[index]` is a `np.float32`, which is not JSON-serialisable —
`json.dumps` raises `TypeError: Object of type float32 is not JSON
serializable`. Since retrieval results routinely get logged or returned from an
API, casting here saves the caller from discovering that at the boundary. Same
class of problem as
[numpy scalars in chunk dicts](../embeddings/step-15-save-corpus.md#boundaries-of-the-contract).

### The double clamp

`min(k, len(chunks))` before calling `top_k_indices`, which clamps again
against `len(scores)`. Redundant when the two lists match, and deliberate: if
they ever *don't* match, this clamps to the shorter one instead of raising
`IndexError` on a chunk lookup.

---

## Boundaries of the contract

**`scores` and `chunks` must be parallel.** Nothing checks it. A mismatch is
either an `IndexError` or, if `chunks` is longer, silently wrong pairings.

**Chunks are returned by reference.** Mutating a returned chunk mutates the
corpus entry.

**Chunks can be anything.** Dicts in practice, but strings work; nothing is read
from them here.

**No thresholding.** Low scores are returned like any other. Filtering is
[step 47](../robustness/step-47-handle-no-context.md)'s job.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Indexing into an already-sorted score array | Right order, wrong scores attached. Looks correct. |
| Returning numpy scalars | `json.dumps` raises at the API boundary. |
| Returning chunks without scores | The caller cannot threshold or display confidence. |
| Sorting `(score, chunk)` tuples directly | On tied scores Python compares the chunks, and dicts are not orderable — `TypeError`. |
| Clamping only `k` and not the chunk count | `IndexError` when the lists disagree. |

---

## Example

```python
>>> scores = np.array([0.1, 0.9, 0.5, 0.7])
>>> chunks = [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}]
>>> top_k_chunks(scores, chunks, 2)
[({'id': 1}, 0.9), ({'id': 3}, 0.7)]
```

---

## Where it fits

```
  (n,) scores ──┐
                ├──▶ [ top_k_chunks ] ──▶ [(chunk, score), ...] ──▶ prompt / rerank / abstain
  chunks list ──┘
```
