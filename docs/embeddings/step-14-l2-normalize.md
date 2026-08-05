# Step 14 · `l2_normalize`

> **Part 3 · Embeddings & Corpus Storage** — step 14 of 51
> Code: [`rag_pipeline/embeddings.py`](../../rag_pipeline/embeddings.py) · Tests: [`tests/test_embeddings.py`](../../tests/test_embeddings.py)
> Previous: [Step 13 · `embed_chunks`](step-13-embed-chunks.md) · Next: [Step 15 · `save_corpus`](step-15-save-corpus.md)

---

## The task

```python
def l2_normalize(matrix: np.ndarray) -> np.ndarray: ...
```

Rescale each row of an `(n, d)` embedding matrix to unit L2 norm, so dot
products between rows equal cosine similarities. Rows that are all zeros must
be left unchanged. Return a new array of the same shape and float dtype.

---

## Why this step exists

Dense retrieval ranks by cosine similarity:

```
cos(x, y) = (x · y) / (‖x‖ ‖y‖)
```

Computing that directly for every query against every chunk means recomputing
norms on every search. But if every vector already has ‖x‖ = 1, the denominator
is 1 and the cosine *is* the dot product:

```
cos(x, y) = x · y
```

So a single matrix multiply — `corpus @ query` — produces every similarity score
at once, at BLAS speed. Normalising once at index time converts a per-query
division into nothing at all.

It is also what lets FAISS work later. FAISS offers an inner-product index
(`IndexFlatIP`) but no cosine index, because with normalised vectors it does not
need one. Feed it unnormalised vectors and it happily returns inner products
that are *not* cosines, ranking long vectors above relevant ones with no error
to warn you.

---

## What's happening

```python
norms = np.linalg.norm(matrix, axis=1, keepdims=True)

return matrix / np.where(norms == 0, 1, norms)
```

Two lines, and both details in them matter.

### `axis=1, keepdims=True`

`axis=1` reduces along the feature axis, giving one norm per row.

`keepdims=True` is the part that is easy to get wrong. Without it the result has
shape `(n,)`, and numpy broadcasting aligns shapes from the **right**:

```python
matrix.shape          (n, d)
norms.shape             (n,)   ->  broadcast as (1, n)
```

That divides along the *feature* axis rather than the row axis. If `n == d` the
shapes are compatible and you get a silently wrong matrix with no error at all;
otherwise you get a confusing broadcast failure. `keepdims=True` produces
`(n, 1)`, which broadcasts across columns — one norm applied to every feature of
its own row, which is what "normalise each row" means.

This is the guide's pitfall, and it is worth restating that on a square matrix
it does not raise. It just returns garbage.

### `np.where(norms == 0, 1, norms)`

Dividing by a zero norm gives `nan`, and a single NaN poisons everything
downstream — comparisons against NaN are always false, so a NaN row silently
never matches anything, and some sorting paths behave arbitrarily.

Replacing the divisor with 1 for zero rows leaves those rows as `0/1 = 0`. The
zero row stays zero, which is the specified behaviour and the honest one: a zero
vector has no direction, so there is no unit vector to map it to.

**Why not an epsilon?** `matrix / (norms + 1e-12)` is the common shortcut. It
avoids the NaN, but it changes *every* row by a tiny amount, so no row is
exactly unit length. Usually harmless, occasionally not — and `np.where` costs
nothing and is exact.

Zero rows are not hypothetical: an empty chunk, a padding row, or a chunk of
pure whitespace can all produce one.

### Returning a new array

`matrix / ...` allocates rather than mutating in place. The caller's array is
untouched, which matters because the unnormalised vectors are sometimes still
wanted — and because in-place division would silently fail on an integer array.

The dtype follows numpy's rules: float32 in gives float32 out, so a matrix
destined for FAISS keeps its dtype. An integer array is promoted to float64,
since there is no sensible integer answer.

---

## Boundaries of the contract

**Rows, not columns.** This normalises each embedding, not each feature.
Feature-wise scaling is a different operation with a different purpose.

**Apply it to queries too.** Normalising only the corpus means
`corpus @ query` returns `cos(x, q) · ‖q‖` — every score scaled by the same
constant. The *ranking* is unaffected, so this bug is invisible in a ranked
list; it only shows up when someone compares a score against a threshold, or
compares scores across queries. Both then behave inexplicably.

**Idempotent.** Normalising twice changes nothing, so it is safe to apply
defensively.

**2D only.** A 1D query vector has no `axis=1`. Either reshape it, or normalise
it directly with `v / np.linalg.norm(v)`.

**No `inf` or `nan` handling.** A row containing infinities produces a NaN norm
and NaN output. Only the all-zero case is guarded.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Omitting `keepdims=True` | Divides along the wrong axis. On a square matrix it does not even raise. |
| Dividing without a zero guard | One empty chunk puts NaNs in the matrix, and NaN rows silently never match. |
| Using `+ 1e-12` instead of `np.where` | No row is exactly unit length; correct in practice, imprecise in principle. |
| Normalising the corpus but not the query | Ranking still works, so it looks fine — until a score threshold is involved. |
| Normalising before saving but re-embedding without it | Half the index normalised, half not. Unnormalised vectors dominate the rankings. |
| Assuming zero rows are impossible | Empty and whitespace-only chunks reach here. |
| Using inner product in FAISS on unnormalised vectors | Returns inner products, not cosines. Longer vectors win regardless of relevance. |

---

## Example

```python
>>> M = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
>>> out = l2_normalize(M)
>>> out.tolist()
[[0.6, 0.8], [1.0, 0.0], [0.0, 0.0]]
>>> np.round(np.linalg.norm(out, axis=1), 4).tolist()
[1.0, 1.0, 0.0]
```

The third norm is `0.0`, not `1.0` — the zero row is unchanged, exactly as
specified.

---

## Where it fits

```
  (N, d) corpus ──▶ [ l2_normalize ] ──┐
                                       ├──▶  corpus @ query  ==  cosine scores
  (d,)   query  ──▶ [ normalise    ] ──┘
```

This is the step that turns similarity search into a matrix multiply. Everything
in the retrieval part assumes it has been applied to both sides.
