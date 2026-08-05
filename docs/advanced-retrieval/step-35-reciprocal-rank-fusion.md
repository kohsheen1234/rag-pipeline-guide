# Step 35 · `reciprocal_rank_fusion`

> **Part 6 · Advanced Retrieval Techniques** — step 35 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 34 · `hyde_retrieve`](step-34-hyde-retrieve.md) · Next: [Step 36 · `bm25_search`](step-36-bm25-search.md)

---

## The task

```python
def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list: ...
```

Fuse several ranked lists of chunk ids into one ranking. Return `(id, score)`
pairs sorted from highest fused score to lowest. The constant `k` damps
low-rank contributions.

---

## Why this step exists

You have a dense ranking, a BM25 ranking, and maybe a HyDE ranking. Combining
them means comparing them, and their scores are not comparable: cosine sits in
roughly `[-1, 1]`, BM25 is unbounded and corpus-dependent, and a cross-encoder
emits arbitrary logits. Min-max scaling them onto a common range —
what [`hybrid_search`](step-37-hybrid-search.md) does — works, but it is
sensitive to outliers and to how many candidates each system returned.

RRF avoids the problem by throwing the scores away. Only *position* is used. A
system that ranks a chunk first contributes the same amount whether it was
99% confident or barely preferred it, so no calibration is needed and no system
can dominate by having a wider numeric range.

That is why it is the default fusion method in most hybrid search stacks: it has
one parameter and no assumptions.

---

## What's happening

```python
scores = {}

for ranked_list in ranked_lists:
    for rank, identifier in enumerate(ranked_list, start=1):
        scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)

return sorted(scores.items(), key=lambda item: -item[1])
```

The formula is `score(d) = Σ_L 1 / (k + rank_L(d))`.

**`start=1`.** The guide's pitfall, and it is not merely cosmetic: with 0-based
ranks the first position contributes `1/k` instead of `1/(k+1)`, which changes
every score in the output. Worse, if `k` were ever 0 the first item would divide
by zero.

**Summed, not maxed.** Appearing in several lists is the signal RRF is built to
reward. An id ranked 1st in one list and 2nd in another scores
`1/61 + 1/62 ≈ 0.0325`, comfortably above an id ranked 1st in one list alone
(`1/61 ≈ 0.0164`). Taking the max would make agreement between systems worth
nothing.

**`.get(identifier, 0.0)`** lets ids appear in any subset of the lists. Nothing
needs to be present everywhere.

**`key=lambda item: -item[1]`** sorts descending on score. Python's sort is
stable, so ids with equal fused scores stay in first-seen order.

---

## What `k` does

`k` flattens the curve. The contribution of rank *r* is `1/(k+r)`, so the ratio
between first and second place is `(k+2)/(k+1)`:

| `k` | 1st vs 2nd | Effect |
| --- | --- | --- |
| 1 | 1.50× | Top position dominates; deep ranks are noise |
| 10 | 1.09× | Still top-weighted |
| 60 | 1.02× | Near-flat; broad agreement across lists wins |
| 1000 | 1.001× | Essentially counts appearances |

At `k = 60` — the value from the original paper, and the near-universal default
— the difference between rank 1 and rank 2 is about 2%, so a chunk that several
systems rank *reasonably* beats one that a single system ranks *first*. That is
usually what you want from fusion: consensus over confidence.

---

## Boundaries of the contract

**Ids in, ids out.** No chunk dicts. You need a lookup to resolve them, and the
ids must be consistent across lists — a dense retriever returning row indices
and a BM25 returning `chunk_id` strings will never agree on anything.

**Absolute scores are meaningless.** They depend only on positions and the
number of lists. Do not threshold on them.

**Order-insensitive across lists.** No list is weighted more than another. If
you trust your cross-encoder more than BM25, RRF gives you no way to say so
short of passing its list twice.

**Position is all it sees.** A system that ranks a chunk first with a score of
0.01 contributes exactly as much as one that ranks it first with 0.99. Usually a
feature; occasionally it discards real information.

**No truncation.** Every id from every list appears in the output. Slice the
result yourself.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| 0-based ranks | Every score changes, and `k=0` divides by zero. |
| Taking the max instead of the sum | Cross-list agreement stops counting, which is the whole point. |
| Fusing lists with different id types | Row indices and string ids never intersect; the fusion silently does nothing. |
| Thresholding on the fused score | It is a function of position and list count, not relevance. |
| Tuning `k` per corpus | It barely matters. 60 is fine; spend the effort elsewhere. |
| Feeding it unranked sets | Order *is* the input. A set has none. |

---

## Example

```python
>>> reciprocal_rank_fusion([['a', 'b', 'c']], k=60)
[('a', 0.01639344262295082), ('b', 0.016129032258064516), ('c', 0.015873015873015872)]
>>> reciprocal_rank_fusion([['a', 'b'], ['c', 'a']], k=60)
[('a', 0.03252247488101534), ('c', 0.01639344262295082), ('b', 0.016129032258064516)]
```

In the second, `a` is ranked 1st then 2nd and wins on the sum; `c` is ranked
1st once and beats `b` at 2nd once.

> The guide quotes `0.032522474916387955` for `a`. In IEEE double,
> `1/61 + 1/62` is `0.03252247488101534` — a difference of 3.5e-11, consistent
> with the reference having been accumulated in float32. The ranking is
> identical either way.

---

## Where it fits

```
  dense ranking ──┐
  bm25 ranking  ──┼──▶ [ reciprocal_rank_fusion ] ──▶ [(id, score), ...]
  hyde ranking  ──┘                                          │
                                                             ▼
                                                  resolve ids ──▶ rerank / prompt
```
