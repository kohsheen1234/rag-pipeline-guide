# Step 43 · `recall_at_k`

> **Part 7 · Evaluation** — step 43 of 51
> Code: [`rag_pipeline/evaluation.py`](../../rag_pipeline/evaluation.py) · Tests: [`tests/test_evaluation.py`](../../tests/test_evaluation.py)
> Previous: [Step 42 · `hit_rate_at_k`](step-42-hit-rate-at-k.md) · Next: [Step 44 · `mean_reciprocal_rank`](step-44-mean-reciprocal-rank.md)

---

## The task

```python
def recall_at_k(retrieved: list, relevant: list, k: int) -> float: ...
```

For each query, compute the fraction of gold ids appearing in the top k, then
return the mean across queries. A query with no gold ids scores `0.0`.

---

## Why this step exists

[Hit rate](step-42-hit-rate-at-k.md) asks whether *anything* useful was found.
Recall asks **how much** of it was.

The difference matters for any question needing more than one passage to answer.
If a question is supported by four chunks and the retriever surfaces one, hit
rate says 1.0 — perfect — and the generator still cannot answer, because it has
a quarter of the evidence. Recall says 0.25 and tells you where the problem is.

For a RAG pipeline, recall@k is usually the number to optimise. Precision
matters less than it does in web search: a few irrelevant chunks in the prompt
cost tokens and some distraction, while a missing chunk costs the answer.

---

## What's happening

```python
if not retrieved:
    return 0.0

total = 0.0
for retrieved_ids, relevant_ids in zip(retrieved, relevant):
    if relevant_ids:
        found = set(retrieved_ids[:k]) & set(relevant_ids)
        total += len(found) / len(relevant_ids)

return total / len(retrieved)
```

**`len(found) / len(relevant_ids)`** — divided by the number of gold ids. This
is the definition, and the guide's headline pitfall: dividing by `k` instead
gives **precision@k**, a different metric that answers a different question.
Both are fractions in `[0, 1]`, both look plausible, and the names are not
interchangeable.

A test pins it: one gold id found in a top-3 list scores 1.0 for recall, where
precision would say 1/3.

**Macro-averaged.** Each query's recall is computed first, then averaged, so
every query counts equally regardless of how many gold chunks it has. The
alternative — pooling all intersections and dividing by all gold ids — would let
one heavily-labelled question dominate.

**`if relevant_ids:`** skips the division for a query with no gold ids, but the
query still counts in the denominator via `len(retrieved)`. So it contributes
`0.0`, exactly as specified. Dropping it from the denominator entirely would
change the metric depending on how many unlabelled queries the set happened to
contain, which makes runs incomparable.

**`[:k]`** truncates before intersecting, as in hit rate.

---

## The ceiling

Worth internalising: recall@k is bounded above by `min(1, k / |R|)`.

A question with 5 gold chunks evaluated at `k=3` cannot score above 0.6, no
matter how good the retriever is. There are only three slots.

So a low recall number is not automatically a retrieval failure — it can be an
evaluation design failure. Check `k` against the gold-set sizes before
concluding anything. If most questions have more gold chunks than `k`, you are
measuring your choice of `k`, not your retriever.

---

## Boundaries of the contract

**Order is ignored.** Once inside the top k, position does not matter. A gold
chunk at rank 1 and at rank `k` score identically. That is what
[MRR](step-44-mean-reciprocal-rank.md) is for.

**Precision is not computed**, so a retriever returning everything scores
perfect recall. Recall alone can always be gamed by raising `k`; report `k`
alongside it, and pin `k` to what the prompt actually receives.

**Unlabelled queries drag the mean down.** They score 0 and stay in the
denominator. Intentional, and it means adding an unlabelled question lowers your
score.

**Duplicates collapse.** The same gold id retrieved twice counts once — right,
since the set intersection is over unique ids.

**Parallel lists, unchecked.** `zip` truncates on mismatch.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Dividing by `k` | That is precision@k. Same range, different meaning. |
| Skipping unlabelled queries entirely | Changes the denominator and makes runs incomparable. |
| Micro-averaging by pooling | One heavily-labelled question dominates the score. |
| Comparing recall at different `k` | Monotone in `k`; the comparison is meaningless without it. |
| Ignoring the `min(1, k/|R|)` ceiling | You conclude the retriever is bad when `k` is too small. |
| Reading it as answer quality | It measures retrieval only. |

---

## Example

```python
>>> retrieved = [['a', 'b', 'c'], ['x', 'y', 'z']]
>>> relevant = [['a', 'd'], ['y']]
>>> recall_at_k(retrieved, relevant, k=2)
0.75
```

First query: top-2 is `{a, b}`, gold is `{a, d}`, so 1 of 2 = 0.5. Second: top-2
is `{x, y}`, gold is `{y}`, so 1 of 1 = 1.0. Mean 0.75.

---

## Where it fits

```
  retrieved ──┐
              ├──▶ [ recall_at_k ] ──▶ "how much of the evidence did we get?"
  gold      ──┘
```

The number to optimise for RAG, read next to hit rate and MRR.
