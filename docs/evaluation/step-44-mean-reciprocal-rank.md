# Step 44 · `mean_reciprocal_rank`

> **Part 7 · Evaluation** — step 44 of 51
> Code: [`rag_pipeline/evaluation.py`](../../rag_pipeline/evaluation.py) · Tests: [`tests/test_evaluation.py`](../../tests/test_evaluation.py)
> Previous: [Step 43 · `recall_at_k`](step-43-recall-at-k.md) · Next: [Step 45 · `faithfulness_score`](step-45-faithfulness-score.md)

---

## The task

```python
def mean_reciprocal_rank(retrieved: list, relevant: list) -> float: ...
```

For each query, find the 1-indexed position of the first retrieved id that is
gold, and average `1 / position` across queries. A query with no gold id in its
list contributes 0.

---

## Why this step exists

Hit rate and recall both treat the top k as a bag: once a chunk is in, where it
sits does not matter. But position matters twice over.

**The prompt is ordered.** [`format_context`](../generation/step-25-format-context.md)
renders chunks in rank order, and models weight earlier and later context
unevenly. A gold chunk at position 9 of 10 is competing for attention with eight
irrelevant passages.

**The budget is finite.** If you retrieve 20 and can only afford to show 5, a
gold chunk at rank 9 is not in the prompt at all — even though recall@20 counted
it.

MRR is the metric that notices the difference between "found it first" and
"found it eventually".

---

## What's happening

```python
if not retrieved:
    return 0.0

total = 0.0
for retrieved_ids, relevant_ids in zip(retrieved, relevant):
    relevant_set = set(relevant_ids)
    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant_set:
            total += 1.0 / rank
            break

return total / len(retrieved)
```

**`start=1`.** Ranks are 1-indexed, and here it is not a convention but a
necessity: a 0-indexed first position would compute `1/0`. The guide names it,
and unlike most off-by-ones this one raises rather than skewing quietly.

**`break` on the first hit.** Only the first gold chunk counts. A query with
gold at positions 1 and 5 scores `1/1 = 1.0`, identical to a query with gold
only at position 1. That is the definition — MRR asks how fast you found
*something*, not how much you found. Coverage is [recall](step-43-recall-at-k.md)'s
job.

**The set is built once per query**, outside the rank loop, so membership is
constant-time as the list is scanned.

**No `k` parameter.** MRR scans the whole retrieved list. In practice you pass
lists already truncated to your serving `k`, which effectively makes it MRR@k —
worth being deliberate about, since a longer list can only raise the score.

**No hit contributes 0**, which falls out of the loop completing without a
`break`.

---

## Reading the number

The reciprocal scale is steep, and that is deliberate:

| First hit at rank | Contribution |
| --- | --- |
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| 10 | 0.100 |

Slipping from rank 1 to rank 2 costs half the score. Slipping from 9 to 10 costs
almost nothing. That matches how retrieval quality is actually experienced —
the difference between first and second place is large, the difference between
ninth and tenth is not.

It also means MRR is dominated by the top few positions, so it is the right
metric when you show the model a handful of chunks and the wrong one when you
care about deep coverage.

An MRR of 0.5 does *not* mean "half correct". It means the average first hit sat
around position 2.

---

## Boundaries of the contract

**Only the first hit.** A retriever putting one gold chunk first and burying
three others scores 1.0. Read alongside recall.

**Unbounded list length.** No `k`; truncate before calling if you want MRR@k.

**Zero for a miss** is harsh but standard — there is no rank, so there is no
reciprocal.

**Unlabelled queries score 0** and stay in the denominator, consistent with
[recall](step-43-recall-at-k.md).

**Ties are invisible.** If two chunks score identically and the tie-break puts
the gold one second, MRR halves. That is one reason
[`top_k_indices`](../retrieval/step-17-top-k-indices.md) breaks ties
deterministically: otherwise this metric fluctuates between runs for no reason.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| 0-indexed ranks | `ZeroDivisionError` on a first-position hit. |
| Summing every hit instead of the first | Scores exceed 1.0; no longer MRR. |
| Forgetting to `break` | Same problem, quietly. |
| Comparing MRR across different list lengths | A longer list can only help. |
| Reading 0.5 as "half right" | It means the first hit averaged position 2. |
| Using MRR alone | Blind to everything after the first hit. |

---

## Example

```python
>>> retrieved = [['a', 'b', 'c'], ['x', 'y', 'z']]
>>> relevant  = [['b'], ['z']]
>>> round(mean_reciprocal_rank(retrieved, relevant), 4)
0.4167
```

First hit at rank 2 → 0.5. Second at rank 3 → 0.333. Mean 0.4167.

---

## Where it fits

```
  retrieved ──┐
              ├──▶ [ mean_reciprocal_rank ] ──▶ "how high was the first good one?"
  gold      ──┘
```

The third of the three retrieval metrics. Hit rate says whether, recall says how
much, MRR says how high.
