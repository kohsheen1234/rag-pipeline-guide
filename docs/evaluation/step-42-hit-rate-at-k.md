# Step 42 · `hit_rate_at_k`

> **Part 7 · Evaluation** — step 42 of 51
> Code: [`rag_pipeline/evaluation.py`](../../rag_pipeline/evaluation.py) · Tests: [`tests/test_evaluation.py`](../../tests/test_evaluation.py)
> Previous: [Step 41 · `build_eval_set`](step-41-build-eval-set.md) · Next: [Step 43 · `recall_at_k`](step-43-recall-at-k.md)

---

## The task

```python
def hit_rate_at_k(retrieved: list, relevant: list, k: int) -> float: ...
```

For each query, given a ranked list of retrieved ids and a list of gold ids,
return the fraction of queries where at least one gold id appears in the top k.
Empty input returns `0.0`.

---

## Why this step exists

The simplest useful question you can ask a retriever: **did anything relevant
make it in front of the reader?**

It is the metric to look at first because a low hit rate makes every other
number moot. If the right chunk is not in the top k, no reranking, no prompt
engineering, and no larger model recovers it — the evidence is simply not in the
context. Fix hit rate before tuning anything else.

It is also the metric that most closely mirrors the user's experience of failure:
they do not care whether three of four gold chunks were retrieved, they care
whether the answer was there at all.

---

## What's happening

```python
if not retrieved:
    return 0.0

hits = sum(
    1
    for retrieved_ids, relevant_ids in zip(retrieved, relevant)
    if set(retrieved_ids[:k]) & set(relevant_ids)
)

return hits / len(retrieved)
```

**Binary per query.** `set(...) & set(...)` produces an intersection; a
non-empty set is truthy, so the generator contributes exactly 1 whether one gold
chunk matched or five. That is the definition, and the guide's pitfall: counting
multiple hits within a query would make this a different, unnamed metric.

**Divided by the query count**, not by the number of gold ids. Every query
contributes equally to the mean regardless of how many gold chunks it has.

**`[:k]` before the intersection.** The truncation is the whole point of the
"@k" — the retrieved list is usually longer than `k`, and a gold chunk at
position 20 should not count as a hit at `k=5`. Slicing a Python list past its
end is safe, so a short list needs no special case.

**Sets, not list scanning.** `in` on a list is linear; the intersection is
effectively constant per element. Irrelevant at these sizes, right by habit.

**`if not retrieved`** guards the division. Zero queries returning 0.0 rather
than raising `ZeroDivisionError` matches the specification.

---

## Reading the number

Hit rate is bounded by `min(1, ...)` in the obvious way and rises monotonically
with `k` — a larger window can only add hits. That makes it easy to make look
good by increasing `k`, which is why the `k` has to be reported alongside it.
`hit_rate@50 = 0.95` on a system that shows the model 5 chunks is not a useful
claim.

Pick `k` to match what you actually put in the prompt.

What it cannot tell you: **how many** gold chunks you found
([recall](step-43-recall-at-k.md)) or **how high** the first one ranked
([MRR](step-44-mean-reciprocal-rank.md)). A hit at position 1 and a hit at
position 10 score identically here, and they are very different systems.

---

## Boundaries of the contract

**Parallel lists, unchecked.** `retrieved[i]` must correspond to `relevant[i]`.
`zip` silently truncates to the shorter one, so a length mismatch quietly
evaluates fewer queries and inflates nothing — it just measures a different set
than you think.

**No gold ids means no hit**, contributing 0. Consistent with
[recall](step-43-recall-at-k.md), and worth knowing since it drags the mean down
for a query that could never have scored.

**Duplicates are collapsed** by the set conversion, which does not matter for a
binary metric.

**Ids must be comparable.** `'c1'` never matches `1`. See
[step 41](step-41-build-eval-set.md#the-ids-have-to-be-real).

**No confidence interval.** On three queries the only possible scores are 0,
0.33, 0.67, and 1.0.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Counting each hit within a query | Turns a bounded [0,1] metric into an unbounded count. |
| Dividing by the number of gold ids | That is recall, not hit rate. |
| Forgetting `[:k]` | Scores the whole retrieved list, so "@k" means nothing. |
| Reporting hit rate without `k` | Trivially improved by raising `k`. |
| Mismatched list lengths | `zip` truncates silently. |
| Treating it as sufficient | Says nothing about rank or coverage. |

---

## Example

```python
>>> retrieved = [[1, 2, 3], [4, 5, 6]]
>>> relevant = [[3], [9]]
>>> hit_rate_at_k(retrieved, relevant, k=3)
0.5
```

First query hits (`3` is in the top 3), second misses. One of two.

---

## Where it fits

```
  retrieved id lists ──┐
                       ├──▶ [ hit_rate_at_k ] ──▶ "did anything useful get in?"
  gold id lists      ──┘
```

Check this first. If it is low, stop tuning the generator.
