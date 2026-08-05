# Step 41 · `build_eval_set`

> **Part 7 · Evaluation** — step 41 of 51
> Code: [`rag_pipeline/evaluation.py`](../../rag_pipeline/evaluation.py) · Tests: [`tests/test_evaluation.py`](../../tests/test_evaluation.py)
> Previous: [Step 40 · `filter_by_metadata`](../advanced-retrieval/step-40-filter-by-metadata.md) · Next: [Step 42 · `hit_rate_at_k`](step-42-hit-rate-at-k.md)

---

## The task

```python
def build_eval_set() -> list: ...
```

Return a small hard-coded evaluation set. Each entry is a dict with the keys
`'question'`, `'answer'`, and `'relevant_ids'`, in that order, where
`relevant_ids` lists the chunk ids that count as gold-relevant. At least three
entries.

---

## Why this step exists

Every metric in this part is a comparison against something. This is the
something.

It is worth noticing that it is the only function in the whole pipeline that
takes no input and does no computation. That is not a placeholder — the ground
truth *is* the deliverable. Deciding which chunks should count as relevant for a
question is a judgement call that cannot be derived from the corpus, which is
exactly why it has to be written down and frozen.

---

## What's happening

```python
return [
    {
        "question": "What is RAG?",
        "answer": "Retrieval-augmented generation combines retrieval with a language model.",
        "relevant_ids": ["c1", "c2"],
    },
    ...
]
```

Three fields, each consumed by a different part of the evaluation:

- **`question`** goes to the retriever, and to
  [`relevance_score`](step-46-relevance-score.md).
- **`relevant_ids`** is the gold set for
  [hit rate](step-42-hit-rate-at-k.md),
  [recall](step-43-recall-at-k.md), and
  [MRR](step-44-mean-reciprocal-rank.md).
- **`answer`** is the reference answer. Notably, *nothing in Part 7 uses it* —
  the two answer metrics here are reference-free. It is there for a human
  reading the eval set, and for a judge-based metric you might add later.

**A list per question, not a single id.** More than one chunk can legitimately
support an answer, and recall is only meaningful when a question can have
several gold chunks — with exactly one, recall@k collapses into hit rate@k.

**Returned fresh each call.** The dicts are constructed inside the function, so
a caller that mutates a returned entry does not corrupt the next call. A
module-level constant would be shared, and an eval set that quietly changes
between runs is the worst possible bug in a measurement harness. There is a test
for this.

**Three entries minimum** so the metrics produce something other than 0 or 1.
With one question every score is binary.

---

## The ids have to be real

The guide's pitfall is the one that actually bites: gold ids that do not exist
in the corpus silently score zero. Every metric here is a set intersection
against the retrieved ids, and an intersection with ids that were never
retrievable is always empty. Recall reads 0.0, MRR reads 0.0, and nothing
distinguishes "the retriever is broken" from "the labels are wrong".

Two things follow.

**Match the id scheme.** These use simple `'c1'` strings. The pipeline's own
chunker emits `'doc1::0'` from
[`attach_chunk_metadata`](../chunking/step-10-attach-chunk-metadata.md). Mixing
the two gives you a permanent zero.

**Ids are not stable across chunking configurations.** This is the subtler
version and it is worth internalising:
[chunk ids are positional](../chunking/step-10-attach-chunk-metadata.md#why-sourceposition-and-not-a-hash),
so re-chunking with a different `chunk_size` makes `doc1::7` point at entirely
different text. Your labels now mark the wrong passages. The scores stay
plausible, which is what makes it dangerous — this is precisely the comparison
(fixed-size vs sentence chunking) that Part 7 exists to enable, and doing it
naively invalidates the labels it depends on.

Relabel per configuration, or define gold at the document level and map ids to
documents before scoring.

---

## Boundaries of the contract

**Three questions is a toy.** Enough to exercise the metrics, nowhere near
enough to distinguish two retrievers. Real evaluation wants tens to hundreds,
and small sets have wide error bars — a single question flipping moves a
3-question score by 33 points.

**Hard-coded, not loaded.** No JSON file, no path. Fine for a fixture, and the
first thing to change for real use.

**Binary relevance.** A chunk is gold or it is not. Graded relevance (perfect /
partial / unrelated) supports better metrics like nDCG.

**No corpus.** The set names ids without saying which corpus they index, so
nothing prevents evaluating against the wrong one.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Empty `relevant_ids` | Recall and MRR read 0 for that query with no indication why. |
| Ids not present in the corpus | Every metric zeroes out. Looks like a broken retriever. |
| Mixing id schemes | `'c1'` and `'doc1::0'` never intersect. |
| Reusing labels across chunk sizes | Ids point at different text; scores are meaningless but plausible. |
| A module-level constant | Callers can mutate the ground truth between runs. |
| Growing the eval set to chase a number | Scores stop being comparable to earlier runs. |

---

## Example

```python
>>> evals = build_eval_set()
>>> evals[0]['question']
'What is RAG?'
>>> evals[0]['relevant_ids']
['c1', 'c2']
>>> len(evals)
3
```

---

## Where it fits

```
  [ build_eval_set ]
         │
         ├── question ──▶ retriever ──▶ retrieved ids ──▶ hit rate / recall / MRR
         │                                                        ▲
         ├── relevant_ids ────────────────────────────────────────┘
         │
         └── answer ──▶ (unused here; for a judge-based metric)
```
