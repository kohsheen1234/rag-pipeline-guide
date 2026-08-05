# Step 40 · `filter_by_metadata`

> **Part 6 · Advanced Retrieval Techniques** — step 40 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 39 · `maximal_marginal_relevance`](step-39-maximal-marginal-relevance.md)

---

## The task

```python
def filter_by_metadata(chunks: list, filter_dict: dict) -> list: ...
```

Keep only chunks whose `metadata` sub-dict matches every key/value pair in
`filter_dict`. A chunk passes when, for every key in the filter, that key exists
in its metadata with exactly that value. An empty filter keeps everything.
Preserve input order.

---

## Why this step exists

Two quite different reasons, and the second is the serious one.

**Precision.** Restricting search to one product manual, one language, or one
date range removes whole categories of wrong answer before ranking starts. It is
usually the cheapest quality improvement available, because it does not depend
on the embedding model being good.

**Access control.** In a multi-tenant system, retrieval reaching another
customer's documents is a data breach, not a relevance problem. This is the
function standing between the two — which is why "a missing key is not a match"
matters far more than it looks.

---

## What's happening

```python
return [
    chunk
    for chunk in chunks
    if all(
        key in chunk.get("metadata", {}) and chunk["metadata"][key] == value
        for key, value in filter_dict.items()
    )
]
```

**`all(...)` is conjunctive** — every condition must hold. An empty filter makes
`all()` vacuously true, so everything passes, which is the specified behaviour
and the right default for "no filter applied".

**`key in ... and ... == value`, not `.get(key) == value`.** These differ in one
case that matters: a chunk whose metadata has the key set to `None`, filtered
against `None`. With `.get`, a chunk *missing* the key also returns `None` and
would match. The explicit membership test keeps "absent" and "present but None"
distinct.

**`chunk.get("metadata", {})`** treats a chunk with no metadata as having empty
metadata, so it fails any non-empty filter rather than raising `KeyError`. The
guide is explicit that a missing key must exclude, not silently keep — for the
access-control reason above, defaulting to "include when unsure" is the
dangerous direction.

**A new list of the same objects.** Order is preserved and the chunks are not
copied, so the result is cheap and downstream code sees the same dicts. If a
matrix is being filtered alongside, the same predicate has to select the same
rows, or the
[row-to-chunk alignment](../embeddings/step-13-embed-chunks.md) breaks.

---

## Filtering and the embedding matrix

The awkward part in practice. This returns chunks; retrieval needs a matrix
whose rows line up with them. Filtering the chunk list alone leaves the matrix
unchanged and the correspondence broken.

You need the indices, not just the chunks:

```python
keep = [i for i, c in enumerate(chunks) if passes(c)]
filtered_chunks = [chunks[i] for i in keep]
filtered_matrix = embeddings[keep]
```

which is the same pattern
[`deduplicate_chunks`](../robustness/step-48-deduplicate-chunks.md) uses, and
why that function returns both halves. Filtering after retrieval avoids the
problem but wastes the ranking — you can ask for `k` and receive fewer.

---

## Boundaries of the contract

**Exact equality only.** No ranges, no `in`, no substring, no case-insensitive
comparison, no `OR` across values of one key. `{'year': 2024}` cannot express
"2020 or later".

**Conjunctive only.** Every key must match; there is no way to express "source
is x *or* lang is en".

**Reads `metadata`, not the top level.** Chunks from
[`attach_chunk_metadata`](../chunking/step-10-attach-chunk-metadata.md) put
`source` and `position` at the *top* level, with no `metadata` sub-dict at all —
so every one of them fails any non-empty filter here. Another instance of the
[schema drift](../generation/step-31-track-source-chunk-ids.md#the-key-name-does-not-match-the-pipeline)
between guide steps; wrap or remap before combining them.

**Linear scan.** Fine for thousands, wrong for millions — a real vector store
pushes the predicate into the index so it never scores excluded vectors.

**Type-sensitive.** `'2024'` does not match `2024`.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Treating a missing key as a match | Chunks leak past the filter. In a tenant filter, that is a breach. |
| `.get(key) == value` | Conflates "absent" with "present and None". |
| Filtering chunks without the matrix rows | Row *i* no longer means chunk *i*. |
| Mutating or copying the chunks | Downstream identity checks and updates break. |
| Expecting range or OR semantics | Exact conjunctive equality only. |
| Assuming step 10's chunks work here | They have no `metadata` sub-dict. |

---

## Example

```python
>>> chunks = [
...   {'text': 'a', 'metadata': {'source': 'x', 'lang': 'en'}},
...   {'text': 'b', 'metadata': {'source': 'y', 'lang': 'en'}},
...   {'text': 'c', 'metadata': {'source': 'x', 'lang': 'fr'}},
... ]
>>> [c['text'] for c in filter_by_metadata(chunks, {'source': 'x'})]
['a', 'c']
>>> [c['text'] for c in filter_by_metadata(chunks, {'source': 'x', 'lang': 'en'})]
['a']
>>> filter_by_metadata(chunks, {'year': 2024})
[]
```

---

## Where it fits

```
  corpus ──▶ [ filter_by_metadata ] ──▶ subset ──▶ retrieval ──▶ rerank ──▶ MMR
                      │
                      └── narrow first: cheaper, and the only correct place
                          for a tenant or permission filter
```

This closes Part 6. [Part 7](../evaluation/00-overview.md) is how you find out
which of these eight actually helped.
