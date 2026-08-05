# Step 31 · `track_source_chunk_ids`

> **Part 5 · Prompting and Answer Generation** — step 31 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 30 · `rag_answer`](step-30-rag-answer.md) · Next: [Step 32 · `append_source_references`](step-32-append-source-references.md)

---

## The task

```python
def track_source_chunk_ids(chunks: list) -> list: ...
```

Collect the `'id'` from each retrieved chunk, in the order they appear. Skip
chunks with no id rather than crashing.

---

## Why this step exists

An answer is only useful if a reader can check it. The id is the handle that
makes that possible: it is stable across the corpus, while the text can be
truncated for display and the score changes with every query.

Keeping the ids as an ordered list is what lets later code render citations,
compute retrieval metrics against gold ids, or fetch the original passage.

---

## What's happening

```python
return [chunk["id"] for chunk in chunks if "id" in chunk]
```

A projection with a guard. Two decisions in it.

### A list, not a set

`set(...)` would look tidier and would destroy two things.

**Order.** The list comes back ranked, best first. That ranking is information —
it is what [MRR](../evaluation/step-44-mean-reciprocal-rank.md) measures, and
what makes `[1]` in the context block mean "the best match". A set has no order,
and in CPython small string sets iterate in an order that *looks* stable enough
to fool you in testing.

**Duplicates.** Two retrieved windows can come from the same source chunk,
especially after
[`chunk_with_overlap`](../chunking/step-09-chunk-with-overlap.md), which
deliberately emits overlapping and even
[fully redundant](../chunking/step-09-chunk-with-overlap.md#the-redundant-tail)
windows. That the same chunk was retrieved twice is a fact worth keeping; a set
throws it away. Deduplicating is a decision for whoever renders the citations,
not for the function that collects them.

### Skipping missing ids

`if "id" in chunk` rather than `chunk["id"]` or `chunk.get("id")`. The first
would raise on partial metadata; the second would put `None` in the output, and
a `None` citation is worse than a missing one because it renders.

This is one of the few places in this repo that deliberately does not
[fail loudly](../ingestion/00-overview.md#design-rules-for-this-part). The
reasoning is that citation is a presentation concern: losing one citation is a
smaller harm than failing to return an answer that was otherwise fine.

---

## The key name does not match the pipeline

Worth flagging clearly, because it will bite.

This step reads `'id'`. But
[`attach_chunk_metadata`](../chunking/step-10-attach-chunk-metadata.md) — the
function in this same pipeline whose entire job is to give chunks an identity —
writes **`'chunk_id'`**:

```python
>>> attach_chunk_metadata(['hello'], 'doc1')
[{'text': 'hello', 'source': 'doc1', 'position': 0, 'chunk_id': 'doc1::0'}]
>>> track_source_chunk_ids(_)
[]
```

Every chunk produced by step 10 is silently skipped. The graceful-skip behaviour
that protects against partial metadata is exactly what hides this: no error, no
warning, just an empty citation list.

The implementation follows the specification for this step, which names `'id'`.
A test pins the mismatch so it cannot be forgotten. If you are wiring the
pipeline end to end, either rename the key in step 10 or read both here.

---

## Boundaries of the contract

**Ids are not validated.** Any type passes through, including `None` if it was
stored under the key explicitly.

**No deduplication**, by design.

**Presentation only.** Nothing checks that the model actually used a source; it
lists what was retrieved. An answer citing five chunks it ignored looks
identical to one that used them all.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning a set | Loses the ranking and collapses meaningful duplicates. |
| `chunk["id"]` with no guard | `KeyError` on partial metadata, mid-render. |
| `chunk.get("id")` | Puts `None` in the list, and `None` renders. |
| Sorting the ids | Destroys the ranking the whole list encodes. |
| Assuming it works on step 10's output | It reads `'id'`; step 10 writes `'chunk_id'`. |

---

## Example

```python
>>> track_source_chunk_ids([{'id': 'doc1::0', 'text': 'a'}, {'id': 'doc1::1', 'text': 'b'}])
['doc1::0', 'doc1::1']
>>> track_source_chunk_ids([])
[]
>>> track_source_chunk_ids([{'id': 'c0'}, {'text': 'no id'}, {'id': 'c1'}])
['c0', 'c1']
```

---

## Where it fits

```
  sources ──▶ [ track_source_chunk_ids ] ──▶ ['c0', 'c1'] ──▶ [ append_source_references ]
                                                            └──▶ retrieval metrics (Part 7)
```
