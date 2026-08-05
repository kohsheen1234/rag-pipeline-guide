# Step 10 · `attach_chunk_metadata`

> **Part 2 · Chunking Strategies** — step 10 of 51
> Code: [`rag_pipeline/chunking.py`](../../rag_pipeline/chunking.py) · Tests: [`tests/test_chunking.py`](../../tests/test_chunking.py)
> Previous: [Step 9 · `chunk_with_overlap`](step-09-chunk-with-overlap.md)

---

## The task

```python
def attach_chunk_metadata(chunks: list[str], source: str) -> list[dict]: ...
```

Wrap each raw chunk string into a dict carrying its text, its `source`, its
0-indexed `position` in the sequence, and a stable `chunk_id` formatted as
`f"{source}::{position}"`.

---

## Why this step exists

This closes the [provenance gap](../ingestion/step-05-make-document.md#the-provenance-gap)
that has been open since step 5. Every chunker in this part takes a string and
returns strings, so by the time a passage reaches the index it has forgotten
which document it came from. A retrieval system that cannot answer *"where did
this come from?"* cannot cite, cannot deduplicate by document, and cannot let a
user click through to the original — which for most RAG applications is the
difference between a usable product and a demo.

The `chunk_id` matters for a second reason: it is the join key. Embeddings live
in a vector index, text lives somewhere else, evaluation labels live in a third
place. All three refer to chunks by id, so the id has to be stable across runs —
recompute it differently and every cached embedding and every gold label is
orphaned.

---

## What's happening

```python
return [
    {
        "text": chunk,
        "source": source,
        "position": position,
        "chunk_id": f"{source}::{position}",
    }
    for position, chunk in enumerate(chunks)
]
```

`enumerate` supplies the 0-indexed position, and the id is derived from it
rather than stored independently — so the two can never disagree.

Everything else about this function is a naming decision, which is exactly what
makes it worth its own step. The code is obvious; the contract is the product.

---

## Why `source::position` and not a hash

The guide flags content hashing as the pitfall, and there are two distinct
reasons it fails.

**Duplicate content collides.** Identical chunks are not rare — boilerplate
headers, repeated disclaimers, and especially the
[redundant tail](step-09-chunk-with-overlap.md#the-redundant-tail) that
`chunk_with_overlap` produces by design. Hash the text and two different
positions map to one id. Whatever you write second overwrites the first, so the
index quietly loses entries, and a citation points at whichever copy survived.

```python
>>> [r["chunk_id"] for r in attach_chunk_metadata(['same', 'same'], 'doc1')]
['doc1::0', 'doc1::1']
```

**Hashes are unreadable.** `doc1::7` in a log tells you the document and roughly
where in it. `a3f8c9e1...` tells you nothing without a lookup. When you are
debugging why the wrong passage was retrieved, that difference is most of the
debugging.

The trade-off, stated honestly: **positional ids are not stable across
re-chunking.** Change `chunk_size` and `doc1::7` refers to entirely different
text, silently. A content hash would at least change when the content changed.
So the id is stable across *runs* but not across *configuration* — which means
a cache keyed on `chunk_id` must be invalidated whenever chunking parameters
change. Including the chunker's settings in the source string is one way to make
that automatic.

---

## Two record shapes

The pipeline now has two dict schemas, and they are not the same:

| Function | Keys |
| --- | --- |
| [`make_document`](../ingestion/step-05-make-document.md) | `text`, `source`, `title` |
| `attach_chunk_metadata` | `text`, `source`, `position`, `chunk_id` |

Both use `text` and `source` to mean the same things, which is what lets a
document's `source` flow into its chunks unchanged. But a chunk record has no
`title`, so **the title is lost at this boundary** — nothing in the current
pipeline carries it from a document onto that document's chunks. Citing "Chapter
4 of *Introduction to RAG*" rather than "chapter4.txt" needs that link, and it
does not exist yet.

That is a gap to watch rather than a bug to fix here: `attach_chunk_metadata`
takes a bare `source` string by specification, so the join has to happen at the
call site or in a later step.

---

## Boundaries of the contract

**No validation.** An empty `source`, or a `source` containing `::`, is accepted
as-is. The latter makes ids ambiguous — `a::b` at position 0 gives `a::b::0`,
which cannot be unambiguously parsed back into source and position by splitting
on `::`. Parse from the right, or do not parse ids at all and keep the fields
you already have.

**Positions are per-call, not global.** Two documents each produce positions
starting at 0. The `source` prefix is what makes ids globally unique, so passing
the same `source` for two different documents silently collides their ids.

**Text is stored verbatim.** No normalisation, no stripping — consistent with
`make_document`.

**Flat, and JSON-serialisable.** Same reasoning as
[step 5](../ingestion/step-05-make-document.md#why-a-flat-dict): most vector
stores take a plain JSON object per record.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Hashing the text for the id | Duplicate chunks collide and overwrite each other in the index. Unreadable in logs. |
| Using the text itself as the id | Same collision problem, plus unbounded key length. |
| 1-indexing positions | Off-by-one against every list index in the pipeline. The example fixes 0. |
| Storing `position` as a string | `"10" < "9"` when sorted. Keep it an `int`. |
| Recomputing ids with different chunk settings | Ids stay the same while the text underneath changes. Caches and gold labels silently point at the wrong passage. |
| Reusing one dict across the loop | Every record ends up being the same object with the last values. |
| Assuming the id round-trips | A `source` containing `::` makes it ambiguous. |

---

## Example

```python
>>> attach_chunk_metadata(['hello', 'world'], 'doc1')
[{'text': 'hello', 'source': 'doc1', 'position': 0, 'chunk_id': 'doc1::0'},
 {'text': 'world', 'source': 'doc1', 'position': 1, 'chunk_id': 'doc1::1'}]
```

Composed with a chunker, which is how it is actually used:

```python
>>> attach_chunk_metadata(chunk_fixed_size('abcdefgh', 3), 'doc1')
[{'text': 'abc', 'source': 'doc1', 'position': 0, 'chunk_id': 'doc1::0'},
 {'text': 'def', 'source': 'doc1', 'position': 1, 'chunk_id': 'doc1::1'},
 {'text': 'gh',  'source': 'doc1', 'position': 2, 'chunk_id': 'doc1::2'}]
```

---

## Where it fits

```
  document ──▶ [ any chunker ] ──▶ list[str] ──▶ [ attach_chunk_metadata ] ──▶ list[dict]
                                                            ▲                       │
                                                         source                     ▼
                                                                            Part 3 · Embedding
```

This is the last step of Part 2, and it is where chunks stop being anonymous
text. From here the pipeline's unit is a record with an identity — which is what
makes it possible to embed it, retrieve it, and say where it came from.
