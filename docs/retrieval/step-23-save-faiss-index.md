# Step 23 · `save_faiss_index`

> **Part 4 · Dense Retrieval with NumPy and FAISS** — step 23 of 51
> Code: [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py) · Tests: [`tests/test_retrieval.py`](../../tests/test_retrieval.py)
> Previous: [Step 22 · `compare_faiss_to_numpy`](step-22-compare-faiss-to-numpy.md)

---

## The task

```python
def save_faiss_index(index, path: str): ...
```

Persist a FAISS index to disk and reload it, returning the reloaded index. The
reloaded object must have the same `ntotal` and `d`, and return the same
nearest neighbours on the same query.

---

## Why this step exists

Same argument as [`save_corpus`](../embeddings/step-15-save-corpus.md), one
level up: a serving process cannot afford to rebuild its index at every restart.
Building is `O(n)` inserts over a corpus you already embedded; loading is a file
read.

---

## What's happening

```python
import faiss

faiss.write_index(index, path)

return faiss.read_index(path)
```

`write_index` serialises the index type, the dimensionality, and the raw vectors
into a single binary file. `read_index` reconstructs the object, dispatching on
the type recorded in the file, so a `IndexFlatIP` comes back as an
`IndexFlatIP`.

### Returning the reloaded object is the whole point

The guide flags this and it is the only interesting decision in the function.
Returning `index` — the one already in memory — would satisfy every assertion
you are likely to write. `ntotal` matches. `d` matches. Search returns the same
neighbours, because it *is* the same object.

And the file on disk could be empty, truncated, or never written at all. You
would find out in a different process, at startup, in production.

Returning `read_index(path)` makes the write path and the read path exercise
each other on every call. The test that pins this asserts `reloaded is not
index`, because identity is the only thing that distinguishes the correct
implementation from the one that proves nothing.

---

## Boundaries of the contract

**A file path, not a directory.** Unlike `save_corpus`, no directory is created.
`faiss.write_index` to a path whose parent does not exist raises.

**The index only.** The chunk list is not saved here, and an index without its
chunks is a pile of row numbers. In practice this pairs with
[`save_corpus`](../embeddings/step-15-save-corpus.md), and nothing links the two
files — no shared id, no manifest. Loading an index next to the wrong
`chunks.json` gives confident, wrong citations. Worth putting them in one
directory and writing the corpus fingerprint alongside.

**No atomic write.** A crash mid-write leaves a corrupt file that
`read_index` will reject at startup.

**Not portable across FAISS majors.** The format is stable in practice but not
guaranteed across versions; treat an index as a build artefact, not an archive
format.

**Reloading costs memory.** For a moment both copies are resident, so peak usage
is roughly double the index size.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning the original in-memory index | Passes every check while verifying nothing about the file. |
| Passing a directory instead of a file path | Raises, unlike `save_corpus` which creates directories. |
| Saving the index without the chunks | Row numbers with nothing to resolve them against. |
| Assuming the index records which model built it | It does not. An index and a query model can drift apart silently. |
| Rebuilding instead of loading at startup | The cost this step exists to avoid. |

---

## Example

```python
>>> M = np.eye(3, dtype=np.float32)
>>> idx = build_faiss_index(M)
>>> with tempfile.TemporaryDirectory() as td:
...     idx2 = save_faiss_index(idx, os.path.join(td, 'i.bin'))
...     print(idx2.ntotal, idx2.d)
3 3
```

---

## Where it fits

```
  offline:  chunks ──▶ embed ──▶ normalise ──▶ build_faiss_index ──▶ save_faiss_index
                                                                            │
                                                                       index.bin
                                                                            │
  serving:                                          faiss.read_index ◀──────┘
                                                            │
                                                            ▼
                                                     [ faiss_search ]
```

With this, Part 4 closes: the corpus is searchable, the search is verified
against a second implementation, and both the vectors and the index survive a
restart.
