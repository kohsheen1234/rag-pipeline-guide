# Step 15 · `save_corpus`

> **Part 3 · Embeddings & Corpus Storage** — step 15 of 51
> Code: [`rag_pipeline/embeddings.py`](../../rag_pipeline/embeddings.py) · Tests: [`tests/test_embeddings.py`](../../tests/test_embeddings.py)
> Previous: [Step 14 · `l2_normalize`](step-14-l2-normalize.md)

---

## The task

```python
def save_corpus(embeddings: np.ndarray, chunks: list, directory: str) -> dict: ...
```

Persist an embedding matrix and its chunk metadata to a directory (creating it
if missing), writing the embeddings as `.npy` and the chunks as `.json`, then
read both back and return `{'embeddings': ..., 'chunks': ...}`.

---

## Why this step exists

Embedding a corpus is the expensive part of the pipeline — minutes to hours for
anything substantial. A serving process that re-embeds on every start is
unusable. Persisting means the expensive work happens once, offline, and
startup is a file read.

**Why the function reloads what it just wrote** is the more interesting design
question. Returning the in-memory objects would be faster and would pass any
shape assertion. But then the first time anyone reads the corpus from disk is in
production, in a different process, and a dtype or ordering bug surfaces there
instead of here. Round-tripping in the same call makes the write path and the
read path exercise each other every time.

---

## What's happening

```python
os.makedirs(directory, exist_ok=True)
np.save(embeddings_path, embeddings)
with open(chunks_path, "w", encoding="utf-8") as file:
    json.dump(chunks, file)

with open(chunks_path, "r", encoding="utf-8") as file:
    return {"embeddings": np.load(embeddings_path), "chunks": json.load(file)}
```

**Two formats, because there are two kinds of data.** `.npy` is a binary format
that stores the raw buffer alongside its shape and dtype, so a matrix comes back
bit-identical. JSON would store the same numbers as decimal text — larger,
slower, and lossy at the last bit of a float. Conversely the chunk dicts are
strings and small integers, where JSON's readability is worth more than
compactness.

**`exist_ok=True`** makes the call idempotent. Without it, re-saving to an
existing directory raises `FileExistsError` — and this is a function you run
repeatedly while iterating on chunk sizes.

**Fixed filenames.** `embeddings.npy` and `chunks.json` are module constants,
so the writer and the reader cannot disagree. The guide names inconsistent
naming as a pitfall, and it is a real one: a save that writes `vectors.npy` and
a load that looks for `embeddings.npy` fails at load time, long after the
expensive work is done.

---

## The invariant this has to preserve

The matrix and the chunk list are joined by **row order and nothing else**. Row
*i* of the matrix is the embedding of `chunks[i]`. There is no id inside the
matrix; the position *is* the link, inherited from
[`embed_chunks`](step-13-embed-chunks.md).

That makes them a single unit that must be written together, read together, and
never independently reordered. Saving the matrix and forgetting to re-save the
chunks after a re-chunk gives you a corpus where every retrieval returns
confidently wrong text — right scores, wrong passages, no error anywhere.

Writing both in one function is the cheapest available defence against that.

---

## Dtype must survive

The guide flags float32 → float64 as a silent corruption, and it is worth being
precise about the consequence: FAISS's index is constructed for a specific
dtype, and float64 vectors either raise on `add` or force a conversion copy that
doubles memory. A corpus saved as float32 and reloaded as float64 breaks an
index built on the original.

`np.save` preserves dtype and shape exactly, so this is handled — but only
because the matrix goes through `.npy`. Round-tripping it through JSON, or
through `np.array(embeddings.tolist())`, would land on float64 by default.

---

## Boundaries of the contract

**Chunks must be JSON-serialisable.** This is the sharp edge in practice. Python
`str`, `int`, `float`, `bool`, `None`, lists and dicts are fine. **numpy scalars
are not**:

```python
>>> save_corpus(matrix, [{"position": np.int64(0)}], directory)
TypeError: Object of type int64 is not JSON serializable
```

Easy to hit — anything derived from a numpy operation carries numpy scalar
types. Cast to `int()` or `float()` when building chunk records. The failure is
at least loud.

**JSON is lossy for types.** Tuples come back as lists; dict keys come back as
strings. A chunk built with a tuple field will not round-trip identically.

**No atomicity.** Two separate writes with no temp-file-and-rename. A crash
between them leaves a directory whose matrix and metadata disagree — and nothing
detects that on load.

**No versioning or validation.** Nothing records which model or chunk settings
produced the corpus, so nothing stops you loading a corpus embedded by one model
and querying it with another. Given
[step 11's warning](step-11-load-embedding-model.md#why-this-step-exists) that
this fails silently, a `config.json` alongside the other two files would be
cheap insurance.

**Overwrites without warning.** Saving to a populated directory replaces both
files.

**Everything is loaded into memory.** `np.load` without `mmap_mode` reads the
whole matrix. For a large corpus, `mmap_mode='r'` would let the OS page it.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning the in-memory objects instead of reloading | Passes every assertion while proving nothing about the files. |
| Forgetting `exist_ok=True` | Re-saving raises on the second run. |
| Inconsistent filenames between save and load | Fails after the expensive work, not before. |
| Saving the matrix as JSON | Larger, slower, and float64 on the way back. |
| numpy scalars in chunk dicts | `TypeError` at save time. Cast to Python types. |
| Saving the matrix without the chunks | Row *i* now points at the wrong text, silently. |
| No record of the embedding model | A corpus and a query model can drift apart with no error. |

---

## Example

```python
>>> emb = np.array([[1.0, 2.0]], dtype=np.float32)
>>> chunks = [{'text': 'hi', 'id': 0}]
>>> out = save_corpus(emb, chunks, tempfile.mkdtemp())
>>> out['embeddings'].tolist()
[[1.0, 2.0]]
>>> out['chunks']
[{'text': 'hi', 'id': 0}]
>>> out['embeddings'].dtype
dtype('float32')
```

---

## Where it fits

```
  chunks ──▶ [ embed_chunks ] ──▶ [ l2_normalize ] ──▶ (N, d) float32 ──┐
     │                                                                  │
     └──────────────────────────────────────────────────────────────────┤
                                                                        ▼
                                                              [ save_corpus ]
                                                                        │
                                              directory/embeddings.npy  │
                                              directory/chunks.json  ◀──┘
                                                                        │
                                                                        ▼
                                                      Part 4 · Dense Retrieval
```

This closes Part 3. The pipeline can now be split in two: an offline job that
ingests, chunks, embeds, and saves; and a serving process that loads and
queries.
