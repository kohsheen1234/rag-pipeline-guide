# Step 13 · `embed_chunks`

> **Part 3 · Embeddings & Corpus Storage** — step 13 of 51
> Code: [`rag_pipeline/embeddings.py`](../../rag_pipeline/embeddings.py) · Tests: [`tests/test_embeddings.py`](../../tests/test_embeddings.py)
> Previous: [Step 12 · `embed_text`](step-12-embed-text.md) · Next: [Step 14 · `l2_normalize`](step-14-l2-normalize.md)

---

## The task

```python
def embed_chunks(model, chunks, batch_size: int = 32) -> np.ndarray: ...
```

Embed a list of chunks into a 2D float32 matrix of shape `(n_chunks, d)`.
Accept either a list of raw strings or a list of chunk dicts containing a
`'text'` field. Batch the inputs through the model and return a single stacked
array. Preserve input order so row *i* corresponds to chunk *i*.

---

## Why this step exists

The corpus is embedded once and queried thousands of times, so this is the
expensive operation that everything else is designed to avoid repeating. Its
output — an `(N, d)` matrix — is the thing the retriever multiplies against, and
it is what [`save_corpus`](step-15-save-corpus.md) persists so a serving process
never has to run it again.

**Batching is why this is a separate function** rather than a loop over
[`embed_text`](step-12-embed-text.md). A transformer forward pass parallelises
across the batch axis, so encoding 32 strings at once is far cheaper than 32
separate passes — the GPU or the vectorised CPU path stays busy instead of
paying fixed per-call overhead 32 times.

---

## What's happening

```python
texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in chunks]
embeddings = model.encode(texts, batch_size=batch_size)

return np.asarray(embeddings, dtype=np.float32)
```

### Accepting two input shapes

The `isinstance` check is the interesting line. Chunks arrive as bare strings
from any of the [Part 2 chunkers](../chunking/00-overview.md), and as dicts once
[`attach_chunk_metadata`](../chunking/step-10-attach-chunk-metadata.md) has run.
Both are legitimate — you prototype with strings and run with records — so the
embedder normalises rather than forcing the caller to.

The guide is right that the alternative breaks *silently*. A dict passed to a
tokenizer does not usually raise; it gets coerced to something, and you end up
with vectors that encode a stringified dict rather than the text. The corpus
looks fine, the shapes are right, and retrieval is garbage.

Only `text` is read. `source`, `position`, and `chunk_id` are metadata for
citation, not content to embed — encoding them would blur every vector with the
filename it came from.

### Returning a stacked array

`model.encode` on a list already returns a stacked `(n, d)` array, so this is
mostly a dtype cast. The point of the contract is what *not* to do: returning a
Python list of 1D arrays would force every later step to call `np.vstack`
itself, and one of them would eventually forget.

### `batch_size`

Passed straight through. Bigger batches are faster until they exhaust memory;
32 is a reasonable default for CPU and small GPUs. It affects speed and memory
only — the output is identical either way.

---

## Boundaries of the contract

**Order is the contract.** Row *i* is chunk *i*, and everything downstream
depends on it — the row index is how a retrieved vector is mapped back to its
text and its `chunk_id`. Nothing in the matrix records which chunk a row came
from; the *position* is the only link. This is the same positional coupling as
[step 2's file ordering](../ingestion/step-02-load-text-directory.md), one level
up.

**Long chunks are truncated silently.** Same as
[`embed_text`](step-12-embed-text.md#boundaries-of-the-contract): text beyond
the model's maximum sequence length is dropped without warning. Worth checking
that your chunk size and the model's limit actually agree — a 2000-character
chunk fed to a 256-token model loses most of itself.

**A missing `text` key raises `KeyError`.** Deliberately loud: a chunk dict
without text is a bug upstream, not something to paper over.

**Empty input is model-dependent.** `embed_chunks(model, [])` returns whatever
the encoder returns for an empty list — which may be shape `(0,)` rather than
`(0, d)`. Guard the empty case at the call site if it can happen.

**No normalisation.** Follow with [`l2_normalize`](step-14-l2-normalize.md).

**Everything is held in memory.** A large corpus produces a large matrix —
1M chunks at 384 float32 dimensions is about 1.5 GB. Streaming to disk in
batches is a different design; this one materialises the whole thing.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Accepting only strings | Breaks the moment metadata is attached — and often silently, by embedding a stringified dict. |
| Embedding the whole dict | The vector encodes the filename and position alongside the text, blurring every chunk toward its neighbours. |
| Looping over `embed_text` | Loses the batch parallelism this function exists for. Correct, and much slower. |
| Returning a list of arrays | Pushes the stacking onto every caller. |
| Sorting or filtering the chunks here | Breaks the row-to-chunk correspondence with no error. |
| Leaving the dtype as float64 | Doubles corpus memory and breaks a FAISS index later. |
| Ignoring the model's sequence limit | Long chunks are half-embedded, invisibly. |

---

## Example

```python
>>> model = load_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
>>> mat = embed_chunks(model, ['hello world', 'goodbye world'])
>>> mat.shape
(2, 384)
>>> mat.dtype
dtype('float32')
```

Both input shapes give the same matrix:

```python
>>> records = attach_chunk_metadata(['hello world', 'goodbye world'], 'doc1')
>>> np.array_equal(embed_chunks(model, records), mat)
True
```

---

## Where it fits

```
  list[str] or list[dict] ──▶ [ embed_chunks ] ──▶ (N, d) float32
                                                          │
                                                          ▼
                                                  [ l2_normalize ]
                                                          │
                                                          ▼
                                                   [ save_corpus ]
```

The matrix and the chunk list travel together from here on, joined only by row
order — which is exactly the invariant [`save_corpus`](step-15-save-corpus.md)
has to preserve across a process boundary.
