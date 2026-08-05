# Step 12 · `embed_text`

> **Part 3 · Embeddings & Corpus Storage** — step 12 of 51
> Code: [`rag_pipeline/embeddings.py`](../../rag_pipeline/embeddings.py) · Tests: [`tests/test_embeddings.py`](../../tests/test_embeddings.py)
> Previous: [Step 11 · `load_embedding_model`](step-11-load-embedding-model.md) · Next: [Step 13 · `embed_chunks`](step-13-embed-chunks.md)

---

## The task

```python
def embed_text(model, text: str) -> np.ndarray: ...
```

Use a loaded sentence-transformers model to produce a single embedding vector
for one text string. The result must be a 1D numpy array of shape `(d,)` and
dtype `float32`.

---

## Why this step exists

This is the query side of retrieval. The corpus is embedded once
([step 13](step-13-embed-chunks.md)); this runs on every search, turning the
user's question into a vector that can be compared against the corpus matrix.

The two requirements in the signature — 1D, and `float32` — are both about
making that comparison work without ceremony downstream. A `(384,)` vector
against an `(N, 384)` matrix broadcasts correctly for `matrix @ vector`, giving
`N` scores. A `(1, 384)` vector does not, and the failure is a confusing shape
error several functions away from the cause.

---

## What's happening

```python
vector = model.encode(text)

return np.asarray(vector, dtype=np.float32).reshape(-1)
```

**`model.encode(text)`** runs the tokenizer, the transformer, and the pooling
layer. From the caller's side it is just text → vector; the model handles
truncation to its maximum sequence length internally (see the boundary note
below).

**`.reshape(-1)`** flattens to one dimension. This is the pitfall the guide
names: batch-style encoders often return `(1, d)` for a single input, because
internally they treat one string as a batch of one. `reshape(-1)` collapses that
leading axis and is a no-op if the array was already 1D — so it is correct
whichever shape the encoder happens to return, rather than depending on which.

`np.squeeze` would also work but is riskier: it removes *every* singleton axis,
so a hypothetical `(1, 1)` result would come back as a 0-d scalar rather than a
1-element vector.

**`dtype=np.float32`** casts down from the float64 some encoders return. Two
reasons this matters and is not just tidiness:

- **Consistency with the corpus.** A float64 query dotted against a float32
  matrix silently promotes the whole matrix to float64 in the process,
  doubling memory for the duration of the multiply.
- **FAISS.** The index built in a later part requires float32. Feeding it
  float64 raises, or silently copies.

`np.asarray` rather than `np.array` avoids a copy when the input already has the
right dtype.

---

## Boundaries of the contract

**Long text is truncated silently.** Every model has a maximum sequence length —
256 word-pieces for `all-MiniLM-L6-v2`. Text beyond that is dropped, with no
warning and no error: you get a perfectly valid vector representing only the
beginning of your input. This is the single most important thing to know about
this function, and it is the real reason [Part 2](../chunking/00-overview.md)
exists. If your chunks exceed the model's limit, the tail of every chunk is
invisible to retrieval.

**No normalisation.** The returned vector is not unit length. Use
[`l2_normalize`](step-14-l2-normalize.md) — and apply it to the query and the
corpus alike, or the dot product is not a cosine.

**The empty string embeds successfully.** It produces a valid vector rather than
raising, which means an empty chunk becomes a retrievable entry with an
arbitrary position in the space.

**One string only.** Passing a list would return a 2D array, and `reshape(-1)`
would flatten it into nonsense rather than failing. Use
[`embed_chunks`](step-13-embed-chunks.md) for lists.

**Not cached.** Repeated identical queries re-encode. Cheap enough not to matter
at low volume; worth a cache if the same queries repeat.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning the raw `(1, d)` array | `matrix @ vector` fails, or worse, broadcasts into an `(N, d)` result that looks like data. |
| `np.squeeze` instead of `reshape(-1)` | Removes all singleton axes; a 1-dimensional model would return a scalar. |
| Leaving the dtype as float64 | Promotes the corpus matrix during the multiply, and FAISS rejects it later. |
| Passing a list of strings | Silently flattened into one long vector by `reshape(-1)`. |
| Forgetting to normalise the query too | Normalising only the corpus makes the dot product a scaled cosine — the ranking survives, but the scores are meaningless and any similarity threshold is wrong. |
| Assuming long input is fully encoded | Truncated at the model's limit, silently. |

---

## Example

```python
>>> model = load_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
>>> v = embed_text(model, 'hello world')
>>> v.shape
(384,)
>>> v.dtype
dtype('float32')
```

---

## Where it fits

```
  query ──▶ [ embed_text ] ──▶ (d,) float32 ──▶ [ l2_normalize ] ──┐
                                                                   ├──▶ dot ──▶ scores
  corpus ─▶ [ embed_chunks ] ▶ (N, d) float32 ▶ [ l2_normalize ] ──┘
```

`embed_text` and `embed_chunks` are the same operation at different arities, and
they must stay in step: same model, same dtype, same normalisation. Any
asymmetry between the query path and the corpus path shows up as retrieval that
is subtly, unexplainably worse.
