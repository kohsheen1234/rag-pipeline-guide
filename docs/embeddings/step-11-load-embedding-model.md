# Step 11 · `load_embedding_model`

> **Part 3 · Embeddings & Corpus Storage** — step 11 of 51
> Code: [`rag_pipeline/embeddings.py`](../../rag_pipeline/embeddings.py) · Tests: [`tests/test_embeddings.py`](../../tests/test_embeddings.py)
> Previous: [Step 10 · `attach_chunk_metadata`](../chunking/step-10-attach-chunk-metadata.md) · Next: [Step 12 · `embed_text`](step-12-embed-text.md)

---

## The task

```python
def load_embedding_model(model_name: str): ...
```

Return a sentence-transformers model instance that can later embed chunks and
queries. Accept any valid sentence-transformers identifier. Keep it a thin
loader — no embedding logic.

---

## Why this step exists

The function is one line, so the step is really about a property of the whole
pipeline: **the corpus and the queries must be encoded by the same model.**

Retrieval works by comparing a query vector to chunk vectors. That comparison
is only meaningful if both live in the same vector space — and a vector space
is defined by the model that produced it. Two models with the same output
dimension produce vectors that are numerically comparable and semantically
unrelated; you get similarity scores that look fine and rank nothing correctly.
There is no error, no shape mismatch, just quietly bad retrieval.

Naming the loader once, and passing the instance around, is what makes it hard
to accidentally use two.

---

## What's happening

```python
from sentence_transformers import SentenceTransformer

return SentenceTransformer(model_name)
```

`SentenceTransformer` wraps a transformer backbone with a pooling layer. The
backbone produces one vector per token; the pooling layer (mean-pooling, for
MiniLM) collapses those into a single fixed-length vector for the whole string.
That collapse is what makes the model a *sentence* encoder rather than a token
encoder, and it is why the output dimension is fixed regardless of input length.

For `all-MiniLM-L6-v2` the dimension is 384. Encoding `N` strings gives an
`(N, 384)` array.

**The identifier is a Hugging Face Hub path.** On first call the weights are
downloaded and cached under `~/.cache/huggingface`; later calls read from the
cache. So the first run of a fresh environment is slow and needs network access,
and every run after that does not.

---

## Why the instance must be reused

The guide's pitfall — reloading per query — is worth quantifying, because "slow"
undersells it. Loading involves reading a few hundred megabytes of weights from
disk, constructing the torch modules, initialising the tokenizer, and moving
tensors to the device. That is on the order of a second or more. Encoding one
short query is on the order of a millisecond.

So `load_embedding_model` inside a request handler makes every query roughly a
thousand times more expensive than it needs to be, and the cost is invisible in
a profile that only looks at "embedding time".

The instance is safe to hold for the process lifetime. Load it once at startup
and pass it into [`embed_text`](step-12-embed-text.md) and
[`embed_chunks`](step-13-embed-chunks.md) — which is exactly why both of those
take the model as their first argument rather than a model *name*.

---

## The lazy import

In this repo the import sits inside the function rather than at the top of the
module:

```python
def load_embedding_model(model_name: str):
    # Imported here so the rest of this module works without torch installed.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
```

`sentence-transformers` pulls in torch — hundreds of megabytes. A module-level
import would make [`l2_normalize`](step-14-l2-normalize.md) and
[`save_corpus`](step-15-save-corpus.md), which need nothing but numpy,
unimportable without it. It also keeps the test suite fast: everything except
this function is tested against a small fake model.

This is a deliberate deviation from the obvious version, and the only one in
this step. If you are following the guide literally, a top-level import is fine
— it just couples the whole module to a heavy dependency.

---

## Boundaries of the contract

**No caching.** Two calls with the same name build two model instances. The
function is a loader, not a registry; deduplication is the caller's job.

**No validation.** An unknown identifier raises from the Hub client, not from
here.

**Device selection is automatic.** sentence-transformers picks CUDA or MPS if
available, else CPU. Pass `device=` to `SentenceTransformer` directly if you
need to pin it — this thin wrapper does not expose that.

**The model is stateful but safe to share** for inference across sequential
calls. Concurrent use from multiple threads is not something this step
addresses.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Loading inside a request handler or a loop | ~1000× the cost of the encode it precedes. |
| Using different models for corpus and queries | Same dimension, unrelated spaces. No error — just wrong rankings. |
| Re-embedding a corpus with a new model but keeping the old vectors | Half the index in one space, half in another. |
| Assuming the first call is fast | It downloads weights. Offline environments need the cache pre-warmed. |
| Comparing dimensions to check compatibility | Many models share 384 or 768. Matching dimension says nothing about matching space. |

---

## Example

```python
>>> model = load_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
>>> type(model).__name__
'SentenceTransformer'
>>> model.get_sentence_embedding_dimension()
384
```

> The test asserting this is skipped unless `sentence-transformers` is
> installed, since it downloads weights. Everything downstream is tested against
> a fake model with the same `encode` surface.

---

## Where it fits

```
  model_name ──▶ [ load_embedding_model ] ──▶ model ──┬──▶ embed_text    (queries)
                                                      └──▶ embed_chunks  (corpus)
```

One model, two call sites, one vector space. The rest of Part 3 assumes that
invariant holds.
