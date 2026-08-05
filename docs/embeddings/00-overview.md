# Part 3 · Embeddings & Corpus Storage

Chunks are text; retrieval is arithmetic. This part is the conversion between
them — and the point at which the pipeline splits into an offline job and a
serving process.

## The idea

An embedding model maps a string to a point in `R^d`, positioned so that texts
meaning similar things land near each other. Encode the corpus once into an
`(N, d)` matrix, encode a query into a `(d,)` vector, and "which chunks are
relevant" becomes "which rows point in the same direction as the query" — a
matrix multiply.

Everything in this part exists to make that multiply correct and cheap.

## The invariants

**One model, one space.** Corpus and queries must be encoded by the same model.
Two models with the same output dimension produce vectors that are numerically
comparable and semantically unrelated, so the failure is silently bad rankings
rather than an error.

**Row order is the only join.** Row *i* of the matrix is the embedding of
`chunks[i]`. Nothing inside the matrix records which chunk a row came from, so
any operation that reorders one without the other corrupts the corpus with no
symptom except wrong answers.

**float32 throughout.** The dtype the matrix is built with is the dtype FAISS
will be constructed for. A stray float64 doubles memory and breaks the index.

**Normalise both sides.** Unit-length vectors turn cosine similarity into a
plain dot product. Normalising only the corpus still ranks correctly, which is
what makes forgetting the query side so easy to miss.

## Steps

All live in [`rag_pipeline/embeddings.py`](../../rag_pipeline/embeddings.py).

| # | Function | What it does |
| --- | --- | --- |
| 11 | [`load_embedding_model`](step-11-load-embedding-model.md) | Load a sentence-transformers model once, reuse the instance. |
| 12 | [`embed_text`](step-12-embed-text.md) | One string → a `(d,)` float32 vector. The query path. |
| 13 | [`embed_chunks`](step-13-embed-chunks.md) | Many chunks → an `(N, d)` float32 matrix. The corpus path. |
| 14 | [`l2_normalize`](step-14-l2-normalize.md) | Unit-length rows, so dot products are cosines. |
| 15 | [`save_corpus`](step-15-save-corpus.md) | Persist matrix and metadata together, and prove it round-trips. |

## A note on dependencies

`sentence-transformers` pulls in torch, which is large. It is imported **inside**
`load_embedding_model` rather than at the top of the module, so `l2_normalize`
and `save_corpus` — which need only numpy — stay importable and testable without
it. Everything except the loader itself is tested against a small fake model
exposing the same `encode` surface; the one test that needs real weights skips
when the package is absent.

## Data flow

```
  Part 2 · Chunking
         │
         ▼
   list[dict]  ──────────────────────────────┐
         │                                   │
         ▼                                   │
  ┌────────────────┐                         │
  │  embed_chunks  │ ◀── model               │
  └───────┬────────┘                         │
          ▼                                  │
     (N, d) float32                          │
          │                                  │
          ▼                                  │
  ┌────────────────┐                         │
  │  l2_normalize  │                         │
  └───────┬────────┘                         │
          ▼                                  ▼
        ┌──────────────────────────────────────┐
        │             save_corpus              │
        └──────────────────┬───────────────────┘
                           ▼
              embeddings.npy + chunks.json
                           │
                           ▼
                 Part 4 · Dense Retrieval

  query ──▶ [ embed_text ] ──▶ (d,) float32 ──▶ normalise ──▶ Part 4
```

The upper path runs offline, once. The lower path runs per query. They meet in
Part 4, and they only work together because both used the same model.
