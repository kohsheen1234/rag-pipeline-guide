# Part 4 · Dense Retrieval with NumPy and FAISS

Part 3 produced a matrix. This part turns it into a search engine.

The whole of dense retrieval is three operations: score the query against every
chunk, keep the best few, and hand back the chunks rather than their row
numbers. Steps 16–19 build exactly that in numpy. Steps 20–23 then do it again
with FAISS and check the two agree.

## Why build it twice

The numpy path is a matrix multiply and an `argsort`. You can read it, and when
a result looks wrong you can print the intermediate scores. It also scales
linearly and becomes the bottleneck somewhere in the tens of thousands of
chunks.

FAISS is the same computation in C++, and for a flat index it is *exactly* the
same computation, not an approximation. So the two should return identical
neighbours, and [step 22](step-22-compare-faiss-to-numpy.md) exists to assert
that rather than assume it. The moment they disagree, something about how the
index was built or queried is wrong, and that is a much easier bug to find at
build time than in production rankings.

## The invariants

**Same model on both sides.** Inherited from
[Part 3](../embeddings/00-overview.md), and still the thing that fails most
silently.

**Row index is identity.** Retrieval returns positions, and positions are
mapped back to chunks by list index. Anything that reorders the chunk list
without reordering the matrix corrupts every result.

**float32 for FAISS.** The index is constructed for a dtype. Feed it float64
and it raises or copies.

**Normalise, then inner product.** FAISS has no cosine index because it does
not need one: for unit vectors, inner product *is* cosine. Skip the
normalisation and `IndexFlatIP` still returns numbers, just not the ones you
wanted.

## Steps

All live in [`rag_pipeline/retrieval.py`](../../rag_pipeline/retrieval.py).

| # | Function | What it does |
| --- | --- | --- |
| 16 | [`cosine_similarity_search`](step-16-cosine-similarity-search.md) | One query against every row, normalising as it goes. |
| 17 | [`top_k_indices`](step-17-top-k-indices.md) | Positions of the k best scores, descending. |
| 18 | [`top_k_chunks`](step-18-top-k-chunks.md) | Those positions resolved to (chunk, score) pairs. |
| 19 | [`retrieve`](step-19-retrieve.md) | Query string in, ranked chunks out. |
| 20 | [`build_faiss_index`](step-20-build-faiss-index.md) | A flat inner-product index over the matrix. |
| 21 | [`faiss_search`](step-21-faiss-search.md) | Single-query search with the batch axis hidden. |
| 22 | [`compare_faiss_to_numpy`](step-22-compare-faiss-to-numpy.md) | Assert the two backends pick the same chunks. |
| 23 | [`save_faiss_index`](step-23-save-faiss-index.md) | Persist the index and prove it reloads. |

## On testing without FAISS

`faiss` is an optional dependency here. The functions that only need an object
with a `.search` method are tested against a small stand-in implementing
`IndexFlatIP` semantics exactly; the ones that construct or serialise a real
index skip unless `faiss` is installed. That keeps the suite fast and
installable, at the cost that the stand-in agrees with numpy by construction —
the real agreement check only runs when FAISS is present.

## Data flow

```
  query string
       │
       ▼
  [ embed_text ] ──▶ (d,) ──┬──────────────────────────────┐
                            ▼                              ▼
              [ cosine_similarity_search ]          [ faiss_search ]
                            │  (n,) scores                 │  (k,) scores
                            ▼                              │  (k,) indices
                    [ top_k_indices ]                      │
                            │                              │
                            ▼                              │
                    [ top_k_chunks ] ◀── chunks            │
                            │                              │
                            ▼                              ▼
                 [(chunk, score), ...]        [ compare_faiss_to_numpy ]
                            │                          same top-k?
                            ▼
                  Part 5 · Generation
```
