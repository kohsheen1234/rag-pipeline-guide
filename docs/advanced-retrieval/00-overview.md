# Part 6 · Advanced Retrieval Techniques

Part 4 built a working retriever: embed the query, take the nearest chunks. This
part is eight ways that is not quite good enough, and what to do about each.

Nothing here replaces dense retrieval. Everything sits around it — cleaning what
goes in, adding a second opinion beside it, or reordering what comes out.

## The four failure modes

**The query does not look like the corpus.** A user types "who founded X?"; the
passage says "X was founded in 1994 by...". Short interrogatives and long
declarative prose sit in different regions of embedding space, so the nearest
neighbour of a question is often another question.
[`query_rewrite`](step-33-query-rewrite.md) strips the conversational packaging;
[`hyde_retrieve`](step-34-hyde-retrieve.md) goes further and searches with a
fabricated *answer* instead.

**Embeddings miss exact terms.** A dense model that has never seen your part
number, error code, or surname will place it nowhere useful. BM25 does not care
what a token means, only how rare it is.
[`bm25_search`](step-36-bm25-search.md) provides that signal;
[`hybrid_search`](step-37-hybrid-search.md) mixes the two;
[`reciprocal_rank_fusion`](step-35-reciprocal-rank-fusion.md) combines rankings
when the scores are not comparable at all.

**Scoring each document alone is imprecise.** A bi-encoder embeds query and
passage separately and compares vectors, which is what makes precomputing the
corpus possible and also what limits it.
[`rerank_cross_encoder`](step-38-rerank-cross-encoder.md) reads the pair
together, far more accurately and far too slowly for a whole corpus — so it
reruns the top-N.

**The top-k are all the same passage.** Overlapping chunks and boilerplate mean
the five best matches are often five views of one paragraph.
[`maximal_marginal_relevance`](step-39-maximal-marginal-relevance.md) trades a
little relevance for coverage. [`filter_by_metadata`](step-40-filter-by-metadata.md)
solves a different narrowing problem: restricting the search to a tenant,
language, or document before ranking at all.

## Where they go

```
              [ query_rewrite ]          clean the query
                     │
                     ▼
              [ hyde_retrieve ]          search with a fake answer
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
 dense           [ bm25_search ]   [ filter_by_metadata ]
    │                │                 (narrow first)
    └──────┬─────────┘
           ▼
  [ hybrid_search ] or [ reciprocal_rank_fusion ]
           │
           ▼
  [ rerank_cross_encoder ]              second pass, expensive
           │
           ▼
  [ maximal_marginal_relevance ]        drop near-duplicates
           │
           ▼
      final top-k
```

You would not use all of them. A common production stack is: rewrite → hybrid →
rerank the top 50 → MMR to 5.

## Steps

All live in
[`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py).

| # | Function | Fixes |
| --- | --- | --- |
| 33 | [`query_rewrite`](step-33-query-rewrite.md) | Filler words diluting the query vector. |
| 34 | [`hyde_retrieve`](step-34-hyde-retrieve.md) | Questions not looking like answers. |
| 35 | [`reciprocal_rank_fusion`](step-35-reciprocal-rank-fusion.md) | Combining rankings with incomparable scores. |
| 36 | [`bm25_search`](step-36-bm25-search.md) | Exact terms embeddings miss. |
| 37 | [`hybrid_search`](step-37-hybrid-search.md) | Needing both signals at once. |
| 38 | [`rerank_cross_encoder`](step-38-rerank-cross-encoder.md) | Bi-encoder imprecision in the top-N. |
| 39 | [`maximal_marginal_relevance`](step-39-maximal-marginal-relevance.md) | Near-duplicate results. |
| 40 | [`filter_by_metadata`](step-40-filter-by-metadata.md) | Searching the wrong subset of the corpus. |

## A note on the BM25 reference value

The guide's example for [step 36](step-36-bm25-search.md) quotes `0.5798`. This
implementation returns `ln(2) = 0.6931` on that corpus, and the write-up shows
why no choice of `k1` or `b` can produce the quoted figure there. Worth reading
before assuming your implementation is wrong.
