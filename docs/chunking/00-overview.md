# Part 2 · Chunking Strategies

Ingestion produces whole documents. Embedding models take a bounded amount of
text, and retrieval wants passages small enough to be specific about one thing.
Chunking is the step in between.

It is also the step with the most leverage in the whole pipeline. **Retrieval
can only ever return a chunk.** If the answer to a question is split across two
chunks, no amount of embedding quality or clever reranking recovers it — the
unit that contained the answer no longer exists. Chunking decides the ceiling
everything downstream operates under.

## The trade-off every strategy is negotiating

**Smaller chunks** are more specific. A chunk about one thing embeds to a vector
that means one thing, so it ranks precisely. But it carries less context, and a
chunk that says "it was discontinued in 2019" is useless without knowing what
"it" refers to.

**Larger chunks** carry their own context and survive being read alone. But
their embedding is an average of everything in them, which is a blurrier target
for any specific query, and they spend more of the generator's context window
per result.

Every strategy in this part is a different answer to that tension.

## Design rules for this part

- **Same signature, swappable policy.** Every chunker takes text and returns
  `list[str]`. Nothing downstream should need to know which one ran, so
  strategies can be compared by measurement rather than argument.
- **Order is preserved.** Chunks come back in document order. Chunk ids are
  derived from position, so this matters for the same reason document order
  did in [step 2](../ingestion/step-02-load-text-directory.md).
- **Verify with properties, not examples.** A chunker is easy to get subtly
  wrong and easy to test convincingly with one input that happens to pass.
  Assert the invariants — that chunks rejoin to the original, that sizes stay
  in bounds — across a spread of lengths.

## Steps

Both live in [`rag_pipeline/chunking.py`](../../rag_pipeline/chunking.py).

| # | Function | Bound on | What it does |
| --- | --- | --- | --- |
| 6 | [`chunk_fixed_size`](step-06-chunk-fixed-size.md) | characters | Consecutive non-overlapping character windows. |
| 7 | [`chunk_by_tokens`](step-07-chunk-by-tokens.md) | tokens | The same windowing, over tokenizer ids instead. |

_Steps are added to this table as the guide progresses._

## Data flow

```
  Part 1 · Ingestion
          │
          ▼
     document text
          │
          ├──────────────────────┐
          ▼                      ▼
  ┌────────────────────┐  ┌──────────────────┐
  │  chunk_fixed_size  │  │  chunk_by_tokens │ ◀── tokenizer
  └─────────┬──────────┘  └────────┬─────────┘
            │  characters          │  tokens
            └──────────┬───────────┘
                       ▼
                   list[str]   ──▶  Part 3 · Embedding
```

Chunks are currently bare strings, so the `source` and `title` attached by
[`make_document`](../ingestion/step-05-make-document.md) do not travel with
them. Until provenance is carried onto each chunk, a retrieved passage cannot
be cited — expect a later step to close that gap.
