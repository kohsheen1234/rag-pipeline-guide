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

All live in [`rag_pipeline/chunking.py`](../../rag_pipeline/chunking.py).

Four strategies, then one step that gives the output an identity.

| # | Function | Bound on | Limit is | What it does |
| --- | --- | --- | --- | --- |
| 6 | [`chunk_fixed_size`](step-06-chunk-fixed-size.md) | characters | hard | Consecutive non-overlapping character windows. |
| 7 | [`chunk_by_tokens`](step-07-chunk-by-tokens.md) | tokens | hard | The same windowing, over tokenizer ids instead. |
| 8 | [`chunk_by_sentences`](step-08-chunk-by-sentences.md) | characters | soft | Packs whole sentences; an overlong one exceeds the limit. |
| 9 | [`chunk_with_overlap`](step-09-chunk-with-overlap.md) | characters | hard | Sliding windows that share `overlap` characters. |
| 10 | [`attach_chunk_metadata`](step-10-attach-chunk-metadata.md) | — | — | Wraps chunks with source, position, and a stable id. |

_Steps are added to this table as the guide progresses._

Note the "limit is" column. Steps 6, 7, and 9 guarantee their bound because they
cut wherever the counter runs out. Step 8 refuses to split a sentence, so its
limit is a target — anything downstream needing a hard bound must enforce it
itself.

The four strategies answer the size trade-off in different ways, but only step 9
addresses the *boundary* problem directly: every disjoint chunker destroys any
fact that straddles a cut, and overlap is the only defence against that — paid
for in index size.

## Data flow

```
  Part 1 · Ingestion
          │
          ▼
     document text
          │
          ├──────────────┬──────────────┬──────────────┐
          ▼              ▼              ▼              ▼
  ┌───────────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐
  │chunk_fixed_   │ │chunk_by_ │ │chunk_by_    │ │chunk_with_   │
  │size           │ │tokens    │ │sentences    │ │overlap       │
  └───────┬───────┘ └────┬─────┘ └──────┬──────┘ └──────┬───────┘
          │ characters   │ tokens       │ sentences     │ sliding
          │ cuts         │ ◀─ tokenizer │ cuts at .!?   │ windows
          │ anywhere     │              │               │ share text
          └──────────────┴──────┬───────┴───────────────┘
                                ▼
                            list[str]
                                │
                 source ──▶ ┌───────────────────────┐
                            │ attach_chunk_metadata │
                            └───────────┬───────────┘
                                        ▼
              {'text', 'source', 'position', 'chunk_id'}
                                        │
                                        ▼
                              Part 3 · Embedding
```

Step 10 closes the provenance gap the chunkers open: they all take a string and
return strings, so a passage arrives at the index having forgotten which
document it came from. Note that `title` is still lost — a chunk record carries
`source` but nothing links it back to the document's human-readable name.
