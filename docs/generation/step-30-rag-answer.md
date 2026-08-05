# Step 30 · `rag_answer`

> **Part 5 · Prompting and Answer Generation** — step 30 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 29 · `generate_answer`](step-29-generate-answer.md) · Next: [Step 31 · `track_source_chunk_ids`](step-31-track-source-chunk-ids.md)

---

## The task

```python
def rag_answer(query, chunks, embeddings, embed_model, generator, tokenizer,
               k=5, max_new_tokens=64) -> dict: ...
```

Embed the query, score it against the chunk matrix, take the top-k, format them
into the prompt template, generate, and return `{'answer', 'sources', 'query'}`.
`sources` is the retrieved chunk dicts, without scores, in ranked order.

---

## Why this step exists

This is the wiring layer. Thirty steps of machinery, one call site.

The returned dict is the interesting part. Returning only the answer string
would be simpler and would make the system unauditable: there would be no way
to tell whether the model used the retrieved passages or invented something. By
returning the sources alongside, every answer carries the evidence it was shown,
and both a human reader and
[`faithfulness_score`](../evaluation/step-45-faithfulness-score.md) can check
one against the other.

---

## What's happening

```python
query_vector = embed_text(embed_model, query)
scores = cosine_similarity_search(query_vector, embeddings)
retrieved = top_k_chunks(scores, chunks, k)

prompt = build_prompt_template().format(
    context=format_context(retrieved),
    question=query,
)
answer = generate_answer(generator, tokenizer, prompt, max_new_tokens)

return {
    "answer": answer,
    "sources": [chunk for chunk, _ in retrieved],
    "query": query,
}
```

Retrieve, format, generate, package. Every line delegates.

### The original query goes in the prompt

The guide's pitfall, and a real one. There are several forms of the query in
flight by this point: the raw string, its embedding, and — if you have wired in
[`query_rewrite`](../advanced-retrieval/step-33-query-rewrite.md) or
[HyDE](../advanced-retrieval/step-34-hyde-retrieve.md) — a rewritten or
hypothetical variant used purely for search.

Those variants exist to retrieve better. They are lossy by design:
`query_rewrite` strips "could you" and lowercases; HyDE replaces the question
with a fabricated answer. Put one of those in `{question}` and the model is
answering a question the user did not ask, grounded in passages retrieved for a
question it is not being shown.

So the raw `query` is used twice, deliberately: in the prompt, and in the
returned dict. The embedding is the only place a transformed variant belongs.

### `sources` drops the scores

`[chunk for chunk, _ in retrieved]` — plain chunk dicts. The scores are useful
for [thresholding](../robustness/step-47-handle-no-context.md), and the caller
does not get them here, which is a real limitation: you cannot decide to abstain
after the fact from this return value. Abstention has to happen before calling
this.

---

## Boundaries of the contract

**No abstention.** The top `k` are used however weak they are. A query about
something absent from the corpus still produces an answer, built from the five
least-irrelevant chunks. Gate with
[`handle_no_context`](../robustness/step-47-handle-no-context.md) first.

**No system instruction.** [`add_system_instruction`](step-27-add-system-instruction.md)
is not applied — the specification for this step lists template formatting and
generation only. The template carries a weaker version of the same guidance.

**No truncation.** [`truncate_context`](step-26-truncate-context.md) is likewise
not applied, so `k=20` over large chunks can overflow the model's window and be
truncated from the wrong end.

**No caching, no dedup, no reranking.** All available in Parts 6 and 8, none
wired in here.

**Eight parameters.** The two models, the corpus in two parallel pieces, and
three knobs. That is a lot of state to thread, and the shape of it hints that a
`Corpus` object holding matrix, chunks, and the model that built them would
prevent several of the mismatches this pipeline warns about.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Putting the rewritten query in `{question}` | The model answers a different question than the user asked. |
| Returning only the answer | Unauditable: no way to tell grounding from invention. |
| Passing a mismatched matrix and chunk list | Row *i* resolves to the wrong text; the sources are confidently wrong. |
| Using a different embed model than built the matrix | Silent, and the rankings are noise. |
| A large `k` with no truncation | Silent prompt truncation, usually removing the instruction. |
| Assuming an answer means grounding | It means the model produced text. Measure it in Part 7. |

---

## Example

```python
>>> chunks = [{'id': 'c0', 'text': 'apple', 'source': 's1'},
...           {'id': 'c1', 'text': 'banana', 'source': 's2'}]
>>> embs = np.array([[1, 0], [0, 1]], dtype=np.float32)
>>> out = rag_answer('q', chunks, embs, FakeEmbed([1, 0]), FakeGen(), FakeTok(), k=1)
>>> out['query'], out['sources'][0]['id']
('q', 'c0')
```

---

## Where it fits

```
  query ──┬──▶ embed ──▶ score ──▶ top_k ──▶ format_context ──┐
          │                                                    ├──▶ template
          └────────────────────────────────────────────────────┘   (raw query)
                                                                        │
                                                                        ▼
                                                              [ generate_answer ]
                                                                        │
                                                                        ▼
                                            {"answer", "sources", "query"}
                                                                        │
                                      ┌─────────────────────────────────┤
                                      ▼                                 ▼
                       [ append_source_references ]        Part 7 · Evaluation
```
