# Step 47 · `handle_no_context`

> **Part 8 · Robustness, Caching, and Chat Memory** — step 47 of 51
> Code: [`rag_pipeline/robustness.py`](../../rag_pipeline/robustness.py) · Tests: [`tests/test_robustness.py`](../../tests/test_robustness.py)
> Previous: [Step 46 · `relevance_score`](../evaluation/step-46-relevance-score.md) · Next: [Step 48 · `deduplicate_chunks`](step-48-deduplicate-chunks.md)

---

## The task

```python
def handle_no_context(scored_chunks: list, threshold: float) -> dict: ...
```

Given scored chunks — either `(chunk, score)` tuples or dicts with a `'score'`
key — return `{'abstain': ..., 'message': ...}`. If no chunk strictly exceeds
the threshold, including the empty case, abstain with `'I do not know'`.
Otherwise `abstain=False` with an empty message.

---

## Why this step exists

Retrieval cannot fail. Ask for the top 5 and you get 5 chunks, ranked, whatever
the corpus contains — there is no "no results" case in a nearest-neighbour
search. Ask a corpus of cooking recipes about tax law and it will confidently
hand you the five recipes closest to tax law in embedding space.

Feed those to a generator with "answer using the context" and you get an answer.
The model has been given a question and some text; producing something is what
it does.

This is the guard. It reads the scores that
[`top_k_chunks`](../retrieval/step-18-top-k-chunks.md) kept precisely so that
somebody could make this decision, and turns a bad retrieval into a refusal
before any tokens are generated.

---

## What's happening

```python
scores = [_score_of(scored_chunk) for scored_chunk in scored_chunks]

if not scores or max(scores) <= threshold:
    return {"abstain": True, "message": REFUSAL}

return {"abstain": False, "message": ""}
```

**`not scores` first, and short-circuit ordering matters.** `max([])` raises
`ValueError`, and an empty retrieval is not exotic — it is what
[`filter_by_metadata`](../advanced-retrieval/step-40-filter-by-metadata.md)
returns when a tenant has no documents. Checking emptiness before calling `max`
handles it, and Python's `or` guarantees the second operand is never evaluated
when the first is true.

**`<= threshold`, so the comparison is strict.** A chunk must beat the bar, not
merely reach it. The guide's reasoning is that exact-threshold values are
usually noise, and the practical reason is that a threshold of 0.0 should not
admit a chunk with cosine exactly 0.0 — an orthogonal, entirely unrelated
passage.

**Only the maximum matters.** If the best chunk is not good enough, the rest are
worse. There is no accumulation of weak evidence here.

**`REFUSAL` is the shared constant**, the same string used by
[`build_prompt_template`](../generation/step-24-build-prompt-template.md) and
[`add_system_instruction`](../generation/step-27-add-system-instruction.md). Two
different mechanisms can now produce a refusal — this gate, and the model
following its instruction — and downstream code matching on one string catches
both. That is the entire reason for the constant.

### Reading either shape

```python
def _score_of(scored_chunk):
    if isinstance(scored_chunk, dict):
        return scored_chunk["score"]
    return scored_chunk[1]
```

Both shapes genuinely occur in this pipeline: `top_k_chunks` returns tuples,
while a reranker or an API client is more likely to return dicts. Accepting both
means the gate can sit at any point in the chain without an adapter.

---

## Choosing the threshold

There is no universal value, and the reason is worth understanding: **cosine
scores are not calibrated across models.** One embedding model may put unrelated
text around 0.1 and relevant text around 0.6; another may compress everything
into 0.7–0.95. A threshold tuned for one is meaningless for the other.

Worse, they are not calibrated across *queries* either — a long query and a
short one produce different score distributions against the same corpus.

The way to pick one is empirical: run [Part 7](../evaluation/00-overview.md)'s
eval set, look at the score distribution for queries with a gold chunk versus
those without, and put the threshold between them. Then re-derive it whenever
the embedding model changes.

Raising it means more abstentions and fewer wrong answers. Which error is worse
is a product decision, not a technical one.

---

## Boundaries of the contract

**It reports, it does not act.** Returning a dict rather than raising or
short-circuiting means the caller has to check `abstain` and actually stop. A
caller that ignores it gets no protection.

**Only the top score.** Five chunks at 0.19 against a threshold of 0.2 abstains,
even though the agreement between them is arguably evidence.

**No `KeyError` guard on dicts.** A dict without `'score'` raises, deliberately —
that is a programming error, not a data condition.

**Absolute scores only.** Cannot express "abstain if the top score is not much
better than the tenth", which is a more robust signal in practice.

**Not wired into [`rag_answer`](../generation/step-30-rag-answer.md).** Call it
yourself, before generating.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `max()` on an empty list | `ValueError`, and empty retrievals are routine after filtering. |
| `>=` instead of `>` | Admits exact-threshold matches, which are usually noise. |
| Supporting only one score shape | Breaks the moment a reranker is inserted. |
| Varying the refusal wording | Downstream abstention detection stops matching. |
| Reusing a threshold across embedding models | Scores are not calibrated between models. |
| Checking after generation | You have already paid for the tokens and risked the hallucination. |

---

## Example

```python
>>> handle_no_context([('a', 0.1), ('b', 0.15)], threshold=0.2)
{'abstain': True, 'message': 'I do not know'}
>>> handle_no_context([('a', 0.5)], threshold=0.2)
{'abstain': False, 'message': ''}
>>> handle_no_context([], threshold=0.2)
{'abstain': True, 'message': 'I do not know'}
```

---

## Where it fits

```
  retrieval ──▶ [(chunk, score), ...] ──▶ [ handle_no_context ]
                                              │            │
                                       abstain│            │proceed
                                              ▼            ▼
                                    "I do not know"    generation
```
