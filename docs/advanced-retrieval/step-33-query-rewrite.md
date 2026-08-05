# Step 33 · `query_rewrite`

> **Part 6 · Advanced Retrieval Techniques** — step 33 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 32 · `append_source_references`](../generation/step-32-append-source-references.md) · Next: [Step 34 · `hyde_retrieve`](step-34-hyde-retrieve.md)

---

## The task

```python
def query_rewrite(query: str) -> str: ...
```

Turn a raw user query into a cleaner search string: lowercase,
whitespace-normalised, with leading conversational filler and trailing
punctuation removed. Reuse
[`normalize_text`](../ingestion/step-04-normalize-text.md).

---

## Why this step exists

`"Please could you tell me what RAG is?"` and `"what is rag"` are the same
information need and different vectors. The filler contributes tokens that mean
nothing about the topic, and in a short query they are most of the tokens — mean
pooling averages them straight into the result, dragging it toward wherever
polite phrasing lives in embedding space.

BM25 suffers differently: `rag?` and `rag` are distinct terms to a
whitespace tokenizer, so the punctuation alone can zero out a lexical match.

This is the cheapest possible fix. No model, no API call, just string work.

---

## What's happening

```python
text = normalize_text(query).lower()

stripped = True
while stripped:
    stripped = False
    for filler in FILLER_PREFIXES:
        if text.startswith(filler + " "):
            text = text[len(filler) + 1 :]
            stripped = True

return text.rstrip(TERMINAL_PUNCTUATION).strip()
```

**`normalize_text` first**, so NFKC folding and whitespace collapsing happen
before any prefix matching. `"Could   you"` with a double space would not match
`"could you "` otherwise.

**Lowercase after normalising.** Matching against lowercase filler phrases needs
the input lowercased, and lowercasing also helps BM25 term matching downstream.

**The loop, not a single pass.** Fillers stack: `"please tell me what is rag"`
needs `"please "` removed and *then* `"tell me "`. One pass over the list would
strip `"please "` and stop, or strip `"tell me "` from a position it no longer
occupies. Looping until nothing changes handles arbitrary combinations without
enumerating them.

**`filler + " "` is load-bearing.** The trailing space is what stops `"please"`
matching the start of `"pleasant weather"`. Without it, `startswith("please")`
is true and you get `"nt weather"`. There is a test for exactly this.

**`rstrip`, not `strip`, for punctuation.** Trailing only. A leading `?` is
strange enough that removing it is not obviously right, and `rstrip` handles
`"FAISS??"` — multiple marks — in one call.

---

## Only the front

The guide is explicit and it matters. `"show me documents that tell me about
X"` uses "tell me" as a genuine part of the sentence. Strip fillers anywhere
and you get `"show me documents that about x"`, which is worse than the input.

Leading position is a decent proxy for "this is packaging, not content". It is
not a perfect one — `"can you"` at the front of `"can you open this file"` is
arguably content — but the failure is mild and the alternative is unbounded.

---

## Boundaries of the contract

**A fixed, English-only list.** Five phrases. `"I'd like to know"`, `"do you
know"`, `"what can you tell me about"` all pass through. Extending the tuple is
the intended way to adapt it.

**Nothing semantic.** No synonyms, no expansion, no spelling correction. An
LLM-based rewriter would do far more; this is deterministic and free, which
makes it testable and safe to run on every query.

**Punctuation stays mid-query.** Only trailing marks go. `"what is rag, really?"`
keeps its comma.

**A query that is only filler survives.** `"please"` has no trailing space to
match, so it comes back as `"please"` rather than an empty string. Reasonable —
returning empty would retrieve nothing.

**Lowercasing costs case information.** Acronyms and proper nouns are flattened,
which is usually right for retrieval and would be wrong if you passed the
rewritten string to a generator. Keep the
[raw query for the prompt](../generation/step-30-rag-answer.md#the-original-query-goes-in-the-prompt).

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Matching a filler without the trailing space | `"please"` eats the start of `"pleasant"`. |
| Stripping fillers anywhere in the string | Mangles sentences that use the phrase meaningfully. |
| A single pass over the filler list | Stacked fillers survive. |
| Lowercasing before normalising | Prefix matching fails on odd whitespace or unicode variants. |
| `strip` instead of `rstrip` for punctuation | Also removes leading marks, which was not asked for. |
| Passing the rewritten query to the generator | The model should see what the user actually typed. |

---

## Example

```python
>>> query_rewrite('Please tell me what is RAG?')
'what is rag'
>>> query_rewrite('  Could you   please explain   FAISS??  ')
'explain faiss'
>>> query_rewrite('pleasant weather')
'pleasant weather'
>>> query_rewrite('show me documents that tell me about x')
'show me documents that tell me about x'
```

---

## Where it fits

```
  raw query ──┬──▶ [ query_rewrite ] ──▶ embed / BM25   (retrieval only)
              │
              └──────────────────────────▶ {question}   (the prompt, unchanged)
```
