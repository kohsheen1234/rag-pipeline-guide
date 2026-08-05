# Step 45 · `faithfulness_score`

> **Part 7 · Evaluation** — step 45 of 51
> Code: [`rag_pipeline/evaluation.py`](../../rag_pipeline/evaluation.py) · Tests: [`tests/test_evaluation.py`](../../tests/test_evaluation.py)
> Previous: [Step 44 · `mean_reciprocal_rank`](step-44-mean-reciprocal-rank.md) · Next: [Step 46 · `relevance_score`](step-46-relevance-score.md)

---

## The task

```python
def faithfulness_score(answer: str, context_chunks: list) -> float: ...
```

Tokenise the answer and the concatenated context (normalised and lowercased),
and return the fraction of answer tokens that also appear somewhere in the
context. Return `0.0` for an empty answer.

---

## Why this step exists

The three retrieval metrics measure whether the right passages were found. This
is the first that asks what the model did with them.

The specific worry is ungrounded generation: the model answering from
pretraining rather than from the context, which is what
[`add_system_instruction`](../generation/step-27-add-system-instruction.md)
tries to prevent and cannot guarantee. If an answer contains a lot of words that
appear nowhere in the retrieved passages, something came from outside.

The appeal is that it costs nothing — no judge model, no reference answer, no
API call — so you can run it on every answer in a test suite or in production.

---

## What's happening

```python
answer_tokens = _tokens(answer)

if not answer_tokens:
    return 0.0

context = set(_tokens(" ".join(chunk["text"] for chunk in context_chunks)))
supported = sum(1 for token in answer_tokens if token in context)

return supported / len(answer_tokens)
```

**A list for the answer, a set for the context.** This asymmetry is the guide's
named pitfall and it is the only subtle line here.

The context becomes a set because all we ask of it is membership: *does this
token appear anywhere?* Position and count are irrelevant, and a set makes the
lookup constant time.

The answer stays a **list** so repetition counts. Use a set on both sides and an
answer of `"paris paris paris paris"` deduplicates to one token, and one
unsupported token repeated ten times costs the same as one repeated once. Keeping
the list means a degenerate, repetitive, ungrounded answer scores as badly as it
deserves. A test covers exactly this: `"cat zzz zzz zzz"` against a context
containing `cat` scores 0.25, not 0.5.

**Normalise both sides identically.** `_tokens` runs
[`normalize_text`](../ingestion/step-04-normalize-text.md) then `.lower()` then
`.split()`, applied to answer and context alike. Casing and whitespace
differences should not cost grounding points, and NFKC folding means a
fullwidth or ligature variant in one is not treated as a different word from the
other.

---

## What this actually measures

Being blunt about it, because the number invites over-reading.

**It measures lexical overlap, not truth.** A few consequences:

- An answer that copies the context verbatim scores 1.0 while contributing
  nothing.
- A correct answer phrased in the model's own words scores poorly. Paraphrase is
  penalised exactly as hard as invention.
- Word order is ignored entirely, so `"Paris is in France"` and `"France is in
  Paris"` are indistinguishable. A perfectly faithful-scoring answer can state
  the opposite of the context.
- Common words carry the score. `the`, `is`, `of`, and `a` will be in almost any
  context, so a long answer of connective tissue floats around 0.5 before saying
  anything. There is no IDF weighting here.
- Punctuation is attached to tokens, so `"France."` does not match `"France"`.
  Same behaviour as [`relevance_score`](step-46-relevance-score.md), inherited
  from `normalize_text` not stripping punctuation.

So a **high score is weak evidence** and a **low score is strong evidence**. If
half the answer's tokens are absent from the context, the model was
demonstrably drawing on something else. Use it as a floor test and a regression
detector — a sudden drop between runs means something changed — not as a
ranking of answer quality.

Proper faithfulness measurement means an LLM judge decomposing the answer into
claims and checking each against the context. This is the free approximation.

---

## Boundaries of the contract

**Empty answer scores 0.0**, not 1.0. Defensible either way — vacuously
grounded, or no evidence of grounding — and the specification chose 0. Note it
means a
[refusal](../generation/step-27-add-system-instruction.md) scores badly despite
being the correct behaviour, so filter abstentions out before averaging.

**Empty context makes everything unsupported**, giving 0.0.

**Requires `text` on each chunk.** `KeyError` otherwise.

**Whole-context matching.** A token is supported if it appears in *any* chunk,
so an answer stitching words from three unrelated passages scores well.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Using a set for the answer too | Repetition stops counting; degenerate answers score well. |
| Normalising only one side | Casing and unicode differences cost points spuriously. |
| Treating a high score as correctness | Verbatim copying scores 1.0 and says nothing. |
| Penalising a paraphrase | The metric cannot tell paraphrase from invention. |
| Averaging over refusals | They score 0 and drag the mean down for behaving correctly. |
| Expecting punctuation to be stripped | `normalize_text` does not strip it. |

---

## Example

```python
>>> ctx = [{'text': 'the cat sat on the mat'}]
>>> faithfulness_score('the cat sat', ctx)
1.0
>>> round(faithfulness_score('the dog sat', ctx), 4)
0.6667
```

`the` and `sat` are supported, `dog` is not: 2 of 3.

---

## Where it fits

```
  answer ──────────┐
                   ├──▶ [ faithfulness_score ] ──▶ "is this grounded in what we showed it?"
  retrieved chunks ┘
```
