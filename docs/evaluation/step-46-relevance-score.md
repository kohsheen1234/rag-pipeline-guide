# Step 46 · `relevance_score`

> **Part 7 · Evaluation** — step 46 of 51
> Code: [`rag_pipeline/evaluation.py`](../../rag_pipeline/evaluation.py) · Tests: [`tests/test_evaluation.py`](../../tests/test_evaluation.py)
> Previous: [Step 45 · `faithfulness_score`](step-45-faithfulness-score.md)

---

## The task

```python
def relevance_score(answer: str, question: str) -> float: ...
```

Normalise and lowercase both strings, split into word tokens, and return the
Jaccard similarity between the two token sets. Return `0.0` if both are empty.

---

## Why this step exists

[Faithfulness](step-45-faithfulness-score.md) checks the answer against the
*context*. It says nothing about whether the answer addresses the *question* — a
perfectly grounded summary of the wrong retrieved passage scores 1.0 there and
is useless.

This is the complementary check: does the answer engage with the entities and
concepts that were asked about? It catches the failure where retrieval went
sideways and the model faithfully answered a question nobody posed.

It is reference-free, so it needs no gold answer, which is why it can run on
production traffic rather than only on a labelled set.

---

## What's happening

```python
answer_tokens = set(_tokens(answer))
question_tokens = set(_tokens(question))
union = answer_tokens | question_tokens

if not union:
    return 0.0

return len(answer_tokens & question_tokens) / len(union)
```

Jaccard: intersection over union.

**Sets on both sides**, unlike
[faithfulness](step-45-faithfulness-score.md#whats-happening). Here repetition
genuinely does not matter — an answer mentioning "capital" three times is not
more on-topic than one mentioning it once. Using lists would also break the
`[0, 1]` bound, since the intersection of two multisets can exceed the size of
either.

**Union in the denominator** is what makes it symmetric and bounded. Dividing by
the answer size instead gives containment, which a one-word answer echoing the
question would max out.

**The empty-union guard** avoids `0/0`. Both empty returns 0.0 per the
specification — arguably they are identical and should score 1.0, but 0.0 is the
safer convention for a quality metric, since two empty strings are not a good
answer to anything.

**Same `_tokens` helper** as faithfulness: NFKC, lowercase, whitespace split.
Sharing it means the two metrics cannot disagree about what a token is.

---

## Symmetry is the problem

Jaccard treats both sides identically, and that is exactly wrong for this
purpose.

A question and its answer are *not* symmetric objects. A good answer contains
information the question does not — that is what makes it an answer. Jaccard
penalises precisely that: every new token the answer introduces grows the union
and lowers the score.

Work through the documented example:

```python
>>> relevance_score('Paris is the capital of France', 'What is the capital of France?')
0.5
```

The answer is essentially perfect. It scores 0.5, and the two tokens costing it
are `paris` — the actual answer — and `france?`, which fails to match `france`
because [`normalize_text`](../ingestion/step-04-normalize-text.md) collapses
whitespace but does not strip punctuation.

So the metric penalises the answer for containing the answer, and for the
question containing a question mark.

This means **a high score is suspicious rather than good.** The way to score 1.0
is to repeat the question back verbatim. Real answers land somewhere in the
middle, and the useful signal is a score near *zero*, which means the answer
shares almost no vocabulary with the question and probably went off-topic.

Read it as an off-topic detector with a low threshold, not as a quality ranking.

---

## Boundaries of the contract

**Stopwords dominate.** `the`, `is`, `of`, `a` match between any two English
sentences, putting a floor under the score that has nothing to do with
relevance. There is no IDF weighting.

**Punctuation blocks matches.** `france?` ≠ `france`. Stripping punctuation in
`_tokens` would improve this metric measurably, at the cost of diverging from
the specification and from `faithfulness_score`.

**No semantics.** A correct answer using entirely different vocabulary scores 0.
Cosine between the answer and question embeddings would capture what this is
reaching for; it would also need the model at evaluation time.

**No length normalisation.** A long, thorough answer scores lower than a terse
one purely for being long.

**Symmetric**, and asserted to be so in the tests, since that is a property of
Jaccard rather than an accident.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| List intersection instead of set | Double-counts repeats and breaks the `[0, 1]` bound. |
| Dividing by the answer size | That is containment, not Jaccard, and is trivially maxed. |
| Forgetting to lowercase | `Paris` and `paris` stop matching. |
| Expecting punctuation to be stripped | It is not, and it costs real matches. |
| Optimising for a high score | The maximum is achieved by parroting the question. |
| Using it as an answer-quality metric | It is an off-topic smoke alarm. |

---

## Example

```python
>>> relevance_score('Paris is the capital of France', 'What is the capital of France?')
0.5
>>> relevance_score('', 'anything')
0.0
```

Intersection `{is, the, capital, of}` = 4; union
`{paris, is, the, capital, of, france, what, france?}` = 8.

---

## Where it fits

```
  answer ────┐
             ├──▶ [ relevance_score ] ──▶ "did it engage with the question at all?"
  question ──┘
```

With this, Part 7 closes. You can now measure retrieval properly and answer
quality approximately, which is enough to compare two configurations and know
which one to keep. [Part 8](../robustness/00-overview.md) is what to do when the
answer is that neither is good enough.
