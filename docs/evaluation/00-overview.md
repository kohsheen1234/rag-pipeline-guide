# Part 7 · Evaluation

Parts 2 through 6 offered choices: four chunkers, two retrieval backends, eight
techniques for improving retrieval, a handful of parameters each. Nothing so far
tells you which combination is any good.

This part is how you stop guessing.

## Retrieval metrics vs answer metrics

The two halves of this part are not equally trustworthy, and it is worth being
blunt about that up front.

**The retrieval metrics** ([hit rate](step-42-hit-rate-at-k.md),
[recall@k](step-43-recall-at-k.md), [MRR](step-44-mean-reciprocal-rank.md)) are
real. Given gold chunk ids, they measure exactly what they claim. They are the
standard measures from information retrieval, and if recall is low nothing
downstream can save you — the answer is not in the context, so the generator can
only invent it.

**The answer metrics** ([faithfulness](step-45-faithfulness-score.md),
[relevance](step-46-relevance-score.md)) are token-overlap heuristics. They are
free, deterministic, and blunt. An answer that copies the context verbatim
scores 1.0 for faithfulness while saying nothing useful; a correct answer
phrased in its own words scores poorly. Treat them as smoke detectors, not
judges: a sudden drop means something changed, and a high score does not mean
the answer is good.

The honest ceiling here is that measuring answer quality properly needs either a
human or a strong model as a judge. This part gives you what you can compute
with no dependencies.

## What the three retrieval metrics each tell you

They form a ladder, and each answers a question the previous one cannot.

| Metric | Question | Blind to |
| --- | --- | --- |
| Hit rate@k | Did *anything* useful reach the top k? | How many, and how high |
| Recall@k | What *fraction* of the gold chunks reached the top k? | Where in the list they landed |
| MRR | How *high* was the first useful one? | Anything after the first hit |

A system can score 1.0 on hit rate and be poor: the right chunk is in the top
10, at position 9, beneath eight irrelevant ones that will crowd it out of the
prompt. MRR catches that. Conversely MRR can look fine while recall is bad, if
one gold chunk ranks first and three others never appear.

Read them together.

## Steps

All live in [`rag_pipeline/evaluation.py`](../../rag_pipeline/evaluation.py).

| # | Function | What it measures |
| --- | --- | --- |
| 41 | [`build_eval_set`](step-41-build-eval-set.md) | Nothing — it is the ground truth everything else consumes. |
| 42 | [`hit_rate_at_k`](step-42-hit-rate-at-k.md) | Fraction of queries with any gold chunk in the top k. |
| 43 | [`recall_at_k`](step-43-recall-at-k.md) | Mean fraction of gold chunks found. |
| 44 | [`mean_reciprocal_rank`](step-44-mean-reciprocal-rank.md) | How high the first gold chunk ranked. |
| 45 | [`faithfulness_score`](step-45-faithfulness-score.md) | How much of the answer appears in the context. |
| 46 | [`relevance_score`](step-46-relevance-score.md) | Token overlap between answer and question. |

## Using them

The point is comparison, not absolute numbers. `recall@5 = 0.6` means nothing on
its own; `0.6` with fixed-size chunks against `0.8` with sentence chunks, on the
same eval set, is a decision.

That also means the eval set has to stay fixed. Change the questions and the
scores stop being comparable across runs, which is much of why
[step 41](step-41-build-eval-set.md) hard-codes them rather than generating
them.

One trap worth naming: the gold ids are
[chunk ids](../chunking/step-10-attach-chunk-metadata.md), and chunk ids are
**not stable across chunking configurations**. Re-chunk with a different size
and `doc1::7` points at different text, so your labels now mark the wrong
passages — silently, with plausible-looking scores. Comparing two chunkers
honestly means relabelling, or defining gold at the document level.

## Data flow

```
  [ build_eval_set ] ──▶ questions, gold ids
            │
            ▼
     run the retriever
            │
            ▼
     retrieved id lists ──┬──▶ [ hit_rate_at_k ]
                          ├──▶ [ recall_at_k ]
                          └──▶ [ mean_reciprocal_rank ]

     generated answer ────┬──▶ [ faithfulness_score ] ◀── retrieved chunks
                          └──▶ [ relevance_score ]    ◀── the question
```
