# Step 39 · `maximal_marginal_relevance`

> **Part 6 · Advanced Retrieval Techniques** — step 39 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 38 · `rerank_cross_encoder`](step-38-rerank-cross-encoder.md) · Next: [Step 40 · `filter_by_metadata`](step-40-filter-by-metadata.md)

---

## The task

```python
def maximal_marginal_relevance(query_vector, candidates, k=5, lambda_param=0.5) -> list: ...
```

Rerank L2-normalised candidate embeddings so the chosen set is both relevant and
internally diverse. Return `min(k, N)` indices in selection order. Similarity to
an empty selection is 0, so the first pick is the most relevant. Ties prefer the
smaller index.

---

## Why this step exists

Pure relevance ranking has a blind spot: it scores each candidate against the
query and never against the others. If your corpus contains five paraphrases of
one paragraph — which it does, because
[`chunk_with_overlap`](../chunking/step-09-chunk-with-overlap.md) manufactures
them by design and real corpora are full of boilerplate — then the top five
results are five copies of the same fact.

That wastes the entire context budget on one thing, and any question needing two
facts to answer cannot be answered.

MMR fixes it by making each pick depend on the picks before it.

---

## What's happening

```
next = argmax over unselected i of:  λ·relevance(i) − (1−λ)·max similarity(i, j) for j already selected
```

```python
relevance = candidates @ query_vector
pairwise = candidates @ candidates.T

selected, remaining = [], list(range(len(candidates)))

for _ in range(min(k, len(candidates))):
    best_index, best_score = None, None
    for index in remaining:
        redundancy = max((pairwise[index][chosen] for chosen in selected), default=0.0)
        score = lambda_param * relevance[index] - (1 - lambda_param) * redundancy
        if best_score is None or score > best_score:
            best_index, best_score = index, score
    selected.append(best_index)
    remaining.remove(best_index)
```

**Two matrix multiplies up front.** Because the candidates are unit vectors,
every cosine is a dot product, so `candidates @ query_vector` gives all
relevances and `candidates @ candidates.T` gives the full `(N, N)` pairwise
similarity matrix in one call each. The greedy loop then only does lookups.

**`default=0.0`** handles the empty selection. On the first iteration there is
nothing to be redundant with, so the redundancy term vanishes and the score is
`λ · relevance` — a monotone transform of relevance, so the first pick is always
the most relevant candidate regardless of `λ`. Tested across `λ` values.

**The redundancy term is recomputed every iteration.** This is what the guide
means by MMR being stateful, and it is the whole algorithm. A candidate's score
is not fixed: it drops the moment something similar to it is selected. Computing
the max once before the loop would give you relevance ranking with extra steps.

**Strict `>`, scanning `remaining` in ascending order**, so on a tie the earlier
index wins.

---

## What lambda does

`λ = 1` is pure relevance — the redundancy term is multiplied by zero, and the
output is just the top-k by cosine. `λ = 0` is pure diversity, ignoring the
query entirely after the first pick.

**There is a sharp edge at `λ = 0.5` worth knowing.** With unit vectors, a
*perfect* duplicate of an already-selected item scores:

```
0.5·1 − 0.5·1 = 0
```

and an orthogonal candidate with zero relevance scores:

```
0.5·0 − 0.5·0 = 0
```

Exactly equal. The two terms cancel, the tie-break by index applies, and the
duplicate is kept if it came first. So at `λ = 0.5` MMR does not reliably demote
even an exact duplicate — you need `λ < 0.5` for the diversity term to
outweigh. There is a test pinning both behaviours.

In practice `λ` between 0.3 and 0.7 is the usual range, and the useful reading
is: below 0.5 diversity dominates, above 0.5 relevance does.

---

## Boundaries of the contract

**Candidates must be L2-normalised.** The dot products are only cosines for unit
vectors. Unnormalised input gives magnitude-weighted similarities and a mix of
two incommensurable quantities. Nothing checks it.

**Greedy, not optimal.** Selecting the globally best diverse set is
combinatorial. Greedy is `O(k·N)` and good enough; it can be beaten.

**Returns indices.** Map them back yourself.

**`O(N²)` memory** for the pairwise matrix. Run it on the reranked top-50, not
the corpus.

**Redundancy uses max, not mean.** One near-duplicate is enough to suppress a
candidate, even if it differs from everything else selected. That is usually the
intent.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Computing redundancy once before the loop | Reduces to relevance ranking. The recomputation *is* MMR. |
| Unnormalised candidates | Dot products stop being cosines; the two terms are on different scales. |
| Expecting `λ = 0.5` to remove duplicates | The terms cancel exactly. Use `λ < 0.5`. |
| Running it over the whole corpus | `O(N²)` similarity matrix. |
| Forgetting the empty-selection case | `max()` on an empty sequence raises. |
| Mean instead of max for redundancy | A candidate similar to one selected item slips through. |

---

## Example

```python
>>> q = np.array([1.0, 0.0])
>>> C = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])
>>> maximal_marginal_relevance(q, C, k=3, lambda_param=1.0)
[0, 1, 2]
>>> maximal_marginal_relevance(q, C, k=3, lambda_param=0.3)
[0, 3, 2]
```

At `λ=1` it is plain relevance order. At `λ=0.3` the second pick becomes index
3 — the *least* relevant candidate — because it is maximally dissimilar to the
one already chosen.

---

## Where it fits

```
  reranked top-50 ──▶ [ maximal_marginal_relevance ] ──▶ 5 diverse indices ──▶ prompt
                                    │
                                    └── each pick changes the scores of the rest
```
