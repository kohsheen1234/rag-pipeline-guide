# Step 36 · `bm25_search`

> **Part 6 · Advanced Retrieval Techniques** — step 36 of 51
> Code: [`rag_pipeline/advanced_retrieval.py`](../../rag_pipeline/advanced_retrieval.py) · Tests: [`tests/test_advanced_retrieval.py`](../../tests/test_advanced_retrieval.py)
> Previous: [Step 35 · `reciprocal_rank_fusion`](step-35-reciprocal-rank-fusion.md) · Next: [Step 37 · `hybrid_search`](step-37-hybrid-search.md)

---

## The task

```python
def bm25_search(query, chunks, k=5, k1=1.5, b=0.75): ...
```

Score chunks against a query with a from-scratch BM25. Tokenise by lowercasing
and splitting on whitespace. Return up to `k` `(chunk_index, score)` tuples
sorted by descending score, omitting chunks with no query-term overlap. Use
`IDF = log((N - df + 0.5) / (df + 0.5) + 1)` so scores stay non-negative.

---

## Why this step exists

Dense retrieval fails on exactly the terms you most need matched literally: part
numbers, error codes, surnames, API names, anything the embedding model never
saw during training. It maps unknown tokens to something bland, so `"error
E4021"` retrieves passages about errors in general.

BM25 does not model meaning at all. It asks how often a query term appears in a
document, how rare that term is across the corpus, and how long the document is.
That is enough to nail exact matches, and useless for paraphrase — which is why
it pairs with dense retrieval rather than replacing it.

---

## What's happening

The score for a document is a sum over query terms:

```
score(q, d) = Σ_t  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
```

Three ideas, each doing one job.

**IDF — how rare is the term?**

```python
idf = math.log((total - document_frequency + 0.5) / (document_frequency + 0.5) + 1)
```

A term in every document carries no information; a term in one document is
highly discriminating. The `+ 1` inside the log is what keeps this non-negative:
without it, a term appearing in more than half the corpus produces a negative
IDF, and a document could be *penalised* for containing a query term.

**Term frequency, saturating.** `tf·(k1+1) / (tf + k1·...)` grows with `tf` but
flattens. Ten occurrences of a term does not make a document ten times more
relevant than one occurrence, and `k1` controls how fast the curve levels off.
This is the main thing BM25 has over raw TF-IDF.

**Length normalisation.** `(1 - b + b·|d|/avgdl)` sits in the denominator, so a
document longer than average is penalised. Without it, long documents win
everything by accumulating more matches. `b = 0.75` applies most, but not all,
of the correction.

### Document frequency counts documents

```python
document_frequency = sum(1 for other in documents if term in other)
```

`term in other` on a list is a membership test, so a term appearing four times
in one document contributes 1, not 4. The guide names this and it is the easiest
way to break IDF: inflate `df` and rare terms start looking common, flattening
the discrimination the metric exists to provide. A test checks that `"the"`,
which appears twice in each of two documents, still scores both identically —
which only holds if `df = 2` rather than 4.

**Zero-overlap chunks are omitted** rather than returned with a score of 0,
which is what `if score > 0` does. Note that
[`hybrid_search`](step-37-hybrid-search.md) has to undo this, since it needs one
entry per chunk.

---

## The guide's reference value does not reproduce

The guide's example gives `bm25_search('cat', chunks, k=2)` → `[(0, 0.5798...)]`
on this corpus:

```python
chunks = [{'text': 'the cat sat on the mat'}, {'text': 'the dog ran in the park'}]
```

This implementation returns **`0.6931`**, and no parameter choice produces
`0.5798` here. The reason is worth walking through, because it is a nice
property of the formula.

Both documents are 6 tokens, so `avgdl = 6` and `|d|/avgdl = 1` exactly. The
length-normalisation factor collapses:

```
1 - b + b·1  =  1 - b + b  =  1        for any b
```

The tf factor with `tf = 1` then becomes:

```
1·(k1 + 1) / (1 + k1·1)  =  (k1 + 1)/(k1 + 1)  =  1        for any k1
```

So the entire term reduces to the IDF, for **every** `k1` and `b`. The tests
verify this across a grid. And the IDF is:

```
log((2 - 1 + 0.5) / (1 + 0.5) + 1)  =  log(1.5/1.5 + 1)  =  log(2)  =  0.6931
```

There is no combination of parameters that yields `0.5798` on a corpus where
both documents are exactly average length. Either the quoted figure came from a
different corpus, or from a different IDF variant. `0.6931` follows from the
formula the guide itself specifies.

[Step 37](step-37-hybrid-search.md)'s reference output is consistent with this
implementation, which is some corroboration.

---

## Boundaries of the contract

**Tokenisation is `text.lower().split()`.** No stemming, so `"running"` and
`"run"` are unrelated. No punctuation stripping, so `"cat."` and `"cat"` are
different terms — the same issue
[`query_rewrite`](step-33-query-rewrite.md) addresses for trailing marks. No
stopword removal, though IDF largely handles that by giving common words near-zero
weight.

**Recomputed per query, `O(N · |q|)` with an inner scan for `df`.** A real
implementation builds an inverted index once. This is written for clarity; it
will not scale.

**Repeated query terms count twice.** `"cat cat"` doubles the contribution.
Arguably correct, arguably not.

**Scores are not comparable across corpora**, since IDF depends on `N` and `df`.
This is exactly why fusion needs
[normalisation](step-37-hybrid-search.md) or
[RRF](step-35-reciprocal-rank-fusion.md).

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Counting occurrences for `df` instead of documents | Inflates `df`, flattens IDF, breaks the whole ranking. |
| Omitting the `+ 1` inside the log | Common terms get negative IDF and *subtract* from the score. |
| Forgetting length normalisation | Long documents win by accumulating matches. |
| Comparing BM25 scores to cosine scores directly | Different scales entirely. Normalise or use RRF. |
| Expecting paraphrase to match | It is lexical. `"car"` does not match `"automobile"`. |
| Rebuilding `df` inside the loop at scale | Fine here, quadratic on a real corpus. |

---

## Example

```python
>>> chunks = [{'text': 'the cat sat on the mat'}, {'text': 'the dog ran in the park'}]
>>> bm25_search('cat', chunks, k=2)
[(0, 0.6931471805599453)]
```

The second document has no query term and is omitted rather than returned at 0.

---

## Where it fits

```
  query ──▶ [ bm25_search ] ──▶ [(index, score), ...]   lexical signal
                                        │
                                        ├──▶ [ hybrid_search ]  (mixed with dense)
                                        └──▶ [ reciprocal_rank_fusion ]
```
