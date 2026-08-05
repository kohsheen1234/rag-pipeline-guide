# Step 9 · `chunk_with_overlap`

> **Part 2 · Chunking Strategies** — step 9 of 51
> Code: [`rag_pipeline/chunking.py`](../../rag_pipeline/chunking.py) · Tests: [`tests/test_chunking.py`](../../tests/test_chunking.py)
> Previous: [Step 8 · `chunk_by_sentences`](step-08-chunk-by-sentences.md) · Next: [Step 10 · `attach_chunk_metadata`](step-10-attach-chunk-metadata.md)

---

## The task

```python
def chunk_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]: ...
```

Slice text into sliding-window character chunks where each successive chunk
shares `overlap` characters with the previous one. The step between chunk
starts is `chunk_size - overlap`. The final chunk may be shorter if the text
runs out.

---

## Why this step exists

Every chunker so far cuts the text into disjoint pieces, which means every
boundary is a place where information can be destroyed. If a document says
*"The treatment was discontinued because of hepatic toxicity"* and the cut lands
after *"because of"*, then no chunk contains the causal link. A query asking why
the treatment stopped will not match anything useful, and the answer is
unreachable no matter how good the embedding model is.

Overlap buys insurance against that. By starting each window before the previous
one ended, any span shorter than `overlap` is guaranteed to appear intact
somewhere. The cost is redundancy — the index gets bigger and the same text is
embedded more than once.

---

## What's happening

```python
step = chunk_size - overlap

return [text[i : i + chunk_size] for i in range(0, len(text), step)]
```

Structurally this is [`chunk_fixed_size`](step-06-chunk-fixed-size.md) with one
line changed: the stride is `chunk_size - overlap` instead of `chunk_size`. Each
window is still `chunk_size` wide, but the starts advance more slowly, so
consecutive windows share their last and first `overlap` characters.

```
text:   a b c d e f g h i j
chunk 0 [a b c d]
chunk 1     [c d e f]        <- shares "cd" with chunk 0
chunk 2         [e f g h]    <- shares "ef" with chunk 1
chunk 3             [g h i j]
chunk 4                 [i j]
```

Setting `overlap=0` makes `step == chunk_size`, and the function reduces exactly
to fixed-size chunking. The tests assert that equivalence across a spread of
inputs rather than just asserting it in prose.

**Why `range` is the right driver.** The starts form an arithmetic sequence and
`range` stops once the start reaches `len(text)`, so no bounds arithmetic is
needed. As in step 6, slicing clamps, so the final short window needs no special
case.

---

## The redundant tail

The specified behaviour produces trailing chunks that are **entirely contained
in the chunk before them**. This is visible in the guide's own example:

```python
>>> chunk_with_overlap('abcdefghij', 4, 2)
['abcd', 'cdef', 'efgh', 'ghij', 'ij']
                                  ^^^^ already inside 'ghij'
```

`'ij'` adds nothing. It is a second index entry containing a strict subset of an
existing one — so it can be retrieved, ranked, and cited as though it were a
distinct passage, while being a worse version of one you already have.

With heavier overlap the tail gets worse, because each successive window
advances less than the amount by which it falls short of the end:

```python
>>> chunk_with_overlap('abcdefghij', 4, 3)
['abcd', 'bcde', 'cdef', 'defg', 'efgh', 'fghi', 'ghij', 'hij', 'ij', 'j']
                                                          ^^^^^  ^^^^  ^^^
                                                          three redundant chunks
```

The last of them is a single character, which will embed to something close to
noise.

This is specified behaviour rather than an implementation bug, so it is left in
place. But if you use this chunker for real, dropping any final chunk that is a
suffix of its predecessor is a one-line filter and pure gain. The general rule:
once the remaining text is shorter than `overlap`, the window has nothing new to
offer.

---

## Choosing the overlap

The parameter is a direct trade of index size against boundary safety, and the
arithmetic is worth internalising: the number of chunks scales as
`N / (chunk_size - overlap)`, so cost grows with the *inverse* of the step, not
linearly with the overlap.

| `chunk_size` | `overlap` | step | chunks for N=10,000 | vs no overlap |
| --- | --- | --- | --- | --- |
| 1000 | 0 | 1000 | 10 | 1× |
| 1000 | 100 | 900 | 12 | 1.2× |
| 1000 | 500 | 500 | 20 | 2× |
| 1000 | 900 | 100 | 100 | 10× |

Ten percent overlap costs almost nothing. Ninety percent multiplies your
embedding bill, your index size, and your query latency by ten — and fills the
top of every result list with near-duplicates of the same passage, crowding out
genuinely different material. Somewhere around 10–20% is the usual answer.

The overlap also needs to exceed the length of the spans you care about
preserving. An overlap of 20 characters does not protect a 100-character
sentence; it only guarantees that spans *shorter than the overlap* survive
intact somewhere.

---

## Invalid parameters

`overlap` must be strictly less than `chunk_size`, and the two ways of getting
it wrong fail differently:

```python
>>> chunk_with_overlap('abcdefgh', 4, 4)
ValueError: range() arg 3 must not be zero
>>> chunk_with_overlap('abcdefgh', 4, 5)
[]
```

Equal values give a step of zero — the window would never advance, an infinite
loop in any hand-rolled version — and `range` refuses. That is the good case.

`overlap > chunk_size` gives a negative step and **silently returns an empty
list**, discarding the document. Same failure mode as
[step 6's negative size](step-06-chunk-fixed-size.md#invalid-sizes), and worth
guarding at the call site if `overlap` is ever computed rather than hard-coded.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Stepping by `chunk_size` | The overlap parameter does nothing. Output looks plausible — correctly sized, in order, covering everything — so it passes a casual eyeball. Compare against `chunk_fixed_size` to catch it. |
| Stepping by `overlap` | The opposite error. With a small overlap the output explodes into near-duplicates; the count is the tell. |
| `text[i : i + step]` in the slice | Uses the step as the width. Chunks come out the wrong size with no overlap at all. |
| Assuming the chunks rejoin | They do not — overlapped regions appear twice. Total output length is greater than the input. |
| Expecting overlap to protect long spans | Only spans shorter than `overlap` are guaranteed intact. |
| Leaving the redundant tail in the index | Duplicate entries compete with their own superset for retrieval slots. |

---

## Example

```python
>>> chunk_with_overlap('abcdefghij', 4, 2)
['abcd', 'cdef', 'efgh', 'ghij', 'ij']
>>> chunk_with_overlap('abcdef', 3, 0)
['abc', 'def']
>>> chunk_with_overlap('abc', 10, 2)
['abc']
>>> chunk_with_overlap('', 4, 2)
[]
```

---

## Where it fits

```
  document text ──▶ [ chunk_with_overlap ] ──▶ list[str] ──▶ Part 3 · Embedding
                             │
                             └── windows of chunk_size, advancing by chunk_size - overlap
```

This is the last of the four chunking strategies, and the only one that trades
*storage* for retrieval quality rather than trading one kind of boundary for
another. It also composes with the others in principle — overlapping sentence
packing is the combination most production systems actually use — though the
guide keeps them separate here.
