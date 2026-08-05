# Step 6 · `chunk_fixed_size`

> **Part 2 · Chunking Strategies** — step 6 of 51
> Code: [`rag_pipeline/chunking.py`](../../rag_pipeline/chunking.py) · Tests: [`tests/test_chunking.py`](../../tests/test_chunking.py)
> Previous: [Step 5 · `make_document`](../ingestion/step-05-make-document.md)

---

## The task

```python
def chunk_fixed_size(text: str, chunk_size: int) -> list[str]: ...
```

Split a string into consecutive non-overlapping pieces of exactly `chunk_size`
characters, with the final piece possibly shorter. The concatenation of the
returned chunks must equal the original text, and every chunk must have length
between 1 and `chunk_size`.

---

## Why this step exists

Retrieval does not return documents; it returns passages. A 50-page manual is
useless as a retrieval unit — it will match almost any query weakly and give the
generator far more text than it can use. Before anything can be retrieved, long
documents have to be broken into units small enough to fit an embedding model's
context and specific enough to be worth ranking.

Fixed-size character windows are the crudest way to do that, and starting here
is deliberate. It establishes the interface — `str` in, `list[str]` out — that
the token, sentence, and overlapping chunkers all share, so the rest of the
pipeline can be built against a chunker that obviously works, and the strategies
can be swapped and compared later without anything downstream changing.

---

## What's happening

```python
return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
```

`range(0, len(text), chunk_size)` yields the start offset of each window: `0`,
`chunk_size`, `2 * chunk_size`, and so on, stopping before `len(text)`. That
gives exactly `ceil(len(text) / chunk_size)` windows.

The trailing partial chunk — the thing most easily got wrong — needs no special
case at all, because **Python slicing clamps instead of raising**. Where
indexing `text[10]` on a 5-character string is an `IndexError`, slicing
`text[3:10]` simply stops at the end and returns what is there:

```python
>>> "abcdefgh"[6:9]
'gh'
```

So the last window truncates itself. An implementation that reaches for
`math.ceil`, or that special-cases the remainder with an `if`, is working
around a problem the language already solved.

Two properties follow, and the tests assert both across a spread of lengths and
sizes rather than trusting a single example:

- **`"".join(chunks) == text`.** Nothing is lost and nothing is duplicated. This
  single assertion catches both classic bugs at once — a dropped tail makes the
  join too short, an overlapping window makes it too long.
- **`1 <= len(chunk) <= chunk_size`** for every chunk. No empty chunks, no
  oversized ones.

---

## Boundaries of the contract

**Chunks split mid-word, mid-sentence, mid-anything.**

```python
>>> chunk_fixed_size('hello world', 4)
['hell', 'o wo', 'rld']
```

This is the strategy's defining weakness, not a bug in the implementation. A
chunk ending `'...the patient was diagnosed with'` and the next beginning
`'diabetes...'` splits the one fact a query needed across two units, and neither
half matches well. Every later chunker in Part 2 exists to address some version
of this.

**No overlap.** Windows are strictly consecutive. Overlap — repeating the last
*n* characters at the start of the next chunk, so a fact spanning a boundary
survives in at least one chunk intact — is a separate strategy with its own
step. Mixing it in here would break the rejoin property that makes this version
easy to verify.

**Characters, not tokens.** `chunk_size` counts Python characters, which is not
what an embedding model's context limit measures. `"東京タワー"` is 5 characters
and considerably more tokens. Sizing chunks against a real model limit needs the
token chunker.

**Characters, not bytes, and not grapheme clusters.** Slicing operates on code
points, so multi-byte characters are never cut in half — good. But a character
plus its combining accent, or an emoji made of several code points, can be split
across a boundary. Rare, and usually harmless once the chunks are embedded
separately, but it is why the function is not safe for byte-exact round-tripping
of every possible input.

**No metadata.** This takes and returns bare strings, so the `source` and
`title` attached in [`make_document`](../ingestion/step-05-make-document.md) do
not travel with the chunks. Carrying provenance onto each chunk is a later
step's job — and until it exists, a retrieved chunk cannot be cited.

---

## Invalid sizes

The function does not validate `chunk_size`, and the two failure modes differ
sharply:

```python
>>> chunk_fixed_size('abcdef', 0)
ValueError: range() arg 3 must not be zero
>>> chunk_fixed_size('abcdef', -1)
[]
```

Zero fails loudly, which is fine. **A negative size silently returns an empty
list** — `range` with a negative step immediately stops, so the entire document
disappears with no error. In a batch ingest that means a document vanishes from
the index and the only symptom is that it never turns up in search results.

That is a real violation of this project's *fail loudly* rule, and it is
documented here rather than fixed only because the guide's contract for this
step specifies no validation. A single `if chunk_size < 1: raise ValueError(...)`
would close it. If a later step does not add one, this is the place to.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Dropping the trailing partial chunk | The last few characters vanish silently. A `while i + chunk_size <= len(text)` loop does exactly this. Caught immediately by the rejoin assertion. |
| Producing overlapping windows | Usually a stride bug — advancing by less than `chunk_size`. Content is duplicated in the index, so the same passage is retrieved twice under different ids. |
| Reaching for `math.ceil` and index arithmetic | Slicing already clamps. The extra arithmetic is where off-by-one errors get in. |
| `textwrap.wrap(text, chunk_size)` | Looks like the right tool and is not — it breaks on words, collapses whitespace, and drops empty lines, so the chunks no longer rejoin to the original. |
| Testing with one example | `'abcdefgh'` at size 3 passes for both a correct implementation and several wrong ones. The exact-multiple case and the empty-string case are where they diverge. |
| Assuming `chunk_size` bounds the token count | It bounds characters. For CJK text the token count can be several times higher. |

---

## Example

```python
>>> chunk_fixed_size('abcdefgh', 3)
['abc', 'def', 'gh']
>>> chunk_fixed_size('hello', 5)
['hello']
>>> chunk_fixed_size('', 3)
[]
>>> chunk_fixed_size('abcdef', 3)     # exact multiple, no short chunk
['abc', 'def']
```

---

## Where it fits

```
  document text ──▶ [ chunk_fixed_size ] ──▶ list[str] ──▶ Part 3 · Embedding
                            │
                            └── ceil(N / c) windows, no overlap, rejoins exactly
```

This is the first of several interchangeable chunkers. They all share the same
signature, which is what lets the pipeline treat chunking as a swappable policy
rather than a fixed behaviour — and what makes it possible, later, to measure
which strategy actually retrieves better instead of arguing about it.
