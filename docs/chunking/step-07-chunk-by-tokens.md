# Step 7 · `chunk_by_tokens`

> **Part 2 · Chunking Strategies** — step 7 of 51
> Code: [`rag_pipeline/chunking.py`](../../rag_pipeline/chunking.py) · Tests: [`tests/test_chunking.py`](../../tests/test_chunking.py)
> Previous: [Step 6 · `chunk_fixed_size`](step-06-chunk-fixed-size.md)

---

## The task

```python
def chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]: ...
```

Split text into consecutive chunks where each chunk corresponds to at most
`max_tokens` token ids under the provided Hugging Face tokenizer, returning the
decoded chunk strings. Empty input yields an empty list; text shorter than
`max_tokens` tokens yields a single chunk.

---

## Why this step exists

[`chunk_fixed_size`](step-06-chunk-fixed-size.md) bounds characters, but no
model has a character limit. Context windows, pricing, and truncation are all
measured in tokens, and the ratio between the two is not stable — roughly 4
characters per token for English prose, closer to 1 for CJK, and far worse for
code, base64, or long URLs.

That variance is the problem. A 1000-character chunk might be 250 tokens of
English or 900 tokens of Japanese. Size for the average and the CJK documents
silently overflow the model; size for the worst case and you waste most of the
window on everything else. Counting the same unit the model counts removes the
guesswork.

---

## What's happening

```python
token_ids = tokenizer.encode(text)

return [
    tokenizer.decode(token_ids[i : i + max_tokens])
    for i in range(0, len(token_ids), max_tokens)
]
```

Structurally this is [step 6](step-06-chunk-fixed-size.md) with one indirection:
encode, slice the id list exactly as before, decode each window back. The
slicing logic is identical — `range` with a stride, and Python's clamping
slices handling the short final window without a special case.

**`encode`, not `__call__`.** Calling a Hugging Face tokenizer directly
(`tokenizer(text)`) returns a `BatchEncoding` — a dict-like object with
`input_ids`, `attention_mask`, and possibly `token_type_ids`. Slicing that gives
you a slice of *dict keys*, not ids. `encode` returns the flat list of integers
this function wants.

**Encode once, not per chunk.** The text is tokenised a single time and the id
list is sliced. Tokenisation is the expensive part, so an implementation that
re-encodes per window does avoidable work proportional to the number of chunks.

---

## The chunks do not rejoin

The invariant that made step 6 easy to verify — `"".join(chunks) == text` — does
**not** hold here, and it is worth being precise about why.

Tokenisation is not a partition of the string into independent pieces. BPE
builds tokens by merging frequent character pairs, so a token can span what you
would call a word boundary, and the merge that produced it depends on
surrounding context. Cut the id sequence in the middle of such a merge and the
two halves decode to something that will not re-encode to the ids you started
with. Decoding also tends to normalise whitespace at the seams — leading spaces
that were part of a token may or may not survive.

So reconstruction is **approximate at the seams**. In practice this is a minor
distortion, and it does not matter much for retrieval, where each chunk is
embedded independently. It matters a great deal if you ever plan to map a chunk
back to a character offset in the source document — for highlighting a citation,
say. Do not build that on this function's output.

> The test suite exercises the slicing with a deliberately simple stub
> tokenizer that *does* round-trip exactly, so it verifies the windowing but not
> this seam behaviour. The seam distortion is a documented property of real BPE
> tokenizers, not something these tests demonstrate.

---

## The empty-input contract is the tokenizer's, not ours

The specification says empty input must yield an empty list. This
implementation satisfies it only because `encode("")` returns `[]` for GPT-2
style tokenizers — the empty id list makes `range` produce nothing.

That is a property of the tokenizer, not of the code. **A BERT-style tokenizer
wraps every encoding in special tokens**, so `encode("")` returns `[CLS], [SEP]`
— two ids, one window, one chunk containing nothing but markers:

```python
>>> chunk_by_tokens("", bert_style_tokenizer, 3)
['[CLS] [SEP]']
```

The same issue affects every chunk, not just the empty case: with special tokens
added, each chunk spends two of its `max_tokens` on markers, and every chunk
after the first carries a spurious `[CLS]`.

Passing `add_special_tokens=False` to `encode` — or an explicit
`if not text: return []` — would make the contract the function's own rather
than borrowed. It is left as-is because the guide's specification assumes the
GPT-2 behaviour, but this is the sharp edge to remember when swapping the
tokenizer.

---

## Boundaries of the contract

**Still no overlap, still no sentence awareness.** This changes the *unit* being
counted, not the strategy. Chunks still cut mid-sentence — just at a token
boundary rather than an arbitrary character.

**`max_tokens` must leave room.** It bounds the chunk, not the prompt. The
generator also has to fit a system prompt, the query, and several retrieved
chunks in the same window, so `max_tokens` should be a fraction of the model
limit, not the limit itself.

**The embedding model's tokenizer is what counts.** Chunks are embedded before
they are generated with, and the two models may tokenise differently. Sizing
against the wrong one gives a bound that does not hold where it matters.

**Invalid sizes behave as in step 6.** `max_tokens=0` raises `ValueError` from
`range`; a negative value returns `[]`, silently discarding the document. The
[same note](step-06-chunk-fixed-size.md#invalid-sizes) applies.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `tokenizer(text)` instead of `tokenizer.encode(text)` | Returns a `BatchEncoding` dict, not a list of ids. Slicing it yields key slices, and the failure is confusing rather than obvious. |
| Returning `[""]` for empty input | An empty chunk embeds to a meaningless vector that can still be retrieved. Zero chunks is the correct answer. |
| Re-encoding inside the loop | Tokenises the text once per chunk. Correct but needlessly slow on long documents. |
| Assuming chunks rejoin to the original | They do not. Any offset mapping built on that assumption drifts at every seam. |
| Ignoring special tokens | Every chunk silently loses two slots and gains markers that were never in the text. |
| Using a character estimate instead | "4 characters per token" is an English-prose average. It is wrong by 4× on CJK and unreliable on code. |

---

## Example

```python
>>> from transformers import AutoTokenizer
>>> tok = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')
>>> chunk_by_tokens('hello world this is a small example', tok, 3)
['hello world this', ' is a small', ' example']
>>> chunk_by_tokens('', tok, 3)
[]
```

The exact decoded strings depend on the tokenizer's whitespace handling — note
the leading spaces, which are part of the tokens rather than separators the
function inserted.

---

## Where it fits

```
  document text ──┬──▶ [ chunk_fixed_size ]  bound in characters
                  │
                  └──▶ [ chunk_by_tokens  ]  bound in tokens ◀── tokenizer
                                │
                                ▼
                            list[str] ──▶ Part 3 · Embedding
```

Two chunkers, one signature apart from the tokenizer argument — which is the
point. The pipeline can swap strategies without anything downstream changing,
so once evaluation exists in Part 7 the question "which chunker retrieves
better" becomes something to measure rather than argue about.
