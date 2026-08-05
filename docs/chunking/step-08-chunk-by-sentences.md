# Step 8 · `chunk_by_sentences`

> **Part 2 · Chunking Strategies** — step 8 of 51
> Code: [`rag_pipeline/chunking.py`](../../rag_pipeline/chunking.py) · Tests: [`tests/test_chunking.py`](../../tests/test_chunking.py)
> Previous: [Step 7 · `chunk_by_tokens`](step-07-chunk-by-tokens.md)

---

## The task

```python
def chunk_by_sentences(text: str, max_chars: int) -> list[str]: ...
```

Split text on sentence boundaries (`.`, `!`, `?`), then greedily pack whole
sentences into chunks so each chunk stays within `max_chars` when possible.
Sentences are joined with a single space inside a chunk. A single sentence
longer than `max_chars` is returned as its own chunk — never split
mid-sentence. Empty or whitespace-only input returns an empty list.

---

## Why this step exists

Steps 6 and 7 cut wherever the counter ran out, which means cutting mid-thought:

```python
>>> chunk_fixed_size('The patient was diagnosed with diabetes.', 20)
['The patient was diag', 'nosed with diabetes.']
```

Neither half means anything. Embed them and you get two vectors that represent
fragments, so a query about diabetes diagnosis matches both weakly and neither
well. The retrieval unit has to be a coherent thought, because the embedding of
a chunk is a summary of whatever is inside it — and half a sentence summarises
to noise.

Sentences are the smallest unit that reliably survives being read alone, which
makes them the natural thing to pack.

---

## What's happening

Two phases: segment, then pack.

### Segmenting

```python
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]*")
```

`[^.!?]+` takes a run of everything that is not a terminator, and `[.!?]*`
takes any terminators immediately following it. Because the terminators are
part of the match, they stay attached:

```python
>>> SENTENCE_PATTERN.findall('Hello world. How are you?')
['Hello world.', ' How are you?']
```

Dropping the terminator is the pitfall the guide names, and the consequence is
worse than cosmetic — a chunk reading `Hello world How are you` is run-on text
that reads as one confused sentence to both a human and a model.

The `*` rather than `?` on the terminator group handles `Wait!!!` as one
sentence instead of one sentence plus two empty ones. Each match is then
stripped, and empties are filtered out — which is what makes whitespace-only
input return `[]` rather than `['']`.

### Packing

```python
for sentence in sentences:
    if not current:
        current = sentence
    elif len(current) + 1 + len(sentence) <= max_chars:
        current += " " + sentence
    else:
        chunks.append(current)
        current = sentence
```

This is first-fit bin packing, in document order. The `+ 1` accounts for the
joining space — omit it and every chunk can end up one character over the
limit, which is exactly the kind of off-by-one that survives casual testing.

The `if not current` branch is doing the real work of the overlong-sentence
rule. An empty `current` accepts the next sentence **unconditionally**, without
consulting `max_chars`. So a 400-character sentence with a limit of 100 becomes
its own 400-character chunk rather than being dropped or split. Content is never
lost; the limit is what gives.

The trailing `if current` flushes the final chunk. Forgetting it silently drops
the last chunk of every document — and a test using only the guide's examples
would still pass, because they happen to end on a flush.

---

## The limit is a target, not a guarantee

Worth stating plainly, because it differs from steps 6 and 7: **a chunk can
exceed `max_chars`.** That is not a bug, it is the "do not split mid-sentence"
rule being honoured. Any downstream code that assumes a hard bound — a model
context window, a database column — needs its own check.

```python
>>> chunk_by_sentences('A very long single sentence that exceeds the limit.', 10)
['A very long single sentence that exceeds the limit.']
```

If a hard bound matters more than sentence integrity, the fix is to run an
overlong chunk through `chunk_fixed_size` afterwards, accepting the mid-sentence
cut only where it is unavoidable.

---

## Where the naive split goes wrong

`.!?` as sentence boundaries is a heuristic, and it is wrong in ways that show
up in ordinary prose.

**Decimals are corrupted.** This one does not just split badly, it *changes the
text*:

```python
>>> chunk_by_sentences('It costs 3.5 dollars.', 50)
['It costs 3. 5 dollars.']
```

The number is split into two "sentences" and then rejoined with a space
inserted. `3.5` has become `3. 5`. Any corpus with prices, versions, or
measurements is being quietly altered before it is ever embedded — and unlike a
bad chunk boundary, this is wrong even when everything fits in one chunk.

**Abbreviations split, but only sometimes.** `Dr.`, `Inc.`, `e.g.`, `U.S.` all
read as sentence ends:

```python
>>> chunk_by_sentences('Dr. Smith went home.', 50)
['Dr. Smith went home.']          # rejoined -- looks fine
>>> chunk_by_sentences('Dr. Smith went home.', 10)
['Dr.', 'Smith went home.']       # same text, different limit
```

The split always happens; whether you *see* it depends on whether packing
happened to put the pieces back together. That makes it a latent bug that
appears when someone tunes `max_chars` down.

**Leading terminators are dropped.** `[^.!?]+` requires at least one
non-terminator character, so terminators with nothing before them match nothing
and are discarded:

```python
>>> chunk_by_sentences('...leading dots then text.', 50)
['leading dots then text.']
```

Ellipses at the start of a quoted fragment vanish. Minor, but it is real content
loss.

**Ellipses mid-text merge sentences.** `Wait... what?` becomes one sentence,
which is arguably right, but it happens for the wrong reason.

Fixing any of this properly needs a real sentence segmenter — the abbreviation
problem in particular is not solvable with a regex, because `U.S.` ending a
sentence and `U.S.` mid-sentence are genuinely ambiguous without a lexicon.
`nltk.tokenize.sent_tokenize`, `pysbd`, and spaCy all handle it. The regex is
the right call for a from-scratch guide with no dependencies; it is not the
right call for a production corpus of prose with numbers in it.

---

## Boundaries of the contract

**Chunks do not rejoin to the original.** Whitespace between sentences is
normalised to a single space, leading terminators are gone, and decimals gain a
space. Unlike [`chunk_fixed_size`](step-06-chunk-fixed-size.md), there is no
reconstruction guarantee.

**Whitespace inside a sentence is untouched.** Only the joins are normalised:
`'Hi.  Lots   of   space.'` keeps its interior runs. Run
[`normalize_text`](../ingestion/step-04-normalize-text.md) first if that
matters.

**First-fit, not optimal.** Greedy packing in document order can leave a chunk
half empty when a long sentence comes next. Optimal bin packing would reorder
sentences, which would destroy the reading order that makes the chunk coherent
in the first place — so first-fit is the correct choice here, not a compromise.

**Characters, not tokens.** Same caveat as step 6: `max_chars` is not a model
limit.

**No overlap, no provenance.** As with the other chunkers.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `re.split(r'[.!?]', text)` | Discards the terminators. Chunks read as run-on text, and the trailing empty string from a text that ends in `.` becomes an empty chunk. |
| Forgetting the `+ 1` for the joining space | Every packed chunk can exceed `max_chars` by one character per join. |
| Forgetting the final flush | The last chunk of every document is silently dropped. The guide's examples still pass. |
| Skipping an overlong sentence | Content loss. It must be emitted whole even though it breaks the limit. |
| Checking `max_chars` before the first sentence | An overlong sentence produces an empty chunk, or an infinite loop in a `while` formulation. |
| Assuming chunks respect `max_chars` | They do not, by design. Check downstream if it matters. |
| Trusting the split on real prose | Decimals are actively corrupted, not merely split. |

---

## Example

```python
>>> chunk_by_sentences('Hello world. How are you? I am fine.', 30)
['Hello world. How are you?', 'I am fine.']
>>> chunk_by_sentences('Hello world. How are you? I am fine.', 20)
['Hello world.', 'How are you?', 'I am fine.']
>>> chunk_by_sentences('   ', 10)
[]
```

---

## Where it fits

```
  document text ──┬──▶ [ chunk_fixed_size    ]  characters, cuts anywhere
                  ├──▶ [ chunk_by_tokens     ]  tokens, cuts anywhere
                  └──▶ [ chunk_by_sentences  ]  characters, cuts at .!?
                                  │
                                  ▼
                              list[str] ──▶ Part 3 · Embedding
```

Three strategies, one signature. This is the first that tries to respect what
the text *means* rather than only how long it is — and it pays for that with a
limit it can no longer guarantee. Which trade is right is not answerable from
the code; it is what evaluation in Part 7 is for.
