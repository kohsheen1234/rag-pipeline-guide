# Step 25 · `format_context`

> **Part 5 · Prompting and Answer Generation** — step 25 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 24 · `build_prompt_template`](step-24-build-prompt-template.md) · Next: [Step 26 · `truncate_context`](step-26-truncate-context.md)

---

## The task

```python
def format_context(retrieved: list) -> str: ...
```

Render a list of `(chunk_dict, score)` tuples as one context block, one chunk
per line, formatted `[i] {text} (source={source})` with `i` 1-based. Return an
empty string when nothing was retrieved.

---

## Why this step exists

Retrieval produces a Python list of tuples. A prompt is one string. This is the
flattening, and the two decisions inside it — number the passages, name their
sources, drop the scores — each earn their place.

**Numbering** gives the model a handle. A model that can write "according to
[2]" produces an answer a reader can check against a specific passage, and Part
7's citation checks have something to match on. Unnumbered passages can only be
cited in aggregate.

**Sources** put the provenance where the model can see it, so the filename can
appear in the answer rather than only in the returned metadata.

---

## What's happening

```python
return "\n".join(
    f"[{position}] {chunk['text']} (source={chunk['source']})"
    for position, (chunk, _) in enumerate(retrieved, start=1)
)
```

`enumerate(..., start=1)` handles the 1-based index directly, rather than
writing `i + 1` at the point of use where it is easier to forget. Humans count
citations from 1, and the numbering is for a reader as much as for the model.

`(chunk, _)` unpacks the tuple and discards the score in the same expression.
Dropping the score is the guide's named pitfall and it is a real one: the score
is a float that helps nobody inside the prompt. It cannot be verified by the
model, it invites spurious reasoning about confidence the number does not
support, and at 8 significant figures across 5 chunks it is pure token cost.
Scores are for [thresholding](../robustness/step-47-handle-no-context.md) and
display, outside the prompt.

`"\n".join(...)` over a generator, rather than `+=` in a loop — one allocation
instead of one per chunk, and no trailing newline to trim.

An empty list joins to `""` naturally, so the empty case needs no branch.

---

## Boundaries of the contract

**Requires `text` and `source`.** A chunk missing either raises `KeyError`. Note
that chunks from
[`attach_chunk_metadata`](../chunking/step-10-attach-chunk-metadata.md) have
both, but their more specific `chunk_id` is *not* rendered — the line shows the
document, not the exact chunk, so two chunks from one file are indistinguishable
in the prompt.

**One line per chunk.** A chunk containing newlines breaks that assumption
visually, and a chunk that has been through
[`normalize_text`](../ingestion/step-04-normalize-text.md) will not have any.

**No truncation.** Ten large chunks produce a very large block. See
[`truncate_context`](step-26-truncate-context.md).

**No escaping.** Retrieved text is inserted verbatim, including any text that
looks like an instruction. Same prompt-injection caveat as
[step 24](step-24-build-prompt-template.md).

**Empty retrieval gives `""`**, which formats into the template as a blank
context. The model then sees a prompt with no evidence and an instruction to use
only the evidence — which is the right setup for a refusal, but abstaining
before generating is cheaper.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| 0-based indices | Citations read `[0]`, which no reader expects. |
| Including scores | Wastes tokens and invites the model to reason about numbers it cannot verify. |
| Concatenating in a loop | Quadratic, and usually leaves a trailing newline. |
| Rendering the whole chunk dict | Puts `position` and `chunk_id` into the prompt as noise. |
| Assuming the block is short | It is as long as the retrieved chunks; nothing here caps it. |

---

## Example

```python
>>> retrieved = [({'text': 'Cats purr.', 'source': 'a.txt'}, 0.9),
...              ({'text': 'Dogs bark.', 'source': 'b.txt'}, 0.7)]
>>> print(format_context(retrieved))
[1] Cats purr. (source=a.txt)
[2] Dogs bark. (source=b.txt)
>>> format_context([])
''
```

---

## Where it fits

```
  [(chunk, score), ...] ──▶ [ format_context ] ──▶ "[1] ... \n[2] ..." ──▶ {context}
```
