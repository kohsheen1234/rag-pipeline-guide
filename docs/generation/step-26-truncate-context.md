# Step 26 · `truncate_context`

> **Part 5 · Prompting and Answer Generation** — step 26 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 25 · `format_context`](step-25-format-context.md) · Next: [Step 27 · `add_system_instruction`](step-27-add-system-instruction.md)

---

## The task

```python
def truncate_context(context: str, max_chars: int) -> str: ...
```

Shrink a formatted context block so it never exceeds `max_chars`. Return it
unchanged if it already fits. Otherwise cut at a whitespace boundary just before
the limit; fall back to a hard cut if there is no whitespace in the budget.

---

## Why this step exists

The context window is a hard ceiling, and nothing upstream respects it.
[`format_context`](step-25-format-context.md) renders however many chunks
retrieval returned, at whatever size chunking produced. Multiply `k` by the
chunk size and it is easy to build a prompt the model cannot read.

What happens then is the reason to cap it deliberately: the model truncates it
for you, from whichever end its tokenizer decides, usually silently. Cutting
here means you choose what gets dropped, and you drop the lowest-ranked
passages, because `format_context` already put the best ones first.

---

## What's happening

```python
if len(context) <= max_chars:
    return context

window = context[: max_chars + 1]
cut = max(window.rfind(character) for character in " \n\t")

if cut == -1:
    return context[:max_chars]

return context[:cut]
```

**`max_chars + 1` is the interesting line.** The window extends one character
past the limit, and it has to. If the character sitting exactly at index
`max_chars` is a space, that space is a legitimate boundary — cutting there
yields exactly `max_chars` characters of content. Searching only `[:max_chars]`
would miss it and fall back to the previous space, throwing away a whole word of
budget for nothing.

Concretely, `truncate_context('the quick brown fox jumps', 15)`: the window is
`'the quick brown '` (16 chars, space at index 15), so the cut lands at 15 and
returns exactly `'the quick brown'`. Search `[:15]` instead and `rfind` finds
the space at index 9, returning `'the quick'`.

**`rfind`, not `find`.** The last whitespace before the limit, not the first.
Using `find` would return everything up to the first space — one word — which is
the other pitfall the guide names.

**`max(...)` over three characters** because
[`format_context`](step-25-format-context.md) joins with `\n`, not spaces. A
`rfind(" ")` alone would never find a line boundary, so a context block of long
unbroken lines would hard-cut mid-word every time. Tabs are included for
completeness.

**The result never includes the boundary character**, so there is no trailing
whitespace, and the length is at most `max_chars`. A test asserts that bound
across every budget from 1 to `len(text) + 5`.

---

## Boundaries of the contract

**Characters, not tokens.** Same caveat as
[`chunk_fixed_size`](../chunking/step-06-chunk-fixed-size.md): a character
budget is a proxy for a token budget, and the ratio varies by several times
across languages. Size conservatively.

**It cuts mid-passage.** The tail of the block is a partial `[4] some text...`
line with no closing source tag. The model sees a truncated citation. Dropping
whole numbered entries would be tidier, and is a different function.

**Not called by [`rag_answer`](step-30-rag-answer.md).** The specification for
step 30 does not include it, so a large `k` can still overflow. Wire it in
yourself between formatting and templating.

**Only the context is budgeted**, not the template, the instruction, or the
question. Leave headroom for those.

**A hard cut can split a multi-byte grapheme**, though not a UTF-8 character —
Python slices code points, not bytes.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Searching `[:max_chars]` instead of `[:max_chars + 1]` | Loses a word whenever the boundary sits exactly at the limit. |
| `find` instead of `rfind` | Cuts at the first space; you get one word. |
| Returning `context[:max_chars]` after finding a boundary | Off by one, and reintroduces the mid-word cut. |
| Only searching for `" "` | Never matches in a newline-joined block. |
| Forgetting the already-fits case | Cheap to check, and avoids scanning entirely. |
| Budgeting the context but not the prompt around it | The prompt still overflows. |

---

## Example

```python
>>> truncate_context('hello world', 50)
'hello world'
>>> truncate_context('the quick brown fox jumps', 15)
'the quick brown'
>>> truncate_context('abcdefghij', 5)
'abcde'
```

The third has no whitespace in the budget, so it falls back to a hard cut.

---

## Where it fits

```
  [ format_context ] ──▶ block ──▶ [ truncate_context ] ──▶ {context}
                                          │
                                          └── drops the lowest-ranked passages,
                                              because the best ones came first
```
