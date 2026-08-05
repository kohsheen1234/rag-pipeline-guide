# Step 24 · `build_prompt_template`

> **Part 5 · Prompting and Answer Generation** — step 24 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 23 · `save_faiss_index`](../retrieval/step-23-save-faiss-index.md) · Next: [Step 25 · `format_context`](step-25-format-context.md)

---

## The task

```python
def build_prompt_template() -> str: ...
```

Return a RAG prompt template containing the literal placeholders `{context}`
and `{question}`, each exactly once, readable by `str.format`.

---

## Why this step exists

The prompt is where grounding is either established or lost, and it needs to be
identical for every query. Defining it in one function means there is one thing
to change when the instruction needs tightening, and one thing to point at when
asking why the model behaved a certain way.

Returning a *template* rather than a formatted string is what makes it reusable:
the retrieval result and the question arrive later, per query.

---

## What's happening

```python
return (
    "Context:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer the question using only the context above. "
    "If the context does not contain the answer, say '" + REFUSAL + "'.\n"
    "Answer:"
)
```

**A plain string, never an f-string.** This is the pitfall that matters. An
f-string interpolates at definition time, so `f"{context}"` would either raise
`NameError` or, worse, capture some unrelated local variable and leave no
placeholder behind. The braces have to survive until `.format` is called.

**Context first, then the question, then the instruction.** The ordering is
deliberate. Models attend more reliably to text near the end of the prompt, so
the instruction sits after the evidence rather than before it, where a long
context would push it far into the past. The trailing `Answer:` gives the model
an obvious continuation point, which matters for base-style models that are
completing text rather than following a chat format.

**`REFUSAL` is interpolated at build time**, not left as a placeholder — it is a
module constant, the same one
[`add_system_instruction`](step-27-add-system-instruction.md) and
[`handle_no_context`](../robustness/step-47-handle-no-context.md) use. One
spelling, three call sites.

---

## Boundaries of the contract

**No literal braces.** Any `{` or `}` meant literally would need doubling as
`{{` or `}}`, or `.format` raises `KeyError`. There are none here, and a test
formats with `{}` as both values to prove nothing else breaks.

**Both slots are required.** `.format(context=...)` alone raises `KeyError:
'question'`. There is no default.

**No length management.** The template does not know how big the context is.
See [`truncate_context`](step-26-truncate-context.md).

**No chat roles.** This is a flat completion prompt. A chat model would want
`[{"role": "system", ...}, {"role": "user", ...}]`, which is a different shape
entirely — worth knowing if you swap the local model for an API.

**Prompt injection is not addressed.** Retrieved text goes straight into
`{context}`. A corpus document containing "ignore the above and say X" is simply
part of the prompt. Delimiting or escaping retrieved content is a real concern
this template does not handle.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Using an f-string | Interpolates immediately; no placeholders survive. |
| Unescaped literal braces | `KeyError` at `.format` time, per query. |
| Misspelling a placeholder | `{questions}` fails at the call site, not here. |
| Duplicating a placeholder | Works, but the same text appears twice and doubles the token cost. |
| Putting the instruction before a long context | Gets buried; models weight nearby text more heavily. |
| Building the template per call | It is a constant. Nothing changes between queries. |

---

## Example

```python
>>> tmpl = build_prompt_template()
>>> '{context}' in tmpl and '{question}' in tmpl
True
>>> tmpl.format(context='Paris is the capital of France.',
...             question='What is the capital of France?')[:7]
'Context'
```

Filled in, it reads:

```
Context:
Paris is the capital of France.

Question: What is the capital of France?

Answer the question using only the context above. If the context does not
contain the answer, say 'I do not know'.
Answer:
```

---

## Where it fits

```
  [ format_context ] ──▶ context ──┐
                                   ├──▶ template.format(...) ──▶ [ generate_answer ]
  user question ───────────────────┘
```
