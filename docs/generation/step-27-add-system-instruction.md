# Step 27 · `add_system_instruction`

> **Part 5 · Prompting and Answer Generation** — step 27 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 26 · `truncate_context`](step-26-truncate-context.md) · Next: [Step 28 · `load_generator`](step-28-load-generator.md)

---

## The task

```python
def add_system_instruction(prompt: str) -> str: ...
```

Prepend a fixed system instruction telling the model to answer only from the
context and to respond with the exact phrase `I do not know` when the context is
insufficient. Return the instruction, a blank line, then the original prompt
unchanged.

---

## Why this step exists

Left alone, a language model answers from its pretraining. That is the whole
hallucination problem in one sentence: the model has no way to distinguish "I
read this in the context" from "I already knew this", and no incentive to
prefer the former.

The system instruction does two separable things:

1. **Restricts the source.** Answer from the context, not from memory.
2. **Provides an exit.** Without a sanctioned way to fail, a model asked an
   unanswerable question will produce *something*. Naming a refusal gives it a
   licence to abstain.

The second is the more useful one, and it only works if the phrase is exact.

---

## What's happening

```python
return f"{SYSTEM_INSTRUCTION}\n\n{prompt}"
```

The instruction is a module constant:

```python
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. If the answer is not in the context, say 'I do not know'."
)
```

**The blank line** separates instruction from prompt. Models are sensitive to
structure, and a double newline reads as a section break in most training data.

**`ONLY` in capitals** is a small, real lever. Emphasis in a prompt measurably
shifts compliance, which is not a principled fact about language models so much
as an artefact of how instruction-tuned data is written.

**The prompt is untouched.** Prefix only, so this composes with anything —
a template-formatted RAG prompt, or a bare question.

### The refusal phrase is the contract

The guide's pitfall is worth taking literally. Downstream, abstention is
detected by string matching. Every variant spelling is a hole:

```python
if REFUSAL in answer:      # "I do not know"
    ...                    # "I don't know"  -> missed
                           # "I'm not sure"  -> missed
```

A missed abstention is worse than no abstention detection at all, because it
gets counted as a substantive answer and scored as one. So the phrase lives in
one constant, `REFUSAL`, used by this instruction, by
[the template](step-24-build-prompt-template.md), and by
[`handle_no_context`](../robustness/step-47-handle-no-context.md). Three call
sites, one spelling, no drift.

---

## Boundaries of the contract

**Not a system message.** Chat APIs have a real `system` role which models are
trained to weight differently. This is plain text prepended to a completion
prompt. Moving to a chat model means restructuring, not just reusing this
string.

**Not enforcement.** A model can ignore every word of this. It shifts
probability; it does not constrain. Measuring whether it worked is
[`faithfulness_score`](../evaluation/step-45-faithfulness-score.md)'s job.

**Duplicated guidance.** The same instruction appears, worded differently,
inside [the template](step-24-build-prompt-template.md). Belt and braces, at the
cost of tokens — and a place for the two to disagree if one is edited.

**Not called by [`rag_answer`](step-30-rag-answer.md)**, per that step's
specification. Apply it yourself around the formatted prompt.

**No injection defence.** A retrieved chunk saying "ignore previous
instructions" is downstream of this text in the prompt, which is the position
that tends to win.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Varying the refusal wording | Breaks every downstream abstention check silently. |
| Appending instead of prepending | Instructions before the task read as setup; after it they read as an afterthought. |
| Omitting the blank line | The instruction runs into the prompt and the boundary blurs. |
| Rebuilding the string per call | It is a constant. |
| Assuming the instruction is obeyed | It is a prior, not a guarantee. Measure it. |

---

## Example

```python
>>> add_system_instruction('Context: Paris is the capital of France.\nQuestion: What is the capital of France?')
"You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say 'I do not know'.\n\nContext: Paris is the capital of France.\nQuestion: What is the capital of France?"
```

---

## Where it fits

```
  template.format(...) ──▶ [ add_system_instruction ] ──▶ [ generate_answer ]
                                     │
                                     └── REFUSAL, shared with handle_no_context
```
