# Step N · `function_name`

> **Part X · Part Title** — step N of 51
> Code: [`rag_pipeline/...`](../../rag_pipeline/...) · Tests: [`tests/...`](../../tests/...)
> Previous: [Step N-1 · `...`](step-NN-....md) · Next: [Step N+1 · `...`](step-NN-....md)

---

## The task

```python
def function_name(arg: T) -> U: ...
```

One paragraph restating the requirement precisely.

---

## Why this step exists

What breaks in the pipeline without it. Not "we need a function that does X" —
the concrete downstream failure this prevents.

---

## What's happening

The implementation, then a walkthrough of each decision in it. Prefer explaining
*why this line and not the obvious alternative* over restating what the line
does.

---

## Boundaries of the contract

What this step deliberately does not handle, and where that concern lives
instead. Omit if there is nothing interesting to say.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| The wrong-but-plausible version | The failure mode, especially if it is silent. |

---

## Example

```python
>>> function_name(...)
...
```

---

## Where it fits

```
  input ──▶ [ function_name ] ──▶ output ──▶ next step
```

One or two sentences on the invariant this step hands to the next one.
