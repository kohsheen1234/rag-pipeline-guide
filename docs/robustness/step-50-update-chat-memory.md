# Step 50 · `update_chat_memory`

> **Part 8 · Robustness, Caching, and Chat Memory** — step 50 of 51
> Code: [`rag_pipeline/robustness.py`](../../rag_pipeline/robustness.py) · Tests: [`tests/test_robustness.py`](../../tests/test_robustness.py)
> Previous: [Step 49 · `cache_query_embedding`](step-49-cache-query-embedding.md) · Next: [Step 51 · `rewrite_followup`](step-51-rewrite-followup.md)

---

## The task

```python
def update_chat_memory(history: list, user_message: str, assistant_message: str) -> list: ...
```

Append the new user turn and the new assistant turn to the history and return
the updated list. Each turn is `{'role': ..., 'content': ...}` with role either
`'user'` or `'assistant'`. Do not mutate the original; return a fresh list.

---

## Why this step exists

Everything up to here has been stateless: a query goes in, an answer comes out,
nothing is remembered. Chat is not like that. "How big is it?" only means
something relative to what was said before, and
[`rewrite_followup`](step-51-rewrite-followup.md) needs somewhere to read that
from.

The shape is deliberately the one every chat API already uses — a list of
`{role, content}` dicts. That means the same history can be fed to an OpenAI or
Anthropic messages endpoint, stored as JSON, or scanned by a rewriter, with no
conversion.

---

## What's happening

```python
return history + [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": assistant_message},
]
```

One expression. `list + list` builds a **new** list, which is the entire point.

### Why not `append`

The obvious version is:

```python
history.append({"role": "user", "content": user_message})       # don't
history.append({"role": "assistant", "content": assistant_message})
return history
```

It works, and it makes the function's return value a lie. The caller's list has
already changed by the time the value is returned, so:

- **Every reference sees the update.** Code holding the old history for
  comparison, logging, or a branch of the conversation finds it silently
  advanced.
- **Test order starts to matter.** A fixture reused across tests accumulates
  turns from whichever tests ran first, and failures depend on ordering.
- **`old = update_chat_memory(history, ...)` is meaningless.** `old is history`,
  so you cannot keep the previous state.

Returning a new list makes each history value immutable in practice, so a
conversation is a sequence of snapshots rather than one mutating object. Cheap
here — the copy is shallow, and the turn dicts are shared rather than
duplicated.

**User first, then assistant.** Chronological, and
[`rewrite_followup`](step-51-rewrite-followup.md) depends on it: it scans for
the *last* user turn, which is only the right one if turns are appended in the
order they happened.

---

## Boundaries of the contract

**Unbounded growth.** Nothing truncates. A long conversation grows until it
exceeds the model's context window, at which point whatever consumes it
truncates silently — usually from the front, dropping the earliest turns.
Windowing (`history[-20:]`) or summarising older turns is the usual fix and is
not here.

**No validation.** Empty strings, `None`, and a 10 MB message are all accepted.

**Two turns, always.** A user message with no reply — a failure, a timeout, an
[abstention](step-47-handle-no-context.md) — cannot be recorded alone. You would
append manually or pass the refusal as the assistant message.

**No system turn.** Only user and assistant. A system prompt would be prepended
elsewhere.

**No timestamps, ids, or metadata.** Not the retrieved sources either, so the
history records what was said and not what it was based on. Auditing a
conversation after the fact means correlating with separate logs.

**Shallow copy.** The new list contains the same turn dicts as the old one, so
mutating a turn in place is visible through every history that contains it.
Nothing here mutates them.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `history.append(...)` then returning it | Leaks state to every holder of the list; makes tests order-dependent. |
| Assistant turn first | Breaks chronology, and `rewrite_followup` picks the wrong turn. |
| Roles other than `user`/`assistant` | Chat APIs reject unknown roles. |
| Letting history grow without bound | Silent truncation at the context limit, from the wrong end. |
| Storing the answer without its sources | The conversation is not auditable later. |
| Assuming the returned turns are private | Shallow copy; the dicts are shared. |

---

## Example

```python
>>> hist = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]
>>> update_chat_memory(hist, 'what is RAG?', 'retrieval augmented generation')
[{'role': 'user', 'content': 'hi'},
 {'role': 'assistant', 'content': 'hello'},
 {'role': 'user', 'content': 'what is RAG?'},
 {'role': 'assistant', 'content': 'retrieval augmented generation'}]
>>> len(hist)
2
```

The original is still two turns long.

---

## Where it fits

```
  turn N:  query ──▶ retrieve ──▶ generate ──▶ answer
                                                  │
                          history ────────────────┴──▶ [ update_chat_memory ]
                                                                  │
  turn N+1: follow-up ──▶ [ rewrite_followup ] ◀───────────────────┘
```
