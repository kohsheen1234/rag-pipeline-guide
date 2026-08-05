# Part 5 · Prompting and Answer Generation

Retrieval hands back passages. This part turns them into an answer, and tries to
keep that answer tied to the passages rather than to whatever the model already
believed.

## The problem this part is solving

A language model asked a question will answer it. It does not need your corpus
to do that, and it will not tell you when it stopped using it. Everything here
is arranged around making the model's job "read these passages and report what
they say" rather than "answer this question":

- the [template](step-24-build-prompt-template.md) puts the context first and
  the instruction next to the question
- the [system instruction](step-27-add-system-instruction.md) says *only* from
  the context, and names a refusal phrase
- the [context block](step-25-format-context.md) numbers the passages so an
  answer can point at one
- the [sources](step-31-track-source-chunk-ids.md) come back with the answer so
  a reader can check it

None of this is enforcement. A model can ignore all of it. It shifts the odds,
and Part 7 measures whether it worked.

## One refusal phrase

`"I do not know"` is defined once, in `generation.py`, and reused by the
template, the system instruction, and
[`handle_no_context`](../robustness/step-47-handle-no-context.md) in Part 8.

That is not tidiness. Abstention is detected downstream by string matching, so
every variant spelling — "I don't know", "unknown", "N/A" — is a hole in the
detection. Naming it once means the prompt and the checker cannot drift apart.

## Steps

All live in [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py).

| # | Function | What it does |
| --- | --- | --- |
| 24 | [`build_prompt_template`](step-24-build-prompt-template.md) | The template, with `{context}` and `{question}` slots. |
| 25 | [`format_context`](step-25-format-context.md) | Retrieved chunks as numbered, sourced lines. |
| 26 | [`truncate_context`](step-26-truncate-context.md) | Cut the block to a budget at a word boundary. |
| 27 | [`add_system_instruction`](step-27-add-system-instruction.md) | Prepend the grounding rules. |
| 28 | [`load_generator`](step-28-load-generator.md) | Load a causal LM and its tokenizer, with a pad token. |
| 29 | [`generate_answer`](step-29-generate-answer.md) | Greedy, deterministic decoding, prompt stripped. |
| 30 | [`rag_answer`](step-30-rag-answer.md) | Query in, grounded answer with sources out. |
| 31 | [`track_source_chunk_ids`](step-31-track-source-chunk-ids.md) | Project sources onto their ids, in order. |
| 32 | [`append_source_references`](step-32-append-source-references.md) | Append a `Sources: [...]` footer. |

## A gap worth knowing

Two of these are not wired into `rag_answer`:

- **`add_system_instruction`** is never called by it. `rag_answer` formats the
  template and generates. The grounding instruction inside the template does
  some of the same work, but the stronger system prefix is left to the caller.
- **`truncate_context`** is likewise not applied, so a large `k` can build a
  prompt longer than the model's context window, which then truncates it
  silently from the wrong end.

Both follow the specifications as written. They are noted here because a reader
assembling a real pipeline from these pieces should wire them in.

## Data flow

```
  Part 4 · Retrieval
         │
         ▼
  [(chunk, score), ...]
         │
         ▼
  [ format_context ] ──▶ "[1] text (source=a.txt)\n[2] ..."
         │
         ▼
  [ truncate_context ]        (available; not called by rag_answer)
         │
         ▼
  build_prompt_template().format(context=..., question=query)
         │
         ▼
  [ add_system_instruction ]  (available; not called by rag_answer)
         │
         ▼
  [ generate_answer ] ◀── model, tokenizer
         │
         ▼
  {"answer", "sources", "query"}
         │
         ▼
  [ append_source_references ] ──▶ "...\nSources: [c0, c1]"
```
