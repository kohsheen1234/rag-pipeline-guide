# Step 32 · `append_source_references`

> **Part 5 · Prompting and Answer Generation** — step 32 of 51
> Code: [`rag_pipeline/generation.py`](../../rag_pipeline/generation.py) · Tests: [`tests/test_generation.py`](../../tests/test_generation.py)
> Previous: [Step 31 · `track_source_chunk_ids`](step-31-track-source-chunk-ids.md)

---

## The task

```python
def append_source_references(answer: str, chunks: list) -> str: ...
```

Use [`track_source_chunk_ids`](step-31-track-source-chunk-ids.md) to collect the
ids, then append a line `Sources: [id1, id2, ...]` to the answer and return the
combined string.

---

## Why this step exists

[`rag_answer`](step-30-rag-answer.md) returns the sources as structured data,
which is right for a program and useless for anything that only shows the user a
string — a CLI, a log line, a chat bubble. This is the rendering step that puts
the provenance where a human will actually see it.

---

## What's happening

```python
ids = track_source_chunk_ids(chunks)

return f"{answer}\nSources: [{', '.join(str(chunk_id) for chunk_id in ids)}]"
```

**Bare ids, not quoted.** The guide's pitfall. The temptation is
`f"Sources: {ids}"`, which uses Python's list repr and gives
`Sources: ['c0', 'c1']` — quotes and all. Worse, the repr changes shape with the
id type: string ids get quotes, integer ids do not, so the same code produces
`['c0', 'c1']` in one corpus and `[0, 1]` in another. Anything parsing the
output has to handle both.

Joining explicitly with `str()` makes the rendering uniform: `[c0, c1]` and
`[0, 1]`, same shape either way.

**`\n`, not `\n\n`.** The footer attaches to the answer rather than reading as a
separate paragraph.

**The answer is untouched** — pure append, so multi-line answers keep their
structure.

**Delegation to step 31** means the ordering and duplicate-keeping decisions
live in one place. It also means this inherits step 31's
[key-name mismatch](step-31-track-source-chunk-ids.md#the-key-name-does-not-match-the-pipeline):
chunks from `attach_chunk_metadata` carry `chunk_id`, not `id`, so this renders
`Sources: []` on them.

---

## Boundaries of the contract

**Empty sources render as `Sources: []`** rather than being omitted. Honest —
it says explicitly that nothing supported the answer — and slightly odd-looking
in a user interface. Suppressing the footer when the list is empty is a
one-line change if you would rather.

**Not machine-readable.** An id containing `,` or `]` breaks any attempt to
parse this back. It is display output; keep the structured `sources` from
`rag_answer` for anything programmatic.

**Lists what was retrieved, not what was used.** The model may have ignored
every one of these chunks. The footer asserts availability of evidence, not use
of it — which is precisely the gap
[`faithfulness_score`](../evaluation/step-45-faithfulness-score.md) tries to
measure.

**No deduplication.** The same id twice appears twice, inherited from step 31.

**No links.** Ids only, no titles or paths. Rendering `doc1::0` as something a
reader can click needs the corpus, which this does not have — and the
[title is lost](../chunking/step-10-attach-chunk-metadata.md#two-record-shapes)
before it ever reaches here.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| `f"Sources: {ids}"` | Python's list repr: quotes on strings, none on ints. Output shape varies with id type. |
| `", ".join(ids)` without `str()` | `TypeError` on integer ids. |
| Prepending instead of appending | The answer should lead; citations follow. |
| Mutating the answer | It is a string; concatenate, do not try to edit. |
| Treating the footer as parseable | Ids can contain the delimiters. |
| Presenting it as proof of grounding | It lists what was shown to the model, not what it used. |

---

## Example

```python
>>> chunks = [{'id': 'c0', 'text': 'a'}, {'id': 'c1', 'text': 'b'}]
>>> append_source_references('The answer is 42.', chunks)
'The answer is 42.\nSources: [c0, c1]'
>>> append_source_references('a', [{'id': 0}, {'id': 1}])
'a\nSources: [0, 1]'
```

Integer and string ids render identically, which is the point.

---

## Where it fits

```
  rag_answer() ──▶ {"answer", "sources", ...}
                          │        │
                          └────────┴──▶ [ append_source_references ] ──▶ display string
```

This closes Part 5. The pipeline now takes a question and returns a grounded
answer with its evidence attached. Whether the grounding is real is
[Part 7](../evaluation/00-overview.md)'s question.
