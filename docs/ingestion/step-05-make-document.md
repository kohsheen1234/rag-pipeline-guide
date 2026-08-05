# Step 5 · `make_document`

> **Part 1 · Document Ingestion & Preprocessing** — step 5 of 51
> Code: [`rag_pipeline/ingestion.py`](../../rag_pipeline/ingestion.py) · Tests: [`tests/test_ingestion.py`](../../tests/test_ingestion.py)
> Previous: [Step 4 · `normalize_text`](step-04-normalize-text.md)

---

## The task

```python
def make_document(text: str, source: str, title: str) -> dict: ...
```

Wrap a raw text string together with its provenance metadata into a single
document dictionary, with the keys `'text'`, `'source'`, and `'title'`, in that
order, mapping to the corresponding arguments.

---

## Why this step exists

Downstream stages all need to know where a piece of text came from. Chunking has
to carry provenance onto each chunk. Retrieval has to return it alongside the
match. Citation has to render it. If provenance is not attached at ingestion, it
cannot be recovered later — by the time a chunk reaches the generator, the file
it came from is long out of scope.

The second reason is uniformity. Fixing the shape now means every later function
takes and returns the same record type, so they compose without adapters. A
three-line function earns its place by being the thing 46 later steps agree on.

---

## What's happening

```python
document = {
    "text": text,
    "source": source,
    "title": title,
}

return document
```

There is no cleverness here, and that is the point. The three things worth
noticing are what the function deliberately does *not* do.

**It does not validate.** `None`, `""`, and a 40 MB string all pass through
untouched. Deciding what counts as a usable source is the caller's problem —
different corpora have genuinely different answers, and a loader that rejects
them here would have to be worked around rather than used.

**It does not transform.** The text is stored exactly as handed in. If you want
it normalised, call [`normalize_text`](step-04-normalize-text.md) first. Folding
normalisation in here would make it impossible to build a document holding raw
text, which the pipeline may well want for display or highlighting.

**It builds a fresh dict every call.** The literal is evaluated on each
invocation, so two calls never share state and callers can mutate the result
freely. (Contrast a mutable default argument, the classic version of this bug.)

---

## Why a flat dict

The pitfall the guide names — nesting under `'meta'`, or renaming `'text'` to
`'content'` — is worth understanding as a design choice rather than an
arbitrary rule.

**Flat beats nested** because every access site stays one subscript deep.
`doc["source"]` in citation code, versus `doc["meta"]["source"]` with a
`KeyError` waiting whenever a document was built by a path that forgot the
sub-dict. Nesting buys namespacing, and with three keys there is nothing to
namespace.

**Fixed key names** are the actual contract. `'text'` versus `'content'` is a
coin flip in isolation; what matters is that all 51 steps flip it the same way.
A stage that emits `'content'` fails at the *next* stage, not its own, which is
the most annoying class of bug to trace.

**A dict rather than a class.** The honest trade-off:

| | dict | dataclass / NamedTuple |
| --- | --- | --- |
| JSON-serialisable | Directly — matters when writing to a vector store's metadata field | Needs conversion |
| Typo protection | None. `doc["titel"]` is a runtime `KeyError` | `doc.titel` caught by a type checker |
| Extra fields later | Just add a key | Schema change |

The dict wins on serialisation, which is the constraint that bites soonest —
most vector stores take a plain JSON object per record. If the typo exposure
starts to hurt, `typing.TypedDict` gives static checking while staying a plain
dict at runtime, with no change to this function or any caller.

---

## The provenance gap

Worth noticing now, because it will come up:
[`load_text_directory`](step-02-load-text-directory.md) returns `list[str]`. It
reads filenames, sorts by them — and then throws them away.

So the `source` that `make_document` needs is not available from the loader as
currently written. Building documents from a directory today means re-deriving
the filename list yourself:

```python
filenames = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
texts = load_text_directory(d)
documents = [
    make_document(normalize_text(t), f, f) for t, f in zip(texts, filenames)
]
```

That works — and it works *only* because both functions sort the same way, which
is precisely the ordering contract step 2 established. But duplicating the scan
is a smell. Expect a later step to introduce a directory loader that returns
documents directly rather than bare strings.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Nesting under `'meta'` | Every consumer needs a second subscript, and any document built without the sub-dict raises `KeyError` at the point of *use*, far from the point of construction. |
| Renaming `'text'` to `'content'` | Breaks the contract silently. The failure surfaces in a later stage, not this one. |
| Testing key order with `==` | Dicts compare equal regardless of insertion order, so `==` cannot catch an ordering regression. Assert on `list(doc)`. |
| Normalising inside this function | Makes it impossible to hold raw text, and hides a transformation the caller cannot see or skip. |
| A mutable default like `def make_document(text, source, title, extra={})` | The default dict is created once at definition time and shared across every call. Not present here — worth knowing why it isn't. |
| Reusing one dict and mutating it in a loop | Every document in the list ends up being the same object with the last values. |

---

## Example

```python
>>> make_document('Hello world.', 'notes.txt', 'Greeting')
{'text': 'Hello world.', 'source': 'notes.txt', 'title': 'Greeting'}
```

Key order is guaranteed: dicts have preserved insertion order since Python 3.7,
so the repr is stable and so is anything serialised from it.

---

## Where it fits

```
  str  ──▶  [ normalize_text ]  ──▶  str  ──┐
                                            ├──▶  [ make_document ]  ──▶  dict
  filename / path / URL  ────────────────────┘                            │
                                                                          ▼
                                              {'text', 'source', 'title'}
                                                          │
                                                          ▼
                                             Part 2 · Chunking, and everything after
```

This is the last step at which text is a bare string. From here on, the
pipeline's unit of currency is the document record — and the reason chunking,
retrieval, and citation can each stay simple is that none of them has to ask
what shape the thing they were handed is in.
