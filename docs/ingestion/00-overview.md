# Part 1 · Document Ingestion & Preprocessing

The job of Part 1 is to take a corpus in whatever shape it happens to exist on
disk and hand the rest of the pipeline something predictable: a list of clean
strings, in a stable order.

Everything downstream depends on two guarantees established here.

**Faithful decoding.** Bytes become text through one explicit codec (UTF-8), in
one place. No stage further up the pipeline should ever have to wonder how a
document was decoded.

**Deterministic order.** The position of a document in the corpus list is part of
its identity. Chunk ids, evaluation labels, and embedding cache keys are all
derived from it, so the order has to be identical on every machine, every run.

## Design rules for this part

- **Read faithfully, normalise explicitly.** Loaders preserve bytes-to-text
  exactly. Whitespace collapsing, case folding, and boilerplate stripping are
  separate, named steps you can see in a call stack and switch off.
- **Only the loaders touch the filesystem.** `load_text_file` is the single
  place in the pipeline that calls `open()`. Every other function takes
  strings, which is what makes them testable without fixtures on disk.
- **Fail loudly.** An unreadable corpus file is a real problem. Nothing here
  swallows exceptions to keep a batch running.

## Steps

All five live in [`rag_pipeline/ingestion.py`](../../rag_pipeline/ingestion.py),
in this order.

| # | Function | What it does |
| --- | --- | --- |
| 1 | [`load_text_file`](step-01-load-text-file.md) | Read one UTF-8 text file into a string, preserving it exactly. |
| 2 | [`load_text_directory`](step-02-load-text-directory.md) | Read every `.txt` file in a folder into a list, ordered by filename. |
| 3 | `extract_text_from_html` | _Not yet implemented._ Strip markup down to readable text. |
| 4 | [`normalize_text`](step-04-normalize-text.md) | NFKC-fold unicode variants, collapse whitespace, strip. |
| 5 | [`make_document`](step-05-make-document.md) | Wrap text with its provenance into the pipeline's document record. |

_Steps are added to this table as the guide progresses._

## Data flow so far

```
  corpus/                 ┌───────────────────────┐
    a.txt        ────────▶│  load_text_directory  │
    b.txt                 └───────────┬───────────┘
    notes.md  (skipped)               │  sorted, one call per .txt
                                      ▼
                            ┌──────────────────┐
                            │  load_text_file  │
                            └────────┬─────────┘
                                     ▼
                                 list[str]   raw, faithful
                                     │
                                     ▼
                            ┌──────────────────┐
                            │  normalize_text  │
                            └────────┬─────────┘
                                     ▼
                                 list[str]   tidy single lines
                                     │
                    source, title ──▶├
                                     ▼
                            ┌──────────────────┐
                            │  make_document   │
                            └────────┬─────────┘
                                     ▼
                     {'text', 'source', 'title'}   the pipeline's record
                                     │
                                     ▼
                             Part 2 · Chunking
```

Reading is separated from transforming by function, not by module: nothing that
opens a file also changes the text, and nothing that changes the text also
reaches the disk. That is the "read faithfully, normalise explicitly" rule, and
it is why `normalize_text` is a step a caller invokes rather than something the
loader quietly does on the way out.
