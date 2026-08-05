# Step 1 · `load_text_file`

> **Part 1 · Document Ingestion & Preprocessing** — step 1 of 51
> Code: [`rag_pipeline/ingestion.py`](../../rag_pipeline/ingestion.py) · Tests: [`tests/test_ingestion.py`](../../tests/test_ingestion.py)
> Next: [Step 2 · `load_text_directory`](step-02-load-text-directory.md)

---

## The task

```python
def load_text_file(path: str) -> str: ...
```

Read the entire contents of a UTF-8 text file at `path` and return it as a
single string. Newlines, leading/trailing whitespace, and unicode characters
must be preserved **exactly** as they appear on disk.

---

## Why this step exists

Before you can chunk, embed, or retrieve anything, you need raw text in memory.
Almost every RAG corpus starts life as a folder of plain `.txt` files, so a
robust file reader is the foundation the other 50 steps stand on.

It is also worth being deliberate about what this function does *not* do. It
does not strip whitespace, collapse newlines, or fix encoding oddities. Those
are normalisation concerns, and folding them in here would mean every later
stage silently inherits a transformation it can neither see nor turn off. The
reader reads; normalisation gets its own step.

---

## What's happening

```python
with open(path, "r", encoding="utf-8") as file:
    return file.read()
```

Three decisions are packed into those two lines.

**`open(path, "r", ...)`** opens the file in text mode. In text mode Python
decodes bytes into a `str` for you and applies universal newline handling, so a
Windows-authored file with `\r\n` line endings comes back with plain `\n`. That
is the one normalisation we *do* want, because it is invisible to the caller and
consistent across platforms.

**`encoding="utf-8"`** is the important part. A text file on disk is just a
sequence of bytes; turning those bytes into a `str` requires a codec. If you
omit the argument, Python falls back to the platform default
(`locale.getpreferredencoding()`), which is UTF-8 on most Linux and macOS
systems but has historically been cp1252 on Windows. The same code then produces
different strings — or crashes — depending on the machine it runs on. Naming the
codec explicitly makes the pipeline reproducible. UTF-8 is the right default: it
is a strict superset of ASCII and handles accented Latin, CJK scripts, and emoji
without surprises.

**`with`** binds the file handle to a context manager, so the handle is closed
when the block exits — including when `read()` raises. Without it, the handle
stays open until the garbage collector happens to reclaim it. On CPython that is
usually immediate, which is exactly what makes the bug so dangerous: it works
fine in testing, then you ingest ten thousand documents and hit the
process file-descriptor limit.

**`file.read()`** with no argument consumes the whole stream and returns one
string. That is fine for the documents a RAG corpus is made of; a genuinely huge
file would want streaming, but chunking (Part 2) needs the full text in memory
anyway.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Omitting `encoding="utf-8"` | Silent. Works on your machine, produces mojibake or `UnicodeDecodeError` on a colleague's, or on the first non-ASCII document. |
| `file = open(path)` with no `with` | Leaks a file descriptor per call. Invisible until you ingest thousands of files, then `OSError: Too many open files`. |
| `.strip()`-ing the result "to be tidy" | Silently changes offsets. Any later step that maps a chunk back to a character position in the source document is now off by however much you removed. |
| `open(path, "rb")` | Returns `bytes`, not `str`. Everything downstream expects `str`. |

---

## Error behaviour

The function deliberately does not catch anything — a corpus file that cannot be
read is a real problem and should be loud, not silently skipped:

- **`FileNotFoundError`** — no file at `path`.
- **`IsADirectoryError`** — `path` points at a directory.
- **`UnicodeDecodeError`** — the bytes are not valid UTF-8 (a PDF or an image
  renamed to `.txt` will trigger this).

---

## Example

```python
>>> from rag_pipeline.ingestion import load_text_file
>>> load_text_file("corpus/intro.txt")
'Retrieval-augmented generation combines...\n'
```

---

## Where it fits

```
       ┌──────────────────┐
 disk ─┤  load_text_file  ├─→ str ─→ (step 2 reads a whole folder)
       └──────────────────┘
```

This is the only function in the pipeline that opens a file. Everything above it
works on strings, which is what makes the rest of the pipeline testable without
fixtures on disk.
