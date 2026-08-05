# Step 2 · `load_text_directory`

> **Part 1 · Document Ingestion & Preprocessing** — step 2 of 51
> Code: [`rag_pipeline/ingestion.py`](../../rag_pipeline/ingestion.py) · Tests: [`tests/test_ingestion.py`](../../tests/test_ingestion.py)
> Previous: [Step 1 · `load_text_file`](step-01-load-text-file.md)

---

## The task

```python
def load_text_directory(directory_path: str) -> list[str]: ...
```

Scan a directory and return the contents of every `.txt` file in it as a list of
strings. Non-`.txt` files are skipped. The list must be ordered by filename in
ascending lexicographic order. Reuse `load_text_file` to read each file.

---

## Why this step exists

A real corpus rarely lives in a single string; it lives as many files in a
folder. Turning a directory into a clean, ordered list of document texts is the
first ingestion primitive the rest of the pipeline builds on — every later stage
takes "a list of documents" as its input.

---

## What's happening

```python
texts = []

for filename in sorted(os.listdir(directory_path)):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)
        texts.append(load_text_file(file_path))

return texts
```

**`os.listdir(directory_path)`** returns the names of the directory's entries —
just the names, not full paths, and not recursively. That is why `os.path.join`
is needed on the next line: `load_text_file` needs a path it can actually open,
and `"a.txt"` on its own resolves against the *current working directory*, not
the directory being scanned. This is the single most common bug in this step, and
it only shows up when the caller runs from somewhere other than the corpus
folder.

**`sorted(...)`** is the load-bearing call, and the reason is worth spelling out
below.

**`filename.endswith(".txt")`** is the filter. Directories with a `.txt` name and
files with an uppercase `.TXT` extension are both edge cases this rule handles
"wrong" in the strict sense — see *Boundaries of the contract*.

**`load_text_file(file_path)`** does the actual reading. Reusing it rather than
inlining another `open()` means the UTF-8 guarantee from step 1 is enforced in
exactly one place. When a later step changes how files are read — adding
`errors="replace"`, say, or BOM stripping — that change lands here for free.

---

## Why sorting is the whole point

`os.listdir` makes **no ordering guarantee**. The order you get back is whatever
the underlying filesystem hands over:

- ext4 with `dir_index` enabled returns entries in *hash* order, which looks
  random and changes as files are added or removed.
- APFS and HFS+ typically return something close to sorted, which is worse in a
  way — it means the bug does not reproduce on the Mac you developed on.
- NTFS returns roughly insertion order.

Downstream, this matters more than it looks. Chunk ids are usually built as
something like `(document_index, chunk_index)`. Evaluation sets label gold
passages by those ids. Embedding caches key on them. If document order shifts
between two machines, every chunk id shifts with it — and the failure is
*silent*: the retriever still returns results, they are simply the wrong
passages, and every gold label now points somewhere else.

Sorting by filename before reading pins the order to something the filesystem
cannot influence. Lexicographic sort is chosen because it is the simplest rule
that is stable, obvious to a reader, and reproducible everywhere.

---

## Boundaries of the contract

Worth knowing, deliberately not handled at this step:

**Lexicographic ≠ numeric.** `doc10.txt` sorts *before* `doc2.txt`, because `"1"
< "2"` character by character. If the corpus uses unpadded numeric filenames the
order will surprise you. The fix is zero-padded names (`doc02.txt`,
`doc10.txt`), not a cleverer sort — a natural-sort key would make the ordering
rule harder to reason about for no real gain.

**Case sensitivity.** `sorted()` compares by codepoint, so all uppercase ASCII
sorts before all lowercase: `Zeta.txt` comes before `alpha.txt`. Consistent, if
not alphabetical in the human sense.

**`.TXT` is skipped.** `endswith(".txt")` is case-sensitive. On a
case-insensitive filesystem like APFS this is an easy trap.

**A directory named `foo.txt` raises.** It passes the extension filter, then
`load_text_file` raises `IsADirectoryError`. Loud failure is the right outcome —
a directory named like a document is a corpus problem worth knowing about.

**The scan is not recursive.** Nested folders are ignored entirely. Recursive
walking is a separate concern with its own ordering questions.

---

## Common pitfalls

| Pitfall | Why it bites |
| --- | --- |
| Returning `os.listdir` order unsorted | The headline bug. OS- and filesystem-dependent, silent, and it invalidates every cached embedding and gold label the moment the machine changes. |
| Forgetting `os.path.join` | `load_text_file("a.txt")` resolves against the CWD. Works when you run from inside the corpus folder, `FileNotFoundError` everywhere else. |
| Sorting *after* reading | Sorting the resulting strings orders by content, not filename. Coincidentally right on small toy inputs, wrong in general. |
| Re-implementing `open()` inline | Duplicates the encoding contract. The two readers drift apart the first time one of them is fixed. |

---

## Example

```python
>>> import tempfile, os
>>> d = tempfile.mkdtemp()
>>> open(os.path.join(d, 'b.txt'), 'w').write('two')
3
>>> open(os.path.join(d, 'a.txt'), 'w').write('one')
3
>>> open(os.path.join(d, 'notes.md'), 'w').write('skipped')
7
>>> load_text_directory(d)
['one', 'two']
```

---

## Where it fits

```
        ┌───────────────────────┐
 folder ┤  load_text_directory  ├─→ list[str]  ─→ (chunking, Part 2)
        └───────────┬───────────┘
                    │ calls, once per .txt file
                    ▼
            ┌──────────────────┐
            │  load_text_file  │
            └──────────────────┘
```

The `list[str]` this returns is the canonical "corpus" object for the rest of
Part 1. Its **index is meaningful** — position `i` is document `i` for every
downstream stage — which is exactly why the ordering contract is enforced here,
at the boundary, rather than assumed later.
