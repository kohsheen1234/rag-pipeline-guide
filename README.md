# rag-pipeline-guide

A retrieval-augmented generation pipeline built from scratch in Python, one step
at a time — 51 steps from reading a text file off disk to evaluating retrieval
quality.

The point is not to ship another RAG library. It is to understand each piece
well enough to know what it guarantees, what it deliberately does not, and what
silently breaks when you get it wrong. Every step has a doc explaining the
reasoning, not just the code.

**Progress: 10 / 51 steps** — ingestion and chunking complete.

---

## Layout

```
rag_pipeline/
  ingestion.py          load, extract, normalise, wrap into documents
  chunking.py           split documents into retrievable passages

tests/
  test_ingestion.py
  test_chunking.py

docs/
  README.md             index of all steps + doc conventions
  ingestion/
    00-overview.md      design rules for the stage
    step-01-load-text-file.md
    step-02-load-text-directory.md
    step-03-extract-text-from-html.md
    step-04-normalize-text.md
    step-05-make-document.md
  chunking/
    00-overview.md
    step-06-chunk-fixed-size.md
    step-07-chunk-by-tokens.md
    step-08-chunk-by-sentences.md
    step-09-chunk-with-overlap.md
    step-10-attach-chunk-metadata.md
```

**One module per pipeline stage**, named for what it does rather than which
part of the guide it came from. Ingestion is a handful of short functions
sharing one concern, so they live together and the stage reads top to bottom in
one file. Chunking, embedding, retrieval, and the rest get their own modules as
the guide reaches them.

**Docs stay one file per step** — the opposite choice, deliberately. The code is
short enough to read at a glance; the reasoning is not. Each step doc is a few
hundred words on why the step exists and what breaks without it, and stacking
five of those into one page would bury them. So `ingestion.py` tells you what
the code does, and `docs/ingestion/step-02-load-text-directory.md` tells you why
it does it that way — and what happens if you do it the obvious wrong way
instead.

---

## Getting started

Requires Python 3.9+. No runtime dependencies so far.

```bash
git clone <this repo>
cd rag-pipeline-guide

python3 -m pip install -r requirements-dev.txt   # pytest
python3 -m pytest                                 # run the suite
```

Then start at [`docs/README.md`](docs/README.md).

---

## Using it

```python
import os
from rag_pipeline.ingestion import (
    load_text_directory,
    make_document,
    normalize_text,
)

corpus = "corpus/"
filenames = sorted(f for f in os.listdir(corpus) if f.endswith(".txt"))
texts = load_text_directory(corpus)          # list[str], same filename order

documents = [
    make_document(normalize_text(text), filename, filename)
    for text, filename in zip(texts, filenames)
]
# [{'text': 'Hello world.', 'source': 'a.txt', 'title': 'a.txt'}, ...]
```

The filename list has to be rebuilt because `load_text_directory` returns bare
strings — see [the provenance gap](docs/ingestion/step-05-make-document.md#the-provenance-gap).

---

## Design rules

These hold across the whole pipeline, not just Part 1.

- **One concern per step.** A loader loads; it does not also normalise. Every
  transformation the text undergoes is a function you can name, test, and skip.
- **Determinism over convenience.** Anything that could vary by machine,
  filesystem, or locale gets pinned explicitly. Non-determinism in a RAG
  pipeline is silent: retrieval still returns results, just the wrong ones.
- **Fail loudly.** No bare `except` to keep a batch running. A corpus file that
  cannot be read is a real problem.
- **Strings in, strings out.** Only the loading module touches the filesystem,
  which keeps everything above it testable without fixtures on disk.

---

## License

MIT — see [LICENSE](LICENSE).
