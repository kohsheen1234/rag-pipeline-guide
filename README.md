# rag-pipeline-guide

A retrieval-augmented generation pipeline written from scratch in Python, in 51
small steps. It starts with reading a `.txt` file off disk and ends with metrics
for measuring whether any of it worked.

There are plenty of good RAG libraries. This isn't trying to be one. It's a
working pipeline where every piece is small enough to read in one sitting, so
you can see what each stage actually guarantees, what it quietly doesn't, and
what breaks when you get it slightly wrong. A lot of RAG bugs don't raise. They
just return the wrong passage and let the model sound confident about it, and
most of the notes here are about those.

All 51 steps are implemented, with 555 tests and a write-up for every one of
them in `docs/`.

## What's in here

The code is organised by pipeline stage, one module each. They run roughly in
this order:

| Module | What it does |
| --- | --- |
| `ingestion.py` | Read files and HTML, normalise the text, wrap it with its source |
| `chunking.py` | Split documents four different ways, then give each chunk an id |
| `embeddings.py` | Encode chunks into a matrix, unit-normalise it, save it to disk |
| `retrieval.py` | Cosine search and top-k, in numpy and in FAISS |
| `generation.py` | Build the prompt, run a local LM, return an answer with sources |
| `advanced_retrieval.py` | Query rewriting, HyDE, BM25, hybrid search, RRF, cross-encoder reranking, MMR, metadata filters |
| `evaluation.py` | Hit rate, recall@k, MRR, plus cheap faithfulness and relevance proxies |
| `robustness.py` | Abstain on weak retrieval, drop duplicate chunks, cache embeddings, keep chat history |

`tests/` mirrors that one-to-one.

The functions are deliberately plain: they take and return lists, dicts, and
numpy arrays, with no classes to inherit from and no config objects. You can
lift any one of them out and use it on its own.

## The docs

`docs/` has one file per step, which is the opposite structure to the code, and
that's on purpose. The code is short enough to skim; the reasoning isn't. So
`chunking.py` shows you that `chunk_by_sentences` splits on `.!?` and packs
greedily, and
[`docs/chunking/step-08-chunk-by-sentences.md`](docs/chunking/step-08-chunk-by-sentences.md)
tells you that this quietly turns `3.5` into `3. 5` and why that matters before
you embed anything.

Each of the eight parts also has an `00-overview.md` covering the decisions that
span its steps — why chunking is the highest-leverage stage, why the same search
is built twice in numpy and FAISS, why the answer-quality metrics are weaker
than the retrieval ones. Start at [`docs/README.md`](docs/README.md).

## Running it

You need Python 3.9+ and numpy. Everything else is optional.

```bash
git clone https://github.com/kohsheen1234/rag-pipeline-guide
cd rag-pipeline-guide
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

The suite runs in about two seconds and doesn't download anything. The three
heavy dependencies (`sentence-transformers`, `transformers`, `faiss-cpu`) are
imported inside the handful of functions that need them, so the rest of the
package imports fine without them, and the tests that would need real model
weights skip themselves instead of failing.

Install them when you want to run the pipeline for real:

```bash
python3 -m pip install sentence-transformers transformers faiss-cpu
```

## A worked example

Ingest a folder, chunk it, embed it, and search it. This uses a toy bag-of-words
encoder so it runs with nothing installed; swap in `load_embedding_model` and
you have the real thing.

```python
import os
import numpy as np
from rag_pipeline.ingestion import load_text_directory, normalize_text
from rag_pipeline.chunking import chunk_by_sentences, attach_chunk_metadata
from rag_pipeline.embeddings import l2_normalize
from rag_pipeline.retrieval import retrieve

corpus_dir = "corpus/"
filenames = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".txt"))
texts = load_text_directory(corpus_dir)

chunks = []
for text, name in zip(texts, filenames):
    pieces = chunk_by_sentences(normalize_text(text), 80)
    chunks += attach_chunk_metadata(pieces, name)

class BagOfWords:
    vocab = sorted({w for c in chunks for w in c["text"].lower().split()})
    def encode(self, text, batch_size=32):
        rows = [text] if isinstance(text, str) else text
        return np.array(
            [[r.lower().split().count(w) for w in self.vocab] for r in rows],
            dtype=np.float32,
        )

model = BagOfWords()
matrix = l2_normalize(model.encode([c["text"] for c in chunks]))

for chunk, score in retrieve("what does chunking do?", model, matrix, chunks, k=2):
    print(f"{score:.3f}  {chunk['chunk_id']}  {chunk['text']}")
```

```
0.378  b.txt::0  Chunking splits long documents into smaller passages.
0.000  a.txt::0  Retrieval-augmented generation combines a search step with a language model.
```

Note that the filename list gets rebuilt separately. `load_text_directory`
returns bare strings and throws the filenames away, so there's nothing to attach
as a `source` without scanning the directory again. It works because both calls
sort the same way, but it's a rough edge, written up in
[the provenance gap](docs/ingestion/step-05-make-document.md#the-provenance-gap).

## Things worth knowing

A few behaviours that surprised me while building this, all pinned by tests:

- HTML extraction glues adjacent block elements together. `<p>one</p><p>two</p>`
  comes out as `onetwo`, and normalising afterwards can't recover a word
  boundary that was never there. It fires constantly on minified HTML.
- Sentence chunking corrupts decimals. `3.5` becomes `3. 5`, because the `.` is
  read as a sentence end and the pieces are rejoined with a space. This happens
  even when everything fits in one chunk.
- A negative `chunk_size` returns an empty list rather than raising, so a
  document silently disappears from the index.
- Chunk ids are stable across runs but not across configuration. Change
  `chunk_size` and `doc1::7` points at completely different text, with no error.
  Anything cached on a chunk id needs invalidating when chunking changes.
- MMR at `lambda=0.5` doesn't demote a perfect duplicate. The two terms cancel
  exactly, so you need `lambda < 0.5` for the diversity half to do anything.
- BM25 on the guide's own example corpus scores `ln(2) = 0.6931`, not the
  `0.5798` the guide quotes. Both documents there are exactly the average
  length, which makes the term-frequency saturation factor identically 1 for any
  `k1` and `b`, leaving the score equal to the IDF. No parameter choice
  reproduces `0.5798`.

## How it's written

Four rules, applied throughout:

**One concern per function.** A loader loads and does not also normalise. Every
transformation the text goes through is a named function you can find in a stack
trace and choose to skip.

**Determinism over convenience.** Anything that could vary by machine,
filesystem, or locale is pinned. Directory listings get sorted, encodings are
named, tie-breaks are explicit. Non-determinism here doesn't crash, it just
returns different passages on someone else's laptop.

**Fail loudly.** Nothing swallows an exception to keep a batch running. Where a
function does fail quietly, that's called out in its doc rather than left to be
discovered.

**Plain data.** Strings, dicts, and arrays. Only the loaders touch the
filesystem, which is what makes everything above them testable without fixtures.

## License

MIT, see [LICENSE](LICENSE).
