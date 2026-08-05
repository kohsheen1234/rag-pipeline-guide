# Guide index

One document per step: what it asks for, why the step exists, what the code is
actually doing, and the pitfalls that make it subtle.

Each part has an overview page covering the design rules that hold across its
steps, so the individual step docs can stay focused on their own function.

---

## Part 1 · [Document Ingestion & Preprocessing](ingestion/00-overview.md)

Raw files and HTML become a deterministically ordered list of document records.

| # | Step | Doc |
| --- | --- | --- |
| 1 | `load_text_file` | [step-01-load-text-file.md](ingestion/step-01-load-text-file.md) |
| 2 | `load_text_directory` | [step-02-load-text-directory.md](ingestion/step-02-load-text-directory.md) |
| 3 | `extract_text_from_html` | [step-03-extract-text-from-html.md](ingestion/step-03-extract-text-from-html.md) |
| 4 | `normalize_text` | [step-04-normalize-text.md](ingestion/step-04-normalize-text.md) |
| 5 | `make_document` | [step-05-make-document.md](ingestion/step-05-make-document.md) |

---

## Part 2 · [Chunking Strategies](chunking/00-overview.md)

Documents become passages small enough to embed and specific enough to rank.

| # | Step | Doc |
| --- | --- | --- |
| 6 | `chunk_fixed_size` | [step-06-chunk-fixed-size.md](chunking/step-06-chunk-fixed-size.md) |
| 7 | `chunk_by_tokens` | [step-07-chunk-by-tokens.md](chunking/step-07-chunk-by-tokens.md) |
| 8 | `chunk_by_sentences` | [step-08-chunk-by-sentences.md](chunking/step-08-chunk-by-sentences.md) |
| 9 | `chunk_with_overlap` | [step-09-chunk-with-overlap.md](chunking/step-09-chunk-with-overlap.md) |
| 10 | `attach_chunk_metadata` | [step-10-attach-chunk-metadata.md](chunking/step-10-attach-chunk-metadata.md) |

---

## Parts 4–8 · implemented, docs pending

All 51 steps are implemented and tested. Step docs so far cover Parts 1–3; the
remaining parts are listed here with their module so the code is still findable.

| Part | Steps | Module |
| --- | --- | --- |
| 4 · Dense Retrieval with NumPy and FAISS | 16–23 | [`retrieval.py`](../rag_pipeline/retrieval.py) |
| 5 · Prompting and Answer Generation | 24–32 | [`generation.py`](../rag_pipeline/generation.py) |
| 6 · Advanced Retrieval Techniques | 33–40 | [`advanced_retrieval.py`](../rag_pipeline/advanced_retrieval.py) |
| 7 · Evaluation | 41–46 | [`evaluation.py`](../rag_pipeline/evaluation.py) |
| 8 · Robustness, Caching, and Chat Memory | 47–51 | [`robustness.py`](../rag_pipeline/robustness.py) |

---

## Progress

**51 / 51 steps implemented**, 15 with step docs. Parts 1–3 are fully
documented; Parts 4–8 have code, docstrings, and tests but no step docs yet. Later parts get their own folder and overview as they are
reached — embedding, indexing, retrieval, generation, and evaluation.

---

## Conventions

**Doc naming.** `step-NN-function-name.md`, zero-padded, inside the folder for
its part. Zero-padding is not cosmetic: it keeps lexicographic file order equal
to step order, the same reason step 2 cares about padded corpus filenames.

**Doc structure.** Every step doc follows the same skeleton, so you can skim to
the section you want without reading the whole page:

| Section | Contains |
| --- | --- |
| The task | The signature and the requirement, restated precisely. |
| Why this step exists | What breaks in the pipeline without it. |
| What's happening | A walkthrough of the implementation, decision by decision. |
| Boundaries of the contract | What the step deliberately does *not* handle. |
| Common pitfalls | Table of the wrong-but-plausible versions and why they bite. |
| Example | A REPL transcript you can paste. |
| Where it fits | An ASCII diagram of the step's place in the data flow. |

Sections that have nothing to say for a given step are omitted rather than left
empty. [`_TEMPLATE.md`](_TEMPLATE.md) is the starting point for a new one.
