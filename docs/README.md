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

## Part 3 · [Embeddings & Corpus Storage](embeddings/00-overview.md)

Chunks become a matrix, and the matrix survives a restart.

| # | Step | Doc |
| --- | --- | --- |
| 11 | `load_embedding_model` | [step-11-load-embedding-model.md](embeddings/step-11-load-embedding-model.md) |
| 12 | `embed_text` | [step-12-embed-text.md](embeddings/step-12-embed-text.md) |
| 13 | `embed_chunks` | [step-13-embed-chunks.md](embeddings/step-13-embed-chunks.md) |
| 14 | `l2_normalize` | [step-14-l2-normalize.md](embeddings/step-14-l2-normalize.md) |
| 15 | `save_corpus` | [step-15-save-corpus.md](embeddings/step-15-save-corpus.md) |

---

## Part 4 · [Dense Retrieval with NumPy and FAISS](retrieval/00-overview.md)

The same search built twice, and checked against itself.

| # | Step | Doc |
| --- | --- | --- |
| 16 | `cosine_similarity_search` | [step-16-cosine-similarity-search.md](retrieval/step-16-cosine-similarity-search.md) |
| 17 | `top_k_indices` | [step-17-top-k-indices.md](retrieval/step-17-top-k-indices.md) |
| 18 | `top_k_chunks` | [step-18-top-k-chunks.md](retrieval/step-18-top-k-chunks.md) |
| 19 | `retrieve` | [step-19-retrieve.md](retrieval/step-19-retrieve.md) |
| 20 | `build_faiss_index` | [step-20-build-faiss-index.md](retrieval/step-20-build-faiss-index.md) |
| 21 | `faiss_search` | [step-21-faiss-search.md](retrieval/step-21-faiss-search.md) |
| 22 | `compare_faiss_to_numpy` | [step-22-compare-faiss-to-numpy.md](retrieval/step-22-compare-faiss-to-numpy.md) |
| 23 | `save_faiss_index` | [step-23-save-faiss-index.md](retrieval/step-23-save-faiss-index.md) |

---

## Part 5 · [Prompting and Answer Generation](generation/00-overview.md)

Passages become a prompt, and the prompt becomes an answer with sources.

| # | Step | Doc |
| --- | --- | --- |
| 24 | `build_prompt_template` | [step-24-build-prompt-template.md](generation/step-24-build-prompt-template.md) |
| 25 | `format_context` | [step-25-format-context.md](generation/step-25-format-context.md) |
| 26 | `truncate_context` | [step-26-truncate-context.md](generation/step-26-truncate-context.md) |
| 27 | `add_system_instruction` | [step-27-add-system-instruction.md](generation/step-27-add-system-instruction.md) |
| 28 | `load_generator` | [step-28-load-generator.md](generation/step-28-load-generator.md) |
| 29 | `generate_answer` | [step-29-generate-answer.md](generation/step-29-generate-answer.md) |
| 30 | `rag_answer` | [step-30-rag-answer.md](generation/step-30-rag-answer.md) |
| 31 | `track_source_chunk_ids` | [step-31-track-source-chunk-ids.md](generation/step-31-track-source-chunk-ids.md) |
| 32 | `append_source_references` | [step-32-append-source-references.md](generation/step-32-append-source-references.md) |

---

## Part 6 · [Advanced Retrieval Techniques](advanced-retrieval/00-overview.md)

Eight ways plain dense retrieval is not quite good enough.

| # | Step | Doc |
| --- | --- | --- |
| 33 | `query_rewrite` | [step-33-query-rewrite.md](advanced-retrieval/step-33-query-rewrite.md) |
| 34 | `hyde_retrieve` | [step-34-hyde-retrieve.md](advanced-retrieval/step-34-hyde-retrieve.md) |
| 35 | `reciprocal_rank_fusion` | [step-35-reciprocal-rank-fusion.md](advanced-retrieval/step-35-reciprocal-rank-fusion.md) |
| 36 | `bm25_search` | [step-36-bm25-search.md](advanced-retrieval/step-36-bm25-search.md) |
| 37 | `hybrid_search` | [step-37-hybrid-search.md](advanced-retrieval/step-37-hybrid-search.md) |
| 38 | `rerank_cross_encoder` | [step-38-rerank-cross-encoder.md](advanced-retrieval/step-38-rerank-cross-encoder.md) |
| 39 | `maximal_marginal_relevance` | [step-39-maximal-marginal-relevance.md](advanced-retrieval/step-39-maximal-marginal-relevance.md) |
| 40 | `filter_by_metadata` | [step-40-filter-by-metadata.md](advanced-retrieval/step-40-filter-by-metadata.md) |

---

## Part 7 · [Evaluation](evaluation/00-overview.md)

How you find out whether any of the previous choices helped.

| # | Step | Doc |
| --- | --- | --- |
| 41 | `build_eval_set` | [step-41-build-eval-set.md](evaluation/step-41-build-eval-set.md) |
| 42 | `hit_rate_at_k` | [step-42-hit-rate-at-k.md](evaluation/step-42-hit-rate-at-k.md) |
| 43 | `recall_at_k` | [step-43-recall-at-k.md](evaluation/step-43-recall-at-k.md) |
| 44 | `mean_reciprocal_rank` | [step-44-mean-reciprocal-rank.md](evaluation/step-44-mean-reciprocal-rank.md) |
| 45 | `faithfulness_score` | [step-45-faithfulness-score.md](evaluation/step-45-faithfulness-score.md) |
| 46 | `relevance_score` | [step-46-relevance-score.md](evaluation/step-46-relevance-score.md) |

---

## Part 8 · [Robustness, Caching, and Chat Memory](robustness/00-overview.md)

What the happy path ignores.

| # | Step | Doc |
| --- | --- | --- |
| 47 | `handle_no_context` | [step-47-handle-no-context.md](robustness/step-47-handle-no-context.md) |
| 48 | `deduplicate_chunks` | [step-48-deduplicate-chunks.md](robustness/step-48-deduplicate-chunks.md) |
| 49 | `cache_query_embedding` | [step-49-cache-query-embedding.md](robustness/step-49-cache-query-embedding.md) |
| 50 | `update_chat_memory` | [step-50-update-chat-memory.md](robustness/step-50-update-chat-memory.md) |
| 51 | `rewrite_followup` | [step-51-rewrite-followup.md](robustness/step-51-rewrite-followup.md) |

---

## Progress

**51 / 51 steps documented.** Every public function has a step doc, and every
part has an overview covering the decisions that span its steps. Later parts get their own folder and overview as they are
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
