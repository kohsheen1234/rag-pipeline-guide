"""Dense retrieval, in numpy and in FAISS.

Two backends over the same corpus matrix. The numpy path is exact and obvious;
the FAISS path is exact and fast. They should agree, and
:func:`compare_faiss_to_numpy` checks that they do. See docs/retrieval/ for the
reasoning behind each function.
"""

import numpy as np

from rag_pipeline.embeddings import embed_text

__all__ = [
    "cosine_similarity_search",
    "top_k_indices",
    "top_k_chunks",
    "retrieve",
    "build_faiss_index",
    "faiss_search",
    "compare_faiss_to_numpy",
    "save_faiss_index",
]


def cosine_similarity_search(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Score a ``(d,)`` query against every row of an ``(n, d)`` matrix.

    Inputs are not assumed normalised, so both norms are divided out here.
    Zero-length rows score 0 rather than producing NaNs.
    """
    query_norm = np.linalg.norm(query)
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm

    return (matrix @ query) / np.where(denominator == 0, 1, denominator)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, highest first.

    Clamped to the number of scores. Ties are broken by index, so the order is
    reproducible across runs and backends.
    """
    order = np.argsort(-scores, kind="stable")

    return order[: min(k, len(scores))]


def top_k_chunks(scores: np.ndarray, chunks: list, k: int) -> list:
    """Pair the top-scoring chunks with their scores, highest first.

    Both the chunk and its score are read with the same index, which is what
    keeps them from being mismatched. Scores are plain floats so the result is
    JSON-serialisable.
    """
    indices = top_k_indices(scores, min(k, len(chunks)))

    return [(chunks[index], float(scores[index])) for index in indices]


def retrieve(query: str, model, chunk_matrix: np.ndarray, chunks: list, k: int) -> list:
    """Embed a query, score it against the corpus, return the top ``k`` chunks."""
    query_vector = embed_text(model, query)
    scores = cosine_similarity_search(query_vector, chunk_matrix)

    return top_k_chunks(scores, chunks, k)


def build_faiss_index(matrix: np.ndarray):
    """Build a flat inner-product FAISS index over an ``(n, d)`` float32 matrix.

    Rows must already be L2-normalised, since inner product only equals cosine
    similarity for unit vectors. The dimensionality is fixed at construction.
    """
    # Imported here so the numpy path works without faiss installed.
    import faiss

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return index


def faiss_search(index, query: np.ndarray, k: int):
    """Search an index with one ``(d,)`` vector, returning flat ``(k,)`` arrays.

    FAISS searches batches, so the query gains a leading axis and the results
    lose theirs.
    """
    scores, indices = index.search(query.reshape(1, -1), k)

    return (
        np.asarray(scores, dtype=np.float32).reshape(-1),
        np.asarray(indices, dtype=np.int64).reshape(-1),
    )


def compare_faiss_to_numpy(query: np.ndarray, matrix: np.ndarray, index, k: int) -> bool:
    """Check that both backends select the same top-``k`` chunks.

    Compared as sets: equal scores can legitimately swap places between
    backends, and that is not a disagreement about which chunks won.
    """
    scores = cosine_similarity_search(query, matrix)
    numpy_indices = top_k_indices(scores, k)
    _, faiss_indices = faiss_search(index, query, k)

    return set(numpy_indices.tolist()) == set(faiss_indices.tolist())


def save_faiss_index(index, path: str):
    """Write a FAISS index to ``path`` and return the copy read back from disk.

    Returning the reloaded object rather than the original is the point: it is
    what proves the file on disk is usable.
    """
    import faiss

    faiss.write_index(index, path)

    return faiss.read_index(path)
