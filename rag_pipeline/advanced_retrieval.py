"""Advanced retrieval: rewriting, fusion, lexical search, reranking, filtering.

Everything here sits around the dense retriever from
:mod:`~rag_pipeline.retrieval` rather than replacing it -- cleaning the query
going in, adding a second signal alongside it, or reordering what comes out.
See docs/advanced-retrieval/ for the reasoning behind each function.
"""

import math

import numpy as np

from rag_pipeline.embeddings import embed_text
from rag_pipeline.ingestion import normalize_text
from rag_pipeline.retrieval import cosine_similarity_search, top_k_chunks

__all__ = [
    "query_rewrite",
    "hyde_retrieve",
    "reciprocal_rank_fusion",
    "bm25_search",
    "hybrid_search",
    "rerank_cross_encoder",
    "maximal_marginal_relevance",
    "filter_by_metadata",
]

#: Conversational lead-ins stripped from the front of a query, never the middle.
FILLER_PREFIXES = ("please", "could you", "can you", "tell me", "i want to know")

TERMINAL_PUNCTUATION = "?.!"


def query_rewrite(query: str) -> str:
    """Reduce a conversational query to its core information need.

    Normalises unicode and whitespace, lowercases, strips leading filler
    phrases, and trims trailing terminal punctuation. Fillers are only removed
    from the front -- "tell me" is meaningful in the middle of a sentence.
    """
    text = normalize_text(query).lower()

    stripped = True
    while stripped:
        stripped = False
        for filler in FILLER_PREFIXES:
            # The trailing space is what stops "please" eating "pleasant".
            if text.startswith(filler + " "):
                text = text[len(filler) + 1 :]
                stripped = True

    return text.rstrip(TERMINAL_PUNCTUATION).strip()


def hyde_retrieve(
    query: str,
    hypothetical_answer: str,
    chunks: list,
    embeddings: np.ndarray,
    model,
    k: int = 5,
) -> list:
    """Retrieve using an embedding of a *hypothetical answer*, not the query.

    A short question and the passage answering it often sit far apart in
    embedding space. A draft answer looks more like a passage, so it lands
    closer to the real one. The query is kept for prompting and logging but
    takes no part in the scoring.
    """
    vector = embed_text(model, hypothetical_answer)
    scores = cosine_similarity_search(vector, embeddings)

    return [chunk for chunk, _ in top_k_chunks(scores, chunks, k)]


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """Fuse ranked id lists into one ranking, scoring by position alone.

    Each list contributes ``1 / (k + rank)`` per id, with ``rank`` 1-based.
    Because only positions matter, lists from different retrievers can be
    combined without calibrating their scores against each other.
    """
    scores = {}

    for ranked_list in ranked_lists:
        for rank, identifier in enumerate(ranked_list, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: -item[1])


def _tokenize(text: str) -> list:
    """Lowercase and split on whitespace. The whole lexical model."""
    return text.lower().split()


def bm25_search(query: str, chunks: list, k: int = 5, k1: float = 1.5, b: float = 0.75):
    """Score chunks against a query with BM25, returning ``(index, score)``.

    Chunks sharing no query term are omitted rather than returned with a zero.
    ``k1`` controls term-frequency saturation, ``b`` how hard long documents
    are penalised.
    """
    documents = [_tokenize(chunk["text"]) for chunk in chunks]
    total = len(documents)

    if total == 0:
        return []

    average_length = sum(len(document) for document in documents) / total
    query_terms = _tokenize(query)

    scored = []
    for index, document in enumerate(documents):
        score = 0.0
        for term in query_terms:
            frequency = document.count(term)
            if frequency == 0:
                continue

            # Document frequency counts documents containing the term, not
            # occurrences -- counting occurrences inflates df and breaks IDF.
            document_frequency = sum(1 for other in documents if term in other)
            idf = math.log(
                (total - document_frequency + 0.5) / (document_frequency + 0.5) + 1
            )
            length_penalty = 1 - b + b * len(document) / average_length
            score += (
                idf
                * frequency
                * (k1 + 1)
                / (frequency + k1 * length_penalty)
            )

        if score > 0:
            scored.append((index, score))

    scored.sort(key=lambda item: -item[1])

    return scored[:k]


def _min_max(scores: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]. A flat vector maps to zeros rather than NaNs."""
    lowest, highest = scores.min(), scores.max()

    if highest == lowest:
        return np.zeros_like(scores)

    return (scores - lowest) / (highest - lowest)


def hybrid_search(
    query: str,
    chunks: list,
    embeddings: np.ndarray,
    model,
    alpha: float = 0.5,
    k: int = 5,
) -> list:
    """Mix dense and lexical scores into one ranking.

    Each score vector is min-max scaled before mixing, because cosine sits in
    roughly [-1, 1] while BM25 is unbounded -- combined raw, whichever has the
    larger magnitude decides the ranking on its own.

    ``alpha`` weights the dense side. Ties keep the original chunk order.
    """
    dense = cosine_similarity_search(embed_text(model, query), embeddings)

    lexical = np.zeros(len(chunks), dtype=float)
    for index, score in bm25_search(query, chunks, k=len(chunks)):
        lexical[index] = score

    combined = alpha * _min_max(dense) + (1 - alpha) * _min_max(lexical)
    order = np.argsort(-combined, kind="stable")[:k]

    return [(int(index), float(combined[index])) for index in order]


def rerank_cross_encoder(query: str, candidate_chunks: list, cross_encoder) -> list:
    """Reorder candidates by a cross-encoder's ``(query, text)`` relevance score.

    A cross-encoder reads the query and the passage together, so it is far more
    accurate than the bi-encoder that produced the candidates -- and far too
    slow to run over a whole corpus. This is the second stage of retrieve-then-
    rerank.
    """
    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]
    scores = cross_encoder.predict(pairs)

    # Sort indices, not (score, chunk) pairs: tied dicts are not comparable.
    order = sorted(range(len(candidate_chunks)), key=lambda i: -scores[i])

    return [candidate_chunks[index] for index in order]


def maximal_marginal_relevance(
    query_vector: np.ndarray,
    candidates: np.ndarray,
    k: int = 5,
    lambda_param: float = 0.5,
) -> list:
    """Greedily pick indices balancing relevance against redundancy.

    Each step takes ``argmax(lambda * relevance - (1 - lambda) * max similarity
    to anything already picked)``. The redundancy term is recomputed every
    iteration -- that it changes as the selection grows is the whole mechanism.
    An empty selection has redundancy 0, so the first pick is the most relevant.

    ``candidates`` must be L2-normalised, so every similarity is a dot product.
    """
    relevance = candidates @ query_vector
    pairwise = candidates @ candidates.T

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(k, len(candidates))):
        best_index, best_score = None, None

        for index in remaining:
            redundancy = max(
                (pairwise[index][chosen] for chosen in selected), default=0.0
            )
            score = lambda_param * relevance[index] - (1 - lambda_param) * redundancy

            # Strict >, scanning ascending, so ties go to the smaller index.
            if best_score is None or score > best_score:
                best_index, best_score = index, score

        selected.append(best_index)
        remaining.remove(best_index)

    return selected


def filter_by_metadata(chunks: list, filter_dict: dict) -> list:
    """Keep chunks whose ``metadata`` matches every key/value in the filter.

    A missing key is not a match. An empty filter keeps everything. Input order
    is preserved and the chunks are returned by reference, not copied.
    """
    return [
        chunk
        for chunk in chunks
        if all(
            key in chunk.get("metadata", {}) and chunk["metadata"][key] == value
            for key, value in filter_dict.items()
        )
    ]
