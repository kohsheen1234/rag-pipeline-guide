"""Evaluation: retrieval metrics and cheap answer-quality proxies.

Retrieval metrics compare ranked id lists against gold ids. The answer metrics
are token-overlap heuristics -- no judge model, no reference answer -- which
makes them fast and free, and blunter than they look. See docs/evaluation/.
"""

from rag_pipeline.ingestion import normalize_text

__all__ = [
    "build_eval_set",
    "hit_rate_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "faithfulness_score",
    "relevance_score",
]


def build_eval_set() -> list:
    """Return the fixed question / answer / relevant-id triples used by the metrics."""
    return [
        {
            "question": "What is RAG?",
            "answer": "Retrieval-augmented generation combines retrieval with a language model.",
            "relevant_ids": ["c1", "c2"],
        },
        {
            "question": "What is chunking?",
            "answer": "Splitting documents into smaller passages that can be embedded and retrieved.",
            "relevant_ids": ["c3"],
        },
        {
            "question": "What is an embedding?",
            "answer": "A dense vector representing text, positioned so similar texts are close together.",
            "relevant_ids": ["c4", "c5"],
        },
    ]


def hit_rate_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Fraction of queries with at least one gold id in their top ``k``.

    Binary per query -- three hits in one query still count once.
    """
    if not retrieved:
        return 0.0

    hits = sum(
        1
        for retrieved_ids, relevant_ids in zip(retrieved, relevant)
        if set(retrieved_ids[:k]) & set(relevant_ids)
    )

    return hits / len(retrieved)


def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Mean fraction of each query's gold ids found in its top ``k``.

    Divided by the number of gold ids, not by ``k`` -- dividing by ``k`` gives
    precision instead. Queries with no gold ids score 0 rather than being
    dropped, so the denominator stays the query count.
    """
    if not retrieved:
        return 0.0

    total = 0.0
    for retrieved_ids, relevant_ids in zip(retrieved, relevant):
        if relevant_ids:
            found = set(retrieved_ids[:k]) & set(relevant_ids)
            total += len(found) / len(relevant_ids)

    return total / len(retrieved)


def mean_reciprocal_rank(retrieved: list, relevant: list) -> float:
    """Mean of ``1 / rank`` of the first gold id in each ranked list.

    Ranks are 1-based. Only the first hit counts; a query with no hit
    contributes 0.
    """
    if not retrieved:
        return 0.0

    total = 0.0
    for retrieved_ids, relevant_ids in zip(retrieved, relevant):
        relevant_set = set(relevant_ids)
        for rank, identifier in enumerate(retrieved_ids, start=1):
            if identifier in relevant_set:
                total += 1.0 / rank
                break

    return total / len(retrieved)


def _tokens(text: str) -> list:
    """Normalise, lowercase, split on whitespace. Punctuation stays attached."""
    return normalize_text(text).lower().split()


def faithfulness_score(answer: str, context_chunks: list) -> float:
    """Fraction of answer tokens that also appear in the retrieved context.

    The answer stays a list so repetition counts; only the context collapses
    to a set, for membership tests.
    """
    answer_tokens = _tokens(answer)

    if not answer_tokens:
        return 0.0

    context = set(_tokens(" ".join(chunk["text"] for chunk in context_chunks)))
    supported = sum(1 for token in answer_tokens if token in context)

    return supported / len(answer_tokens)


def relevance_score(answer: str, question: str) -> float:
    """Jaccard similarity between the answer's and the question's token sets."""
    answer_tokens = set(_tokens(answer))
    question_tokens = set(_tokens(question))
    union = answer_tokens | question_tokens

    if not union:
        return 0.0

    return len(answer_tokens & question_tokens) / len(union)
