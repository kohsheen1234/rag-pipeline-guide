"""Robustness, caching, and chat memory.

The parts of a RAG system that are not retrieval or generation: refusing when
retrieval came up empty, pruning duplicate chunks, avoiding repeated work, and
carrying enough conversation state to make follow-ups searchable.
See docs/robustness/ for the reasoning behind each function.
"""

import numpy as np

from rag_pipeline.embeddings import embed_text
from rag_pipeline.generation import REFUSAL
from rag_pipeline.ingestion import normalize_text

__all__ = [
    "handle_no_context",
    "deduplicate_chunks",
    "cache_query_embedding",
    "update_chat_memory",
    "rewrite_followup",
]


def _score_of(scored_chunk) -> float:
    """Read a score from either a ``(chunk, score)`` tuple or a ``{'score': ...}``."""
    if isinstance(scored_chunk, dict):
        return scored_chunk["score"]

    return scored_chunk[1]


def handle_no_context(scored_chunks: list, threshold: float) -> dict:
    """Decide whether retrieval was confident enough to answer from.

    Abstains unless some chunk scores *strictly* above ``threshold``. An empty
    retrieval abstains rather than raising on ``max([])``.
    """
    scores = [_score_of(scored_chunk) for scored_chunk in scored_chunks]

    if not scores or max(scores) <= threshold:
        return {"abstain": True, "message": REFUSAL}

    return {"abstain": False, "message": ""}


def deduplicate_chunks(chunks: list, embeddings: np.ndarray, similarity_threshold: float):
    """Drop chunks too similar to one already kept, keeping the first of each group.

    Embeddings must be L2-normalised, so cosine similarity is a dot product. A
    candidate is compared only against the *kept* set, never against itself.

    Returns the surviving chunks and the matching rows of the matrix.
    """
    kept_indices = []

    for index in range(len(chunks)):
        vector = embeddings[index]
        duplicate = any(
            float(vector @ embeddings[kept]) > similarity_threshold
            for kept in kept_indices
        )

        if not duplicate:
            kept_indices.append(index)

    return [chunks[index] for index in kept_indices], embeddings[kept_indices]


def cache_query_embedding(query: str, model, cache: dict) -> np.ndarray:
    """Embed ``query``, memoising the result in ``cache`` keyed by the raw string.

    The cache is mutated in place so callers share it across turns. Keys are
    used exactly as given -- "Hi" and "hi" are different entries.
    """
    if query in cache:
        return cache[query]

    vector = embed_text(model, query)
    cache[query] = vector

    return vector


def update_chat_memory(history: list, user_message: str, assistant_message: str) -> list:
    """Append the user and assistant turns, returning a new list.

    The caller's history is not mutated, so a caller that keeps the old list
    keeps the old conversation.
    """
    return history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]


def rewrite_followup(followup: str, history: list) -> str:
    """Stitch the most recent user turn onto a follow-up to make it standalone.

    "How big is it?" retrieves nothing useful because "it" has no referent.
    Prepending the previous *user* question restores it. The assistant's replies
    are skipped -- they are answers, not information needs.
    """
    previous = [turn["content"] for turn in history if turn.get("role") == "user"]

    if not previous:
        return normalize_text(followup)

    return normalize_text(f"{previous[-1]} {followup}")
